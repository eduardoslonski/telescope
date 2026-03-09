"""
Prime Sandbox provider.

Wraps the ``prime_sandboxes`` SDK to implement :class:`SandboxProvider`.

Requires: ``uv add prime-sandboxes``
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .base import (
    ExecResult,
    SandboxCommandTimeoutError,
    SandboxConfig,
    SandboxError,
    SandboxHandle,
    SandboxImagePullError,
    SandboxNotRunningError,
    SandboxOOMError,
    SandboxProvider,
)

logger = logging.getLogger(__name__)

# Timeout for individual sandbox API calls (upload/execute).
_SANDBOX_API_TIMEOUT = 60.0  # seconds


class PrimeSandboxProvider(SandboxProvider):
    """Sandbox provider backed by Prime Sandboxes."""

    def __init__(self, **kwargs: Any) -> None:
        try:
            from prime_sandboxes import AsyncSandboxClient
        except ImportError:
            raise ImportError(
                "prime_sandboxes is required for the Prime sandbox provider. "
                "Install it with: uv add prime-sandboxes"
            ) from None

        client_kwargs: dict[str, Any] = {
            "max_connections": kwargs.get("max_connections", 500),
            "max_keepalive_connections": kwargs.get("max_keepalive_connections", 250),
        }
        if "api_key" in kwargs:
            client_kwargs["api_key"] = kwargs["api_key"]

        # Validate credentials eagerly so missing API key fails at startup,
        # not in the background producer loop minutes later.
        import os
        from pathlib import Path

        has_key = (
            kwargs.get("api_key")
            or os.environ.get("PRIME_API_KEY")
        )
        if not has_key:
            # Check ~/.prime/config.json (written by `prime login`)
            config_file = Path.home() / ".prime" / "config.json"
            if config_file.exists():
                import json
                try:
                    has_key = bool(json.loads(config_file.read_text()).get("api_key"))
                except Exception:
                    pass
        if not has_key:
            from prime_sandboxes.core import APIError
            raise APIError(
                "No API key configured for Prime. Either:\n"
                "  1. Set PRIME_API_KEY environment variable, or\n"
                "  2. Run `prime login` to authenticate via CLI, or\n"
                "  3. Pass api_key in provider_kwargs in your YAML config."
            )

        self._client_kwargs = client_kwargs
        # Lazily created per event loop — AsyncSandboxClient uses httpx/asyncio
        # internals that have loop affinity, so sharing across loops fails with
        # "bound to a different event loop".
        self._clients: dict[int, AsyncSandboxClient] = {}

    def _get_client(self):
        """Get or create an AsyncSandboxClient for the current event loop."""
        from prime_sandboxes import AsyncSandboxClient

        loop_id = id(asyncio.get_running_loop())
        client = self._clients.get(loop_id)
        if client is None:
            client = AsyncSandboxClient(**self._client_kwargs)
            self._clients[loop_id] = client
        return client

    # ── create ────────────────────────────────────────────────────────

    async def create(self, config: SandboxConfig) -> SandboxHandle:
        from prime_sandboxes import (
            CreateSandboxRequest,
            SandboxImagePullError as PrimeImagePull,
            SandboxOOMError as PrimeOOM,
            SandboxTimeoutError as PrimeTimeout,
        )

        request = CreateSandboxRequest(
            name=config.name or "telescope-sandbox",
            docker_image=config.image,
            start_command=config.extra.get("start_command", "tail -f /dev/null"),
            cpu_cores=int(config.cpu),
            memory_gb=max(1, config.memory_mb // 1024),
            disk_size_gb=config.disk_size_gb,
            gpu_count=config.gpu_count,
            timeout_minutes=max(1, config.timeout_seconds // 60),
            environment_vars=config.environment_vars or None,
            team_id=config.extra.get("team_id"),
            labels=config.extra.get("labels") or [],
        )

        try:
            client = self._get_client()
            sandbox = await client.create(request)
            sandbox_id = sandbox.id
            await client.wait_for_creation(sandbox_id)
            return SandboxHandle(id=sandbox_id, provider_data=None)
        except PrimeImagePull as e:
            raise SandboxImagePullError(str(e)) from e
        except PrimeOOM as e:
            raise SandboxOOMError(str(e)) from e
        except PrimeTimeout as e:
            raise SandboxNotRunningError(str(e)) from e

    # ── execute ───────────────────────────────────────────────────────

    async def execute(
        self,
        handle: SandboxHandle,
        command: str,
        timeout: int = 60,
        working_dir: str | None = None,
    ) -> ExecResult:
        from prime_sandboxes import (
            CommandTimeoutError as PrimeCommandTimeout,
            SandboxNotRunningError as PrimeNotRunning,
            SandboxOOMError as PrimeOOM,
            SandboxTimeoutError as PrimeTimeout,
            SandboxImagePullError as PrimeImagePull,
        )

        try:
            resp = await self._get_client().execute_command(
                sandbox_id=handle.id,
                command=command,
                timeout=timeout,
                working_dir=working_dir,
            )
            return ExecResult(
                exit_code=resp.exit_code,
                stdout=resp.stdout or "",
                stderr=resp.stderr or "",
            )
        except PrimeCommandTimeout:
            raise SandboxCommandTimeoutError(
                command=command, timeout=timeout, sandbox_id=handle.id,
            )
        except PrimeOOM as e:
            raise SandboxOOMError(str(e)) from e
        except PrimeTimeout as e:
            raise SandboxNotRunningError(str(e)) from e
        except PrimeNotRunning as e:
            raise SandboxNotRunningError(str(e)) from e
        except PrimeImagePull as e:
            raise SandboxImagePullError(str(e)) from e

    # ── upload_bytes ──────────────────────────────────────────────────

    async def upload_bytes(
        self,
        handle: SandboxHandle,
        remote_path: str,
        content: bytes,
    ) -> None:
        try:
            await asyncio.wait_for(
                self._get_client().upload_bytes(
                    sandbox_id=handle.id,
                    file_path=remote_path,
                    file_bytes=content,
                    filename=remote_path.split("/")[-1],
                ),
                timeout=_SANDBOX_API_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise SandboxError(
                f"Timeout uploading to {remote_path} in sandbox {handle.id}"
            )

    # ── upload_file ───────────────────────────────────────────────────

    async def upload_file(
        self,
        handle: SandboxHandle,
        remote_path: str,
        local_path: str,
    ) -> None:
        await self._get_client().upload_file(handle.id, remote_path, local_path)

    # ── destroy ───────────────────────────────────────────────────────

    async def destroy(self, handle: SandboxHandle) -> None:
        try:
            await asyncio.wait_for(
                self._get_client().delete(handle.id), timeout=10.0,
            )
        except Exception as e:
            logger.warning(f"Failed to delete sandbox {handle.id}: {e!r}")

    async def bulk_destroy(self, handles: list[SandboxHandle]) -> None:
        ids = [h.id for h in handles]
        if ids:
            try:
                await self._get_client().bulk_delete(sandbox_ids=ids)
            except Exception as e:
                logger.warning(f"Error bulk deleting sandboxes: {e!r}")

    def destroy_sync(self, handle: SandboxHandle) -> None:
        from prime_sandboxes import APIClient, SandboxClient

        try:
            SandboxClient(APIClient()).delete(handle.id)
        except Exception as e:
            logger.warning(f"Sync delete failed for {handle.id}: {e!r}")

    def bulk_destroy_sync(self, handles: list[SandboxHandle]) -> None:
        from prime_sandboxes import APIClient, SandboxClient

        ids = [h.id for h in handles]
        if ids:
            try:
                SandboxClient(APIClient()).bulk_delete(sandbox_ids=ids)
            except Exception as e:
                logger.warning(f"Sync bulk delete failed: {e!r}")

    # ── Background job helpers (Prime-specific) ───────────────────────

    async def start_background_job(
        self, handle: SandboxHandle, command: str, working_dir: str | None = None
    ) -> Any:
        """Start a background job (Prime Sandbox specific)."""
        from prime_sandboxes import (
            CommandTimeoutError as PrimeCommandTimeout,
            SandboxOOMError as PrimeOOM,
            SandboxTimeoutError as PrimeTimeout,
        )

        try:
            return await self._get_client().start_background_job(
                sandbox_id=handle.id, command=command, working_dir=working_dir,
            )
        except PrimeOOM as e:
            raise SandboxOOMError(str(e)) from e
        except PrimeTimeout as e:
            raise SandboxNotRunningError(str(e)) from e
        except PrimeCommandTimeout as e:
            raise SandboxCommandTimeoutError(
                command=command, sandbox_id=handle.id,
            ) from e

    async def get_background_job(self, handle: SandboxHandle, job: Any) -> Any:
        """Poll a background job (Prime Sandbox specific)."""
        from prime_sandboxes import (
            CommandTimeoutError as PrimeCommandTimeout,
            SandboxOOMError as PrimeOOM,
            SandboxTimeoutError as PrimeTimeout,
        )

        try:
            return await self._get_client().get_background_job(handle.id, job)
        except PrimeOOM as e:
            raise SandboxOOMError(str(e)) from e
        except PrimeTimeout as e:
            raise SandboxNotRunningError(str(e)) from e
        except PrimeCommandTimeout as e:
            raise SandboxCommandTimeoutError(sandbox_id=handle.id) from e
