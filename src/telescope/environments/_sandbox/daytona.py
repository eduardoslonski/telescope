"""
Daytona Sandbox provider.

Wraps the ``daytona_sdk`` to implement :class:`SandboxProvider`.
The Daytona SDK is natively async, so no executor threads are needed.

Requires: ``uv add daytona-sdk``

Credentials:
  - ``DAYTONA_API_KEY`` / ``DAYTONA_API_URL`` / ``DAYTONA_TARGET`` env vars
    (read natively by the SDK), or
  - Pass ``api_key`` / ``api_url`` / ``target`` as constructor kwargs.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    ExecResult,
    SandboxCommandTimeoutError,
    SandboxConfig,
    SandboxError,
    SandboxHandle,
    SandboxNotRunningError,
    SandboxOOMError,
    SandboxProvider,
)

logger = logging.getLogger(__name__)


class DaytonaSandboxProvider(SandboxProvider):
    """Sandbox provider backed by Daytona."""

    def __init__(self, **kwargs: Any) -> None:
        try:
            from daytona_sdk import AsyncDaytona, DaytonaConfig
        except ImportError:
            raise ImportError(
                "daytona_sdk is required for the Daytona sandbox provider. "
                "Install it with: uv add daytona-sdk"
            ) from None

        config_kwargs = {}
        for key in ("api_key", "api_url", "target"):
            if key in kwargs:
                config_kwargs[key] = kwargs[key]

        self._config_kwargs = config_kwargs

        # Validate credentials eagerly so missing API key fails at startup,
        # not in the background producer loop minutes later.
        # (The actual AsyncDaytona client must be created lazily inside the
        # producer thread's event loop — aiohttp has loop affinity.)
        import os
        has_key = (
            config_kwargs.get("api_key")
            or os.environ.get("DAYTONA_API_KEY")
            or os.environ.get("DAYTONA_JWT_TOKEN")
        )
        if not has_key:
            from daytona_sdk import DaytonaError
            raise DaytonaError(
                "API key or JWT token is required. "
                "Set DAYTONA_API_KEY env var or pass api_key in provider_kwargs."
            )

        self._client = None  # lazily initialized in producer's event loop

    async def _get_client(self):
        """Lazily create the Daytona async client (must run in the event loop that will use it)."""
        if self._client is None:
            from daytona_sdk import AsyncDaytona, DaytonaConfig

            if self._config_kwargs:
                config = DaytonaConfig(**self._config_kwargs)
                self._client = AsyncDaytona(config)
            else:
                self._client = AsyncDaytona()
        return self._client

    # ── create ────────────────────────────────────────────────────────

    async def create(self, config: SandboxConfig) -> SandboxHandle:
        from daytona_sdk import CreateSandboxFromImageParams, Resources

        client = await self._get_client()
        params = CreateSandboxFromImageParams(
            image=config.image,
            resources=Resources(
                cpu=int(config.cpu),
                memory=max(1, config.memory_mb // 1024),  # GiB
                disk=config.disk_size_gb,
                gpu=config.gpu_count or None,
            ),
            env_vars=config.environment_vars or {},
            auto_stop_interval=0,  # disable auto-stop for training sandboxes
        )
        sandbox = await client.create(params, timeout=300)
        return SandboxHandle(id=sandbox.id, provider_data=sandbox)

    # ── execute ───────────────────────────────────────────────────────

    async def execute(
        self,
        handle: SandboxHandle,
        command: str,
        timeout: int = 60,
        working_dir: str | None = None,
    ) -> ExecResult:
        sandbox = handle.provider_data
        try:
            resp = await sandbox.process.exec(
                command, cwd=working_dir, timeout=timeout,
            )
            return ExecResult(
                exit_code=resp.exit_code,
                stdout=resp.result or "",
                stderr="",
            )
        except Exception as e:
            error_str = str(e).lower()
            if "timeout" in error_str:
                raise SandboxCommandTimeoutError(
                    command=command, timeout=timeout, sandbox_id=handle.id,
                ) from e
            if "oom" in error_str or "memory" in error_str:
                raise SandboxOOMError(str(e)) from e
            if "not running" in error_str or "terminated" in error_str:
                raise SandboxNotRunningError(str(e)) from e
            raise SandboxError(str(e)) from e

    # ── upload_bytes ──────────────────────────────────────────────────

    async def upload_bytes(
        self,
        handle: SandboxHandle,
        remote_path: str,
        content: bytes,
    ) -> None:
        sandbox = handle.provider_data
        try:
            await sandbox.fs.upload_file(content, remote_path)
        except Exception as e:
            raise SandboxError(
                f"Failed to upload to {remote_path}: {e!r}"
            ) from e

    # ── upload_file ───────────────────────────────────────────────────

    async def upload_file(
        self,
        handle: SandboxHandle,
        remote_path: str,
        local_path: str,
    ) -> None:
        with open(local_path, "rb") as f:
            content = f.read()
        await self.upload_bytes(handle, remote_path, content)

    # ── destroy ───────────────────────────────────────────────────────

    async def destroy(self, handle: SandboxHandle) -> None:
        client = await self._get_client()
        sandbox = handle.provider_data
        try:
            await client.delete(sandbox)
        except Exception as e:
            logger.warning(f"Failed to delete Daytona sandbox {handle.id}: {e!r}")

    async def bulk_destroy(self, handles: list[SandboxHandle]) -> None:
        for h in handles:
            try:
                await self.destroy(h)
            except Exception:
                pass

    def _get_sync_client(self):
        """Lazily create a sync Daytona client for shutdown cleanup."""
        if not hasattr(self, "_sync_client") or self._sync_client is None:
            from daytona_sdk import Daytona, DaytonaConfig

            if self._config_kwargs:
                config = DaytonaConfig(**self._config_kwargs)
                self._sync_client = Daytona(config)
            else:
                self._sync_client = Daytona()
        return self._sync_client

    def destroy_sync(self, handle: SandboxHandle) -> None:
        try:
            client = self._get_sync_client()
            sandbox = client.get(handle.id)
            sandbox.delete(timeout=30)
        except Exception as e:
            logger.warning(f"Failed to delete Daytona sandbox {handle.id}: {e!r}")

    def bulk_destroy_sync(self, handles: list[SandboxHandle]) -> None:
        for h in handles:
            try:
                self.destroy_sync(h)
            except Exception:
                pass
