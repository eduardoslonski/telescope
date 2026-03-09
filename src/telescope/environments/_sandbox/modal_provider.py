"""
Modal Sandbox provider.

Wraps the ``modal`` SDK to implement :class:`SandboxProvider`.
Modal's API is synchronous, so all calls are run via ``run_in_executor``.

Requires: ``uv add modal``
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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

# Thread pool for Modal sync API calls (exec, wait, read)
_MODAL_EXECUTOR = ThreadPoolExecutor(max_workers=500, thread_name_prefix="modal-ops")


# ── Sync helpers (run in executor threads) ────────────────────────────

def _create_sandbox_sync(
    app, image, timeout_seconds: int, cpu: float, memory_mb: int,
):
    """Create and warm up a Modal sandbox (sync)."""
    import modal

    sb = modal.Sandbox.create(
        app=app, image=image, timeout=timeout_seconds,
        cpu=cpu, memory=memory_mb,
    )
    # Warmup: ensure the container is actually running
    p = sb.exec("echo", "ready")
    p.wait()
    return sb


def _exec_command_sync(sandbox, command: str, timeout: int, working_dir: str | None = None):
    """Execute a bash command in a Modal sandbox (sync)."""
    kwargs: dict[str, Any] = {"timeout": timeout}
    if working_dir:
        kwargs["workdir"] = working_dir
    p = sandbox.exec("bash", "-c", command, **kwargs)
    stdout = p.stdout.read()
    stderr = p.stderr.read()
    p.wait()
    return ExecResult(
        exit_code=p.returncode if p.returncode is not None else -1,
        stdout=stdout,
        stderr=stderr,
    )


def _upload_bytes_sync(sandbox, remote_path: str, content: bytes) -> None:
    """Upload bytes to a file in the sandbox via exec stdin piping."""
    parent = str(Path(remote_path).parent)
    p = sandbox.exec("bash", "-c", f"mkdir -p {parent}", timeout=30)
    p.wait()

    p = sandbox.exec("bash", "-c", f"cat > {remote_path}", timeout=120)
    p.stdin.write(content)
    p.stdin.write_eof()
    p.stdin.drain()
    p.wait()
    if p.returncode != 0:
        stderr_out = p.stderr.read()
        raise RuntimeError(
            f"Failed to upload to {remote_path} "
            f"(exit={p.returncode}): {stderr_out[:500]}"
        )


def _upload_file_sync(sandbox, remote_path: str, local_path: str) -> None:
    """Upload a local file to the sandbox."""
    content = Path(local_path).read_bytes()
    _upload_bytes_sync(sandbox, remote_path, content)


def _terminate_sandbox_sync(sandbox) -> None:
    """Terminate a Modal sandbox (sync)."""
    try:
        sandbox.terminate()
    except Exception:
        pass


# ── Provider class ────────────────────────────────────────────────────

class ModalSandboxProvider(SandboxProvider):
    """Sandbox provider backed by Modal."""

    def __init__(self, **kwargs: Any) -> None:
        try:
            import modal
        except ImportError:
            raise ImportError(
                "modal is required for the Modal sandbox provider. "
                "Install it with: uv add modal"
            ) from None

        app_name = kwargs.get("modal_app_name", "telescope-modal")
        self._app = modal.App.lookup(app_name, create_if_missing=True)
        self._default_image = kwargs.get("image")  # may be None

    def _get_image(self, config: SandboxConfig):
        """Resolve the Modal image for a given config."""
        import modal

        # If a pre-built image was passed, use it
        if self._default_image is not None and not config.image:
            return self._default_image

        # Build from registry if image is specified
        if config.image:
            return modal.Image.from_registry(config.image)

        # Fallback default
        return modal.Image.debian_slim(python_version="3.11").pip_install(
            "numpy", "pandas"
        )

    # ── create ────────────────────────────────────────────────────────

    async def create(self, config: SandboxConfig) -> SandboxHandle:
        loop = asyncio.get_running_loop()
        image = self._get_image(config)

        try:
            sb = await loop.run_in_executor(
                _MODAL_EXECUTOR,
                _create_sandbox_sync,
                self._app,
                image,
                config.timeout_seconds,
                config.cpu,
                config.memory_mb,
            )
        except Exception as e:
            error_str = str(e).lower()
            if "pull" in error_str or "not found" in error_str:
                from .base import SandboxImagePullError
                raise SandboxImagePullError(str(e)) from e
            raise
        return SandboxHandle(id=sb.object_id, provider_data=sb)

    # ── execute ───────────────────────────────────────────────────────

    async def execute(
        self,
        handle: SandboxHandle,
        command: str,
        timeout: int = 60,
        working_dir: str | None = None,
    ) -> ExecResult:
        sandbox = handle.provider_data
        loop = asyncio.get_running_loop()

        try:
            result = await loop.run_in_executor(
                _MODAL_EXECUTOR,
                _exec_command_sync,
                sandbox,
                command,
                timeout,
                working_dir,
            )
            return result
        except Exception as e:
            classified = self._classify_error(e)
            raise classified from e

    def _classify_error(self, error: Exception) -> Exception:
        """Classify a Modal exception into standard error types."""
        error_str = str(error).lower()
        if "oom" in error_str or "out of memory" in error_str or "killed" in error_str:
            return SandboxOOMError(str(error))
        if "terminated" in error_str or "lifetime" in error_str:
            return SandboxNotRunningError(str(error))
        if "timeout" in error_str:
            return SandboxCommandTimeoutError(command="", timeout=0, sandbox_id="")
        return SandboxError(str(error))

    # ── upload_bytes ──────────────────────────────────────────────────

    async def upload_bytes(
        self,
        handle: SandboxHandle,
        remote_path: str,
        content: bytes,
    ) -> None:
        sandbox = handle.provider_data
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _MODAL_EXECUTOR, _upload_bytes_sync, sandbox, remote_path, content,
        )

    # ── upload_file ───────────────────────────────────────────────────

    async def upload_file(
        self,
        handle: SandboxHandle,
        remote_path: str,
        local_path: str,
    ) -> None:
        sandbox = handle.provider_data
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _MODAL_EXECUTOR, _upload_file_sync, sandbox, remote_path, local_path,
        )

    # ── destroy ───────────────────────────────────────────────────────

    async def destroy(self, handle: SandboxHandle) -> None:
        sandbox = handle.provider_data
        if sandbox is None:
            return
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                _MODAL_EXECUTOR, _terminate_sandbox_sync, sandbox,
            )
        except Exception as e:
            logger.warning(f"Failed to terminate sandbox {handle.id}: {e!r}")

    async def bulk_destroy(self, handles: list[SandboxHandle]) -> None:
        for h in handles:
            try:
                await self.destroy(h)
            except Exception:
                pass

    def destroy_sync(self, handle: SandboxHandle) -> None:
        sandbox = handle.provider_data
        if sandbox is not None:
            _terminate_sandbox_sync(sandbox)

    def bulk_destroy_sync(self, handles: list[SandboxHandle]) -> None:
        for h in handles:
            self.destroy_sync(h)
