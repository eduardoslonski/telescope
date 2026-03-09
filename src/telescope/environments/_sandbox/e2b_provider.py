"""
E2B Sandbox provider.

Wraps the ``e2b`` SDK to implement :class:`SandboxProvider`.
The E2B SDK provides a natively async ``AsyncSandbox`` class.

Requires: ``uv add e2b``

Credentials:
  - ``E2B_API_KEY`` env var (read natively by the SDK), or
  - Pass ``api_key`` as a constructor kwarg.
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


class E2BSandboxProvider(SandboxProvider):
    """Sandbox provider backed by E2B."""

    def __init__(self, **kwargs: Any) -> None:
        try:
            import e2b  # noqa: F401
        except ImportError:
            raise ImportError(
                "e2b is required for the E2B sandbox provider. "
                "Install it with: uv add e2b"
            ) from None
        self._api_key: str | None = kwargs.get("api_key")
        self._template: str | None = kwargs.get("template")
        self._built_template: str | None = None  # lazily built from config.image

        # Validate credentials eagerly so missing API key fails at startup,
        # not in the background producer loop minutes later.
        import os
        from e2b.exceptions import AuthenticationException

        if not self._api_key and not os.environ.get("E2B_API_KEY"):
            raise AuthenticationException(
                "API key is required, please visit the Team tab at "
                "https://e2b.dev/dashboard to get your API key. "
                "You can either set the environment variable `E2B_API_KEY` "
                'or pass api_key in provider_kwargs.'
            )

    # ── template building ─────────────────────────────────────────────

    def _ensure_template(self, config: SandboxConfig) -> str | None:
        """Resolve the E2B template to use.

        Priority:
        1. Explicit template from config.extra or constructor kwarg
        2. Template built from config.image (Docker image), built once and cached
        3. None (E2B default)
        """
        # Explicit template always wins
        explicit = config.extra.get("template") or self._template
        if explicit:
            return explicit

        # Already built from image in a previous call
        if self._built_template is not None:
            return self._built_template

        # No Docker image to build from — use E2B default
        if not config.image:
            return None

        # Build an E2B template from the Docker image (one-time, cached
        # server-side so subsequent runs skip the build).
        from e2b import Template

        # Derive a stable template name from the image
        # e.g. "amancevice/pandas:slim" -> "telescope-amancevice-pandas-slim"
        template_name = "telescope-" + config.image.replace("/", "-").replace(":", "-")

        build_kwargs: dict[str, Any] = {}
        if self._api_key:
            build_kwargs["api_key"] = self._api_key

        logger.info(
            f"Building E2B template '{template_name}' from image '{config.image}' "
            f"(cached server-side, first build may take a few minutes)..."
        )
        template_def = Template().from_image(config.image)
        Template.build(template_def, template_name, **build_kwargs)
        logger.info(f"E2B template '{template_name}' ready")

        self._built_template = template_name
        return self._built_template

    # ── create ────────────────────────────────────────────────────────

    async def create(self, config: SandboxConfig) -> SandboxHandle:
        from e2b import AsyncSandbox

        template = self._ensure_template(config)

        create_kwargs: dict[str, Any] = {}
        if template:
            create_kwargs["template"] = template
        if self._api_key:
            create_kwargs["api_key"] = self._api_key
        if config.timeout_seconds:
            create_kwargs["timeout"] = config.timeout_seconds
        if config.environment_vars:
            create_kwargs["envs"] = config.environment_vars

        # Pass metadata with sandbox name for identification
        metadata: dict[str, str] = {}
        if config.name:
            metadata["name"] = config.name
        if config.extra.get("metadata"):
            metadata.update(config.extra["metadata"])
        if metadata:
            create_kwargs["metadata"] = metadata

        try:
            sandbox = await AsyncSandbox.create(**create_kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if "template" in error_str or "not found" in error_str:
                from .base import SandboxImagePullError

                raise SandboxImagePullError(str(e)) from e
            raise SandboxError(f"Failed to create E2B sandbox: {e!r}") from e

        return SandboxHandle(id=sandbox.sandbox_id, provider_data=sandbox)

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
            run_kwargs: dict[str, Any] = {"timeout": float(timeout)}
            if working_dir:
                run_kwargs["cwd"] = working_dir

            result = await sandbox.commands.run(command, **run_kwargs)
            return ExecResult(
                exit_code=result.exit_code,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            )
        except Exception as e:
            raise self._classify_error(e, command, timeout, handle.id) from e

    def _classify_error(
        self, error: Exception, command: str = "", timeout: int = 0, sandbox_id: str = "",
    ) -> SandboxError:
        """Classify an E2B exception into the standard error hierarchy."""
        error_str = str(error).lower()
        if "timeout" in error_str:
            return SandboxCommandTimeoutError(
                command=command, timeout=timeout, sandbox_id=sandbox_id,
            )
        if "oom" in error_str or "out of memory" in error_str or "killed" in error_str:
            return SandboxOOMError(str(error))
        if (
            "not running" in error_str
            or "not found" in error_str
            or "terminated" in error_str
            or "expired" in error_str
        ):
            return SandboxNotRunningError(str(error))
        return SandboxError(str(error))

    # ── upload_bytes ──────────────────────────────────────────────────

    async def upload_bytes(
        self,
        handle: SandboxHandle,
        remote_path: str,
        content: bytes,
    ) -> None:
        sandbox = handle.provider_data
        try:
            await sandbox.files.write(remote_path, content)
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
        sandbox = handle.provider_data
        if sandbox is None:
            return
        try:
            await sandbox.kill()
        except Exception as e:
            logger.warning(f"Failed to kill E2B sandbox {handle.id}: {e!r}")

    async def bulk_destroy(self, handles: list[SandboxHandle]) -> None:
        for h in handles:
            try:
                await self.destroy(h)
            except Exception:
                pass

    def destroy_sync(self, handle: SandboxHandle) -> None:
        from e2b import Sandbox

        try:
            Sandbox.kill(
                sandbox_id=handle.id,
                api_key=self._api_key,
            )
        except Exception as e:
            logger.warning(f"Failed to kill E2B sandbox {handle.id}: {e!r}")

    def bulk_destroy_sync(self, handles: list[SandboxHandle]) -> None:
        for h in handles:
            try:
                self.destroy_sync(h)
            except Exception:
                pass
