"""
Abstract sandbox provider interface and shared types.

All sandbox providers (Prime, Modal, Daytona) implement the
:class:`SandboxProvider` ABC so that environment code can be written
against a single API regardless of the underlying infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ── Shared data types ─────────────────────────────────────────────────

@dataclass
class SandboxHandle:
    """Opaque handle returned by :meth:`SandboxProvider.create`.

    ``id`` is a human-readable identifier (sandbox ID or object_id).
    ``provider_data`` carries the native object (e.g. ``modal.Sandbox``,
    ``daytona.Sandbox``) for provider-specific operations.
    """

    id: str
    provider_data: Any = None


@dataclass
class SandboxConfig:
    """Provider-agnostic configuration for creating a sandbox."""

    image: str = ""
    cpu: float = 1.0
    memory_mb: int = 2048
    disk_size_gb: int = 3
    gpu_count: int = 0
    timeout_seconds: int = 3600  # 1 hour
    environment_vars: dict[str, str] = field(default_factory=dict)
    name: str = ""
    # Provider-specific overrides (e.g. team_id, labels, start_command)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecResult:
    """Result of executing a command in a sandbox."""

    exit_code: int
    stdout: str
    stderr: str


# ── Standard error hierarchy ──────────────────────────────────────────

class SandboxError(Exception):
    """Base class for all sandbox infrastructure errors."""


class SandboxCommandTimeoutError(SandboxError):
    """A command exceeded its per-command timeout."""

    def __init__(self, command: str = "", timeout: int = 0, sandbox_id: str = ""):
        self.command = command
        self.timeout = timeout
        self.sandbox_id = sandbox_id
        super().__init__(
            f"Command timed out after {timeout}s in sandbox {sandbox_id}: "
            f"{command[:120]}"
        )


class SandboxOOMError(SandboxError):
    """The sandbox ran out of memory."""


class SandboxNotRunningError(SandboxError):
    """The sandbox is no longer running (terminated / expired)."""


class SandboxImagePullError(SandboxError):
    """Failed to pull the Docker image for the sandbox."""


# ── Abstract provider ─────────────────────────────────────────────────

class SandboxProvider(ABC):
    """Abstract interface that all sandbox providers must implement."""

    @abstractmethod
    async def create(self, config: SandboxConfig) -> SandboxHandle:
        """Create a new sandbox and return a handle once it is ready."""

    @abstractmethod
    async def execute(
        self,
        handle: SandboxHandle,
        command: str,
        timeout: int = 60,
        working_dir: str | None = None,
    ) -> ExecResult:
        """Execute a shell command inside the sandbox."""

    @abstractmethod
    async def upload_bytes(
        self,
        handle: SandboxHandle,
        remote_path: str,
        content: bytes,
    ) -> None:
        """Upload raw bytes to *remote_path* inside the sandbox."""

    @abstractmethod
    async def upload_file(
        self,
        handle: SandboxHandle,
        remote_path: str,
        local_path: str,
    ) -> None:
        """Upload a local file to *remote_path* inside the sandbox."""

    @abstractmethod
    async def destroy(self, handle: SandboxHandle) -> None:
        """Destroy / terminate a single sandbox."""

    async def bulk_destroy(self, handles: list[SandboxHandle]) -> None:
        """Destroy multiple sandboxes.  Default: sequential destroy."""
        for h in handles:
            try:
                await self.destroy(h)
            except Exception:
                pass

    def destroy_sync(self, handle: SandboxHandle) -> None:
        """Synchronous destroy for atexit handlers.  Override if provider
        offers a native sync API; default is a no-op with a warning."""
        import logging

        logging.getLogger(__name__).warning(
            f"destroy_sync not implemented for {type(self).__name__}; "
            f"sandbox {handle.id} may leak"
        )

    def bulk_destroy_sync(self, handles: list[SandboxHandle]) -> None:
        """Synchronous bulk destroy for atexit handlers."""
        for h in handles:
            try:
                self.destroy_sync(h)
            except Exception:
                pass
