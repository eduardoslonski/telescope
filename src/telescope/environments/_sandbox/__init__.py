"""
Sandbox provider abstraction layer.

Usage::

    from telescope.environments._sandbox import get_provider, SandboxConfig

    provider = get_provider("daytona")  # or "prime", "modal", "e2b", etc.
    handle = await provider.create(SandboxConfig(image="debian:latest"))
    result = await provider.execute(handle, "echo hello")
    await provider.destroy(handle)
"""

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
from .pool import GenericSandboxPool

__all__ = [
    "ExecResult",
    "GenericSandboxPool",
    "SandboxCommandTimeoutError",
    "SandboxConfig",
    "SandboxError",
    "SandboxHandle",
    "SandboxImagePullError",
    "SandboxNotRunningError",
    "SandboxOOMError",
    "SandboxProvider",
    "get_provider",
]


_BUILTIN_PROVIDERS: dict[str, tuple[str, str]] = {
    "prime": (".prime", "PrimeSandboxProvider"),
    "modal": (".modal_provider", "ModalSandboxProvider"),
    "daytona": (".daytona", "DaytonaSandboxProvider"),
    "e2b": (".e2b_provider", "E2BSandboxProvider"),
}


def get_provider(name: str, **kwargs) -> SandboxProvider:
    """
    Factory function to get a sandbox provider by name.

    Built-in providers: ``"daytona"``, ``"prime"``, ``"modal"``, ``"e2b"``.

    Custom providers can be specified as a fully-qualified class name
    (e.g. ``"my_package.providers.MySandboxProvider"``).

    Args:
        name: Built-in provider name or fully-qualified class path.
        **kwargs: Provider-specific keyword arguments.

    Returns:
        A :class:`SandboxProvider` instance.
    """
    if name in _BUILTIN_PROVIDERS:
        module_path, class_name = _BUILTIN_PROVIDERS[name]
        import importlib

        mod = importlib.import_module(module_path, package=__package__)
        cls = getattr(mod, class_name)
        return cls(**kwargs)

    # Try loading as a fully-qualified class name (e.g. "my_pkg.MyProvider")
    if "." in name:
        import importlib

        module_path, class_name = name.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            return cls(**kwargs)
        except (ImportError, AttributeError) as exc:
            raise ValueError(
                f"Could not load sandbox provider '{name}': {exc}"
            ) from exc

    raise ValueError(
        f"Unknown sandbox provider '{name}'.\n"
        f"Built-in providers: {', '.join(sorted(_BUILTIN_PROVIDERS))}.\n"
        f"You can also pass a fully-qualified class name for custom providers."
    )
