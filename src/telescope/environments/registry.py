"""
Auto-discovery environment loader.

Environments are discovered automatically from subfolders of the
``telescope/environments/`` package.  Each subfolder must contain an
``environment.py`` with a concrete :class:`Environment` subclass.

To add a new environment, create a folder under ``environments/``
with an ``environment.py`` — no registration or ``__init__.py`` needed.

Usage::

    from telescope.environments import get_environment

    env = get_environment("countdown")           # environments/countdown/environment.py
    env = get_environment("hendrycks_math")       # environments/hendrycks_math/environment.py
"""

import ast
import importlib
import importlib.util
import inspect
import logging
from pathlib import Path

from telescope.environments.base import Environment


logger = logging.getLogger(__name__)

_CLASS_CACHE: dict[str, type[Environment]] = {}

_BASE_CLASSES = frozenset({
    "Environment",
    "SingleTurnEnvironment",
    "MultiTurnEnvironment",
    "ToolEnvironment",
})


def _find_env_class(mod) -> type[Environment] | None:
    """Find the concrete Environment subclass in a module."""
    for attr_name in dir(mod):
        if attr_name.startswith("_"):
            continue
        attr = getattr(mod, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, Environment)
            and attr.__name__ not in _BASE_CLASSES
            and not inspect.isabstract(attr)
        ):
            return attr
    return None


def _read_packages(env_path: Path) -> tuple[list[str], list[str]]:
    """Read REQUIRED_PACKAGES and OPTIONAL_PACKAGES from an environment.py without importing it.

    Uses AST parsing so it works even when the module's dependencies are missing.
    """
    required: list[str] = []
    optional: list[str] = []
    try:
        tree = ast.parse(env_path.read_text())
    except SyntaxError:
        return required, optional
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.List):
                values = [
                    elt.value for elt in node.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
                if target.id == "REQUIRED_PACKAGES":
                    required = values
                elif target.id == "OPTIONAL_PACKAGES":
                    optional = values
    return required, optional


def _check_required_packages(name: str) -> None:
    """Check REQUIRED_PACKAGES are installed *before* importing the environment module.

    This catches missing dependencies even when the environment defers imports
    (e.g. inside functions), which would otherwise let the module import succeed
    and only fail at runtime.
    """
    env_path = Path(__file__).parent / name / "environment.py"
    if not env_path.exists():
        return
    required, optional = _read_packages(env_path)
    if not required:
        return
    # PyPI package names use hyphens, but Python import names use underscores
    missing = [
        pkg for pkg in required
        if importlib.util.find_spec(pkg.replace("-", "_")) is None
    ]
    if missing:
        install_cmd = f"uv add {' '.join(missing)}"
        msg = (
            f"Environment '{name}' requires optional dependencies: {missing}\n"
            f"Install with: {install_cmd}"
        )
        if optional:
            msg += (
                f"\n\nFor optional sandbox providers, also install: "
                f"uv add {' '.join(optional)}"
            )
        raise ImportError(msg)


def _load_env_class(name: str) -> type[Environment]:
    """Import ``telescope.environments.<name>.environment`` and return the Environment subclass."""
    # Check required packages before importing so we get a clear error even when
    # the environment defers its imports (lazy imports inside functions).
    available = list_environments()
    if name not in available:
        raise ValueError(
            f"Environment '{name}' not found. "
            f"Available: {available}"
        )
    _check_required_packages(name)

    fqn = f"telescope.environments.{name}.environment"
    try:
        mod = importlib.import_module(fqn)
    except ImportError as exc:
        env_path = Path(__file__).parent / name / "environment.py"
        required, optional = _read_packages(env_path)
        if required:
            install_cmd = f"uv add {' '.join(required)}"
            msg = (
                f"Environment '{name}' requires optional dependencies.\n"
                f"Install with: {install_cmd}"
            )
            if optional:
                msg += (
                    f"\n\nFor optional sandbox providers, also install: "
                    f"uv add {' '.join(optional)}"
                )
            raise ImportError(msg) from exc
        raise ImportError(
            f"Environment '{name}' failed to import: {exc}\n"
            f"Check that all required dependencies are installed."
        ) from exc

    cls = _find_env_class(mod)
    if cls is None:
        raise ValueError(
            f"No concrete Environment subclass found in {fqn}"
        )
    return cls


def get_environment(name: str, **kwargs) -> Environment:
    """
    Load an environment by folder name.

    The *name* must match a subfolder under ``telescope/environments/``
    exactly (e.g. ``"countdown"``, ``"hendrycks_math"``).
    """
    if name not in _CLASS_CACHE:
        _CLASS_CACHE[name] = _load_env_class(name)

    env = _CLASS_CACHE[name](**kwargs)
    # Set the name from the folder if the subclass didn't override it
    if env._name is None:
        env._name = name
    return env


def list_environments() -> list[str]:
    """Return sorted list of environment folder names."""
    env_dir = Path(__file__).parent
    return sorted(
        item.name
        for item in env_dir.iterdir()
        if item.is_dir()
        and (item / "environment.py").exists()
        and item.name != "__pycache__"
    )


def check_environments() -> dict[str, dict[str, object]]:
    """Check which environments are available and which have missing dependencies.

    Returns a dict mapping environment name to its status::

        {
            "countdown": {"available": True},
            "i3_code":   {"available": False, "error": "...", "install": "uv add prime-sandboxes"},
        }

    Also prints a human-readable summary to stdout.
    """
    results: dict[str, dict[str, object]] = {}

    for name in list_environments():
        fqn = f"telescope.environments.{name}.environment"
        try:
            _check_required_packages(name)
            mod = importlib.import_module(fqn)
            cls = _find_env_class(mod)
            if cls is None:
                results[name] = {
                    "available": False,
                    "error": f"No concrete Environment subclass in {fqn}",
                }
            else:
                results[name] = {"available": True}
        except ImportError as exc:
            entry: dict[str, object] = {"available": False, "error": str(exc)}
            env_path = Path(__file__).parent / name / "environment.py"
            required, _ = _read_packages(env_path)
            if required:
                entry["install"] = f"uv add {' '.join(required)}"
            results[name] = entry

    # Print summary
    print(f"\n{'Environment':<25} {'Status':<12} {'Install command'}")
    print("-" * 70)
    for name, info in sorted(results.items()):
        if info["available"]:
            status = "OK"
            install = ""
        else:
            status = "MISSING"
            install = info.get("install", "")
        print(f"{name:<25} {status:<12} {install}")
    print()

    return results
