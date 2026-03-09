"""
Auto-discovery eval loader.

Evals are discovered automatically from subfolders of the
``telescope/evals/`` package.  Each subfolder must contain an ``eval.py``
with a concrete :class:`Eval` subclass.

Resolution order used by :func:`resolve`:
    1. Check for an eval folder under ``telescope/evals/<name>/eval.py``
    2. Fall back to ``telescope/environments/<name>/environment.py``

This lets ``config.cfg.evals`` reference either dedicated evals *or* plain
environments by the same ``"name"`` key.

Usage::

    from telescope.evals import resolve, list_evals

    source = resolve("aime")           # evals/aime/eval.py  (dedicated eval)
    source = resolve("countdown")      # environments/countdown/environment.py  (no eval override)
"""

import importlib
import inspect
import logging
from pathlib import Path

from telescope.evals.base import Eval


logger = logging.getLogger(__name__)

_CLASS_CACHE: dict[str, type[Eval]] = {}


_BASE_CLASSES = frozenset({"Eval"})


def _find_eval_class(mod) -> type[Eval] | None:
    """Find the concrete Eval subclass in a module."""
    for attr_name in dir(mod):
        if attr_name.startswith("_"):
            continue
        attr = getattr(mod, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, Eval)
            and attr.__name__ not in _BASE_CLASSES
            and not inspect.isabstract(attr)
        ):
            return attr
    return None


def _load_eval_class(name: str) -> type[Eval]:
    """Import ``telescope.evals.<name>.eval`` and return the Eval subclass."""
    fqn = f"telescope.evals.{name}.eval"
    try:
        mod = importlib.import_module(fqn)
    except ImportError as exc:
        available = list_evals()
        raise ValueError(
            f"Eval '{name}' not found. "
            f"Available evals: {available}"
        ) from exc

    cls = _find_eval_class(mod)
    if cls is None:
        raise ValueError(
            f"No concrete Eval subclass found in {fqn}"
        )
    return cls


def get_eval(name: str, **kwargs) -> Eval:
    """
    Load an eval by folder name.

    The *name* must match a subfolder under ``telescope/evals/``
    exactly (e.g. ``"aime"``, ``"countdown_hard"``).
    """
    if name not in _CLASS_CACHE:
        _CLASS_CACHE[name] = _load_eval_class(name)
    return _CLASS_CACHE[name](**kwargs)


def list_evals() -> list[str]:
    """Return sorted list of eval folder names."""
    eval_dir = Path(__file__).parent
    return sorted(
        item.name
        for item in eval_dir.iterdir()
        if item.is_dir()
        and (item / "eval.py").exists()
        and item.name != "__pycache__"
    )


def resolve(name: str, **kwargs):
    """
    Load an eval or environment by name.

    Tries the eval registry first.  If no dedicated eval exists, falls back
    to the environment registry so that plain ``"countdown"`` still works
    in ``config.cfg.evals``.

    Returns either an :class:`Eval` or an :class:`~telescope.environments.base.Environment`.
    """
    if name in list_evals():
        return get_eval(name, **kwargs)

    from telescope.environments import get_environment
    return get_environment(name, **kwargs)
