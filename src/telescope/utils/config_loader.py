"""Three-layer config loading: defaults/default_train.yaml -> run config -> CLI overrides.

``--config`` takes a path (relative to cwd or absolute).  Without
``--config``, ``configs/run.yaml`` is used if it exists, otherwise
only ``configs/defaults/default_train.yaml`` is used.

Usage from ``train.py``::

    from telescope.utils.config_loader import parse_args_and_load
    cfg = parse_args_and_load()          # returns TelescopeConfig
    from telescope.utils import config
    config._cfg = cfg                     # install singleton

Or programmatically::

    cfg = load_config(run_yaml="configs/run.yaml",
                      cli_overrides=[("learning_rate", "5e-7")])
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from telescope.utils.config_schema import TelescopeConfig


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _PROJECT_ROOT / "configs"
DEFAULT_YAML = _CONFIG_DIR / "defaults" / "default_train.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins).

    Lists are replaced wholesale (not concatenated).
    """
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _coerce_value(raw: str) -> Any:
    """Best-effort type coercion for CLI values.

    Tries (in order): JSON literal, int, float, then falls back to string.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _set_dotpath(data: dict, dotpath: str, raw_value: str) -> None:
    """Set a nested dict value via dot-separated path with type coercion."""
    parts = dotpath.split(".")
    d = data
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = _coerce_value(raw_value)


def _resolve_flag(flag: str) -> str:
    """Resolve a CLI flag to a config key.

    - Dashes become underscores (``--learning-rate`` → ``learning_rate``)
    - Validated against TelescopeConfig fields
    """
    canonical = flag.replace("-", "_").lower()
    if canonical in TelescopeConfig.model_fields:
        return canonical
    raise SystemExit(f"error: unknown flag --{flag}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _resolve_config_path(run_yaml: str | Path | None) -> Path | None:
    """Resolve the run config path.

    - If *run_yaml* is given, use it as-is (relative to cwd or absolute).
    - If *run_yaml* is ``None``, try ``configs/run.yaml`` automatically.
    - Returns ``None`` when no run config is found.
    """
    if run_yaml is not None:
        path = Path(run_yaml)
        if not path.exists():
            raise SystemExit(f"error: config file not found: {path}")
        return path.resolve()
    # No --config flag: try configs/run.yaml by default
    default_run = _CONFIG_DIR / "run.yaml"
    if default_run.exists():
        return default_run
    return None


def load_config(
    run_yaml: str | Path | None = None,
    cli_overrides: list[tuple[str, str]] | None = None,
) -> TelescopeConfig:
    """Load config with 3-layer merge: defaults/default_train.yaml -> run.yaml -> CLI overrides.

    Returns a validated ``TelescopeConfig`` instance.
    """
    # Layer 1: defaults
    with open(DEFAULT_YAML) as f:
        data = yaml.safe_load(f) or {}

    # Layer 2: run overrides
    resolved = _resolve_config_path(run_yaml)
    if resolved is not None:
        print(f"[config] Using run config: {resolved}")
        with open(resolved) as f:
            run_data = yaml.safe_load(f) or {}
        data = _deep_merge(data, run_data)
    else:
        print(f"[config] No run config found, using {DEFAULT_YAML} only")

    # Layer 3: CLI overrides
    for key, raw_value in (cli_overrides or []):
        _set_dotpath(data, key, raw_value)

    return TelescopeConfig(**data)


def _parse_extra_args(remaining: list[str]) -> list[tuple[str, str]]:
    """Parse unknown args as ``--flag value`` pairs.

    Supports flat names (``--learning_rate 5e-7``), dashed names
    (``--learning-rate 5e-7``), and ``=`` syntax (``--model=Qwen3``).
    """
    overrides: list[tuple[str, str]] = []
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if not arg.startswith("--"):
            raise SystemExit(f"error: unexpected argument: {arg}")
        arg = arg[2:]  # strip --
        if "=" in arg:
            key, value = arg.split("=", 1)
            overrides.append((_resolve_flag(key), value))
        else:
            if i + 1 >= len(remaining):
                raise SystemExit(f"error: --{arg} requires a value")
            overrides.append((_resolve_flag(arg), remaining[i + 1]))
            i += 1
        i += 1
    return overrides


def parse_args_and_load(argv: list[str] | None = None) -> TelescopeConfig:
    """Parse ``--config`` and config overrides from *argv*.

    ``--config`` takes a path relative to cwd or absolute.
    Without it, ``configs/run.yaml`` is used if present, otherwise
    only ``configs/defaults/default_train.yaml`` is used.

    Usage::

        python train.py --config configs/run.yaml --model Qwen3 --learning_rate 5e-7
    """
    parser = argparse.ArgumentParser(
        description="Telescope — post-training with RL",
        add_help=True,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config file (relative to cwd or absolute; default: configs/run.yaml)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode (full verbose output from all components)",
    )
    args, remaining = parser.parse_known_args(argv)

    cli_overrides = _parse_extra_args(remaining)
    if args.debug:
        cli_overrides.append(("debug", "true"))
    return load_config(run_yaml=args.config, cli_overrides=cli_overrides)
