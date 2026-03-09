"""Three-layer config loading for standalone eval.

Same pattern as ``telescope/utils/config_loader.py``:

1. ``configs/defaults/default_eval.yaml`` — base defaults
2. ``--config <path>`` — run-level overrides (path relative to cwd or absolute)
3. ``--flag value`` — CLI overrides

Usage::

    from telescope.eval_standalone.config_loader import parse_args_and_load
    eval_cfg = parse_args_and_load()   # returns EvalStandaloneConfig
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from telescope.eval_standalone.config_schema import EvalStandaloneConfig
from telescope.utils.config_loader import (
    _coerce_value,
    _deep_merge,
    _set_dotpath,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _PROJECT_ROOT / "configs" / "evals"
DEFAULT_YAML = _PROJECT_ROOT / "configs" / "defaults" / "default_eval.yaml"


def _resolve_flag(flag: str) -> str:
    """Resolve a CLI flag to a config key.

    Dashes become underscores.
    Validated against ``EvalStandaloneConfig`` fields.
    """
    canonical = flag.replace("-", "_").lower()
    if canonical in EvalStandaloneConfig.model_fields:
        return canonical
    raise SystemExit(f"error: unknown flag --{flag}")


def _parse_extra_args(remaining: list[str]) -> list[tuple[str, str]]:
    """Parse unknown args as ``--flag value`` pairs."""
    overrides: list[tuple[str, str]] = []
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if not arg.startswith("--"):
            raise SystemExit(f"error: unexpected argument: {arg}")
        arg = arg[2:]
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


def _resolve_config_path(run_yaml: str | Path | None) -> Path | None:
    """Resolve the eval run config path.

    - If *run_yaml* is given, use it as-is (relative to cwd or absolute).
    - If *run_yaml* is ``None``, try ``configs/evals/eval.yaml`` automatically.
    - Returns ``None`` when no run config is found.
    """
    if run_yaml is not None:
        path = Path(run_yaml)
        if not path.exists():
            raise SystemExit(f"error: config file not found: {path}")
        return path.resolve()
    # No --config flag: try configs/evals/eval.yaml by default
    default_run = _CONFIG_DIR / "eval.yaml"
    if default_run.exists():
        return default_run
    return None


def load_config(
    run_yaml: str | Path | None = None,
    cli_overrides: list[tuple[str, str]] | None = None,
) -> EvalStandaloneConfig:
    """Load eval config with 3-layer merge: defaults/default_eval.yaml -> run config -> CLI."""
    # Layer 1: defaults
    with open(DEFAULT_YAML) as f:
        data = yaml.safe_load(f) or {}

    # Layer 2: run overrides
    resolved = _resolve_config_path(run_yaml)
    if resolved is not None:
        print(f"[eval-config] Using run config: {resolved}")
        with open(resolved) as f:
            run_data = yaml.safe_load(f) or {}
        data = _deep_merge(data, run_data)
    else:
        print(f"[eval-config] No run config found, using {DEFAULT_YAML} only")

    # Layer 3: CLI overrides
    for key, raw_value in (cli_overrides or []):
        _set_dotpath(data, key, raw_value)

    return EvalStandaloneConfig(**data)


def parse_args_and_load(argv: list[str] | None = None) -> EvalStandaloneConfig:
    """Parse ``--config`` and overrides from *argv*, return ``EvalStandaloneConfig``."""
    parser = argparse.ArgumentParser(
        description="Telescope — standalone eval",
        add_help=True,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to an eval config file (relative to cwd or absolute; default: configs/evals/eval.yaml)",
    )
    args, remaining = parser.parse_known_args(argv)

    cli_overrides = _parse_extra_args(remaining)
    return load_config(run_yaml=args.config, cli_overrides=cli_overrides)
