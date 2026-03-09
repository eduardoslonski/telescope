"""
Centralized path management for telescope.

All data, checkpoint, and log paths are defined here to avoid
duplication and inconsistencies across modules.
"""
import os
from pathlib import Path

# Base directories
TELESCOPE_ROOT = Path(__file__).resolve().parents[1]  # …/telescope
RUN_DIR = Path(os.environ.get("TELESCOPE_RUN_DIR", os.getcwd())).resolve()
_ckpt_env = os.environ.get("TELESCOPE_CHECKPOINT_DIR")
CHECKPOINT_DIR = Path(_ckpt_env).resolve() if _ckpt_env else RUN_DIR / "checkpoints"
LOGS_DIR = RUN_DIR / "logs"

SANDBOX_LOGS_DIR = LOGS_DIR / "sandboxes"

# Stdout directories for process output (nested under logs/)
STDOUT_DIR = LOGS_DIR / "stdout"
STDOUT_INFERENCE_DIR = STDOUT_DIR / "inference"
STDOUT_TRAINER_DIR = STDOUT_DIR / "trainer"


def ensure_stdout_dirs():
    """Create all stdout directories if they don't exist."""
    STDOUT_INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    STDOUT_TRAINER_DIR.mkdir(parents=True, exist_ok=True)

