"""Suppress noisy third-party log output in non-debug mode."""
from __future__ import annotations

import logging
import os
import warnings


def suppress_third_party_noise() -> None:
    """Quiet Ray, vLLM, torch, wandb, and NCCL output.

    Call once at startup when ``--debug`` is **not** set.
    """
    # --- Python logging: push chatty loggers to WARNING+ ---
    for name in (
        "ray",
        "ray.data",
        "ray.serve",
        "ray.rllib",
        "ray.tune",
        "ray.train",
        "ray.air",
        "ray._private",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    for name in ("wandb", "wandb.sdk", "wandb.run"):
        logging.getLogger(name).setLevel(logging.WARNING)

    for name in ("opentelemetry", "opentelemetry.sdk", "opentelemetry.exporter"):
        logging.getLogger(name).setLevel(logging.ERROR)

    # vLLM loggers
    for name in ("vllm", "vllm.config", "vllm.engine", "vllm.worker"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # HuggingFace datasets / transformers
    for name in ("datasets", "huggingface_hub", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Telescope sub-modules that are chatty at INFO
    logging.getLogger("telescope.environments").setLevel(logging.WARNING)

    # --- Python warnings: suppress known noisy categories ---
    warnings.filterwarnings("ignore", category=FutureWarning, module="vllm")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    # --- Environment variables ---
    os.environ.setdefault("RAY_DISABLE_DOCKER_CPU_WARNING", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
