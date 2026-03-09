#!/usr/bin/env python3
"""Convert a training checkpoint to HuggingFace format for vLLM / standalone eval.

Reads native checkpoint files (DCP or Megatron dist_checkpointing) plus
the ``hf_meta/`` directory saved alongside, and produces a standard HF
model directory (config.json, model.safetensors, tokenizer files).

Usage
-----
Single checkpoint::

    python tools/convert_checkpoint_to_hf.py \\
        --checkpoint-dir checkpoints/step_100 \\
        --output-dir converted/step_100

Batch (all step_N dirs under a root)::

    python tools/convert_checkpoint_to_hf.py \\
        --checkpoint-root checkpoints \\
        --output-root converted
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from telescope.utils.checkpoint_converter import convert_single

_STEP_RE = re.compile(r"^step_(\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert telescope training checkpoints to HuggingFace format",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Single checkpoint directory (e.g., checkpoints/step_100)",
    )
    group.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Root containing step_N/ directories (batch mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (required with --checkpoint-dir)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output root (required with --checkpoint-root). "
             "Creates step_N/ subdirectories inside.",
    )
    args = parser.parse_args()

    if args.checkpoint_dir:
        if not args.output_dir:
            parser.error("--output-dir is required with --checkpoint-dir")
        convert_single(args.checkpoint_dir, args.output_dir)
    else:
        if not args.output_root:
            parser.error("--output-root is required with --checkpoint-root")
        root = args.checkpoint_root
        found = 0
        for name in sorted(os.listdir(root)):
            if _STEP_RE.match(name) and (root / name / "meta.json").is_file():
                out = args.output_root / name
                if out.exists() and (out / "config.json").is_file():
                    print(f"Skipping {name}: already converted")
                    continue
                convert_single(root / name, out)
                found += 1
        if found == 0:
            print(f"No unconverted checkpoints found in {root}")
            sys.exit(1)
        print(f"\nDone: converted {found} checkpoint(s)")


if __name__ == "__main__":
    main()
