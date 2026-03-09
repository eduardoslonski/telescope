"""
Evals module.

Each eval lives in its own subfolder and defines how to evaluate a model.
Evals can wrap an existing environment (reusing dataset, prompts, metrics)
or be fully standalone for eval-only benchmarks.

Evals are auto-discovered from subfolders of this package.  To use one,
specify the folder name in ``config.cfg.evals``::

    EVALS = [
        {"name": "aime", "eval_every": 50, ...},           # evals/aime/eval.py
        {"name": "countdown", "eval_every": 10, ...},      # falls back to environments/countdown/
    ]

To add a new eval, create a subfolder with an ``eval.py``
that contains a concrete ``Eval`` subclass.  No registration needed.
"""
from telescope.evals.base import Eval
from telescope.evals.registry import get_eval, list_evals, resolve

__all__ = [
    "Eval",
    "get_eval",
    "list_evals",
    "resolve",
]
