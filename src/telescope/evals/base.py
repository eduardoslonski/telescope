"""
Base eval class.

An eval defines how to evaluate a model. It can either wrap an existing
environment (reusing its dataset, prompts, and metrics) or be fully standalone.

Two ways to create an eval:

1. **Wrap an environment** — set ``environment_name`` and override only what
   you want. Everything else delegates to the environment automatically::

       class CountdownHardEval(Eval):
           environment_name = "countdown"

           def load_dataset(self, num_samples=-1, **kwargs):
               super().load_dataset(num_samples, **kwargs)
               self._samples = [s for s in self._samples if s.metadata["target"] > 100]
               return self._samples

2. **Standalone** — implement all required methods directly. Useful for
   eval-only benchmarks like AIME that have no training environment::

       class AIMEEval(Eval):
           name = "aime"

           def load_dataset(self, num_samples=-1, **kwargs):
               ...

           def compute_eval_metrics(self, completion, sample, eos_token=""):
               ...

Any attribute or method not defined on the eval subclass is automatically
forwarded to the wrapped environment via ``__getattr__``.
"""
from __future__ import annotations

from typing import Any

from telescope.environments.base import Sample


class Eval:
    """
    Base class for evals.

    Set ``environment_name`` to wrap an existing environment and inherit all
    its behavior.  Override any method to customize.  Leave
    ``environment_name`` as ``None`` for a fully standalone eval.

    Attributes:
        environment_name: Folder name under ``telescope/environments/`` to
            wrap, or ``None`` for standalone evals.
        environment_kwargs: Default kwargs forwarded to the wrapped
            environment's ``__init__``.  Merged with per-instance kwargs
            (per-instance wins).
    """

    environment_name: str | None = None
    environment_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs):
        self._env = None
        self._samples: list[Sample] | None = None

        if self.environment_name is not None:
            from telescope.environments import get_environment

            merged_kwargs = {**self.environment_kwargs, **kwargs}
            self._env = get_environment(self.environment_name, **merged_kwargs)
        else:
            for key, value in kwargs.items():
                setattr(self, key, value)

    @property
    def env(self):
        """The wrapped environment instance, or ``None`` for standalone evals."""
        return self._env

    # ------------------------------------------------------------------
    # Dataset loading — explicit because we need to sync ``_samples``
    # ------------------------------------------------------------------

    def load_dataset(self, num_samples: int = -1, **kwargs) -> list[Sample]:
        """Load evaluation dataset.

        When wrapping an environment, delegates to it and copies the
        ``_samples`` reference so that ``get_sample`` / ``__len__`` work on
        the eval object directly.

        Standalone evals must override this and set ``self._samples``.
        """
        if self._env is not None:
            result = self._env.load_dataset(num_samples, **kwargs)
            self._samples = self._env._samples
            return result
        raise NotImplementedError(
            f"{type(self).__name__} is a standalone eval — implement load_dataset()"
        )

    # ------------------------------------------------------------------
    # Sample access
    # ------------------------------------------------------------------

    def get_sample(self, idx: int) -> Sample:
        if self._samples is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        return self._samples[idx]

    def __len__(self) -> int:
        if self._samples is None:
            return 0
        return len(self._samples)

    # ------------------------------------------------------------------
    # Delegation — anything not found on the eval forwards to the env
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        # Guard against infinite recursion during __init__
        if name == "_env":
            raise AttributeError(name)
        env = self.__dict__.get("_env")
        if env is not None:
            return getattr(env, name)
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'. "
            f"Either set 'environment_name' to wrap an environment, "
            f"or implement this attribute on your eval subclass."
        )
