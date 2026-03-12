"""Scheduler for sampling dataset items during training."""
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from telescope.utils import config
from telescope.environments import get_environment
from telescope.environments.base import Environment, Sample
from telescope.utils.tlog import get_logger

_log = get_logger("orchestrator")


class Scheduler:
    """
    Infinite scheduler for sampling dataset items using a seeded RNG.
    
    Generates random indices lazily on each call to get_next_sample(),
    so it never runs out. The orchestrator controls when to stop based
    on training steps, not sample count.
    
    Reproducible: given the same numpy global seed, the sequence of
    samples is deterministic regardless of how many are requested.
    """

    def __init__(self, environment: Environment, excluded_indices: set[int] | None = None):
        """
        Args:
            environment: Environment to sample from
            excluded_indices: Sample indices reserved for eval (skipped during training)
        """
        self.environment = environment
        self._dataset_size = len(environment)
        # Cap excluded indices to valid dataset range
        if excluded_indices:
            self._excluded = {i for i in excluded_indices if i < self._dataset_size}
        else:
            self._excluded = set()
        self._rng = np.random.RandomState(np.random.randint(0, 2**31))
        self.current_idx = 0
        if self._excluded:
            effective = self._dataset_size - len(self._excluded)
            _log.info(
                f"Scheduler: excluding {len(self._excluded)} eval samples, "
                f"{effective}/{self._dataset_size} available for training"
            )
        _log.debug(f"Scheduler initialized (infinite): env={environment.name}, dataset_size={self._dataset_size}")

    def get_next_sample(self) -> dict:
        """
        Get the next sample from the dataset.
        
        Returns dict with:
            - prompt: The raw question text (chat template applied at rollout time)
            - sample: The Sample object (for reward computation)
            - env_name: Environment name for logging
            - env: Environment instance (for formatting and multi-turn)
        """
        idx = self._rng.randint(0, self._dataset_size)
        if self._excluded:
            # Resample to avoid eval-reserved indices
            for _ in range(100):
                if idx not in self._excluded:
                    break
                idx = self._rng.randint(0, self._dataset_size)
            else:
                if idx in self._excluded:
                    _log.warning(
                        f"Rejection sampling failed after 100 attempts "
                        f"(excluded {len(self._excluded)}/{self._dataset_size}). "
                        f"Using eval-reserved index {idx}."
                    )
        progress = self.current_idx
        self.current_idx += 1
        
        sample = self.environment.get_sample(idx)
        prompt_text = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)
        prompt_preview = prompt_text[:60].replace("\n", " ") if prompt_text else ""
        _log.debug(f"Scheduler.get_next_sample: sample_num={progress}, dataset_idx={idx}")
        _log.debug(f"  env={self.environment.name}, prompt_preview: {prompt_preview}...")
        
        return {
            "prompt": sample.prompt,
            "sample": sample,
            "env_name": self.environment.name,
            "env": self.environment,
        }

    def compute_reward(self, completion: str, sample: Sample, eos_token: str):
        """
        Compute reward using the environment's reward function.
        
        Returns RewardResult with total_reward and components dict.
        """
        return self.environment.compute_reward(completion, sample, eos_token)


class MultiEnvScheduler:
    """
    Infinite scheduler that samples from multiple environments with weighted sampling.
    
    Generates (env_idx, sample_idx) pairs lazily using a seeded RNG,
    so it never runs out. The orchestrator controls when to stop.
    """

    def __init__(
        self,
        environments: list[Environment],
        weights: list[float],
        excluded_indices: dict[str, set[int]] | None = None,
    ):
        """
        Args:
            environments: List of environments to sample from
            weights: Sampling weights for each environment (will be normalized)
            excluded_indices: Per-environment sample indices reserved for eval
        """
        self.environments = environments
        self._dataset_sizes = [len(env) for env in environments]
        # Cap excluded indices to valid dataset range per environment
        env_sizes = {env.name: len(env) for env in environments}
        if excluded_indices:
            self._excluded = {
                name: {i for i in exc if i < env_sizes.get(name, 0)}
                for name, exc in excluded_indices.items()
            }
        else:
            self._excluded = {}

        # Normalize weights to probabilities
        weights_arr = np.array(weights, dtype=float)
        self.probs = weights_arr / weights_arr.sum()

        self._rng = np.random.RandomState(np.random.randint(0, 2**31))
        self.current_idx = 0

        env_names = [e.name for e in environments]
        for env_name, exc in self._excluded.items():
            if exc:
                _log.info(
                    f"MultiEnvScheduler: excluding {len(exc)} eval samples "
                    f"from '{env_name}'"
                )
        _log.debug(f"MultiEnvScheduler initialized (infinite): environments={env_names}")
        _log.debug(f"  weights={weights}, normalized_probs={self.probs.tolist()}")

    def get_next_sample(self) -> dict:
        """
        Get the next sample from a randomly selected environment.
        
        Returns dict with:
            - prompt: The raw question text (chat template applied at rollout time)
            - sample: The Sample object (for reward computation)
            - env_name: Environment name for logging
            - env: Environment instance (for formatting and multi-turn)
        """
        env_idx = self._rng.choice(len(self.environments), p=self.probs)
        sample_idx = self._rng.randint(0, self._dataset_sizes[env_idx])
        env = self.environments[env_idx]
        env_excluded = self._excluded.get(env.name)
        if env_excluded:
            for _ in range(100):
                if sample_idx not in env_excluded:
                    break
                sample_idx = self._rng.randint(0, self._dataset_sizes[env_idx])
            else:
                if sample_idx in env_excluded:
                    _log.warning(
                        f"Rejection sampling failed after 100 attempts for '{env.name}' "
                        f"(excluded {len(env_excluded)}/{self._dataset_sizes[env_idx]}). "
                        f"Using eval-reserved index {sample_idx}."
                    )
        progress = self.current_idx
        self.current_idx += 1
        sample = env.get_sample(sample_idx)
        prompt_text = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)
        prompt_preview = prompt_text[:60].replace("\n", " ") if prompt_text else ""
        
        _log.debug(f"MultiEnvScheduler.get_next_sample: sample_num={progress}")
        _log.debug(f"  env_idx={env_idx}, env={env.name}, dataset_idx={sample_idx}")
        _log.debug(f"  prompt_preview: {prompt_preview}...")
        
        return {
            "prompt": sample.prompt,
            "sample": sample,
            "env_name": env.name,
            "env": env,
        }

    def compute_reward(self, completion: str, sample: Sample, eos_token: str):
        """
        Compute reward using the sample's environment.
        
        The sample knows which environment it came from via metadata.
        """
        # Find environment by name (stored in sample metadata)
        env_name = sample.metadata.get("_env_name")
        for env in self.environments:
            if env.name == env_name:
                return env.compute_reward(completion, sample, eos_token)
        raise ValueError(f"No environment found for env_name={env_name!r}")


def _parse_env_config(env_config) -> list[dict]:
    """
    Parse environment config into a normalized list of dicts.

    Accepts a list of EnvironmentEntry Pydantic models (from config.cfg.environments).
    """
    def _parse_single(item):
        return {
            "name": item.name,
            "kwargs": dict(item.kwargs),
            "weight": item.weight,
            "reward_min": item.reward_min,
            "reward_max": item.reward_max,
        }

    return [_parse_single(item) for item in env_config]


def create_scheduler(
    excluded_indices: dict[str, set[int]] | None = None,
) -> tuple[Scheduler | MultiEnvScheduler, str, Any]:
    """
    Create an infinite scheduler for training.

    The scheduler never runs out of samples — it generates random indices
    lazily using a seeded RNG. The orchestrator is responsible for stopping
    rollout based on the number of training steps.

    Args:
        excluded_indices: Per-environment sample indices reserved for eval
            (computed by the orchestrator from eval configs with
            ``separate_eval_samples=True``).

    Returns:
        Tuple of (scheduler, eos_token, tokenizer)
    """
    _log.debug("=== Creating scheduler ===")
    
    # Get environment config
    env_config = config.cfg.environments
    _log.debug(f"Environment config: {env_config}")
    env_specs = _parse_env_config(env_config)
    _log.debug(f"Parsed environment specs: {env_specs}")
    
    # Load tokenizer for multi-turn chat template formatting
    _log.debug(f"Loading tokenizer for model: {config.cfg.model}")
    _log.debug(f"Loading tokenizer with trust_remote_code=True")
    tokenizer = AutoTokenizer.from_pretrained(config.cfg.model, trust_remote_code=True)
    if config.cfg.chat_template is not None:
        _log.info(f"Overriding tokenizer chat_template from config")
        tokenizer.chat_template = config.cfg.chat_template
    elif not getattr(tokenizer, "chat_template", None):
        _log.warning(
            f"Tokenizer for {config.cfg.model} has no chat_template. "
            f"Set 'chat_template' in your config to provide one."
        )
    _log.debug(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, eos_token={repr(tokenizer.eos_token)}")
    
    # Load environments
    environments = []
    weights = []

    for spec in env_specs:
        name = spec["name"]
        kwargs = spec["kwargs"]
        weight = spec["weight"]
        reward_min = spec.get("reward_min")
        reward_max = spec.get("reward_max")
        
        _log.debug(f"Loading environment: {name}")
        if kwargs:
            _log.debug(f"  with kwargs: {kwargs}")
        _log.debug(f"  weight: {weight}")
        if reward_min is not None or reward_max is not None:
            _log.debug(f"  reward_min: {reward_min}, reward_max: {reward_max}")
        
        env = get_environment(name, **kwargs)
        
        # Store reward range on the environment (accessible in compute_reward)
        env.reward_min = reward_min
        env.reward_max = reward_max
        
        env.load_dataset()
        
        # Tag samples with environment name for reward computation
        if env._samples:
            for sample in env._samples:
                sample.metadata["_env_name"] = env.name
        
        environments.append(env)
        weights.append(weight)
        
        _log.debug(f"  loaded {len(env)} samples")
        
        # Check if multi-turn
        if getattr(env, 'is_multi_turn', False):
            _log.info(f"  [multi-turn environment, max_turns={getattr(env, 'max_turns', '?')}]")
    
    # Create appropriate scheduler
    excluded = excluded_indices or {}
    if len(environments) == 1:
        env_exc = excluded.get(environments[0].name, set())
        scheduler = Scheduler(environments[0], excluded_indices=env_exc or None)
        _log.debug("Scheduler created: single-env infinite")
        _log.debug(f"Environment: {environments[0].name} ({len(environments[0])} dataset samples)")
    else:
        scheduler = MultiEnvScheduler(environments, weights, excluded_indices=excluded or None)
        # Normalize weights for display
        total_weight = sum(weights)
        _log.debug("Scheduler created: multi-env infinite")
        for env, w in zip(environments, weights):
            pct = 100 * w / total_weight
            _log.debug(f"  - {env.name}: {pct:.1f}% ({len(env)} dataset samples)")
    
    # Get EOS token from first environment
    eos_token = environments[0].eos_token
    _log.debug(f"eos_token={eos_token!r} (from {environments[0].name})")

    return scheduler, eos_token, tokenizer

