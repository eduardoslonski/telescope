"""
Base environment interface for RL tasks.

An environment encapsulates everything needed for a specific RL task:
- How to load and preprocess the dataset
- How to format prompts
- How to compute rewards

Multi-turn environments extend this to support:
- Iterative model-environment interaction loops
- Trajectory tracking across turns
- Environment-specific stop conditions
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset


# Type aliases for chat messages
ChatMessage = dict[str, Any]  # {"role": "user"|"assistant"|"system", "content": str, ...}
Messages = str | list[ChatMessage]


@dataclass
class Sample:
    """A single sample from the environment."""
    prompt: str | list[ChatMessage]  # Can be string or chat messages for multi-turn
    answer: str = ""  # Ground truth answer for reward computation
    metadata: dict[str, Any] = field(default_factory=dict)  # Task-specific data (e.g., info dict)


@dataclass
class RewardResult:
    """Result of reward computation."""
    total_reward: float
    sample_metrics: dict[str, float] = field(default_factory=dict)  # Per-sample metrics: reward components + any other metrics
    golden_answers: dict[str, str | None] = field(default_factory=dict)  # Golden answer per reward component
    info_turns: list[dict[str, Any]] = field(default_factory=list)  # Per-turn text info (e.g. stderr, summaries)
    # Each dict: {"turn_order": int, "info_key": str, "info_value": str, "info_type": str}
    # info_type defaults to "text", can also be "stderr", "stdout", etc.
    sample_tags: dict[str, str] = field(default_factory=dict)  # Per-sample string tags for filtering (e.g. {"style": "4 numbers", "task": "coding"})


@dataclass
class EvalMetricsResult:
    """Result of eval metrics computation (separate from training rewards)."""
    metrics: dict[str, float] = field(default_factory=dict)
    golden_answers: dict[str, str | None] = field(default_factory=dict)
    info_turns: list[dict[str, Any]] = field(default_factory=list)
    sample_tags: dict[str, str] = field(default_factory=dict)


@dataclass
class TrajectoryStep:
    """
    A single step in a multi-turn trajectory.
    
    Each step represents one model rollout request/response pair.
    For training, we track token IDs and logprobs to compute importance sampling ratios.
    """
    prompt: Messages  # Messages sent to model for this turn
    completion: Messages  # Model's response (usually assistant message)
    prompt_token_ids: list[int] = field(default_factory=list)
    completion_token_ids: list[int] = field(default_factory=list)
    completion_logprobs: list[float] = field(default_factory=list)
    is_truncated: bool = False  # Whether rollout was cut off by max_tokens


@dataclass 
class RolloutState:
    """State maintained during a multi-turn rollout."""
    # Input fields
    sample: Sample
    env_name: str
    
    # Rollout tracking
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    is_completed: bool = False
    stop_reason: str | None = None  # Name of stop condition that ended rollout
    error: str | None = None
    
    # Custom state (environment-specific)
    custom: dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_turns(self) -> int:
        return len(self.trajectory)
    
    def get_full_completion_ids(self) -> list[int]:
        """Get all completion token IDs concatenated across turns."""
        ids = []
        for step in self.trajectory:
            ids.extend(step.completion_token_ids)
        return ids
    
    def get_full_logprobs(self) -> list[float]:
        """Get all logprobs concatenated across turns."""
        logprobs = []
        for step in self.trajectory:
            logprobs.extend(step.completion_logprobs)
        return logprobs


class Environment(ABC):
    """
    Base class for RL environments.
    
    Environments are responsible for:
    1. Loading and preprocessing datasets
    2. Computing rewards for model completions
    3. Providing metadata for logging
    """
    
    # Default EOS token (can be overridden by subclasses or via kwargs)
    eos_token: str = "<|endoftext|>"
    
    # Expected reward range (can be set via environment config: reward_min / reward_max)
    reward_min: float | None = None
    reward_max: float | None = None
    
    # Known ranges for sample_metrics: {"metric_name": {"min": float, "max": float, "invert": bool}, ...}
    # Override in subclasses to declare known min/max bounds for metrics.
    # Used by the UI to display normalized charts.
    # Set "invert": True when lower values are better (e.g. num_errors), so the UI
    # colors min as green and max as red instead of the default min=red, max=green.
    metrics_ranges: dict[str, dict[str, float | bool]] = {}

    # Known options for sample_tags: {"tag_name": ["option1", "option2", ...], ...}
    # Override in subclasses to declare known tag names and their possible values.
    # Used by the UI for filtering (e.g. show only samples with tag "task" == "coding").
    tags_options: dict[str, list[str]] = {}

    @property
    def is_multi_turn(self) -> bool:
        """Whether this environment requires multi-turn rollouts."""
        return False
    
    def __init__(self, eos_token: str | None = None, add_thinking_prefix: bool = False, **kwargs):
        self._dataset: Dataset | None = None
        self._samples: list[Sample] | None = None
        self._name: str | None = None
        if eos_token is not None:
            self.eos_token = eos_token
        self.add_thinking_prefix = add_thinking_prefix
        # Store any extra kwargs for subclass use
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def name(self) -> str:
        """Environment name for logging. Defaults to the folder name (set by the registry)."""
        if self._name is not None:
            return self._name
        raise ValueError(
            "Environment name not set. Either override the `name` property "
            "or load this environment via get_environment() which sets it automatically."
        )
    
    @abstractmethod
    def load_dataset(self, num_samples: int = -1, **kwargs) -> list[Sample]:
        """
        Load and preprocess the dataset.
        
        Args:
            num_samples: Number of samples to load. -1 for all.
            **kwargs: Additional dataset loading parameters.
            
        Returns:
            List of Sample objects.
        """
        pass
    
    @abstractmethod
    async def compute_reward(
        self,
        completion: str,
        sample: Sample,
        eos_token: str = ""
    ) -> RewardResult:
        """
        Compute reward for a completion given the original sample.

        Args:
            completion: Model's completion string.
            sample: Original sample with prompt and metadata.
            eos_token: End of sequence token to strip.

        Returns:
            RewardResult with total reward and component breakdown.
        """
        pass

    async def compute_eval_metrics(
        self,
        completion: str,
        sample: Sample,
        eos_token: str = "",
    ) -> EvalMetricsResult:
        """
        Compute eval-specific metrics for a completion.

        Separated from ``compute_reward`` so eval metrics stay independent of
        training rewards/advantages.  Default implementation delegates to
        ``compute_reward`` and lifts ``sample_metrics`` into an
        ``EvalMetricsResult``.  Override in subclasses for eval-specific logic.
        """
        reward_result = await self.compute_reward(completion, sample, eos_token)
        return EvalMetricsResult(
            metrics=reward_result.sample_metrics,
            golden_answers=reward_result.golden_answers,
            info_turns=reward_result.info_turns,
            sample_tags=reward_result.sample_tags,
        )

    async def prepare_resources(self):
        """Optional hook to pre-warm expensive resources (e.g. sandbox pools).

        Called by the orchestrator before the rollout loop starts.
        Default is a no-op; override in subclasses that need it.
        """
        pass

    def get_sample(self, idx: int) -> Sample:
        """Get a sample by index."""
        if self._samples is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        return self._samples[idx]
    
    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        if self._samples is None:
            return 0
        return len(self._samples)


class SingleTurnEnvironment(Environment):
    """
    Base class for single-turn (Q&A style) environments.
    
    These environments expect:
    - A question/prompt
    - A model response
    - Reward computation based on correctness
    """
    
    def __init__(
        self,
        system_prompt: str | None = None,
        instruction_prompt: str | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.instruction_prompt = instruction_prompt
    
    def format_prompt(
        self,
        question: str,
        tokenizer,
        add_thinking_prefix: bool | None = None,
        chat_template_kwargs: dict | None = None,
    ) -> str:
        """
        Format a prompt for the task using the tokenizer's chat template.
        
        Args:
            question: The question/task text.
            tokenizer: HuggingFace tokenizer for chat template. REQUIRED.
            add_thinking_prefix: Whether to add thinking prefix to assistant message.
                If None, uses environment default (self.add_thinking_prefix).
            chat_template_kwargs: Additional kwargs for apply_chat_template (e.g., enable_thinking).
            
        Returns:
            Formatted prompt string.
            
        Raises:
            ValueError: If tokenizer is not provided.
        """
        if tokenizer is None:
            raise ValueError(
                "Tokenizer is required for format_prompt(). "
                "Chat template formatting cannot be done without a tokenizer."
            )
        
        # Build the full question with instruction
        if self.instruction_prompt:
            full_question = f"{self.instruction_prompt}\n\n{question}"
        else:
            full_question = question

        use_thinking_prefix = (
            self.add_thinking_prefix
            if add_thinking_prefix is None
            else add_thinking_prefix
        )
        
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": full_question})
        
        if use_thinking_prefix:
            messages.append({"role": "assistant", "content": "<think>"})
        
        kwargs = chat_template_kwargs or {}
        return tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            continue_final_message=use_thinking_prefix,
            add_generation_prompt=not use_thinking_prefix,  # Add gen prompt when not continuing
            **kwargs,
        )


class MultiTurnEnvironment(Environment):
    """
    Base class for multi-turn interactive environments.
    
    Multi-turn environments support iterative model-environment interaction:
    1. Model generates a response
    2. Environment provides feedback/next prompt via env_response()
    3. Repeat until a stop condition is met
    
    Subclasses must implement:
    - env_response(): Generate environment feedback after model response
    - compute_reward(): Compute reward for the full trajectory
    
    Optionally override stop conditions by implementing methods with @stop decorator pattern.
    
    Example usage:
        class GameEnv(MultiTurnEnvironment):
            async def env_response(self, messages: list, state: RolloutState) -> list:
                # Parse model action, update game state, return feedback
                action = self.parse_action(messages[-1]["content"])
                result = self.game.step(action)
                state.custom["score"] = result.score
                return [{"role": "user", "content": result.feedback}]
            
            def is_done(self, state: RolloutState) -> tuple[bool, str | None]:
                if state.custom.get("game_over"):
                    return True, "game_over"
                return False, None
    """
    
    def __init__(
        self,
        max_turns: int = 10,
        system_prompt: str | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_turns = max_turns
        self.system_prompt = system_prompt
    
    @property
    def is_multi_turn(self) -> bool:
        """Indicates this is a multi-turn environment."""
        return True
    
    def create_initial_state(self, sample: Sample) -> RolloutState:
        """
        Create initial rollout state for a sample.
        
        Override to add environment-specific initial state.
        """
        return RolloutState(
            sample=sample,
            env_name=self.name,
        )
    
    def get_initial_prompt(self, sample: Sample, tokenizer=None) -> list[ChatMessage]:
        """
        Get initial prompt messages for the first turn.
        
        Args:
            sample: The sample with prompt data
            tokenizer: Optional tokenizer for formatting
            
        Returns:
            List of chat messages for initial prompt
        """
        messages = []
        
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Handle both string and message list prompts
        if isinstance(sample.prompt, str):
            messages.append({"role": "user", "content": sample.prompt})
        else:
            # Already a list of messages
            messages.extend(sample.prompt)
        
        return messages
    
    async def env_response(
        self,
        messages: list[ChatMessage],
        state: RolloutState,
    ) -> list[ChatMessage]:
        """
        Generate environment response after model completion.

        This is called after each model rollout. The returned messages
        are appended to the conversation for the next turn.

        Args:
            messages: Full conversation so far (including latest model response)
            state: Current rollout state (can be mutated to track custom state)

        Returns:
            List of messages to append (typically one user message with feedback).
            Return empty list to signal no response needed (for final turn).

            Messages can include an optional "turn_type" key to specify the type
            of this turn for logging purposes (e.g., "tool_call", "tool_result",
            "context", "feedback"). If not specified, defaults to "env_response".
            Example: {"role": "user", "content": "...", "turn_type": "tool_result"}
        """
        raise NotImplementedError("Subclasses must implement env_response()")
    
    def is_done(self, state: RolloutState) -> tuple[bool, str | None]:
        """
        Check if rollout should terminate.
        
        Called after each model response and after env_response has processed
        the action. This means state changes from env_response (e.g. game_won
        flags, success markers) are visible here.
        
        Override to add custom stop conditions.
        
        Args:
            state: Current rollout state
            
        Returns:
            Tuple of (is_done, reason). If is_done=True, reason should describe why.
        """
        # Default: just check max turns
        if state.num_turns >= self.max_turns:
            return True, "max_turns_reached"
        return False, None
    
    def get_next_prompt_messages(
        self,
        state: RolloutState,
        env_response_messages: list[ChatMessage],
    ) -> list[ChatMessage]:
        """
        Build the full prompt for the next turn.
        
        By default, uses the last turn's full prompt plus its completion, then
        appends the new environment response. This avoids duplicating earlier
        turns because each trajectory step stores the full prompt for that turn.
        
        Override for custom prompt construction (e.g., context window management).
        
        Args:
            state: Current rollout state with trajectory
            env_response_messages: Environment's response to append
            
        Returns:
            Full message list for next model request
        """
        messages: list[ChatMessage] = []
        
        if state.trajectory:
            last_step = state.trajectory[-1]
            if isinstance(last_step.prompt, list):
                messages.extend(last_step.prompt)
            elif isinstance(last_step.prompt, str):
                messages.append({"role": "user", "content": last_step.prompt})
            
            if isinstance(last_step.completion, list):
                messages.extend(last_step.completion)
            elif isinstance(last_step.completion, str):
                messages.append({"role": "assistant", "content": last_step.completion})
        
        # Add new env response
        messages.extend(env_response_messages)
        
        return messages
    
    async def compute_reward(
        self,
        state: RolloutState,
        eos_token: str = "",
    ) -> RewardResult:
        """
        Compute reward for a completed trajectory.

        Override this to define task-specific reward computation.

        Args:
            state: Completed rollout state with full trajectory
            eos_token: EOS token to strip from completions

        Returns:
            RewardResult with total reward and component breakdown
        """
        raise NotImplementedError("Subclasses must implement compute_reward()")

    async def compute_eval_metrics(
        self,
        state: RolloutState,
        eos_token: str = "",
    ) -> EvalMetricsResult:
        """
        Compute eval-specific metrics for a completed multi-turn trajectory.

        Default implementation delegates to ``compute_reward`` and lifts
        ``sample_metrics``.  Override for eval-specific logic.
        """
        reward_result = await self.compute_reward(state, eos_token)
        return EvalMetricsResult(
            metrics=reward_result.sample_metrics,
            golden_answers=reward_result.golden_answers,
            info_turns=reward_result.info_turns,
            sample_tags=reward_result.sample_tags,
        )

    def load_dataset(self, num_samples: int = -1, **kwargs) -> list[Sample]:
        """Load and preprocess the dataset."""
        raise NotImplementedError("Subclasses must implement load_dataset()")


