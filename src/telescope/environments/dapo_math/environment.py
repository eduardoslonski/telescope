"""
DAPO Math environment implementation.

Mathematical reasoning environment using the DAPO Math 17K dataset.
Supports rule-based math verification via math_verify library.

Requires: uv add math-verify
"""
REQUIRED_PACKAGES = ["math-verify"]

import json
import logging

from datasets import load_dataset

from telescope.environments.base import Sample, RewardResult, SingleTurnEnvironment
from telescope.environments.parsers import (
    extract_boxed_answer,
    strip_think_tags,
    verify_math_answer,
)


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in the mind and then provide the user with the answer"
)

INSTRUCTION_PROMPT = (
    "Show your work and put your final answer within \\boxed{}."
)


class DapoMathEnvironment(SingleTurnEnvironment):
    """
    Environment for DAPO Math 17K mathematical reasoning tasks.

    Uses the BytedTsinghua-SIA/DAPO-Math-17k dataset.
    Reward is based on correctness of the answer in \\boxed{} format.
    """

    def __init__(
        self,
        dataset_name: str = "BytedTsinghua-SIA/DAPO-Math-17k",
        dataset_subset: str = "default",
        dataset_split: str = "train",
        **kwargs
    ):
        super().__init__(
            system_prompt=SYSTEM_PROMPT,
            instruction_prompt=INSTRUCTION_PROMPT,
            **kwargs
        )
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.dataset_split = dataset_split

    metrics_ranges = {
        "correct": {"min": 0, "max": 1},
    }

    @property
    def name(self) -> str:
        return "dapo-math"

    def load_dataset(
        self,
        num_samples: int = -1,
        shuffle: bool = False,
        seed: int = 42,
        **kwargs
    ) -> list[Sample]:
        """
        Load the DAPO Math 17K dataset.

        Args:
            num_samples: Number of samples to load. -1 for all.
            shuffle: Whether to shuffle the dataset.
            seed: Random seed for shuffling.

        Returns:
            List of Sample objects.
        """
        logger.info(f"Loading {self.dataset_name}/{self.dataset_subset}...")

        dataset = load_dataset(
            self.dataset_name,
            self.dataset_subset,
            split=self.dataset_split
        )

        # Filter out invalid samples
        dataset = dataset.filter(
            lambda x: (
                isinstance(x.get("prompt"), list) and
                len(x["prompt"]) > 0 and
                isinstance(x["prompt"][0], dict) and
                "content" in x["prompt"][0] and
                x.get("reward_model") is not None
            )
        )

        if shuffle:
            dataset = dataset.shuffle(seed=seed)

        if num_samples > 0:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        # Convert to samples
        samples = []
        for item in dataset:
            question = item["prompt"][0]["content"]

            # Handle reward_model as dict or JSON string
            reward_model = item["reward_model"]
            if isinstance(reward_model, str):
                reward_model = json.loads(reward_model)

            answer = reward_model.get("ground_truth", "")

            # Skip if answer is invalid
            if not isinstance(answer, str) or not answer:
                continue

            sample = Sample(
                prompt=question,
                answer=answer,
                metadata={
                    "question": question,
                    "data_source": item.get("data_source", ""),
                    "task": "dapo-math",
                }
            )
            samples.append(sample)

        self._samples = samples
        self._dataset = dataset

        logger.info(f"Loaded {len(samples)} samples from {self.name}")
        return samples

    async def compute_reward(
        self,
        completion: str,
        sample: Sample,
        eos_token: str = ""
    ) -> RewardResult:
        """
        Compute reward for a math completion.

        Uses math_verify for symbolic comparison if available,
        falls back to exact string matching otherwise.
        """
        # Strip EOS token if present
        if eos_token and completion.endswith(eos_token):
            completion = completion[:-len(eos_token)]

        # Extract answer from completion
        text = strip_think_tags(completion)
        predicted = extract_boxed_answer(text)
        ground_truth = sample.answer

        # Compute correctness
        correct = verify_math_answer(predicted, ground_truth)

        reward = 1.0 if correct else 0.0

        return RewardResult(
            total_reward=reward,
            sample_metrics={
                "correct": reward,
            },
            golden_answers={
                "correct": ground_truth,
            },
        )
