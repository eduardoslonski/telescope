"""
DeepScaler environment implementation.

Mathematical reasoning environment using the DeepScaleR dataset.
Supports rule-based math verification via math_verify library.

Requires: uv add math-verify
"""
REQUIRED_PACKAGES = ["math-verify"]

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


class DeepScalerEnvironment(SingleTurnEnvironment):
    """
    Environment for DeepScaleR mathematical reasoning tasks.

    Uses the agentica-org/DeepScaleR-Preview-Dataset dataset.
    Reward is based on correctness of the answer in \\boxed{} format.
    """

    def __init__(
        self,
        dataset_name: str = "agentica-org/DeepScaleR-Preview-Dataset",
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
        return "deepscaler"

    def load_dataset(
        self,
        num_samples: int = -1,
        shuffle: bool = False,
        seed: int = 42,
        **kwargs
    ) -> list[Sample]:
        """
        Load the DeepScaleR dataset.

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
                isinstance(x.get("problem"), str) and
                len(x["problem"]) > 0 and
                isinstance(x.get("answer"), str) and
                len(x["answer"]) > 0
            )
        )

        if shuffle:
            dataset = dataset.shuffle(seed=seed)

        if num_samples > 0:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        # Convert to samples
        samples = []
        for item in dataset:
            question = item["problem"]
            answer = item["answer"]

            sample = Sample(
                prompt=question,
                answer=answer,
                metadata={
                    "question": question,
                    "task": "deepscaler",
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
