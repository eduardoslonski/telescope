"""
countdown environment implementation.

Adapted from https://github.com/Jiayi-Pan/TinyZero

Mathematical reasoning environment using the Countdown task.
The task is to create an equation using given numbers that equals a target.
"""
import logging

from datasets import load_dataset

from telescope.environments.base import Sample, SingleTurnEnvironment
from telescope.environments.countdown.rewards import compute_format_reward, compute_equation_reward
from telescope.environments.rewards import Rubric


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in the mind and then provide the user with the answer"
)

INSTRUCTION_PROMPT = (
    "Show your work in <think> </think> tags. And return the final equation and answer in "
    "<answer> </answer> tags, for example <answer>(1 + 2) / (3 * 5)</answer>"
)


class CountdownEnvironment(SingleTurnEnvironment):
    """
    Environment for Countdown math task.
    
    Uses the Jiayi-Pan/Countdown-Tasks-3to4 dataset.
    The task is to create an equation using given numbers that equals a target.
    Reward is based on format correctness and equation validity.
    """
    
    def __init__(
        self,
        dataset_name: str = "Jiayi-Pan/Countdown-Tasks-3to4",
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

        self.rubric = Rubric()
        self.rubric.add_reward(self._format_reward, range_min=0, range_max=1)
        self.rubric.add_reward(self._equation_reward, range_min=0, range_max=1)

    @property
    def metrics_ranges(self):
        return self.rubric.metrics_ranges
    
    def load_dataset(
        self,
        num_samples: int = -1,
        shuffle: bool = False,
        seed: int = 42,
        **kwargs
    ) -> list[Sample]:
        """
        Load the Countdown dataset.
        
        Args:
            num_samples: Number of samples to load. -1 for all.
            shuffle: Whether to shuffle the dataset.
            seed: Random seed for shuffling.
            
        Returns:
            List of Sample objects.
        """
        logger.info(f"Loading {self.dataset_name}...")
        
        dataset = load_dataset(self.dataset_name, self.dataset_subset, split=self.dataset_split)
        
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        
        if num_samples > 0:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        # Convert to samples
        samples = []
        for item in dataset:
            nums = item["nums"]
            target = item["target"]
            
            # Format the question with numbers and target
            # Store raw question - formatting happens at rollout time with proper tokenizer
            question = (
                f"Using the numbers {nums}, create an equation that equals {target}. "
                f"You can use basic arithmetic operations (+, -, *, /) and each number can only be used once."
            )
            
            sample = Sample(
                prompt=question,  # Raw question, not pre-formatted
                answer=str(target),
                metadata={
                    "question": question,
                    "nums": nums,
                    "target": target,
                }
            )
            samples.append(sample)
        
        self._samples = samples
        self._dataset = dataset
        
        logger.info(f"Loaded {len(samples)} samples from {self.name}")
        return samples
    
    @staticmethod
    def _format_reward(completion: str) -> float:
        return compute_format_reward(completion)

    @staticmethod
    def _equation_reward(completion: str, sample: Sample) -> tuple[float, str]:
        nums = sample.metadata["nums"]
        target = sample.metadata["target"]
        return compute_equation_reward(completion, nums, target), str(target)

    async def compute_reward(self, completion: str, sample: Sample, eos_token: str = ""):
        if eos_token and completion.endswith(eos_token):
            completion = completion[: -len(eos_token)]
        return await self.rubric.score(completion=completion, sample=sample)
