"""
MATH-500 evaluation.

500 problems from the MATH benchmark (HuggingFaceH4/MATH-500, test split).
Wraps the hendrycks_math environment for answer verification via math_verify,
but loads the MATH-500 dataset instead.
"""
import logging

from datasets import load_dataset

from telescope.evals import Eval
from telescope.environments.base import EvalMetricsResult, Sample

logger = logging.getLogger(__name__)

INSTRUCTION_PROMPT_POST = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


class Math500Eval(Eval):
    environment_name = "hendrycks_math"
    name = "math500"

    metrics_ranges = {
        "correct": {"min": 0, "max": 1},
    }

    def __init__(
        self,
        dataset_name: str = "HuggingFaceH4/MATH-500",
        dataset_split: str = "test",
        instruction_prompt_post: str = INSTRUCTION_PROMPT_POST,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.instruction_prompt_post = instruction_prompt_post

    def load_dataset(self, num_samples: int = -1, **kwargs) -> list[Sample]:
        logger.info(f"Loading {self.dataset_name} ({self.dataset_split})...")

        dataset = load_dataset(self.dataset_name, split=self.dataset_split)

        if num_samples > 0:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        samples = []
        for item in dataset:
            question = item["problem"] + self.instruction_prompt_post
            answer = item["answer"]

            samples.append(Sample(
                prompt=question,
                answer=answer,
                metadata={
                    "question": item["problem"],
                    "subject": item.get("subject", ""),
                    "level": item.get("level", ""),
                },
            ))

        self._samples = samples
        logger.info(f"Loaded {len(samples)} samples for {self.name}")
        return samples

    def compute_eval_metrics(
        self,
        completion: str,
        sample: Sample,
        eos_token: str = "",
    ) -> EvalMetricsResult:
        reward_result = self.env.compute_reward(completion, sample, eos_token)
        metrics = dict(reward_result.sample_metrics)

        level_str = sample.metadata.get("level", "")
        try:
            metrics["level"] = float(level_str.replace("Level ", ""))
        except (ValueError, AttributeError):
            pass

        return EvalMetricsResult(
            metrics=metrics,
            golden_answers={
                **reward_result.golden_answers,
                "subject": sample.metadata.get("subject", ""),
            },
            info_turns=reward_result.info_turns,
        )
