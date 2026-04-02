"""
AIME 2024 evaluation.

30 problems from the 2024 American Invitational Mathematics Examination.
Uses math_verify for symbolic answer verification.

Requires: uv add math-verify
"""
import logging

from datasets import load_dataset

from telescope.evals import Eval
from telescope.environments.base import EvalMetricsResult, Sample
from telescope.environments.parsers import (
    extract_boxed_answer,
    strip_think_tags,
    verify_math_answer,
)

logger = logging.getLogger(__name__)

INSTRUCTION_PROMPT_POST = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


class Aime2024Eval(Eval):
    name = "aime_2024"

    metrics_ranges = {
        "correct": {"min": 0, "max": 1},
    }

    def __init__(
        self,
        dataset_name: str = "HuggingFaceH4/aime_2024",
        dataset_split: str = "train",
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
            answer = str(item["answer"])

            samples.append(Sample(
                prompt=question,
                answer=answer,
                metadata={
                    "question": item["problem"],
                },
            ))

        self._samples = samples
        logger.info(f"Loaded {len(samples)} samples for {self.name}")
        return samples

    async def compute_eval_metrics(
        self,
        completion: str,
        sample: Sample,
        eos_token: str = "",
    ) -> EvalMetricsResult:
        if eos_token and completion.endswith(eos_token):
            completion = completion[:-len(eos_token)]

        text = strip_think_tags(completion)
        predicted = extract_boxed_answer(text)
        ground_truth = sample.answer

        correct = 1.0 if verify_math_answer(predicted, ground_truth) else 0.0

        return EvalMetricsResult(
            metrics={"correct": correct},
            golden_answers={"correct": ground_truth},
        )
