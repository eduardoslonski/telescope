"""
GPQA Diamond evaluation.

198 graduate-level multiple-choice questions testing scientific reasoning
across physics, chemistry, and biology.

The dataset may require authentication: huggingface-cli login
"""
import logging
import random
import re

from datasets import load_dataset

from telescope.evals import Eval
from telescope.environments.base import EvalMetricsResult, Sample
from telescope.environments.parsers import strip_think_tags

logger = logging.getLogger(__name__)

ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*\$?([A-D])\$?")

INSTRUCTION_PROMPT_PRE = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n"
)


def extract_multiple_choice(text: str) -> str:
    """
    Extract multiple-choice answer letter from text.

    Looks for 'Answer: X' pattern (case-insensitive), returns the last match.
    Returns empty string if no match found.
    """
    matches = ANSWER_PATTERN.findall(text)
    if matches:
        return matches[-1].upper()
    return ""


class GpqaDiamondEval(Eval):
    name = "gpqa_diamond"

    metrics_ranges = {
        "correct": {"min": 0, "max": 1},
    }

    def __init__(
        self,
        dataset_name: str = "Idavidrein/gpqa",
        dataset_subset: str = "gpqa_diamond",
        dataset_split: str = "train",
        shuffle_seed: int = 42,
        instruction_prompt_pre: str = INSTRUCTION_PROMPT_PRE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.dataset_split = dataset_split
        self.shuffle_seed = shuffle_seed
        self.instruction_prompt_pre = instruction_prompt_pre

    def load_dataset(self, num_samples: int = -1, **kwargs) -> list[Sample]:
        logger.info(f"Loading {self.dataset_name}/{self.dataset_subset}...")

        dataset = load_dataset(
            self.dataset_name,
            self.dataset_subset,
            split=self.dataset_split,
            trust_remote_code=True,
        )

        if num_samples > 0:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        rng = random.Random(self.shuffle_seed)

        samples = []
        for item in dataset:
            choices = [
                item["Correct Answer"],
                item["Incorrect Answer 1"],
                item["Incorrect Answer 2"],
                item["Incorrect Answer 3"],
            ]

            # Shuffle choices and track correct answer position
            indices = list(range(4))
            rng.shuffle(indices)
            shuffled = [choices[i] for i in indices]
            correct_idx = indices.index(0)
            correct_letter = "ABCD"[correct_idx]

            # Format question with instruction and choices
            question = (
                f"{self.instruction_prompt_pre}"
                f"{item['Question']}\n\n"
                f"A) {shuffled[0]}\n"
                f"B) {shuffled[1]}\n"
                f"C) {shuffled[2]}\n"
                f"D) {shuffled[3]}"
            )

            samples.append(Sample(
                prompt=question,
                answer=correct_letter,
                metadata={
                    "question": item["Question"],
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
        if eos_token and completion.endswith(eos_token):
            completion = completion[:-len(eos_token)]

        text = strip_think_tags(completion)
        predicted = extract_multiple_choice(text)
        ground_truth = sample.answer

        correct = 1.0 if predicted == ground_truth else 0.0

        return EvalMetricsResult(
            metrics={"correct": correct},
            golden_answers={"correct": ground_truth},
        )
