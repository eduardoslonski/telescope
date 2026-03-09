"""Reward functions for the Countdown task."""
import re
import warnings


def compute_format_reward(completion: str) -> float:
    """
    Check if the completion follows the expected format.

    Handles both cases:
    - completion starts with <think> (model generated it, e.g. add_thinking_prefix=false)
    - completion starts after <think> (prefix was in prompt, e.g. add_thinking_prefix=true)

    Returns:
        1.0 if format is correct with valid equation characters
        0.5 if format is correct but answer contains unexpected characters
        0.0 if format is incorrect
    """
    try:
        # Only prepend <think> if the completion doesn't already start with it
        completion = completion.lstrip()
        if not completion.startswith("<think>"):
            completion = "<think>" + completion

        # Expected format:
        # <think>...contents...</think>
        # <answer>...equation...</answer>
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\s*<answer>([\s\S]*?)<\/answer>\s*$"
        match = re.search(regex, completion, re.DOTALL)

        if match is None or len(match.groups()) != 2:
            return 0.0

        answer_content = match.group(2).strip()
        # Only numbers, operators, parentheses, decimal points, and whitespace allowed
        allowed_pattern = r"^[\d+\-*/().\s]+$"

        if re.match(allowed_pattern, answer_content):
            return 1.0
        else:
            return 0.5

    except Exception:
        return 0.0


def compute_equation_reward(completion: str, nums: list[int], target: int) -> float:
    """
    Check if the equation in the answer is correct.

    Returns:
        1.0 if equation uses exactly the given numbers and evaluates to target
        0.0 otherwise
    """
    try:
        # Take the last <answer> tag (the model's final answer)
        matches = re.findall(r"<answer>(.*?)<\/answer>", completion)
        if not matches:
            return 0.0

        equation = matches[-1].strip()

        # Reject overly long equations to prevent eval() DoS
        if len(equation) > 200:
            return 0.0

        # Extract numbers used in equation
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

        # Must use exactly the given numbers
        if sorted(used_numbers) != sorted(nums):
            return 0.0

        # Validate equation only contains safe characters
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Evaluate safely (suppress SyntaxWarning from model-generated
        # expressions like "4(3 + 2)" that Python 3.12+ warns about)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            result = eval(equation, {"__builtins__": None}, {})

        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        else:
            return 0.0

    except Exception:
        return 0.0
