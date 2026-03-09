"""
Answer parsing utilities for extracting answers from model completions.

Utility functions for various answer formats:
- \\boxed{...} for math
- <answer>...</answer> for structured responses
- ```python ... ``` for code blocks
- Think tags removal
- Math answer verification (requires math_verify: uv add math-verify)
"""
import logging
import re

_logger = logging.getLogger(__name__)


# ============================================================================
# Utility functions
# ============================================================================

def strip_think_tags(text: str) -> str:
    """
    Remove <think>...</think> content from text.
    
    Returns content after </think> if present, otherwise original text.
    """
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def extract_boxed_answer(text: str) -> str:
    """
    Extract content from \\boxed{...}, handling nested braces.
    
    Args:
        text: Text containing \\boxed{...}
        
    Returns:
        Content inside the last \\boxed{}, or empty string if not found.
    """
    def find_matching_brace(s: str, start: int) -> int:
        count = 1
        i = start
        while i < len(s) and count > 0:
            if s[i] == "{":
                count += 1
            elif s[i] == "}":
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    # Find last \boxed{
    boxed_start = text.rfind("\\boxed{")
    if boxed_start == -1:
        # Also try without backslash
        boxed_start = text.rfind("boxed{")
        if boxed_start == -1:
            return ""
        content_start = boxed_start + 6  # len('boxed{')
    else:
        content_start = boxed_start + 7  # len('\\boxed{')

    closing_brace = find_matching_brace(text, content_start)

    if closing_brace == -1:
        return ""

    return text[content_start:closing_brace].strip()


def extract_answer_tags(text: str) -> str:
    """
    Extract content from <answer>...</answer> tags.

    Returns content inside the last <answer> tag, or empty string if not found.
    Uses the last match (consistent with extract_boxed_answer) since models
    may revise their answer across multiple tags.
    """
    pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return ""


def extract_code_block(text: str, language: str = "python") -> str:
    """
    Extract code from markdown code blocks.

    Tries language-specific first, then generic code blocks.
    Returns content from the last matching block (consistent with
    extract_boxed_answer/extract_answer_tags), since models may
    revise their answer across multiple blocks.

    Args:
        text: Text containing code blocks.
        language: Language to look for (e.g., "python").

    Returns:
        Code content, or empty string if not found.
    """
    # Try language-specific block first (last match, case-insensitive)
    pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # Try generic code block (skip optional language tag)
    pattern = r"```(?:\w+)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    return ""


def extract_xml_tag(text: str, tag: str) -> str:
    """
    Extract content from XML-style tags like <tag>...</tag>.

    Args:
        text: Text containing XML tags.
        tag: Tag name to extract (e.g., "guess", "think").

    Returns:
        Content inside the last matching tag, or empty string if not found.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return ""


# ============================================================================
# Math answer verification
# ============================================================================

def verify_exact_match(predicted: str, ground_truth: str) -> bool:
    """Normalized exact string matching for math answers."""
    predicted = " ".join(predicted.split()).strip()
    ground_truth = " ".join(ground_truth.split()).strip()
    return predicted == ground_truth


def verify_math_answer(predicted: str, ground_truth: str) -> bool:
    """
    Verify a math answer using math_verify for symbolic comparison,
    falling back to exact string matching on parse/verify errors.

    Handles equivalent mathematical expressions (e.g. 1/2 == 0.5).
    Returns False for empty or excessively long predictions.

    Requires: uv add math-verify
    """
    if not predicted or len(predicted) > 500:
        return False

    from math_verify import parse as _mv_parse, verify as _mv_verify

    try:
        parsed_pred = _mv_parse(f"\\boxed{{{predicted}}}", parsing_timeout=5)
        parsed_gt = _mv_parse(f"\\boxed{{{ground_truth}}}", parsing_timeout=5)
        return bool(_mv_verify(parsed_gt, parsed_pred, timeout_seconds=5))
    except Exception:
        return verify_exact_match(predicted, ground_truth)


