import re


LETTER_PATTERN = re.compile(r"\b([ABC])\b", re.IGNORECASE)


def extract_choice(text: str) -> str | None:
    """Return first occurrence of option letter A/B/C."""
    if not text:
        return None
    match = LETTER_PATTERN.search(text)
    return match.group(1).upper() if match else None


def verify_func(sample: dict, ground_truth: str, timeout_score: float = 0.0) -> float:
    predicted = extract_choice(sample.get("response", ""))
    if predicted is None:
        return timeout_score
    return 1.0 if predicted == ground_truth.upper() else 0.0
