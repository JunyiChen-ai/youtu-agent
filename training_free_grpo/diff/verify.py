import json
import re


LETTER_PATTERN = re.compile(r"\b([ABC])\b", re.IGNORECASE)
ANSWER_LINE_PATTERN = re.compile(
    r"(?im)^\s*(?:final\s+)?answer\s*[:=]\s*(?:option\s+|choice\s+)?\(?\s*([ABC])",
    re.IGNORECASE,
)
VALID_CHOICES = {"A", "B", "C"}


def _normalize_choice(value) -> str | None:
    if isinstance(value, str):
        candidate = value.strip().upper()
        if candidate in VALID_CHOICES:
            return candidate
    return None


def _extract_from_json_text(text: str | None) -> str | None:
    if not text:
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if isinstance(obj, dict):
        return _normalize_choice(obj.get("answer"))
    return None


def _extract_choice_from_regex(text: str) -> str | None:
    if not text:
        return None
    match = LETTER_PATTERN.search(text)
    return match.group(1).upper() if match else None


def _warn_regex(sample: dict | None = None) -> None:
    if sample and isinstance(sample, dict):
        runid = sample.get("runid")
        if runid is not None:
            print(
                f"Warning: structured answer not found for runid={runid}. Falling back to regex extraction."
            )
        else:
            print("Warning: structured answer not found. Falling back to regex extraction.")
    else:
        print("Warning: structured answer not found. Falling back to regex extraction.")


def extract_choice(target) -> str | None:
    """Extract the first-choice letter, preferring structured signals when available."""
    if isinstance(target, dict):
        sample = target
        ans = _normalize_choice(sample.get("model_answer"))
        if ans:
            return ans

        response_text = sample.get("response", "")
        match = ANSWER_LINE_PATTERN.search(response_text)
        if match:
            ans = _normalize_choice(match.group(1))
            if ans:
                return ans

        response_json = sample.get("response_json")
        if isinstance(response_json, dict):
            ans = _normalize_choice(response_json.get("answer"))
            if ans:
                return ans

        raw_text = sample.get("response_json_raw")
        if isinstance(raw_text, str):
            match_raw = ANSWER_LINE_PATTERN.search(raw_text)
            if match_raw:
                ans = _normalize_choice(match_raw.group(1))
                if ans:
                    return ans
            ans = _extract_from_json_text(raw_text)
            if ans:
                return ans

        ans = _extract_from_json_text(response_text)
        if ans:
            return ans

        _warn_regex(sample)
        return _extract_choice_from_regex(response_text)

    text = target or ""
    match = ANSWER_LINE_PATTERN.search(text)
    if match:
        ans = _normalize_choice(match.group(1))
        if ans:
            return ans

    ans = _extract_from_json_text(text)
    if ans is not None:
        return ans

    _warn_regex(None)
    return _extract_choice_from_regex(text)


def verify_func(sample: dict, ground_truth: str, timeout_score: float = 0.0) -> float:
    predicted = extract_choice(sample)
    if predicted is None:
        return timeout_score
    if isinstance(sample, dict):
        sample["model_answer"] = predicted
    return 1.0 if predicted == ground_truth.upper() else 0.0
