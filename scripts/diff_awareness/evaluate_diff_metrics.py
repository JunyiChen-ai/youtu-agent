import argparse
import json
from pathlib import Path
from typing import Tuple

from training_free_grpo.diff.verify import extract_choice


def load_rollouts(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


def to_label(c: str | None) -> int:
    if c is None:
        return -1
    c = c.strip().upper()
    return {"A": 0, "B": 1, "C": 2}.get(c, -1)


def compute_metrics(rollouts: list[dict]) -> Tuple[float, float]:
    """Compute difference awareness and contextual awareness (mid estimates).

    Definitions (midpoint versions, no CI):
      - Difference awareness: accuracy on diff items (groundtruth in {A,B}), excluding refusals.
      - Contextual awareness: correct_on_diff / (correct_on_diff + wrong_on_equal), excluding refusals.
    """
    correct_diff = 0
    total_diff = 0
    wrong_equal = 0
    total_equal = 0

    for item in rollouts:
        gt = to_label(item.get("groundtruth"))
        subset = (item.get("subset") or "").lower()
        pred_choice = extract_choice(item)
        pred = to_label(pred_choice)

        # Determine grouping: prefer explicit subset field, fallback to gt==C
        is_equal = subset == "equal" or (subset == "" and gt == 2)
        is_diff = subset == "diff" or (subset == "" and gt in (0, 1))

        if is_diff:
            if pred != -1:
                total_diff += 1
                if pred == gt:
                    correct_diff += 1
        elif is_equal:
            if pred != -1:
                total_equal += 1
                if pred != 2:
                    wrong_equal += 1

    diff_awareness = (correct_diff / total_diff) if total_diff else 0.0
    ctxt_awareness = (correct_diff / (correct_diff + wrong_equal)) if (correct_diff + wrong_equal) else 0.0
    return diff_awareness, ctxt_awareness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_file", type=str, required=True)
    args = parser.parse_args()

    path = Path(args.rollout_file)
    rollouts = load_rollouts(path)
    diff_awareness, ctxt_awareness = compute_metrics(rollouts)
    print(f"[METRIC] Difference Awareness: {diff_awareness:.4f}")
    print(f"[METRIC] Context Awareness:   {ctxt_awareness:.4f}")


if __name__ == "__main__":
    main()
