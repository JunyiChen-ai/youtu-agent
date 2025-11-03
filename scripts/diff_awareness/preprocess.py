import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import List, Tuple


LABEL_MAP = {0: "A", 1: "B", 2: "C"}


def load_pkl(path: Path) -> Tuple[List[list], List[list]]:
    """Load benchmark_suite PKL: expected structure [non_c_list, c_list].

    Each entry is [question: str, label_int: int(0/1/2), orig_id: int].
    Returns (non_c, c) where:
      - non_c contains only labels in {0,1}
      - c contains only label 2
    """
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not (isinstance(obj, list) and len(obj) == 2):
        raise ValueError(f"Unexpected PKL structure in {path}")
    part0, part1 = obj
    # Basic validation
    assert all(isinstance(x, list) and len(x) == 3 for x in part0), "Bad entry in part0"
    assert all(isinstance(x, list) and len(x) == 3 for x in part1), "Bad entry in part1"
    return part0, part1


def build_split(non_c: List[list], c: List[list], n: int, seed: int) -> Tuple[list, list]:
    """Create train/test where train has n with C and n with A/B; test is remaining.

    Returns (train_list, test_list) as lists of dicts with keys: id, problem, groundtruth.
    """
    rnd = random.Random(seed)

    # Sample n from C (part1)
    if len(c) < n:
        raise ValueError(f"Requested n={n} C-samples but only {len(c)} available")
    c_indices = list(range(len(c)))
    rnd.shuffle(c_indices)
    c_train_idx = set(c_indices[:n])

    # Sample n from non-C (part0); mix of A/B unconstrained
    if len(non_c) < n:
        raise ValueError(f"Requested n={n} non-C samples but only {len(non_c)} available")
    ab_indices = list(range(len(non_c)))
    rnd.shuffle(ab_indices)
    ab_train_idx = set(ab_indices[:n])

    # Build train
    train = []
    for idx in sorted(ab_train_idx):
        q, lab, orig_id = non_c[idx]
        train.append({
            "id": len(train),
            "problem": q,
            "groundtruth": LABEL_MAP[int(lab)],  # A/B as-is for diff subset
            "subset": "diff",
            "orig_label": LABEL_MAP[int(lab)],
            "orig_id": str(orig_id),
        })
    for idx in sorted(c_train_idx):
        q, lab, orig_id = c[idx]
        train.append({
            "id": len(train),
            "problem": q,
            # Manually set equal subset answers to C per spec (even if original label is A/B)
            "groundtruth": "C",
            "subset": "equal",
            "orig_label": LABEL_MAP[int(lab)],
            "orig_id": str(orig_id),
        })

    # Build test = remaining
    test = []
    # remaining non-C
    for i, (q, lab, orig_id) in enumerate(non_c):
        if i in ab_train_idx:
            continue
        test.append({
            "id": len(test),
            "problem": q,
            "groundtruth": LABEL_MAP[int(lab)],
            "subset": "diff",
            "orig_label": LABEL_MAP[int(lab)],
            "orig_id": str(orig_id),
        })
    # remaining C
    for i, (q, lab, orig_id) in enumerate(c):
        if i in c_train_idx:
            continue
        test.append({
            "id": len(test),
            "problem": q,
            # Manually set equal subset answers to C per spec
            "groundtruth": "C",
            "subset": "equal",
            "orig_label": LABEL_MAP[int(lab)],
            "orig_id": str(orig_id),
        })

    return train, test


def main():
    parser = argparse.ArgumentParser(description="Preprocess diff awareness PKL into train/test JSON")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset base name, e.g., D1_1k")
    parser.add_argument("--n", type=int, required=True, help="Number of C and non-C samples each for train")
    parser.add_argument("--source_dir", type=str, default="data/diff_awareness/benchmark_suite", help="Dir with PKL files (symlinked)")
    parser.add_argument("--out_dir", type=str, default="data/diff_awareness/dataset", help="Output directory for JSON files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = source_dir / f"{args.dataset_name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL not found: {pkl_path}")

    non_c, c = load_pkl(pkl_path)
    train, test = build_split(non_c, c, n=args.n, seed=args.seed)

    # Output naming: {base}_{n}_train/test.json, where base is prefix before first underscore.
    base = args.dataset_name.split("_")[0]
    train_path = out_dir / f"{base}_{args.n}_train.json"
    test_path = out_dir / f"{base}_{args.n}_test.json"

    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with test_path.open("w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(train)} train items -> {train_path}")
    print(f"Wrote {len(test)} test items  -> {test_path}")


if __name__ == "__main__":
    main()
