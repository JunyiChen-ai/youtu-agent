import json
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path("data/diff_awareness/dataset")


def _load_json(name: str) -> List[Dict[str, Any]]:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Diff dataset file not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_data(name: str) -> List[Dict[str, Any]]:
    if name == "D1_1k_train":
        return _load_json("D1_1k_train.json")
    if name == "D1_1k_test":
        return _load_json("D1_1k_test.json")
    raise ValueError(f"Unsupported dataset: {name}")
