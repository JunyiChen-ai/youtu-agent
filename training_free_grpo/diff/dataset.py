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
    """Load arbitrary diff dataset split by name.

    Expected names like: D1_1k_train, D1_1k_test, D1_100_train, D1_100_test, etc.
    The function directly maps to a JSON file of the same name under DATA_DIR.
    """
    filename = f"{name}.json"
    return _load_json(filename)
