from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

from .verify import extract_choice


class ExperienceUpdater:
    """A lightweight updater that keeps successful answer snippets."""

    def __init__(self, max_experiences: int = 50):
        self.max_experiences = max_experiences

    def run(
        self,
        rollouts: List[Dict[str, Any]],
        experiences: Dict[str, str],
        save_dir: str | Path,
        **_: Any,
    ) -> Dict[str, str]:
        updated = OrderedDict(experiences)

        success_snippets: List[str] = []
        for entry in rollouts:
            if entry.get("reward", 0) <= 0:
                continue
            choice = extract_choice(entry.get("response", ""))
            if not choice:
                continue
            problem = entry.get("problem", "")
            summary = f"For similar questions, choose option {choice}. Question snippet: {problem[:160]}"
            success_snippets.append(summary)

        idx = len(updated)
        for snippet in success_snippets:
            updated[f"G{idx}"] = snippet
            idx += 1

        if len(updated) > self.max_experiences:
            # keep the latest entries
            trimmed_items = list(updated.items())[-self.max_experiences :]
            updated = OrderedDict(trimmed_items)

        save_path = Path(save_dir) / "experience_log.json"
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2)

        return dict(updated)
