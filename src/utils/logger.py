"""Simple JSON logger for experiment tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class JsonLogger:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "metrics.jsonl"

    def log(self, metrics: Dict[str, Any]) -> None:
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    def save_config(self, config: Dict[str, Any]) -> None:
        (self.log_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
