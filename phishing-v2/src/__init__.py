from __future__ import annotations
"""
Shared utilities: config loading, logging, git commit tracking, result writing.
"""
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def get_project_root() -> Path:
    """Return the project root (phishing-v2/) directory."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | None = None) -> dict:
    """Load config.yaml from the project root or a specified path."""
    if config_path is None:
        config_path = get_project_root() / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(name: str = "phishing-v2", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_git_commit() -> str:
    """Return the current short git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(get_project_root())
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def result_metadata() -> dict:
    """Return standard metadata dict for every result file."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "python_version": sys.version,
    }


def save_result_json(data: dict, filepath: str | Path, *, extra_meta: dict | None = None) -> None:
    """Save a result dict as JSON with metadata."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    payload = {**result_metadata(), **(extra_meta or {}), "results": data}
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def save_result_csv(df, filepath: str | Path) -> None:
    """Save a DataFrame as CSV with a metadata header comment."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    meta = result_metadata()
    with open(filepath, "w") as f:
        f.write(f"# timestamp: {meta['timestamp']}\n")
        f.write(f"# git_commit: {meta['git_commit']}\n")
        df.to_csv(f, index=False)


def ensure_dirs(config: dict) -> None:
    """Create all directories listed in config['paths']."""
    root = get_project_root()
    for key, rel_path in config["paths"].items():
        if key == "ontology":
            (root / rel_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            (root / rel_path).mkdir(parents=True, exist_ok=True)
