from __future__ import annotations
"""
Fetch phishing URLs from PhishTank's open dataset.

PhishTank provides a CSV of verified phishing URLs.  If the live download
is unavailable (rate-limited or offline), the script falls back to a local
copy in data/raw/ if one exists.
"""
import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import load_config, setup_logging, get_project_root

logger = setup_logging("fetch_phishtank")


def fetch_phishtank(config: dict | None = None, force: bool = False) -> Path:
    """
    Download the PhishTank verified-online CSV to data/raw/phishtank.csv.

    Returns the path to the saved file.  If the file already exists and
    `force` is False, the download is skipped.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    raw_dir = root / config["paths"]["raw_data"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "phishtank.csv"

    if output_path.exists() and not force:
        logger.info("PhishTank data already present at %s — skipping download.", output_path)
        return output_path

    url = config["data"]["phishtank_url"]
    logger.info("Downloading PhishTank data from %s …", url)

    try:
        resp = requests.get(url, timeout=120, headers={"User-Agent": "phishing-v2-research"})
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to download PhishTank data: %s", exc)
        if output_path.exists():
            logger.warning("Using existing local copy at %s", output_path)
            return output_path
        raise RuntimeError(
            f"Cannot fetch PhishTank data and no local copy exists at {output_path}. "
            "Download manually from https://phishtank.org/developer_info.php and place "
            "the CSV at data/raw/phishtank.csv"
        ) from exc

    # PhishTank CSV columns: phish_id, url, phish_detail_url, submission_time,
    # verified, verified_time, online, target
    df = pd.read_csv(io.StringIO(resp.text))
    logger.info("Downloaded %d rows from PhishTank.", len(df))

    # Keep only verified-online phishing URLs
    if "verified" in df.columns and "online" in df.columns:
        before = len(df)
        df = df[(df["verified"] == "yes") & (df["online"] == "yes")]
        logger.info("Filtered to %d verified-online URLs (dropped %d).", len(df), before - len(df))

    # Standardise columns
    rename_map = {}
    if "url" in df.columns:
        rename_map["url"] = "url"
    if "submission_time" in df.columns:
        rename_map["submission_time"] = "submission_time"
    if "verified_time" in df.columns:
        rename_map["verified_time"] = "verified_time"
    if "target" in df.columns:
        rename_map["target"] = "target"

    df = df.rename(columns=rename_map)
    df["label"] = 1  # phishing

    # Apply dev-loop sample limit
    sample_size = config["data"].get("dev_sample_size")
    if sample_size is not None:
        df = df.head(sample_size)
        logger.info("Dev mode: limited to %d rows.", sample_size)

    df.to_csv(output_path, index=False)
    logger.info("Saved PhishTank data to %s (%d rows).", output_path, len(df))
    return output_path


if __name__ == "__main__":
    fetch_phishtank(force="--force" in sys.argv)
