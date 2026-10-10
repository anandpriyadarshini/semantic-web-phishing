from __future__ import annotations
"""
Extract lexical and structural features from URLs.

Features are computed entirely from the URL string itself (no network
lookups), corresponding to the features in Table 5 of the original paper.
"""
import math
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import tldextract

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import load_config, setup_logging, get_project_root

logger = setup_logging("extract_features")

# Regex for detecting IP addresses in hostnames
IP_PATTERN = re.compile(
    r"^(?:\d{1,3}\.){3}\d{1,3}$"  # IPv4
    r"|^\[?[0-9a-fA-F:]+\]?$"      # IPv6
)

# Regex for hex-encoded characters
HEX_PATTERN = re.compile(r"%[0-9a-fA-F]{2}")


def _entropy(s: str) -> float:
    """Shannon entropy of a string."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum(
        (c / length) * math.log2(c / length)
        for c in counts.values()
    )


def _tokenize_url(url: str) -> list[str]:
    """Split URL into tokens on common delimiters."""
    return re.split(r"[/\.\-_?=&@#:]+", url)


def extract_single_url(url: str) -> dict:
    """
    Extract all lexical/structural features from a single URL string.

    Returns a dict of feature_name -> value.
    """
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
    except Exception:
        parsed = urlparse(f"http://{url}")

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    ext = tldextract.extract(url)
    subdomains = ext.subdomain.split(".") if ext.subdomain else []
    tld = ext.suffix or ""

    tokens = _tokenize_url(url)
    tokens = [t for t in tokens if t]  # remove empties

    # Query parameters
    try:
        query_params = parse_qs(query)
    except Exception:
        query_params = {}

    features = {
        # Length features
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),

        # Count features
        "num_dots_in_url": url.count("."),
        "num_dots_in_hostname": hostname.count("."),
        "num_hyphens_in_url": url.count("-"),
        "num_hyphens_in_hostname": hostname.count("-"),
        "num_underscores_in_url": url.count("_"),
        "num_slashes_in_path": path.count("/"),
        "num_query_params": len(query_params),
        "num_fragments": 1 if fragment else 0,

        # Digit features
        "num_digits_in_url": sum(c.isdigit() for c in url),
        "num_digits_in_hostname": sum(c.isdigit() for c in hostname),
        "ratio_digits_url": (
            sum(c.isdigit() for c in url) / len(url) if url else 0.0
        ),
        "ratio_digits_hostname": (
            sum(c.isdigit() for c in hostname) / len(hostname) if hostname else 0.0
        ),

        # Boolean features (cast to int for ML)
        "has_ip_address": int(bool(IP_PATTERN.match(hostname))),
        "has_at_symbol": int("@" in url),
        "has_double_slash_redirect": int("//" in path),
        "has_hex_encoding": int(bool(HEX_PATTERN.search(url))),
        "has_www": int(hostname.startswith("www.")),
        "has_https": int(parsed.scheme == "https"),

        # Entropy features
        "url_entropy": _entropy(url),
        "hostname_entropy": _entropy(hostname),

        # Token features
        "longest_token_length": max((len(t) for t in tokens), default=0),
        "avg_token_length": (
            np.mean([len(t) for t in tokens]) if tokens else 0.0
        ),

        # Structural features
        "num_subdomains": len(subdomains),
        "tld_length": len(tld),
        "path_depth": len([p for p in path.split("/") if p]),
    }

    return features


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from a DataFrame with a 'url' column.

    Returns a new DataFrame with feature columns appended.
    """
    logger.info("Extracting features from %d URLs…", len(df))
    feature_dicts = df["url"].apply(extract_single_url)
    features_df = pd.DataFrame(feature_dicts.tolist(), index=df.index)
    result = pd.concat([df, features_df], axis=1)
    logger.info("Extracted %d features per URL.", len(features_df.columns))
    return result


def get_feature_names(config: dict | None = None) -> list[str]:
    """Return the list of feature column names from config."""
    if config is None:
        config = load_config()
    return config["features"]["lexical"]


def run_feature_extraction(config: dict | None = None) -> None:
    """
    Run feature extraction on all splits and save to processed/.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    splits_dir = root / config["paths"]["splits"]
    processed_dir = root / config["paths"]["processed_data"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    split_files = list(splits_dir.glob("*.csv"))
    if not split_files:
        raise FileNotFoundError(f"No split files found in {splits_dir}. Run dedupe_and_split.py first.")

    for split_file in split_files:
        df = pd.read_csv(split_file)
        df_feat = extract_features(df)
        out_path = processed_dir / f"{split_file.stem}_features.csv"
        df_feat.to_csv(out_path, index=False)
        logger.info("Saved features for %s → %s (%d rows, %d cols).",
                     split_file.name, out_path.name, len(df_feat), len(df_feat.columns))


if __name__ == "__main__":
    run_feature_extraction()
