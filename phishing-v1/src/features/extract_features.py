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
from src import load_config, setup_logging, get_project_root, get_tld_extractor

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

    try:
        ext = get_tld_extractor()(url)
    except Exception:
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

    digits_url = sum(c.isdigit() for c in url)
    digits_host = sum(c.isdigit() for c in hostname)
    ratio_digits_u = digits_url / len(url) if url else 0.0
    ratio_digits_h = digits_host / len(hostname) if hostname else 0.0

    features = {
        # ── 28 Lexical / structural features (original paper Table 5) ──
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
        "num_digits_in_url": digits_url,
        "num_digits_in_hostname": digits_host,
        "ratio_digits_url": ratio_digits_u,
        "ratio_digits_hostname": ratio_digits_h,

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
        "avg_token_length": np.mean([len(t) for t in tokens]) if tokens else 0.0,

        # Structural features
        "num_subdomains": len(subdomains),
        "tld_length": len(tld),
        "path_depth": len([p for p in path.split("/") if p]),

        # ── Refined CSV dataset features (when inferred per URL) ──────
        "length_url": len(url),
        "length_hostname": len(hostname),
        "ip": int(bool(IP_PATTERN.match(hostname))),
        "nb_dots": url.count("."),
        "nb_hyphens": url.count("-"),
        "nb_at": url.count("@"),
        "nb_qm": url.count("?"),
        "nb_and": url.count("&"),
        "nb_or": url.count("|"),
        "nb_eq": url.count("="),
        "nb_underscore": url.count("_"),
        "nb_tilde": url.count("~"),
        "nb_percent": url.count("%"),
        "nb_slash": url.count("/"),
        "nb_star": url.count("*"),
        "nb_colon": url.count(":"),
        "nb_comma": url.count(","),
        "nb_semicolumn": url.count(";"),
        "nb_dollar": url.count("$"),
        "nb_space": url.count(" "),
        "nb_www": int("www" in hostname.lower()),
        "nb_com": int(".com" in url.lower()),
        "nb_dslash": url.count("//"),
        "http_in_path": int("http" in path.lower()),
        "https_token": int(parsed.scheme == "https"),
        "ratio_digits_host": ratio_digits_h,
        "tld_in_path": int(bool(tld and tld in path)),
        "tld_in_subdomain": int(bool(tld and any(tld in s for s in subdomains))),
        "abnormal_subdomain": int(len(subdomains) > 2),
        "nb_subdomains": len(subdomains),
        "prefix_suffix": int("-" in hostname),
        "random_domain": 0,
        "shortening_service": int(any(s in hostname.lower() for s in ["bit.ly", "goo.gl", "tinyurl", "t.co", "is.gd", "buff.ly"])),
        "nb_redirection": 0,
        "nb_external_redirection": 0,
        "length_words_raw": len(tokens),
        "phish_hints": int(any(h in url.lower() for h in ["secure", "account", "update", "banking", "login", "verify", "signin"])),
        "domain_in_brand": 0,
        "brand_in_subdomain": 0,
        "brand_in_path": 0,
        "suspecious_tld": int(tld in ["zip", "cricket", "link", "work", "party", "gq", "kim", "country", "science", "tk"]),
        "statistical_report": 0,
    }

    return features


def extract_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    Extract features from a DataFrame with a 'url' column.

    If features are already present in df (from final_refined_dataset_phishing.csv),
    preserves them. Also appends any missing features needed by config.
    """
    if config is None:
        config = load_config()

    feature_set = config.get("features", {}).get("feature_set", "dataset")
    needed_features = get_feature_names(config)
    missing = [f for f in needed_features if f not in df.columns]

    if not missing and feature_set == "dataset":
        logger.info("All %d requested features already present in dataset.", len(needed_features))
        return df

    logger.info("Extracting features from %d URLs…", len(df))
    feature_dicts = df["url"].apply(extract_single_url)
    features_df = pd.DataFrame(feature_dicts.tolist(), index=df.index)

    # Only add columns not already in df to prevent duplication
    new_cols = [c for c in features_df.columns if c not in df.columns]
    result = pd.concat([df, features_df[new_cols]], axis=1)
    logger.info("Extracted %d features per URL (%d total columns).", len(new_cols), len(result.columns))
    return result


def get_feature_names(config: dict | None = None) -> list[str]:
    """Return the list of feature column names from config based on feature_set."""
    if config is None:
        config = load_config()
    features_cfg = config.get("features", {})
    feature_set = features_cfg.get("feature_set", "dataset")

    if feature_set == "dataset" and "dataset" in features_cfg:
        return features_cfg["dataset"]
    elif feature_set == "all":
        dataset_feats = features_cfg.get("dataset", [])
        lex_feats = features_cfg.get("lexical", [])
        combined = list(dataset_feats)
        for f in lex_feats:
            if f not in combined:
                combined.append(f)
        return combined
    return features_cfg.get("lexical", [])


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
        df_feat = extract_features(df, config=config)
        out_path = processed_dir / f"{split_file.stem}_features.csv"
        df_feat.to_csv(out_path, index=False)
        logger.info("Saved features for %s → %s (%d rows, %d cols).",
                     split_file.name, out_path.name, len(df_feat), len(df_feat.columns))


if __name__ == "__main__":
    run_feature_extraction()
