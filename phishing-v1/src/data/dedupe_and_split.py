from __future__ import annotations
"""
Deduplication and splitting with leakage prevention.

This module enforces two critical invariants:
1. No exact URL appears in both train and test.
2. All URLs sharing a registered domain go entirely into train OR test.

It produces both domain-based and time-based splits.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tldextract

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import load_config, setup_logging, get_project_root, get_tld_extractor

logger = setup_logging("dedupe_and_split")


def extract_registered_domain(url: str) -> str:
    """Extract the registered domain (domain + suffix) from a URL."""
    try:
        ext = get_tld_extractor()(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
        return ext.domain or url
    except Exception:
        return url


def dedupe_and_split(config: dict | None = None) -> dict[str, Path]:
    """
    Load phishing + legitimate data, deduplicate, and produce splits.

    Supports both 'refined_csv' (final_refined_dataset_phishing.csv)
    and 'phishtank' data sources.

    Returns a dict mapping split names to file paths.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    raw_dir = root / config["paths"]["raw_data"]
    processed_dir = root / config["paths"]["processed_data"]
    splits_dir = root / config["paths"]["splits"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    seed = config["random_seed"]
    rng = np.random.RandomState(seed)

    # ── Load data based on source ──────────────────────────────────────
    source = config.get("data", {}).get("source", "refined_csv")

    if source == "refined_csv":
        dataset_name = config.get("data", {}).get("refined_csv_path", "final_refined_dataset_phishing.csv")
        possible_paths = [
            root / dataset_name,
            root / "final_refined_dataset_phishing.csv",
            root / "final_ refined_dataset_phishing.csv",
            raw_dir / dataset_name,
            raw_dir / "final_refined_dataset_phishing.csv",
            raw_dir / "final_ refined_dataset_phishing.csv",
        ]
        csv_path = None
        for p in possible_paths:
            if p.exists():
                csv_path = p
                break
        if csv_path is None:
            raise FileNotFoundError(
                f"Refined dataset not found at any of: {[str(p) for p in possible_paths]}"
            )

        logger.info("Loading refined dataset from %s…", csv_path)
        df = pd.read_csv(csv_path)
        logger.info("Loaded %d rows from %s.", len(df), csv_path.name)

        if "label" not in df.columns:
            if "status" in df.columns:
                df["label"] = df["status"].apply(
                    lambda s: 1 if str(s).strip().lower() in ("phishing", "1", "true") else 0
                )
            else:
                raise ValueError("Refined CSV must contain 'status' or 'label' column.")

        # Optional dev sampling
        sample_size = config["data"].get("dev_sample_size")
        if sample_size is not None and sample_size > 0:
            df = df.groupby("label", group_keys=False).apply(
                lambda x: x.sample(min(len(x), sample_size), random_state=seed)
            ).reset_index(drop=True)
            logger.info("Dev mode: sampled %d rows (%d per class).", len(df), sample_size)

    else:
        # PhishTank + Tranco raw data
        phish_path = raw_dir / "phishtank.csv"
        legit_path = raw_dir / "legitimate.csv"

        if not phish_path.exists():
            raise FileNotFoundError(f"PhishTank data not found at {phish_path}. Run fetch_phishtank.py first.")
        if not legit_path.exists():
            raise FileNotFoundError(f"Legitimate data not found at {legit_path}. Run fetch_legitimate.py first.")

        df_phish = pd.read_csv(phish_path, comment="#")
        df_legit = pd.read_csv(legit_path, comment="#")

        logger.info("Loaded %d phishing and %d legitimate URLs.", len(df_phish), len(df_legit))

        assert "url" in df_phish.columns, "PhishTank CSV must have a 'url' column"
        assert "url" in df_legit.columns, "Legitimate CSV must have a 'url' column"

        if "label" not in df_phish.columns:
            df_phish["label"] = 1
        if "label" not in df_legit.columns:
            df_legit["label"] = 0

        df = pd.concat([df_phish, df_legit], ignore_index=True)

    # ── Step 1: Deduplicate on exact URL ──────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first")
    logger.info("Exact-URL dedup: %d → %d (dropped %d).", before, len(df), before - len(df))

    # ── Step 2: Extract registered domains ────────────────────────────
    df["registered_domain"] = df["url"].apply(extract_registered_domain)

    # ── Step 3: Deduplicate on registered domain within each class ────
    # (keep all URLs, but ensure domain consistency)
    domain_labels = df.groupby("registered_domain")["label"].nunique()
    ambiguous_domains = domain_labels[domain_labels > 1].index
    if len(ambiguous_domains) > 0:
        logger.warning(
            "%d domains appear in BOTH phishing and legitimate sets. "
            "Removing these domains to prevent label ambiguity: %s",
            len(ambiguous_domains),
            list(ambiguous_domains)[:10],
        )
        df = df[~df["registered_domain"].isin(ambiguous_domains)]

    # ── Save processed data ───────────────────────────────────────────
    processed_path = processed_dir / "combined.csv"
    df.to_csv(processed_path, index=False)
    logger.info("Saved processed data to %s (%d rows).", processed_path, len(df))

    # ── Step 4: Domain-based split ────────────────────────────────────
    output_paths = {}
    output_paths.update(_domain_split(df, splits_dir, config, rng))

    # ── Step 5: Time-based split (phishing only has timestamps) ───────
    output_paths.update(_time_split(df, raw_dir, splits_dir, config, rng))

    return output_paths


def _domain_split(df: pd.DataFrame, splits_dir: Path, config: dict,
                  rng: np.random.RandomState) -> dict[str, Path]:
    """
    Split by registered domain: all URLs from a domain go to the same split.
    """
    test_frac = config["splitting"]["test_fraction"]
    val_frac = config["splitting"]["val_fraction"]

    # Get unique domains per class
    domains = df["registered_domain"].unique()
    rng.shuffle(domains)

    n_test = int(len(domains) * test_frac)
    n_val = int(len(domains) * val_frac)

    test_domains = set(domains[:n_test])
    val_domains = set(domains[n_test:n_test + n_val])
    train_domains = set(domains[n_test + n_val:])

    df_train = df[df["registered_domain"].isin(train_domains)].copy()
    df_val = df[df["registered_domain"].isin(val_domains)].copy()
    df_test = df[df["registered_domain"].isin(test_domains)].copy()

    # Verify no domain overlap
    assert len(train_domains & test_domains) == 0, "LEAKAGE: domain overlap train/test"
    assert len(train_domains & val_domains) == 0, "LEAKAGE: domain overlap train/val"
    assert len(val_domains & test_domains) == 0, "LEAKAGE: domain overlap val/test"

    # Verify no URL overlap
    train_urls = set(df_train["url"])
    val_urls = set(df_val["url"])
    test_urls = set(df_test["url"])
    assert len(train_urls & test_urls) == 0, "LEAKAGE: URL overlap train/test"
    assert len(train_urls & val_urls) == 0, "LEAKAGE: URL overlap train/val"
    assert len(val_urls & test_urls) == 0, "LEAKAGE: URL overlap val/test"

    paths = {}
    for name, split_df in [("domain_train", df_train), ("domain_val", df_val), ("domain_test", df_test)]:
        p = splits_dir / f"{name}.csv"
        split_df.to_csv(p, index=False)
        paths[name] = p
        logger.info(
            "Domain split '%s': %d URLs, %d phishing, %d legit, %d domains.",
            name, len(split_df),
            (split_df["label"] == 1).sum(),
            (split_df["label"] == 0).sum(),
            split_df["registered_domain"].nunique(),
        )

    return paths


def _time_split(df: pd.DataFrame, raw_dir: Path, splits_dir: Path,
                config: dict, rng: np.random.RandomState) -> dict[str, Path]:
    """
    Time-based split: train on older PhishTank entries, test on newer ones.

    Legitimate URLs have no inherent timestamp, so we assign them to both
    splits proportionally, still respecting the domain constraint.
    """
    train_pct = config["splitting"]["time_split_train_percentile"]

    source = config.get("data", {}).get("source", "refined_csv")
    phish_path = raw_dir / "phishtank.csv"
    time_col = None
    if source == "phishtank" and phish_path.exists():
        try:
            df_phish_raw = pd.read_csv(phish_path, comment="#")
            for candidate in ["submission_time", "verified_time"]:
                if candidate in df_phish_raw.columns:
                    time_col = candidate
                    break
        except Exception as e:
            logger.warning("Could not read PhishTank timestamps: %s", e)
            time_col = None

    phish_train_urls = set()
    phish_test_urls = set()

    if time_col is not None:
        df_phish_raw[time_col] = pd.to_datetime(df_phish_raw[time_col], errors="coerce")
        df_phish_raw = df_phish_raw.dropna(subset=[time_col])
        cutoff_time = df_phish_raw[time_col].quantile(train_pct / 100.0)
        url_to_time = dict(zip(df_phish_raw["url"], df_phish_raw[time_col]))

        df_phish_urls = df[df["label"] == 1].copy()
        df_phish_urls["_timestamp"] = df_phish_urls["url"].map(url_to_time)
        matched = df_phish_urls.dropna(subset=["_timestamp"])
        if len(matched) > 0.5 * len(df_phish_urls):
            logger.info("Time split cutoff: %s (percentile %d).", cutoff_time, train_pct)
            phish_train_urls = set(matched[matched["_timestamp"] <= cutoff_time]["url"])
            phish_test_urls = set(matched[matched["_timestamp"] > cutoff_time]["url"])

    if not phish_train_urls:
        logger.info(
            "Using index order proxy for time split (train percentile: %d%%).",
            train_pct,
        )
        df_phish_urls = df[df["label"] == 1].copy()
        df_phish_urls["_time_proxy"] = range(len(df_phish_urls))
        cutoff = np.percentile(df_phish_urls["_time_proxy"], train_pct)
        phish_train_urls = set(df_phish_urls[df_phish_urls["_time_proxy"] <= cutoff]["url"])
        phish_test_urls = set(df_phish_urls[df_phish_urls["_time_proxy"] > cutoff]["url"])

    # Legitimate URLs: split proportionally by domain
    df_legit = df[df["label"] == 0].copy()
    legit_domains = df_legit["registered_domain"].unique()
    rng.shuffle(legit_domains)
    n_legit_train = int(len(legit_domains) * (train_pct / 100.0))
    legit_train_domains = set(legit_domains[:n_legit_train])
    legit_test_domains = set(legit_domains[n_legit_train:])

    # Build final splits
    train_mask = (
        (df["url"].isin(phish_train_urls)) |
        ((df["label"] == 0) & df["registered_domain"].isin(legit_train_domains))
    )
    test_mask = (
        (df["url"].isin(phish_test_urls)) |
        ((df["label"] == 0) & df["registered_domain"].isin(legit_test_domains))
    )

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    # Enforce domain constraint: if a domain has URLs in both, assign all to train
    train_domains_phish = set(df_train[df_train["label"] == 1]["registered_domain"])
    test_domains_phish = set(df_test[df_test["label"] == 1]["registered_domain"])
    overlap_domains = train_domains_phish & test_domains_phish
    if overlap_domains:
        logger.warning(
            "Time split: %d phishing domains span the time boundary. "
            "Assigning all their URLs to train to prevent leakage.",
            len(overlap_domains),
        )
        move_mask = df_test["registered_domain"].isin(overlap_domains) & (df_test["label"] == 1)
        df_train = pd.concat([df_train, df_test[move_mask]], ignore_index=True)
        df_test = df_test[~move_mask]

    paths = {}
    for name, split_df in [("time_train", df_train), ("time_test", df_test)]:
        p = splits_dir / f"{name}.csv"
        split_df.to_csv(p, index=False)
        paths[name] = p
        logger.info(
            "Time split '%s': %d URLs, %d phishing, %d legit.",
            name, len(split_df),
            (split_df["label"] == 1).sum(),
            (split_df["label"] == 0).sum(),
        )

    return paths


if __name__ == "__main__":
    dedupe_and_split()
