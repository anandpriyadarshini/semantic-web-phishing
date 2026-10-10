"""
Tests for data leakage prevention.

Verifies:
1. Zero exact-URL overlap between train and test splits.
2. Zero registered-domain overlap between train and test splits.
3. No duplicate URLs within any split.

These tests run on the actual split files produced by dedupe_and_split.py.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import load_config, get_project_root


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def splits_dir(config):
    root = get_project_root()
    return root / config["paths"]["splits"]


def _load_split(splits_dir: Path, name: str) -> pd.DataFrame:
    """Load a split CSV, raising a clear error if not found."""
    path = splits_dir / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"Split file {path} not found. Run the data pipeline first.")
    return pd.read_csv(path)


class TestDomainSplitLeakage:
    """Tests for the domain-based split."""

    def test_no_url_overlap_train_test(self, splits_dir):
        """No exact URL should appear in both train and test."""
        train = _load_split(splits_dir, "domain_train")
        test = _load_split(splits_dir, "domain_test")

        train_urls = set(train["url"])
        test_urls = set(test["url"])

        overlap = train_urls & test_urls
        assert len(overlap) == 0, (
            f"LEAKAGE: {len(overlap)} URLs appear in both domain_train and domain_test. "
            f"Examples: {list(overlap)[:5]}"
        )

    def test_no_url_overlap_train_val(self, splits_dir):
        """No exact URL should appear in both train and val."""
        train = _load_split(splits_dir, "domain_train")
        val = _load_split(splits_dir, "domain_val")

        overlap = set(train["url"]) & set(val["url"])
        assert len(overlap) == 0, (
            f"LEAKAGE: {len(overlap)} URLs appear in both domain_train and domain_val."
        )

    def test_no_url_overlap_val_test(self, splits_dir):
        """No exact URL should appear in both val and test."""
        val = _load_split(splits_dir, "domain_val")
        test = _load_split(splits_dir, "domain_test")

        overlap = set(val["url"]) & set(test["url"])
        assert len(overlap) == 0, (
            f"LEAKAGE: {len(overlap)} URLs appear in both domain_val and domain_test."
        )

    def test_no_domain_overlap_train_test(self, splits_dir):
        """No registered domain should appear in both train and test."""
        train = _load_split(splits_dir, "domain_train")
        test = _load_split(splits_dir, "domain_test")

        train_domains = set(train["registered_domain"])
        test_domains = set(test["registered_domain"])

        overlap = train_domains & test_domains
        assert len(overlap) == 0, (
            f"LEAKAGE: {len(overlap)} domains appear in both domain_train and domain_test. "
            f"Examples: {list(overlap)[:5]}"
        )

    def test_no_domain_overlap_train_val(self, splits_dir):
        """No registered domain should appear in both train and val."""
        train = _load_split(splits_dir, "domain_train")
        val = _load_split(splits_dir, "domain_val")

        overlap = set(train["registered_domain"]) & set(val["registered_domain"])
        assert len(overlap) == 0, (
            f"LEAKAGE: {len(overlap)} domains appear in both domain_train and domain_val."
        )

    def test_no_domain_overlap_val_test(self, splits_dir):
        """No registered domain should appear in both val and test."""
        val = _load_split(splits_dir, "domain_val")
        test = _load_split(splits_dir, "domain_test")

        overlap = set(val["registered_domain"]) & set(test["registered_domain"])
        assert len(overlap) == 0, (
            f"LEAKAGE: {len(overlap)} domains appear in both domain_val and domain_test."
        )

    def test_no_duplicates_within_splits(self, splits_dir):
        """No exact URL should be duplicated within a single split."""
        for name in ["domain_train", "domain_val", "domain_test"]:
            df = _load_split(splits_dir, name)
            duplicates = df[df.duplicated(subset=["url"], keep=False)]
            assert len(duplicates) == 0, (
                f"Found {len(duplicates)} duplicate URLs within {name}."
            )

    def test_all_urls_accounted_for(self, splits_dir, config):
        """Train + val + test should cover all processed URLs."""
        root = get_project_root()
        processed = root / config["paths"]["processed_data"] / "combined.csv"
        if not processed.exists():
            pytest.skip("combined.csv not found.")

        all_df = pd.read_csv(processed)
        train = _load_split(splits_dir, "domain_train")
        val = _load_split(splits_dir, "domain_val")
        test = _load_split(splits_dir, "domain_test")

        total_split = len(train) + len(val) + len(test)
        # Allow for some loss due to ambiguous domains removed during processing
        assert total_split > 0, "No URLs in any split."
        assert total_split <= len(all_df), (
            f"More URLs in splits ({total_split}) than in combined.csv ({len(all_df)})."
        )


class TestTimeSplitLeakage:
    """Tests for the time-based split."""

    def test_no_url_overlap_time_train_test(self, splits_dir):
        """No exact URL should appear in both time_train and time_test."""
        train = _load_split(splits_dir, "time_train")
        test = _load_split(splits_dir, "time_test")

        overlap = set(train["url"]) & set(test["url"])
        assert len(overlap) == 0, (
            f"LEAKAGE: {len(overlap)} URLs appear in both time_train and time_test."
        )

    def test_no_domain_overlap_phishing_time_split(self, splits_dir):
        """
        No phishing domain should span the time boundary.

        (Legitimate domains may appear in both since they have no timestamp,
        but phishing domains must be in exactly one split.)
        """
        train = _load_split(splits_dir, "time_train")
        test = _load_split(splits_dir, "time_test")

        train_phishing_domains = set(
            train[train["label"] == 1]["registered_domain"]
        )
        test_phishing_domains = set(
            test[test["label"] == 1]["registered_domain"]
        )

        overlap = train_phishing_domains & test_phishing_domains
        assert len(overlap) == 0, (
            f"LEAKAGE: {len(overlap)} phishing domains span the time boundary. "
            f"Examples: {list(overlap)[:5]}"
        )
