"""
Tests for XAI metric computation.

Validates that:
1. Fidelity metrics (deletion/insertion) produce values in [0, 1].
2. Stability (Jaccard@k) produces values in [0, 1].
3. Kendall's tau is computed correctly on known inputs.
4. Sparsity is computed correctly.
5. Metrics degrade gracefully on edge-case inputs (zero attribution, constant model).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xai.xai_metrics import (
    fidelity_deletion,
    fidelity_insertion,
    rank_agreement,
    sparsity,
)


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def toy_model_and_data():
    """Fit a small RandomForest on a toy dataset."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42,
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=20, random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe, X, y, feature_names


@pytest.fixture(scope="module")
def toy_attributions(toy_model_and_data):
    """Generate random attributions for testing."""
    _, X, _, feature_names = toy_model_and_data
    rng = np.random.RandomState(42)
    # Simulate SHAP-like attributions: random values, some near-zero
    attributions = rng.randn(50, X.shape[1])
    attributions[:, :3] = 0.001 * rng.randn(50, 3)  # Make some near-zero for sparsity
    return attributions


# ── Fidelity Tests ─────────────────────────────────────────────────

class TestFidelityDeletion:
    def test_returns_expected_keys(self, toy_model_and_data, toy_attributions):
        model, X, _, feature_names = toy_model_and_data
        result = fidelity_deletion(model, X[:50], toy_attributions, feature_names, top_k_values=[3, 5])
        assert "deletion@3" in result
        assert "deletion@5" in result

    def test_mean_drop_in_range(self, toy_model_and_data, toy_attributions):
        """Mean prediction drop should be non-negative (can't guarantee > 0 on toy data)."""
        model, X, _, feature_names = toy_model_and_data
        result = fidelity_deletion(model, X[:50], toy_attributions, feature_names, top_k_values=[3])
        drop = result["deletion@3"]["mean_drop"]
        assert drop >= 0.0, f"Mean drop should be non-negative, got {drop}"
        assert drop <= 1.0, f"Mean drop should be ≤ 1.0, got {drop}"

    def test_handles_k_larger_than_features(self, toy_model_and_data, toy_attributions):
        """Should not raise when k > number of features."""
        model, X, _, feature_names = toy_model_and_data
        result = fidelity_deletion(model, X[:20], toy_attributions[:20], feature_names, top_k_values=[1000])
        assert "deletion@10" in result  # Capped at n_features


class TestFidelityInsertion:
    def test_returns_expected_keys(self, toy_model_and_data, toy_attributions):
        model, X, _, feature_names = toy_model_and_data
        result = fidelity_insertion(model, X[:50], toy_attributions, feature_names, top_k_values=[3, 5])
        assert "insertion@3" in result
        assert "insertion@5" in result

    def test_mean_rise_in_range(self, toy_model_and_data, toy_attributions):
        model, X, _, feature_names = toy_model_and_data
        result = fidelity_insertion(model, X[:50], toy_attributions, feature_names, top_k_values=[5])
        rise = result["insertion@5"]["mean_rise"]
        assert rise >= 0.0, f"Mean rise should be non-negative, got {rise}"
        assert rise <= 1.0, f"Mean rise should be ≤ 1.0, got {rise}"


# ── Rank Agreement Tests ───────────────────────────────────────────

class TestRankAgreement:
    def test_identical_attributions_tau_equals_one(self):
        """Identical attributions should give Kendall's tau = 1.0."""
        attrs = np.array([
            [0.5, 0.3, 0.1, 0.8, 0.2],
            [0.9, 0.1, 0.4, 0.6, 0.3],
        ])
        result = rank_agreement(attrs, attrs, [f"f{i}" for i in range(5)])
        assert abs(result["kendall_tau_mean"] - 1.0) < 1e-6, (
            f"Identical attributions should give tau=1, got {result['kendall_tau_mean']}"
        )

    def test_reversed_attributions_tau_negative(self):
        """Reversed rank ordering should give negative tau."""
        n_features = 5
        attrs_a = np.array([[float(i) for i in range(n_features)]])
        attrs_b = np.array([[float(n_features - 1 - i) for i in range(n_features)]])
        result = rank_agreement(attrs_a, attrs_b, [f"f{i}" for i in range(n_features)])
        assert result["kendall_tau_mean"] < 0, (
            f"Reversed ranks should give negative tau, got {result['kendall_tau_mean']}"
        )

    def test_tau_in_range(self, toy_attributions):
        """Tau should always be in [-1, 1]."""
        rng = np.random.RandomState(99)
        other_attrs = rng.randn(*toy_attributions.shape)
        result = rank_agreement(toy_attributions, other_attrs, [f"f{i}" for i in range(toy_attributions.shape[1])])
        tau = result["kendall_tau_mean"]
        assert -1.0 <= tau <= 1.0, f"Kendall's tau out of range: {tau}"

    def test_handles_mismatched_lengths(self):
        """Should handle arrays of different lengths gracefully."""
        attrs_a = np.array([[0.5, 0.3, 0.1]])
        attrs_b = np.array([[0.5, 0.3, 0.1], [0.9, 0.1, 0.4]])
        result = rank_agreement(attrs_a, attrs_b, ["f0", "f1", "f2"])
        assert "kendall_tau_mean" in result

    def test_n_samples_reported_correctly(self, toy_attributions):
        """n_samples_compared should equal min(len(a), len(b))."""
        result = rank_agreement(toy_attributions, toy_attributions, [f"f{i}" for i in range(toy_attributions.shape[1])])
        assert result["n_samples_compared"] == len(toy_attributions)


# ── Sparsity Tests ─────────────────────────────────────────────────

class TestSparsity:
    def test_all_equal_attribution_low_sparsity(self):
        """Uniform attributions should produce low sparsity."""
        attrs = np.ones((10, 5))
        result = sparsity(attrs, threshold=0.01)
        # All features equal, so relative threshold doesn't distinguish — sparsity = 0
        assert result["mean_sparsity"] < 0.2, (
            f"Uniform attributions should have low sparsity, got {result['mean_sparsity']}"
        )

    def test_one_hot_attribution_high_sparsity(self):
        """Single-feature attribution should produce high sparsity."""
        attrs = np.zeros((10, 5))
        attrs[:, 0] = 1.0  # Only first feature has attribution
        result = sparsity(attrs, threshold=0.01)
        # 4/5 features are near-zero → sparsity = 0.8
        assert result["mean_sparsity"] > 0.7, (
            f"One-hot attribution should have high sparsity, got {result['mean_sparsity']}"
        )

    def test_sparsity_in_range(self, toy_attributions):
        """Sparsity should be in [0, 1]."""
        result = sparsity(toy_attributions)
        assert 0.0 <= result["mean_sparsity"] <= 1.0

    def test_all_zero_attribution_handled(self):
        """Zero attribution should not cause division-by-zero."""
        attrs = np.zeros((5, 4))
        result = sparsity(attrs)
        assert result["mean_sparsity"] == 1.0  # All near-zero relative to max=0

    def test_threshold_reported(self):
        """Threshold should be included in output."""
        attrs = np.random.randn(5, 4)
        result = sparsity(attrs, threshold=0.05)
        assert result["threshold"] == 0.05


# ── Integration Test ───────────────────────────────────────────────

class TestMetricsIntegration:
    def test_full_pipeline_on_toy_data(self, toy_model_and_data, toy_attributions):
        """Run all metrics end-to-end and ensure no exceptions."""
        model, X, _, feature_names = toy_model_and_data
        X_sub = X[:50]

        fd = fidelity_deletion(model, X_sub, toy_attributions, feature_names, [3])
        fi = fidelity_insertion(model, X_sub, toy_attributions, feature_names, [3])
        ra = rank_agreement(toy_attributions, toy_attributions, feature_names)
        sp = sparsity(toy_attributions)

        assert isinstance(fd, dict)
        assert isinstance(fi, dict)
        assert isinstance(ra, dict)
        assert isinstance(sp, dict)
