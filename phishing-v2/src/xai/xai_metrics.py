from __future__ import annotations
"""
Quantitative XAI evaluation metrics.

All metrics are objectively computable — NO subjective 1–10 scoring.

Metrics implemented:
- Fidelity (deletion & insertion): how much does the model's prediction change
  when the top-k most important features are removed/added?
- Stability (Jaccard@k): how consistent are the top-k explanations across
  different random seeds / perturbations?
- SHAP–LIME rank agreement (Kendall's tau): how well do the two explanation
  methods agree on feature importance rankings?
- Sparsity: what fraction of features have near-zero attribution?
- Runtime: wall-clock time comparison.
"""
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import load_config, setup_logging, get_project_root, save_result_json
from src.features.extract_features import get_feature_names
from src.xai.run_shap_lime import run_shap, run_lime

logger = setup_logging("xai_metrics")


def fidelity_deletion(model, X: np.ndarray, attributions: np.ndarray,
                      feature_names: list[str], top_k_values: list[int]) -> dict:
    """
    Deletion fidelity: remove top-k features and measure prediction drop.

    Higher drop = higher fidelity (the explanation correctly identifies
    the most important features).
    """
    scaler = model.named_steps.get("scaler", None)
    clf = model.named_steps.get("clf", model)

    def predict_proba_fn(x):
        if scaler is not None:
            x = scaler.transform(x)
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(x)[:, 1]
        return clf.predict(x).astype(float)

    original_proba = predict_proba_fn(X)
    results = {}

    for k in top_k_values:
        if k > X.shape[1]:
            k = X.shape[1]

        drops = []
        for i in range(len(X)):
            # Rank features by absolute attribution
            abs_attr = np.abs(attributions[i]) if len(attributions.shape) > 1 else np.abs(attributions)
            top_indices = np.argsort(abs_attr)[-k:]

            # Zero out top-k features (replace with training mean = 0 after scaling)
            x_modified = X[i].copy()
            col_means = X.mean(axis=0)
            x_modified[top_indices] = col_means[top_indices]

            modified_proba = predict_proba_fn(x_modified.reshape(1, -1))[0]
            drop = abs(original_proba[i] - modified_proba)
            drops.append(drop)

        results[f"deletion@{k}"] = {
            "mean_drop": float(np.mean(drops)),
            "std_drop": float(np.std(drops)),
            "median_drop": float(np.median(drops)),
        }

    return results


def fidelity_insertion(model, X: np.ndarray, attributions: np.ndarray,
                       feature_names: list[str], top_k_values: list[int]) -> dict:
    """
    Insertion fidelity: start with mean features, add top-k, measure prediction rise.

    Higher rise = higher fidelity.
    """
    scaler = model.named_steps.get("scaler", None)
    clf = model.named_steps.get("clf", model)

    def predict_proba_fn(x):
        if scaler is not None:
            x = scaler.transform(x)
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(x)[:, 1]
        return clf.predict(x).astype(float)

    original_proba = predict_proba_fn(X)
    col_means = X.mean(axis=0)
    results = {}

    for k in top_k_values:
        if k > X.shape[1]:
            k = X.shape[1]

        rises = []
        for i in range(len(X)):
            abs_attr = np.abs(attributions[i]) if len(attributions.shape) > 1 else np.abs(attributions)
            top_indices = np.argsort(abs_attr)[-k:]

            # Start with all-mean, insert top-k from original
            x_base = col_means.copy()
            x_base[top_indices] = X[i, top_indices]

            base_proba = predict_proba_fn(col_means.reshape(1, -1))[0]
            inserted_proba = predict_proba_fn(x_base.reshape(1, -1))[0]
            rise = abs(inserted_proba - base_proba)
            rises.append(rise)

        results[f"insertion@{k}"] = {
            "mean_rise": float(np.mean(rises)),
            "std_rise": float(np.std(rises)),
            "median_rise": float(np.median(rises)),
        }

    return results


def stability_jaccard(model, X: np.ndarray, feature_names: list[str],
                      k: int = 5, n_runs: int = 5,
                      explanation_fn: str = "shap",
                      base_seed: int = 42) -> dict:
    """
    Stability via Jaccard@k: run explanations multiple times with different
    seeds and measure overlap of top-k features.
    """
    all_top_k_sets = []

    for run in range(n_runs):
        seed = base_seed + run

        if explanation_fn == "shap":
            result = run_shap(model, X, feature_names, n_samples=min(50, len(X)), seed=seed)
            if "error" in result:
                continue
            vals = np.array(result["shap_values"])
        else:
            result = run_lime(model, X, feature_names,
                            n_samples=min(50, len(X)), n_features=len(feature_names),
                            lime_n_samples=1000, seed=seed)
            vals = np.array(result["lime_weights"])

        # For each sample, get top-k feature indices
        run_sets = []
        for i in range(len(vals)):
            top_k_idx = set(np.argsort(np.abs(vals[i]))[-k:])
            run_sets.append(top_k_idx)
        all_top_k_sets.append(run_sets)

    if len(all_top_k_sets) < 2:
        return {"jaccard_mean": float("nan"), "note": "Not enough successful runs"}

    # Pairwise Jaccard across runs
    n_samples_min = min(len(s) for s in all_top_k_sets)
    jaccards = []
    for i in range(n_samples_min):
        for r1 in range(len(all_top_k_sets)):
            for r2 in range(r1 + 1, len(all_top_k_sets)):
                s1 = all_top_k_sets[r1][i]
                s2 = all_top_k_sets[r2][i]
                if len(s1 | s2) == 0:
                    continue
                j = len(s1 & s2) / len(s1 | s2)
                jaccards.append(j)

    return {
        "jaccard_mean": float(np.mean(jaccards)) if jaccards else float("nan"),
        "jaccard_std": float(np.std(jaccards)) if jaccards else float("nan"),
        "jaccard_median": float(np.median(jaccards)) if jaccards else float("nan"),
        "n_comparisons": len(jaccards),
        "k": k,
        "n_runs": n_runs,
    }


def rank_agreement(shap_attributions: np.ndarray, lime_attributions: np.ndarray,
                   feature_names: list[str]) -> dict:
    """
    SHAP vs LIME rank agreement via Kendall's tau.

    Computes per-sample ranking correlation and aggregates.
    """
    n_samples = min(len(shap_attributions), len(lime_attributions))
    taus = []
    p_values = []

    for i in range(n_samples):
        shap_ranks = np.argsort(np.argsort(-np.abs(shap_attributions[i])))
        lime_ranks = np.argsort(np.argsort(-np.abs(lime_attributions[i])))

        tau, p = kendalltau(shap_ranks, lime_ranks)
        if not np.isnan(tau):
            taus.append(tau)
            p_values.append(p)

    return {
        "kendall_tau_mean": float(np.mean(taus)) if taus else float("nan"),
        "kendall_tau_std": float(np.std(taus)) if taus else float("nan"),
        "kendall_tau_median": float(np.median(taus)) if taus else float("nan"),
        "mean_p_value": float(np.mean(p_values)) if p_values else float("nan"),
        "n_samples_compared": len(taus),
        "interpretation": (
            "Kendall's tau ranges from -1 (complete disagreement) to +1 "
            "(complete agreement). Values near 0 indicate independence."
        ),
    }


def sparsity(attributions: np.ndarray, threshold: float = 0.01) -> dict:
    """
    Compute sparsity of explanations.

    Sparsity = fraction of features with |attribution| < threshold * max.
    Higher sparsity = more focused explanations.
    """
    sparsities = []
    for i in range(len(attributions)):
        abs_attr = np.abs(attributions[i])
        max_attr = abs_attr.max()
        if max_attr == 0:
            sparsities.append(1.0)
        else:
            near_zero = (abs_attr < threshold * max_attr).sum()
            sparsities.append(near_zero / len(abs_attr))

    return {
        "mean_sparsity": float(np.mean(sparsities)),
        "std_sparsity": float(np.std(sparsities)),
        "threshold": threshold,
        "interpretation": (
            "Sparsity measures what fraction of features have near-zero "
            "attribution. Higher = more focused, easier to interpret."
        ),
    }


def run_xai_metrics(config: dict | None = None) -> dict:
    """
    Run all quantitative XAI metrics using pre-computed explanations.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    results_dir = root / config["paths"]["results"]
    models_dir = root / config["paths"]["models"]
    processed_dir = root / config["paths"]["processed_data"]
    feature_names = get_feature_names(config)
    seed = config["random_seed"]
    top_k_values = config["xai"]["fidelity_top_k"]
    stability_n_runs = config["xai"]["stability_n_runs"]

    # Load pre-computed XAI results
    xai_results_path = results_dir / "xai_results.json"
    if not xai_results_path.exists():
        logger.error("XAI results not found at %s. Run run_shap_lime.py first.", xai_results_path)
        return {}

    import json
    with open(xai_results_path) as f:
        raw = json.load(f)

    xai_data = raw.get("results", raw)
    all_metrics = {}

    for key, xai_result in xai_data.items():
        if not isinstance(xai_result, dict):
            continue
        if "shap" not in xai_result or "lime" not in xai_result:
            continue

        parts = key.rsplit("_", 1)
        if len(parts) != 2:
            continue
        clf_name, split_type = parts

        # Load model and test data
        model_path = models_dir / f"{clf_name}_{split_type}.joblib"
        test_path = processed_dir / f"{split_type}_test_features.csv"

        if not model_path.exists() or not test_path.exists():
            logger.warning("Missing model or test data for %s. Skipping metrics.", key)
            continue

        model = joblib.load(model_path)
        df_test = pd.read_csv(test_path)
        X_test = df_test[feature_names].values
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        shap_result = xai_result["shap"]
        lime_result = xai_result["lime"]

        if "error" in shap_result:
            logger.warning("SHAP had errors for %s: %s", key, shap_result["error"])
            continue

        shap_vals = np.array(shap_result["shap_values"])
        lime_vals = np.array(lime_result["lime_weights"])

        indices = shap_result["sample_indices"]
        X_subset = X_test[indices]

        # Trim to matching lengths
        min_n = min(len(shap_vals), len(lime_vals), len(X_subset))
        shap_vals = shap_vals[:min_n]
        lime_vals = lime_vals[:min_n]
        X_subset = X_subset[:min_n]

        logger.info("Computing XAI metrics for %s (%d samples)…", key, min_n)

        metrics = {}

        # 1. Fidelity (deletion)
        logger.info("  Computing deletion fidelity…")
        metrics["fidelity_deletion_shap"] = fidelity_deletion(
            model, X_subset, shap_vals, feature_names, top_k_values
        )
        metrics["fidelity_deletion_lime"] = fidelity_deletion(
            model, X_subset, lime_vals, feature_names, top_k_values
        )

        # 2. Fidelity (insertion)
        logger.info("  Computing insertion fidelity…")
        metrics["fidelity_insertion_shap"] = fidelity_insertion(
            model, X_subset, shap_vals, feature_names, top_k_values
        )
        metrics["fidelity_insertion_lime"] = fidelity_insertion(
            model, X_subset, lime_vals, feature_names, top_k_values
        )

        # 3. Stability (Jaccard@k)
        logger.info("  Computing stability…")
        for k in [3, 5]:
            metrics[f"stability_shap_jaccard@{k}"] = stability_jaccard(
                model, X_subset, feature_names, k=k,
                n_runs=stability_n_runs, explanation_fn="shap",
                base_seed=seed,
            )
            metrics[f"stability_lime_jaccard@{k}"] = stability_jaccard(
                model, X_subset, feature_names, k=k,
                n_runs=stability_n_runs, explanation_fn="lime",
                base_seed=seed,
            )

        # 4. Rank agreement (Kendall's tau)
        logger.info("  Computing rank agreement…")
        metrics["rank_agreement"] = rank_agreement(shap_vals, lime_vals, feature_names)

        # 5. Sparsity
        logger.info("  Computing sparsity…")
        metrics["sparsity_shap"] = sparsity(shap_vals)
        metrics["sparsity_lime"] = sparsity(lime_vals)

        # 6. Runtime comparison
        metrics["runtime"] = {
            "shap_seconds": shap_result.get("runtime_seconds", float("nan")),
            "lime_seconds": lime_result.get("runtime_seconds", float("nan")),
            "shap_n_samples": shap_result.get("n_samples", 0),
            "lime_n_samples": lime_result.get("n_samples", 0),
        }

        all_metrics[key] = metrics
        logger.info("  Done with %s.", key)

    save_result_json(all_metrics, results_dir / "xai_metrics.json")
    logger.info("XAI metrics saved to %s", results_dir / "xai_metrics.json")
    return all_metrics


if __name__ == "__main__":
    run_xai_metrics()
