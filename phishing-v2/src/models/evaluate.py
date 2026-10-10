from __future__ import annotations
"""
Evaluate trained models on held-out test sets.

Produces: confusion matrices, ROC/PR curves, per-class metrics,
McNemar's test for pairwise model comparison, and feature importance
dominance checks.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import load_config, setup_logging, get_project_root, save_result_json
from src.features.extract_features import get_feature_names

logger = setup_logging("evaluate")


def _confidence_interval(scores: np.ndarray, confidence: float = 0.95) -> tuple:
    """Compute confidence interval for an array of scores."""
    n = len(scores)
    mean = np.mean(scores)
    se = stats.sem(scores)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return float(mean - h), float(mean + h)


def _mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray,
                  y_pred_b: np.ndarray) -> dict:
    """
    McNemar's test comparing two classifiers.

    Returns test statistic and p-value.
    """
    # Build contingency table
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b: A correct, B wrong; c: A wrong, B correct
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "note": "Both models agree on all samples"}

    # McNemar's test with continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {"statistic": float(statistic), "p_value": float(p_value)}


def _check_feature_dominance(model, feature_names: list[str],
                             threshold: float = 0.5) -> list[dict]:
    """
    Check if any single feature dominates importance.

    Flags features with importance > threshold of total.
    """
    warnings_list = []

    clf = model.named_steps.get("clf", model)

    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_).flatten()

    if importances is not None:
        total = importances.sum()
        if total > 0:
            normalized = importances / total
            for i, (name, imp) in enumerate(zip(feature_names, normalized)):
                if imp > threshold:
                    warnings_list.append({
                        "feature": name,
                        "relative_importance": float(imp),
                        "warning": (
                            f"Feature '{name}' has {imp:.1%} of total importance. "
                            f"This may indicate data leakage, a trivially-predictive "
                            f"proxy, or a dataset artifact. Investigate whether this "
                            f"feature genuinely captures phishing behavior or is "
                            f"simply an artifact of the data collection process."
                        ),
                    })

                # Also flag if importance exceeds 3x the uniform baseline
                uniform = 1.0 / len(feature_names)
                if imp > 3 * uniform:
                    warnings_list.append({
                        "feature": name,
                        "relative_importance": float(imp),
                        "note": (
                            f"Feature '{name}' importance ({imp:.3f}) is >{3*uniform:.3f} "
                            f"(3× uniform baseline of {uniform:.3f}). Marked for review."
                        ),
                    })

    return warnings_list


def evaluate_all(config: dict | None = None) -> dict:
    """
    Evaluate all trained models on their respective test sets.

    Returns a dict of model_name -> evaluation_results.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    models_dir = root / config["paths"]["models"]
    processed_dir = root / config["paths"]["processed_data"]
    results_dir = root / config["paths"]["results"]
    figures_dir = root / config["paths"]["figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_names = get_feature_names(config)
    all_results = {}
    predictions_by_split = {}  # For McNemar tests

    model_files = list(models_dir.glob("*.joblib"))
    if not model_files:
        logger.warning("No trained models found in %s.", models_dir)
        return {}

    for model_path in model_files:
        name = model_path.stem  # e.g. "RandomForest_domain"
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        clf_name, split_type = parts

        # Load test data
        test_path = processed_dir / f"{split_type}_test_features.csv"
        if not test_path.exists():
            logger.warning("Test features not found at %s. Skipping %s.", test_path, name)
            continue

        df_test = pd.read_csv(test_path)
        missing = [f for f in feature_names if f not in df_test.columns]
        if missing:
            logger.warning("Missing features in %s: %s. Skipping.", test_path, missing)
            continue

        X_test = df_test[feature_names].values
        y_test = df_test["label"].values

        # Drop rows with NaN labels
        valid_mask = ~np.isnan(y_test)
        if valid_mask.sum() < len(y_test):
            logger.warning("Dropping %d NaN-label rows from test set %s.", len(y_test) - valid_mask.sum(), name)
            X_test = X_test[valid_mask]
            y_test = y_test[valid_mask]
        y_test = y_test.astype(int)

        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Load model
        model = joblib.load(model_path)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        # Store predictions for McNemar
        if split_type not in predictions_by_split:
            predictions_by_split[split_type] = {"y_true": y_test}
        predictions_by_split[split_type][clf_name] = y_pred

        # Metrics
        result = {
            "classifier": clf_name,
            "split_type": split_type,
            "n_test": int(len(y_test)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
        }

        if y_proba is not None:
            result["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            result["avg_precision"] = float(average_precision_score(y_test, y_proba))

        # Feature importance dominance check
        dominance_warnings = _check_feature_dominance(model, feature_names)
        if dominance_warnings:
            result["feature_dominance_warnings"] = dominance_warnings
            for w in dominance_warnings:
                logger.warning("[%s] %s", name, w.get("warning", w.get("note", "")))

        # Save feature importances
        clf = model.named_steps.get("clf", model)
        if hasattr(clf, "feature_importances_"):
            importances = dict(zip(feature_names, clf.feature_importances_.tolist()))
            result["feature_importances"] = importances

        all_results[name] = result
        logger.info(
            "%s: Acc=%.4f, P=%.4f, R=%.4f, F1=%.4f%s",
            name, result["accuracy"], result["precision"],
            result["recall"], result["f1"],
            f", AUC={result.get('roc_auc', 'N/A')}" if y_proba is not None else "",
        )

        # ── Plot confusion matrix ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legitimate", "Phishing"],
                    yticklabels=["Legitimate", "Phishing"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {clf_name} ({split_type} split)")
        fig.tight_layout()
        fig.savefig(figures_dir / f"cm_{name}.png", dpi=150)
        plt.close(fig)

        # ── Plot ROC curve ────────────────────────────────────────────
        if y_proba is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            ax.plot(fpr, tpr, lw=2, label=f"{clf_name} (AUC={result['roc_auc']:.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve — {clf_name} ({split_type} split)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures_dir / f"roc_{name}.png", dpi=150)
            plt.close(fig)

            # ── Plot PR curve ─────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(6, 5))
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
            ax.plot(recall_vals, precision_vals, lw=2,
                    label=f"{clf_name} (AP={result['avg_precision']:.3f})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"PR Curve — {clf_name} ({split_type} split)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures_dir / f"pr_{name}.png", dpi=150)
            plt.close(fig)

    # ── McNemar tests between all pairs within each split ──────────
    mcnemar_results = {}
    for split_type, preds in predictions_by_split.items():
        y_true = preds["y_true"]
        model_names = [k for k in preds if k != "y_true"]
        for i, name_a in enumerate(model_names):
            for name_b in model_names[i + 1:]:
                key = f"{name_a}_vs_{name_b}_{split_type}"
                mcnemar_results[key] = _mcnemar_test(
                    y_true, preds[name_a], preds[name_b]
                )
                logger.info(
                    "McNemar %s vs %s (%s): stat=%.3f, p=%.4f",
                    name_a, name_b, split_type,
                    mcnemar_results[key]["statistic"],
                    mcnemar_results[key]["p_value"],
                )

    all_results["mcnemar_tests"] = mcnemar_results

    # Save results
    save_result_json(all_results, results_dir / "evaluation_results.json")
    logger.info("Evaluation complete. Results at %s", results_dir / "evaluation_results.json")

    return all_results


if __name__ == "__main__":
    evaluate_all()
