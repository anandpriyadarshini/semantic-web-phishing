from __future__ import annotations
"""
Run SHAP and LIME explanations on trained models.

Generates per-sample explanations for a subset of test data and saves
raw explanation values for downstream quantitative evaluation.
"""
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import load_config, setup_logging, get_project_root, save_result_json
from src.features.extract_features import get_feature_names

logger = setup_logging("run_shap_lime")


def run_shap(model, X: np.ndarray, feature_names: list[str],
             n_samples: int = 100, seed: int = 42) -> dict:
    """
    Compute SHAP values for a model on a sample of data.

    Uses the fastest available explainer:
    - TreeExplainer for tree-based models (RF, DT, XGBoost) — exact, fast
    - LinearExplainer for linear models (LogisticRegression)
    - KernelExplainer as fallback (slow, use small n_samples)

    Returns dict with shap_values, base_value, and timing.
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_sample = X[indices]

    clf = model.named_steps.get("clf", model)
    scaler = model.named_steps.get("scaler", None)

    if scaler is not None:
        X_scaled = scaler.transform(X_sample)
        X_bg = scaler.transform(X[rng.choice(len(X), size=min(50, len(X)), replace=False)])
    else:
        X_scaled = X_sample
        X_bg = X[rng.choice(len(X), size=min(50, len(X)), replace=False)]

    start_time = time.perf_counter()
    clf_type = type(clf).__name__

    shap_values = None
    base_value = None

    # Strategy 1: TreeExplainer (for tree-based models — much faster than Kernel)
    tree_types = ("RandomForestClassifier", "DecisionTreeClassifier",
                  "XGBClassifier", "GradientBoostingClassifier", "ExtraTreesClassifier")
    if clf_type in tree_types:
        try:
            # Use path_dependent (default) perturbation — works with scaled data,
            # no additivity check issues. check_additivity=False suppresses the
            # known approximate RF warning (SHAP docs acknowledge this).
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_scaled, check_additivity=False)
            # For RF/binary classification shap_values is a list of 2 arrays
            if isinstance(sv, list) and len(sv) == 2:
                sv = sv[1]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]
            shap_values = sv
            ev = explainer.expected_value
            base_value = float(ev[1]) if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else float(ev)
            logger.info("Used TreeExplainer for %s (%d samples).", clf_type, len(X_sample))
        except Exception as e:
            logger.warning("TreeExplainer failed for %s: %s. Trying alternatives.", clf_type, e)

    # Strategy 2: LinearExplainer (for logistic regression)
    if shap_values is None and clf_type in ("LogisticRegression", "LinearSVC", "SGDClassifier"):
        try:
            explainer = shap.LinearExplainer(clf, X_bg)
            shap_values = explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            base_value = float(explainer.expected_value)
            logger.info("Used LinearExplainer for %s.", clf_type)
        except Exception as e:
            logger.warning("LinearExplainer failed: %s", e)

    # Strategy 3: KernelExplainer (slow fallback — limit to 20 samples)
    if shap_values is None:
        kernel_samples = min(20, len(X_sample))  # Hard cap for performance
        X_kernel = X_scaled[:kernel_samples]
        try:
            def predict_fn(x):
                if hasattr(clf, "predict_proba"):
                    return clf.predict_proba(x)[:, 1]
                return clf.predict(x).astype(float)
            explainer = shap.KernelExplainer(predict_fn, X_bg[:20])
            shap_values = explainer.shap_values(X_kernel, silent=True)
            base_value = float(explainer.expected_value)
            logger.warning("Used KernelExplainer (slow) for %s on %d samples.", clf_type, kernel_samples)
        except Exception as e:
            logger.error("All SHAP explainers failed: %s", e)
            return {"error": str(e)}

    elapsed = time.perf_counter() - start_time

    return {
        "shap_values": shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
        "base_value": float(base_value) if base_value is not None else 0.0,
        "feature_names": feature_names,
        "sample_indices": indices.tolist(),
        "n_samples": len(X_sample),
        "runtime_seconds": float(elapsed),
    }



def run_lime(model, X: np.ndarray, feature_names: list[str],
             n_samples: int = 100, n_features: int = 10,
             lime_n_samples: int = 5000, seed: int = 42) -> dict:
    """
    Compute LIME explanations for a model on a sample of data.

    Returns dict with explanation weights and timing.
    """
    import lime.lime_tabular

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_sample = X[indices]

    scaler = model.named_steps.get("scaler", None)
    clf = model.named_steps.get("clf", model)

    # Create the full prediction function
    def predict_fn(x):
        if scaler is not None:
            x = scaler.transform(x)
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(x)
        else:
            preds = clf.predict(x)
            return np.column_stack([1 - preds, preds])

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X,
        feature_names=feature_names,
        class_names=["legitimate", "phishing"],
        mode="classification",
        random_state=seed,
    )

    start_time = time.perf_counter()

    all_weights = []
    for i, x in enumerate(X_sample):
        try:
            exp = explainer.explain_instance(
                x, predict_fn,
                num_features=n_features,
                num_samples=lime_n_samples,
            )
            # Use as_map() to get feature weights by exact index
            weights = [0.0] * len(feature_names)
            exp_map = exp.as_map()
            target_cls = 1 if 1 in exp_map else (list(exp_map.keys())[0] if exp_map else None)
            if target_cls is not None:
                for feat_idx, w in exp_map[target_cls]:
                    if 0 <= feat_idx < len(weights):
                        weights[feat_idx] = float(w)
            all_weights.append(weights)
        except Exception as e:
            logger.warning("LIME failed for sample %d: %s", i, e)
            all_weights.append([0.0] * len(feature_names))

    elapsed = time.perf_counter() - start_time

    return {
        "lime_weights": all_weights,
        "feature_names": feature_names,
        "sample_indices": indices.tolist(),
        "n_samples": len(X_sample),
        "n_features_explained": n_features,
        "runtime_seconds": float(elapsed),
    }


def run_all_xai(config: dict | None = None) -> dict:
    """
    Run SHAP and LIME on the best model from each split type.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    models_dir = root / config["paths"]["models"]
    processed_dir = root / config["paths"]["processed_data"]
    results_dir = root / config["paths"]["results"]
    figures_dir = root / config["paths"]["figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    feature_names = get_feature_names(config)
    seed = config["random_seed"]
    n_samples = config["xai"]["shap_n_samples"]
    lime_n_features = config["xai"]["lime_n_features"]
    lime_n_samples = config["xai"]["lime_n_samples"]

    all_xai_results = {}
    
    # Run XAI for all available classifiers on both splits
    classifiers = [
        "RandomForest",
        "DecisionTree",
        "XGBoost",
        "KNN",
        "LogisticRegression",
    ]

    for split_type in ["domain", "time"]:
        test_path = processed_dir / f"{split_type}_test_features.csv"

        if not test_path.exists():
            logger.warning("Test features missing: %s", test_path)
            continue

        df_test = pd.read_csv(test_path)
        X_test = df_test[feature_names].values
        X_test = np.nan_to_num(
            X_test, nan=0.0, posinf=0.0, neginf=0.0
        )

        for clf_name in classifiers:
            model_path = models_dir / f"{clf_name}_{split_type}.joblib"

            if not model_path.exists():
                logger.warning("Model missing: %s", model_path.name)
                continue

            model = joblib.load(model_path)
            key = f"{clf_name}_{split_type}"

            logger.info("Running SHAP for %s on %s split", clf_name, split_type)
            shap_result = run_shap(
                model, X_test, feature_names, n_samples, seed
            )

            logger.info("Running LIME for %s on %s split", clf_name, split_type)
            lime_result = run_lime(
                model, X_test, feature_names,
                n_samples, lime_n_features, lime_n_samples, seed
            )

            all_xai_results[key] = {
                "shap": shap_result,
                "lime": lime_result,
            }

            # Save a SHAP summary image
            if "shap_values" in shap_result and "error" not in shap_result:
                try:
                    shap_vals = np.asarray(shap_result["shap_values"])
                    indices = shap_result["sample_indices"]
                    X_plot = X_test[indices]

                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(
                        shap_vals,
                        X_plot,
                        feature_names=feature_names,
                        show=False,
                    )

                    fig = plt.gcf()
                    fig.tight_layout()
                    fig.savefig(
                        figures_dir / f"shap_summary_{key}.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close("all")

                    logger.info("Saved SHAP plot for %s", key)

                except Exception:
                    logger.exception("SHAP plot failed for %s", key)


    

        # Plot SHAP summary
        if "shap_values" in shap_result and "error" not in shap_result:
            try:
                shap_vals = np.array(shap_result["shap_values"])
                indices = shap_result["sample_indices"]
                X_plot = X_test[indices]

                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(
                    shap_vals, X_plot,
                    feature_names=feature_names,
                    show=False,
                )
                fig = plt.gcf()
                fig.tight_layout()
                fig.savefig(figures_dir / f"shap_summary_{key}.png", dpi=150, bbox_inches="tight")
                plt.close("all")
            except Exception as e:
                logger.warning("Failed to plot SHAP summary: %s", e)

    save_result_json(all_xai_results, results_dir / "xai_results.json")
    logger.info("XAI analysis complete.")
    return all_xai_results


if __name__ == "__main__":
    run_all_xai()
