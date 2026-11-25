from __future__ import annotations
"""
Train classifiers (KNN, DecisionTree, RandomForest) and baselines
(LogisticRegression, XGBoost) with hyperparameter search via cross-validation.

All training uses the domain-based split to prevent leakage.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import load_config, setup_logging, get_project_root, save_result_json
from src.features.extract_features import get_feature_names

logger = setup_logging("train")


def _build_classifier(name: str, params: dict):
    """Instantiate a classifier by name."""
    classifiers = {
        "KNN": KNeighborsClassifier,
        "DecisionTree": DecisionTreeClassifier,
        "RandomForest": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
    }
    if name == "XGBoost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(use_label_encoder=False, eval_metric="logloss", **params)
        except ImportError:
            logger.warning("XGBoost not installed, skipping.")
            return None
    cls = classifiers.get(name)
    if cls is None:
        raise ValueError(f"Unknown classifier: {name}")
    return cls(**params)


def _get_param_grid(name: str, config: dict) -> dict:
    """Build sklearn-compatible param grid from config."""
    model_config = config["models"]
    if name in model_config.get("classifiers", {}):
        raw = model_config["classifiers"][name]
    elif name in model_config.get("baselines", {}):
        raw = model_config["baselines"][name]
    else:
        return {}

    grid = {}
    for param, values in raw.items():
        if isinstance(values, list):
            grid[f"clf__{param}"] = values
        # Single values are fixed, not searched
    return grid


def _get_fixed_params(name: str, config: dict) -> dict:
    """Get non-list (fixed) params from config."""
    model_config = config["models"]
    if name in model_config.get("classifiers", {}):
        raw = model_config["classifiers"][name]
    elif name in model_config.get("baselines", {}):
        raw = model_config["baselines"][name]
    else:
        return {}

    return {k: v for k, v in raw.items() if not isinstance(v, list)}


def train_all(config: dict | None = None) -> dict:
    """
    Train all classifiers on both domain-based and time-based splits.

    Returns a dict of model_name -> {best_params, cv_results, model_path}.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_data"]
    models_dir = root / config["paths"]["models"]
    results_dir = root / config["paths"]["results"]
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_names = get_feature_names(config)
    seed = config["random_seed"]
    cv_folds = config["splitting"]["cv_folds"]

    all_results = {}

    # Train on each split type
    for split_type in ["domain", "time"]:
        train_path = processed_dir / f"{split_type}_train_features.csv"
        if not train_path.exists():
            logger.warning("Train features not found for %s split at %s. Skipping.", split_type, train_path)
            continue

        df_train = pd.read_csv(train_path)

        # Validate features exist
        missing = [f for f in feature_names if f not in df_train.columns]
        if missing:
            raise ValueError(f"Missing features in training data: {missing}")

        X_train = df_train[feature_names].values
        y_train = df_train["label"].values

        # Drop rows with NaN labels (may occur if ambiguous-domain rows remain)
        valid_mask = ~np.isnan(y_train)
        if valid_mask.sum() < len(y_train):
            dropped = len(y_train) - valid_mask.sum()
            logger.warning("Dropping %d rows with NaN labels from %s split.", dropped, split_type)
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]

        y_train = y_train.astype(int)

        # Handle NaN/Inf in features
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

        class_counts = dict(zip(*np.unique(y_train, return_counts=True)))
        logger.info(
            "Training on %s split: %d samples, %d features, class balance: %s",
            split_type, len(X_train), X_train.shape[1], class_counts,
        )

        # Build classifiers to train
        classifier_names = list(config["models"].get("classifiers", {}).keys()) + \
                          list(config["models"].get("baselines", {}).keys())

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

        for clf_name in classifier_names:
            logger.info("Training %s on %s split…", clf_name, split_type)

            fixed_params = _get_fixed_params(clf_name, config)

            # Add random_state where applicable
            if clf_name in ("DecisionTree", "RandomForest", "LogisticRegression"):
                fixed_params["random_state"] = seed
            if clf_name == "XGBoost":
                fixed_params["random_state"] = seed

            clf = _build_classifier(clf_name, fixed_params)
            if clf is None:
                continue

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf),
            ])

            param_grid = _get_param_grid(clf_name, config)

            if param_grid:
                search = GridSearchCV(
                    pipe, param_grid,
                    cv=cv, scoring="f1",
                    n_jobs=-1, refit=True,
                    return_train_score=True,
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                best_params = search.best_params_
                cv_results_raw = search.cv_results_

                # Extract mean ± std for best model
                best_idx = search.best_index_
                cv_summary = {
                    "mean_test_f1": float(cv_results_raw["mean_test_score"][best_idx]),
                    "std_test_f1": float(cv_results_raw["std_test_score"][best_idx]),
                    "mean_train_f1": float(cv_results_raw["mean_train_score"][best_idx]),
                    "std_train_f1": float(cv_results_raw["std_train_score"][best_idx]),
                }
            else:
                # No grid search, just fit with CV for metrics
                from sklearn.model_selection import cross_validate
                cv_out = cross_validate(
                    pipe, X_train, y_train, cv=cv,
                    scoring="f1", return_train_score=True, n_jobs=-1,
                )
                pipe.fit(X_train, y_train)
                best_model = pipe
                best_params = fixed_params
                cv_summary = {
                    "mean_test_f1": float(cv_out["test_score"].mean()),
                    "std_test_f1": float(cv_out["test_score"].std()),
                    "mean_train_f1": float(cv_out["train_score"].mean()),
                    "std_train_f1": float(cv_out["train_score"].std()),
                }

            # Save model
            model_path = models_dir / f"{clf_name}_{split_type}.joblib"
            joblib.dump(best_model, model_path)

            result_key = f"{clf_name}_{split_type}"
            all_results[result_key] = {
                "classifier": clf_name,
                "split_type": split_type,
                "best_params": {k: str(v) for k, v in best_params.items()},
                "cv_summary": cv_summary,
                "model_path": str(model_path),
                "n_train": int(len(X_train)),
            }

            logger.info(
                "%s on %s: F1 = %.4f ± %.4f",
                clf_name, split_type,
                cv_summary["mean_test_f1"], cv_summary["std_test_f1"],
            )

    # Save all training results
    save_result_json(all_results, results_dir / "training_results.json")
    logger.info("Training complete. Results saved to %s", results_dir / "training_results.json")

    return all_results


if __name__ == "__main__":
    train_all()
