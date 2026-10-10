from __future__ import annotations
"""
Cluster evaluation metrics.

Reports purity/NMI/ARI ONLY if ground-truth campaign labels are available.
Otherwise, reports descriptive cluster statistics (size distribution, density,
modularity) and explicitly states that no ground truth is available.

NEVER fabricates ground-truth labels or metrics.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import load_config, setup_logging, get_project_root, save_result_json

logger = setup_logging("cluster_eval")


def _purity(labels_true, labels_pred):
    """Compute cluster purity."""
    from sklearn.metrics import confusion_matrix as cm
    contingency = cm(labels_true, labels_pred)
    return float(np.sum(np.max(contingency, axis=0)) / np.sum(contingency))


def evaluate_clusters(config: dict | None = None) -> dict:
    """
    Evaluate campaign clusters.

    If ground-truth labels exist, compute purity/NMI/ARI.
    Otherwise, compute only descriptive statistics and explicitly note
    the absence of ground truth.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    results_dir = root / config["paths"]["results"]

    # Load campaign results
    import json
    campaign_path = results_dir / "campaign_results.json"
    if not campaign_path.exists():
        logger.warning("Campaign results not found. Run cluster_campaigns.py first.")
        return {}

    with open(campaign_path) as f:
        campaign_data = json.load(f)

    campaign_results = campaign_data.get("results", campaign_data)

    # Check for ground-truth labels
    ground_truth_path = root / "data" / "raw" / "campaign_ground_truth.csv"
    has_ground_truth = ground_truth_path.exists()

    result = {
        "has_ground_truth": has_ground_truth,
    }

    if has_ground_truth:
        # Load ground truth and compute supervised metrics
        import pandas as pd
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
        )

        gt = pd.read_csv(ground_truth_path)
        logger.info("Found ground-truth campaign labels for %d URLs.", len(gt))

        # Match predicted clusters to ground truth
        # This requires URL-to-cluster mapping from campaign_results
        campaigns = campaign_results.get("campaigns", {})
        url_to_cluster = {}
        for cluster_id, data in campaigns.items():
            for url in data.get("urls", []):
                url_to_cluster[url] = int(cluster_id)

        # Only evaluate URLs that appear in both
        matched = gt[gt["url"].isin(url_to_cluster)]
        if len(matched) < 10:
            result["note"] = (
                f"Only {len(matched)} URLs have both ground-truth labels and "
                f"predicted cluster assignments. Metrics may be unreliable."
            )
            result["confidence"] = "low"

        if len(matched) > 0:
            labels_true = matched["campaign_id"].values
            labels_pred = np.array([url_to_cluster[u] for u in matched["url"]])

            result["purity"] = float(_purity(labels_true, labels_pred))
            result["nmi"] = float(normalized_mutual_info_score(labels_true, labels_pred))
            result["ari"] = float(adjusted_rand_score(labels_true, labels_pred))
            result["n_evaluated"] = int(len(matched))
        else:
            result["note"] = "No overlap between ground truth and predicted clusters."

    else:
        result["note"] = (
            "NO ground-truth campaign labels available. "
            "Cannot compute purity, NMI, or ARI. "
            "Reporting only descriptive cluster statistics. "
            "To enable supervised evaluation, provide campaign_ground_truth.csv "
            "in data/raw/ with columns: url, campaign_id"
        )
        logger.warning(result["note"])

    # Always compute descriptive statistics
    descriptive = {}

    campaigns = campaign_results.get("campaigns", {})
    sizes = [d.get("size", 0) for d in campaigns.values()] if campaigns else []

    descriptive["n_campaigns"] = len(campaigns)
    descriptive["total_urls_in_campaigns"] = int(sum(sizes))

    if sizes:
        descriptive["size_distribution"] = {
            "min": int(min(sizes)),
            "max": int(max(sizes)),
            "mean": float(np.mean(sizes)),
            "median": float(np.median(sizes)),
            "std": float(np.std(sizes)),
            "q25": float(np.percentile(sizes, 25)),
            "q75": float(np.percentile(sizes, 75)),
        }
    else:
        descriptive["size_distribution"] = "No campaigns found."

    descriptive["graph_density"] = campaign_results.get("graph_density", "N/A")
    descriptive["n_connected_components"] = campaign_results.get("n_connected_components", "N/A")
    descriptive["clustering_method"] = campaign_results.get("method", "N/A")

    result["descriptive_statistics"] = descriptive

    save_result_json(result, results_dir / "cluster_evaluation.json")
    logger.info("Cluster evaluation complete. Ground truth available: %s", has_ground_truth)

    return result


if __name__ == "__main__":
    evaluate_clusters()
