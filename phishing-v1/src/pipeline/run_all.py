#!/usr/bin/env python3
"""
End-to-end pipeline orchestrator.

Run with: python src/pipeline/run_all.py

This script runs every stage of the pipeline in order:
1. Data acquisition (PhishTank + legitimate URLs)
2. Deduplication and splitting
3. Feature extraction
4. Model training (with cross-validation)
5. Model evaluation (confusion matrices, ROC/PR, McNemar)
6. XAI explanations (SHAP + LIME)
7. XAI metrics (fidelity, stability, rank agreement, sparsity)
8. Knowledge graph construction (enrichment + RDF)
9. Campaign clustering
10. Cluster evaluation
11. Latency benchmark
12. Results aggregation (summary.md)
"""
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src import load_config, setup_logging, get_project_root, ensure_dirs, result_metadata

logger = setup_logging("run_all")


def _run_stage(name: str, func, *args, **kwargs):
    """Run a pipeline stage with error handling and timing."""
    import time
    logger.info("=" * 60)
    logger.info("STAGE: %s", name)
    logger.info("=" * 60)
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("✓ %s completed in %.1fs", name, elapsed)
        return result, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error("✗ %s FAILED after %.1fs: %s", name, elapsed, e)
        traceback.print_exc()
        return None, elapsed


def generate_summary(config: dict) -> None:
    """
    Generate results/summary.md from all result JSON files.

    Every metric is sourced from actual result files — nothing is hand-typed.
    """
    root = get_project_root()
    results_dir = root / config["paths"]["results"]

    lines = [
        "# Pipeline Results Summary",
        "",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}  ",
        f"**Git commit**: {result_metadata()['git_commit']}  ",
        f"**Python**: {sys.version.split()[0]}",
        "",
        "---",
        "",
    ]

    # ── Classification Results ─────────────────────────────────────
    eval_path = results_dir / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)
        results = eval_data.get("results", eval_data)

        lines.append("## Classification Results")
        lines.append("")
        lines.append("| Model | Split | Accuracy | Precision | Recall | F1 | AUC | n_test |")
        lines.append("|-------|-------|----------|-----------|--------|-----|-----|--------|")

        for key, r in results.items():
            if key == "mcnemar_tests" or not isinstance(r, dict) or "accuracy" not in r:
                continue
            roc_auc_val = r.get("roc_auc")
            roc_auc_str = f"{roc_auc_val:.4f}" if isinstance(roc_auc_val, float) else "N/A"
            lines.append(
                f"| {r.get('classifier', key)} | {r.get('split_type', '?')} | "
                f"{r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                f"{r['f1']:.4f} | {roc_auc_str} | "
                f"{r['n_test']} |"
            )

        # McNemar tests
        mcnemar = results.get("mcnemar_tests", {})
        if mcnemar:
            lines.append("")
            lines.append("### McNemar Pairwise Tests")
            lines.append("")
            lines.append("| Comparison | χ² statistic | p-value | Significant (α=0.05) |")
            lines.append("|------------|-------------|---------|---------------------|")
            for key, m in mcnemar.items():
                sig = "Yes" if m.get("p_value", 1) < 0.05 else "No"
                lines.append(f"| {key} | {m.get('statistic', 'N/A'):.3f} | {m.get('p_value', 'N/A'):.4f} | {sig} |")

        # Feature dominance warnings
        dominance_found = False
        for key, r in results.items():
            if isinstance(r, dict) and "feature_dominance_warnings" in r:
                if not dominance_found:
                    lines.append("")
                    lines.append("### ⚠️ Feature Dominance Warnings")
                    lines.append("")
                    dominance_found = True
                for w in r["feature_dominance_warnings"]:
                    lines.append(f"- **{key}**: {w.get('warning', w.get('note', ''))}")

        lines.append("")

    # ── Training Results (CV) ──────────────────────────────────────
    train_path = results_dir / "training_results.json"
    if train_path.exists():
        with open(train_path) as f:
            train_data = json.load(f)
        results = train_data.get("results", train_data)

        lines.append("## Cross-Validation Results")
        lines.append("")
        lines.append("| Model | Split | CV F1 (mean ± std) | n_train |")
        lines.append("|-------|-------|---------------------|---------|")

        for key, r in results.items():
            if not isinstance(r, dict) or "cv_summary" not in r:
                continue
            cv = r["cv_summary"]
            lines.append(
                f"| {r.get('classifier', key)} | {r.get('split_type', '?')} | "
                f"{cv['mean_test_f1']:.4f} ± {cv['std_test_f1']:.4f} | "
                f"{r.get('n_train', '?')} |"
            )
        lines.append("")

    # ── XAI Metrics ────────────────────────────────────────────────
    xai_metrics_path = results_dir / "xai_metrics.json"
    if xai_metrics_path.exists():
        with open(xai_metrics_path) as f:
            xai_data = json.load(f)
        xai_results = xai_data.get("results", xai_data)

        lines.append("## Explainability (XAI) Metrics")
        lines.append("")
        lines.append("All metrics are quantitative and objectively computable — no subjective scoring.")
        lines.append("")

        for model_key, metrics in xai_results.items():
            if not isinstance(metrics, dict):
                continue
            lines.append(f"### {model_key}")
            lines.append("")

            # Rank agreement
            ra = metrics.get("rank_agreement", {})
            if ra:
                lines.append(f"**SHAP–LIME Rank Agreement (Kendall's τ)**: "
                           f"{ra.get('kendall_tau_mean', 'N/A'):.4f} ± "
                           f"{ra.get('kendall_tau_std', 'N/A'):.4f} "
                           f"(n={ra.get('n_samples_compared', '?')})")
                lines.append("")

            # Sparsity
            for method in ["shap", "lime"]:
                sp = metrics.get(f"sparsity_{method}", {})
                if sp:
                    lines.append(f"**{method.upper()} Sparsity**: "
                               f"{sp.get('mean_sparsity', 'N/A'):.4f} ± "
                               f"{sp.get('std_sparsity', 'N/A'):.4f}")

            # Runtime
            rt = metrics.get("runtime", {})
            if rt:
                lines.append(f"**Runtime**: SHAP={rt.get('shap_seconds', 'N/A'):.1f}s, "
                           f"LIME={rt.get('lime_seconds', 'N/A'):.1f}s")

            # Fidelity
            for method in ["shap", "lime"]:
                fd = metrics.get(f"fidelity_deletion_{method}", {})
                if fd:
                    lines.append(f"**{method.upper()} Fidelity (deletion)**: " +
                               ", ".join(f"{k}: {v.get('mean_drop', 0):.4f}" for k, v in fd.items()))

            lines.append("")

    # ── Campaign Clustering ────────────────────────────────────────
    campaign_path = results_dir / "campaign_results.json"
    if campaign_path.exists():
        with open(campaign_path) as f:
            camp_data = json.load(f)
        camp = camp_data.get("results", camp_data)

        lines.append("## Campaign Clustering (Infrastructure Correlation)")
        lines.append("")
        lines.append(f"- **Method**: {camp.get('method', 'N/A')}")
        lines.append(f"- **Graph nodes**: {camp.get('n_nodes', 'N/A')}")
        lines.append(f"- **Graph edges**: {camp.get('n_edges', 'N/A')}")
        lines.append(f"- **Graph density**: {camp.get('graph_density', 'N/A')}")
        lines.append(f"- **Total campaigns**: {camp.get('n_campaigns_total', 'N/A')}")
        lines.append(f"- **Significant campaigns (size ≥ {camp.get('min_campaign_size_threshold', '?')})**: "
                     f"{camp.get('n_campaigns_significant', 'N/A')}")

        stats = camp.get("campaign_size_stats", {}).get("significant", {})
        if isinstance(stats, dict) and stats:
            lines.append(f"- **Campaign sizes**: mean={stats.get('mean', 'N/A'):.1f}, "
                         f"median={stats.get('median', 'N/A'):.1f}, "
                         f"max={stats.get('max', 'N/A')}")
        lines.append("")

    # ── Cluster Evaluation ─────────────────────────────────────────
    cluster_eval_path = results_dir / "cluster_evaluation.json"
    if cluster_eval_path.exists():
        with open(cluster_eval_path) as f:
            ce_data = json.load(f)
        ce = ce_data.get("results", ce_data)

        lines.append("### Cluster Evaluation")
        lines.append("")
        if ce.get("has_ground_truth"):
            lines.append(f"- **Purity**: {ce.get('purity', 'N/A')}")
            lines.append(f"- **NMI**: {ce.get('nmi', 'N/A')}")
            lines.append(f"- **ARI**: {ce.get('ari', 'N/A')}")
        else:
            lines.append(f"> **Note**: {ce.get('note', 'No ground truth available.')}")
        lines.append("")

    # ── Latency Benchmark ──────────────────────────────────────────
    latency_path = results_dir / "latency_benchmark.json"
    if latency_path.exists():
        with open(latency_path) as f:
            lat_data = json.load(f)
        lat = lat_data.get("results", lat_data)

        lines.append("## Latency Benchmark")
        lines.append("")
        lines.append(f"- **URLs benchmarked**: {lat.get('n_urls_benchmarked', 'N/A')}")
        lines.append(f"- **Measurements**: {lat.get('n_measurements', 'N/A')}")
        lines.append(f"- **Throughput**: {lat.get('throughput_urls_per_second', 'N/A'):.1f} URLs/s")
        lines.append("")

        stages = lat.get("stages", {})
        if stages:
            lines.append("| Stage | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |")
            lines.append("|-------|----------|----------|----------|----------|")
            for stage, stats in stages.items():
                lines.append(
                    f"| {stage} | {stats.get('p50_ms', 0):.1f} | "
                    f"{stats.get('p95_ms', 0):.1f} | {stats.get('p99_ms', 0):.1f} | "
                    f"{stats.get('max_ms', 0):.1f} |"
                )
        lines.append("")
        if lat.get("note"):
            lines.append(f"> {lat['note']}")
        lines.append("")

    # ── Graph Stats ────────────────────────────────────────────────
    graph_stats_path = results_dir / "graph" / "graph_stats.json"
    if graph_stats_path.exists():
        with open(graph_stats_path) as f:
            gs_data = json.load(f)
        gs = gs_data.get("results", gs_data)

        lines.append("## Knowledge Graph Statistics")
        lines.append("")
        lines.append(f"- **Total RDF triples**: {gs.get('total_triples', 'N/A')}")
        lines.append(f"- **URLs processed**: {gs.get('urls_processed', 'N/A')}")
        lines.append(f"- **URLs with DNS**: {gs.get('urls_with_dns', 'N/A')}")
        lines.append(f"- **URLs with WHOIS**: {gs.get('urls_with_whois', 'N/A')}")
        lines.append(f"- **URLs with ASN**: {gs.get('urls_with_asn', 'N/A')}")
        lines.append(f"- **URLs with Certificates**: {gs.get('urls_with_certs', 'N/A')}")
        lines.append("")

    # ── Caveats ────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Caveats and Limitations")
    lines.append("")
    lines.append("- Classification metrics are from k-fold CV (mean ± std). "
                "Single-split results on held-out test sets may differ.")
    lines.append("- XAI stability depends on the number of perturbation runs and "
                "sample size. Results with <50 samples should be interpreted cautiously.")
    lines.append("- Campaign clustering uses infrastructure correlation only. "
                "Without ground-truth campaign labels, these are **candidate clusters**, "
                "not confirmed attributions.")
    lines.append("- Latency measurements include real network lookups (cached after first run). "
                "First-run latency will be significantly higher than cached-run latency.")
    lines.append("- Enrichment coverage depends on DNS/WHOIS/crt.sh availability at runtime.")
    lines.append("")

    summary_path = results_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Summary written to %s", summary_path)


def main():
    """Run the full pipeline."""
    config = load_config()
    ensure_dirs(config)

    stage_times = {}

    # Stage 1: Data acquisition / validation
    source = config.get("data", {}).get("source", "refined_csv")
    if source == "refined_csv":
        def validate_refined_dataset(cfg):
            root = get_project_root()
            name = cfg.get("data", {}).get("refined_csv_path", "final_refined_dataset_phishing.csv")
            candidates = [
                root / name,
                root / "final_refined_dataset_phishing.csv",
                root / "final_ refined_dataset_phishing.csv",
            ]
            for p in candidates:
                if p.exists():
                    logger.info("Using refined dataset at %s (%d bytes).", p.name, p.stat().st_size)
                    return p
            raise FileNotFoundError(f"Refined dataset not found at any of: {[str(c) for c in candidates]}")

        _, t = _run_stage("Load Refined Dataset", validate_refined_dataset, config)
        stage_times["load_refined_dataset"] = t
    else:
        from src.data.fetch_phishtank import fetch_phishtank
        from src.data.fetch_legitimate import fetch_legitimate
        _, t = _run_stage("Fetch PhishTank", fetch_phishtank, config)
        stage_times["fetch_phishtank"] = t
        _, t = _run_stage("Fetch Legitimate URLs", fetch_legitimate, config)
        stage_times["fetch_legitimate"] = t

    # Stage 2: Deduplicate and split
    from src.data.dedupe_and_split import dedupe_and_split
    _, t = _run_stage("Deduplicate and Split", dedupe_and_split, config)
    stage_times["dedupe_and_split"] = t

    # Stage 3: Feature extraction
    from src.features.extract_features import run_feature_extraction
    _, t = _run_stage("Feature Extraction", run_feature_extraction, config)
    stage_times["feature_extraction"] = t

    # Stage 4: Model training
    from src.models.train import train_all
    _, t = _run_stage("Model Training", train_all, config)
    stage_times["train"] = t

    # Stage 5: Model evaluation
    from src.models.evaluate import evaluate_all
    _, t = _run_stage("Model Evaluation", evaluate_all, config)
    stage_times["evaluate"] = t

    # Stage 6: XAI explanations
    from src.xai.run_shap_lime import run_all_xai
    _, t = _run_stage("XAI Explanations (SHAP + LIME)", run_all_xai, config)
    stage_times["xai_explanations"] = t

    # Stage 7: XAI metrics
    from src.xai.xai_metrics import run_xai_metrics
    _, t = _run_stage("XAI Metrics", run_xai_metrics, config)
    stage_times["xai_metrics"] = t

    # Stage 8: Knowledge graph
    from src.graph.build_graph import build_graph
    _, t = _run_stage("Build Knowledge Graph", build_graph, config)
    stage_times["build_graph"] = t

    # Stage 9: Campaign clustering
    from src.graph.cluster_campaigns import cluster_campaigns
    _, t = _run_stage("Campaign Clustering", cluster_campaigns, config)
    stage_times["cluster_campaigns"] = t

    # Stage 10: Cluster evaluation
    from src.eval.cluster_eval import evaluate_clusters
    _, t = _run_stage("Cluster Evaluation", evaluate_clusters, config)
    stage_times["cluster_eval"] = t

    # Stage 11: Latency benchmark
    from src.eval.latency_benchmark import run_latency_benchmark
    _, t = _run_stage("Latency Benchmark", run_latency_benchmark, config)
    stage_times["latency_benchmark"] = t

    # Stage 12: Generate summary
    _run_stage("Generate Summary", generate_summary, config)

    # Final report
    total = sum(stage_times.values())
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — Total time: %.1fs", total)
    logger.info("=" * 60)
    for stage, t in stage_times.items():
        logger.info("  %-30s %7.1fs", stage, t)


if __name__ == "__main__":
    main()
