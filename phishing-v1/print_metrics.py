"""
Visual metrics printer for phishing detection pipeline results.
Run: python phishing-v2/print_metrics.py
"""
import json, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(BASE, "results")

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

def bar(value, width=30, color=GREEN):
    filled = int(round(value * width))
    return color + "█" * filled + DIM + "░" * (width - filled) + RESET

def header(title):
    line = "═" * 68
    print(f"\n{BOLD}{CYAN}╔{line}╗{RESET}")
    print(f"{BOLD}{CYAN}║  {title:<66}║{RESET}")
    print(f"{BOLD}{CYAN}╚{line}╝{RESET}")

def section(title):
    print(f"\n{BOLD}{YELLOW}  ▶  {title}{RESET}")
    print(f"  {'─'*62}")

# ─────────────────────────────────────────────────────────────
# 1. EVALUATION METRICS
# ─────────────────────────────────────────────────────────────
header("MODEL EVALUATION RESULTS  (final_refined_dataset_phishing.csv)")

with open(os.path.join(R, "evaluation_results.json")) as f:
    ev = json.load(f)["results"]

MODEL_KEYS = [k for k in ev if not k.startswith("mcnemar")]
metrics = ["accuracy","precision","recall","f1","roc_auc"]
labels  = ["Accuracy ","Precision","Recall   ","F1 Score ","ROC-AUC  "]

section("Domain Split")
dom_models = sorted([k for k in MODEL_KEYS if k.endswith("_domain")])
print(f"  {'Model':<28} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print(f"  {'─'*28} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
for k in dom_models:
    d = ev[k]
    name = k.replace("_domain","")
    acc, pre, rec, f1, auc = d["accuracy"], d["precision"], d["recall"], d["f1"], d["roc_auc"]
    best = f1 == max(ev[m]["f1"] for m in dom_models)
    star = f" {YELLOW}★{RESET}" if best else "  "
    print(f"  {BOLD}{name:<28}{RESET} {acc:.4f} {pre:.4f} {rec:.4f} {GREEN}{f1:.4f}{RESET} {auc:.4f}{star}")

section("Time Split")
time_models = sorted([k for k in MODEL_KEYS if k.endswith("_time")])
print(f"  {'Model':<28} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print(f"  {'─'*28} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
for k in time_models:
    d = ev[k]
    name = k.replace("_time","")
    acc, pre, rec, f1, auc = d["accuracy"], d["precision"], d["recall"], d["f1"], d["roc_auc"]
    best = f1 == max(ev[m]["f1"] for m in time_models)
    star = f" {YELLOW}★{RESET}" if best else "  "
    print(f"  {BOLD}{name:<28}{RESET} {acc:.4f} {pre:.4f} {rec:.4f} {GREEN}{f1:.4f}{RESET} {auc:.4f}{star}")

# Best model bar chart
section("F1 Score Comparison (bar)")
all_models = dom_models + time_models
max_f1 = max(ev[k]["f1"] for k in all_models)
for k in sorted(all_models):
    f1 = ev[k]["f1"]
    color = GREEN if f1 == max_f1 else (CYAN if f1 > 0.95 else YELLOW)
    label = k.replace("_domain"," [D]").replace("_time"," [T]")
    print(f"  {label:<36} {bar(f1,30,color)} {f1:.4f}")

# ─────────────────────────────────────────────────────────────
# 2. XAI METRICS
# ─────────────────────────────────────────────────────────────
header("XAI METRICS  (SHAP & LIME — Fidelity, Stability, Sparsity)")

with open(os.path.join(R, "xai_metrics.json")) as f:
    xm = json.load(f)["results"]

section("Fidelity — Deletion@10 (mean probability drop when top-10 features removed)")
print(f"  {'Model':<36} {'SHAP':>8} {'LIME':>8}")
print(f"  {'─'*36} {'─'*8} {'─'*8}")
for k, d in xm.items():
    shap_d = d.get("fidelity_deletion_shap",{}).get("deletion@10",{}).get("mean_drop", None)
    lime_d = d.get("fidelity_deletion_lime",{}).get("deletion@10",{}).get("mean_drop", None)
    if shap_d is None: continue
    label = k.replace("_domain"," [D]").replace("_time"," [T]")
    print(f"  {label:<36} {shap_d:>8.4f} {lime_d:>8.4f}")

section("Stability — Jaccard@5 (feature-set overlap across 5 runs; higher=more stable)")
print(f"  {'Model':<36} {'SHAP':>8} {'LIME':>8}")
print(f"  {'─'*36} {'─'*8} {'─'*8}")
for k, d in xm.items():
    sv = d.get("stability_shap_jaccard@5",{}).get("jaccard_mean", None)
    lv = d.get("stability_lime_jaccard@5",{}).get("jaccard_mean", None)
    if sv is None: continue
    label = k.replace("_domain"," [D]").replace("_time"," [T]")
    sc = GREEN if sv > 0.6 else (YELLOW if sv > 0.4 else RED)
    lc = GREEN if lv > 0.6 else (YELLOW if lv > 0.4 else RED)
    print(f"  {label:<36} {sc}{sv:>8.4f}{RESET} {lc}{lv:>8.4f}{RESET}")

section("Sparsity (fraction of near-zero attributions; higher = more interpretable)")
print(f"  {'Model':<36} {'SHAP':>8} {'LIME':>8}")
print(f"  {'─'*36} {'─'*8} {'─'*8}")
for k, d in xm.items():
    sv = d.get("sparsity_shap",{}).get("mean_sparsity", None)
    lv = d.get("sparsity_lime",{}).get("mean_sparsity", None)
    if sv is None: continue
    label = k.replace("_domain"," [D]").replace("_time"," [T]")
    print(f"  {label:<36} {sv:>8.4f} {lv:>8.4f}")

# ─────────────────────────────────────────────────────────────
# 3. LATENCY BENCHMARK
# ─────────────────────────────────────────────────────────────
header("LATENCY BENCHMARK  (100 URLs × 3 repetitions)")

with open(os.path.join(R, "latency_benchmark.json")) as f:
    lb = json.load(f)["results"]

stages = lb["stages"]
section("Per-stage latency (ms)")
print(f"  {'Stage':<26} {'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8}")
print(f"  {'─'*26} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
order = ["feature_extraction","enrichment_total","inference","graph_insertion","correlation_query","end_to_end"]
for s in order:
    d = stages.get(s)
    if not d: continue
    name = s.replace("_"," ").title()
    color = GREEN if d["p50_ms"] < 10 else (YELLOW if d["p50_ms"] < 500 else RED)
    print(f"  {name:<26} {color}{d['p50_ms']:>8.1f}{RESET} {d['p95_ms']:>8.1f} {d['p99_ms']:>8.1f} {d['mean_ms']:>8.1f}")

print(f"\n  {BOLD}Throughput:{RESET} {lb['throughput_urls_per_second']:.4f} URLs/s  "
      f"({YELLOW}end-to-end p50 = {stages['end_to_end']['p50_ms']:.1f} ms{RESET})")

# ─────────────────────────────────────────────────────────────
# 4. CAMPAIGN CLUSTERING
# ─────────────────────────────────────────────────────────────
header("CAMPAIGN CLUSTERING RESULTS")

with open(os.path.join(R, "campaign_results.json")) as f:
    cr = json.load(f)

section("Cluster summary")
clusters = cr.get("clusters", cr.get("results", {}).get("clusters", {}))
stats = cr.get("stats", cr.get("results", {}).get("stats", {}))

if stats:
    for k,v in stats.items():
        if isinstance(v,(int,float)):
            print(f"  {k:<35} {CYAN}{v}{RESET}")

if clusters:
    top_clusters = sorted(clusters.items(), key=lambda x: x[1].get("size",0), reverse=True)[:10]
    print(f"\n  {'Cluster':<20} {'Size':>6} {'Label':>12}")
    print(f"  {'─'*20} {'─'*6} {'─'*12}")
    for cid, cdata in top_clusters:
        size = cdata.get("size", "?")
        label = cdata.get("dominant_label", cdata.get("label","?"))
        print(f"  {str(cid):<20} {str(size):>6} {str(label):>12}")

# ─────────────────────────────────────────────────────────────
# 5. FOOTER
# ─────────────────────────────────────────────────────────────
header("QUICK REFERENCE — Key Files")
files = [
    ("Summary Report",    "results/summary.md"),
    ("Evaluation JSON",   "results/evaluation_results.json"),
    ("Training JSON",     "results/training_results.json"),
    ("XAI Metrics JSON",  "results/xai_metrics.json"),
    ("Campaign JSON",     "results/campaign_results.json"),
    ("Latency JSON",      "results/latency_benchmark.json"),
    ("SHAP/CM Plots",     "results/figures/"),
    ("Trained Models",    "results/models/"),
    ("KG (Turtle)",       "results/graph/phishing_graph.ttl"),
]
for label, path in files:
    full = os.path.join(BASE, path.replace("/", os.sep))
    exists = "✓" if os.path.exists(full) else "✗"
    color = GREEN if exists == "✓" else RED
    print(f"  {color}{exists}{RESET}  {label:<24} {DIM}{path}{RESET}")

print(f"\n  {DIM}★ = best F1  |  [D] = domain split  |  [T] = time split{RESET}\n")
