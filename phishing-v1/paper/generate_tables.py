"""
Auto-generate LaTeX table fragments from results/ JSON files.

Run after the full pipeline to produce camera-ready table fragments.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import load_config, get_project_root, result_metadata


def _fmt(v, decimals=4):
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v) if v is not None else "N/A"


def generate_classification_table(results: dict, outfile: Path) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Classification Results — Domain-Based and Time-Based Splits}",
        r"  \label{tab:classification}",
        r"  \begin{tabular}{llrrrr}",
        r"    \toprule",
        r"    Model & Split & Accuracy & Precision & Recall & F1 \\",
        r"    \midrule",
    ]
    for key, r in results.items():
        if key == "mcnemar_tests" or not isinstance(r, dict) or "accuracy" not in r:
            continue
        lines.append(
            f"    {r.get('classifier', key)} & {r.get('split_type', '?')} & "
            f"{_fmt(r['accuracy'])} & {_fmt(r['precision'])} & "
            f"{_fmt(r['recall'])} & {_fmt(r['f1'])} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    outfile.write_text("\n".join(lines))
    print(f"Written: {outfile}")


def generate_latency_table(stages: dict, outfile: Path) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Pipeline Latency Benchmark (ms)}",
        r"  \label{tab:latency}",
        r"  \begin{tabular}{lrrr}",
        r"    \toprule",
        r"    Stage & p50 & p95 & max \\",
        r"    \midrule",
    ]
    for stage, s in stages.items():
        lines.append(
            f"    {stage.replace('_', ' ')} & {_fmt(s.get('p50_ms', 0), 1)} & "
            f"{_fmt(s.get('p95_ms', 0), 1)} & {_fmt(s.get('max_ms', 0), 1)} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    outfile.write_text("\n".join(lines))
    print(f"Written: {outfile}")


def main():
    config = load_config()
    root = get_project_root()
    results_dir = root / config["paths"]["results"]
    tables_dir = root / config["paths"]["tables"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Classification table
    eval_path = results_dir / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            data = json.load(f)
        results = data.get("results", data)
        generate_classification_table(results, tables_dir / "classification.tex")

    # Latency table
    lat_path = results_dir / "latency_benchmark.json"
    if lat_path.exists():
        with open(lat_path) as f:
            data = json.load(f)
        lat = data.get("results", data)
        stages = lat.get("stages", {})
        if stages:
            generate_latency_table(stages, tables_dir / "latency.tex")

    print("LaTeX tables generated in", tables_dir)


if __name__ == "__main__":
    main()
