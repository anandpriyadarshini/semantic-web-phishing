# Response to Reviewers

> **Note**: This document is a skeleton. Fill in the numbered sections
> after running the full pipeline and reviewing `results/summary.md`.
> Every claim in this response must reference a specific entry in `results/`.

---

## Overview of Revisions

The manuscript was substantially revised to address the three core reviewer
concerns: (1) missing implementation of Semantic Web modeling, (2) missing
attacker identification methodology, and (3) unverified near-real-time claims.

We restructruct all three components from scratch in a new codebase
(`phishing-v2/`) with the following guarantees:
- Every reported number is produced by a script in this repository.
- Every script writes its output to `results/` with a git commit hash.
- No metric is hand-typed or fabricated.

---

## Reviewer 1

### Comment R1.1: "The paper claims Semantic Web modeling but implements only a flat feature vector."

**Response**: We have implemented a full OWL ontology (`src/graph/ontology.ttl`)
with [N] classes and [M] properties. The graph is populated by real DNS, WHOIS,
ASN, and certificate lookups and queried with SPARQL for infrastructure
correlation. Statistics are in `results/graph/graph_stats.json`:
- Total RDF triples: [from results]
- URLs with resolved infrastructure: [from results]

See `docs/ontology_design.md` for the design rationale of each class and
property.

### Comment R1.2: "The near-real-time claim is unverified — no latency data provided."

**Response**: We implemented a latency benchmark (`src/eval/latency_benchmark.py`)
that measures real wall-clock timings across all pipeline stages on [N] URLs
with [K] repetitions. Results from `results/latency_benchmark.json`:

| Stage | p50 | p95 |
|-------|-----|-----|
| Feature extraction | [value] ms | [value] ms |
| DNS lookup | [value] ms | [value] ms |
| WHOIS lookup | [value] ms | [value] ms |
| ASN lookup | [value] ms | [value] ms |
| Certificate lookup | [value] ms | [value] ms |
| ML inference | [value] ms | [value] ms |
| Graph insertion | [value] ms | [value] ms |
| Correlation query | [value] ms | [value] ms |
| **End-to-end** | **[value] ms** | **[value] ms** |

Throughput: [value] URLs/second.

---

## Reviewer 2

### Comment R2.1: "Attacker identification is claimed but not validated against any ground truth."

**Response**: We have removed all claims of "attacker identification" from
the manuscript. The methodology is now described as **infrastructure correlation**
and **candidate campaign clustering**. We explicitly state in the paper and in
`src/eval/cluster_eval.py` that clusters are candidate groupings based on
shared infrastructure, not confirmed attributions.

`results/cluster_evaluation.json` documents whether ground-truth labels were
available for evaluation. At submission time: [ground truth available: YES/NO].
[If NO: only descriptive cluster statistics are reported, as explained in the code.]

### Comment R2.2: "Single train/test split results are not statistically reliable."

**Response**: All classification results now use [K]-fold stratified
cross-validation. Results report mean ± standard deviation across folds.
See `results/training_results.json` and `results/evaluation_results.json`.

Additionally, McNemar's test is applied to all pairwise model comparisons
to assess statistical significance of performance differences.
Results are in `results/evaluation_results.json` under `mcnemar_tests`.

### Comment R2.3: "Feature importance from Figure 5 appears dominated by a single feature (`has_www`)."

**Response**: We investigated this. The evaluation script (`src/models/evaluate.py`)
now automatically checks for feature dominance and flags any feature exceeding
50% of total importance or 3× the uniform baseline.

Results from the revised run are in `results/evaluation_results.json` under
`feature_dominance_warnings`. [Insert finding here after running pipeline.]

---

## Reviewer 3

### Comment R3.1: "XAI comparison uses subjective 1–10 scoring with no clear methodology."

**Response**: All XAI evaluation metrics have been replaced with objective,
computable measures:
- **Fidelity** (deletion/insertion): prediction change when top-k features
  are removed or added.
- **Stability** (Jaccard@k): overlap of top-k explanation sets across seeds.
- **Rank agreement** (Kendall's τ): correlation between SHAP and LIME
  feature rankings.
- **Sparsity**: fraction of features with near-zero attribution.
- **Runtime**: wall-clock comparison.

Results in `results/xai_metrics.json`. No subjective scoring appears anywhere.

---

## Reproducibility Statement

All results can be reproduced from a clean checkout with:

```bash
pip install -r requirements.txt
python run_all.py
```

Expected runtime: [fill in after running full pipeline].
All results land in `results/summary.md` with timestamp and git commit hash.
