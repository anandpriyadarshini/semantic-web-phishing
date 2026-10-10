# Semantic-Web Phishing Detection Framework

A fully reproducible research pipeline implementing phishing URL classification,
explainability evaluation, RDF knowledge graph modeling, and near-real-time
latency benchmarking. Every number in the paper comes from code run on real
data — nothing is fabricated or hardcoded.

---

## Setup

### Requirements
- Python 3.11+
- ~2 GB disk (data + models)
- Internet access for initial data download (PhishTank, Tranco, DNS/WHOIS/crt.sh)

### Installation

```bash
cd phishing-v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Reproducing Every Result

### Full pipeline (all stages)

```bash
make pipeline
# or equivalently:
python run_all.py
```

This runs all 12 stages in order. All outputs land in `results/`.  
The final `results/summary.md` lists every reported metric.

### Individual stages

```bash
make fetch        # Download PhishTank + Tranco data
make split        # Deduplicate and split by domain + time
make features     # Extract 28 lexical/structural features
make train        # GridSearchCV over KNN, DT, RF, LR, XGBoost
make evaluate     # Confusion matrices, ROC/PR, McNemar tests
make xai          # SHAP + LIME explanations + quantitative metrics
make graph        # Build RDF knowledge graph + campaign clustering
make benchmark    # Latency benchmark (p50/p95/throughput)
```

### Tests

```bash
# Run XAI unit tests (no data needed):
make test-xai

# Run leakage tests (requires data pipeline to have run):
make test-leakage

# Run all tests:
make test
```

---

## Data Sources

| Source | URL | License |
|--------|-----|---------|
| PhishTank | http://data.phishtank.com/data/online-valid.csv | CC BY-SA 2.5 |
| Tranco | https://tranco-list.eu/top-1m.csv.zip | MIT |
| ASN lookup | Team Cymru DNS service (free) | — |
| Cert transparency | https://crt.sh (Sectigo public API) | Public |

> **Dev mode**: `config.yaml` sets `dev_sample_size: 500`. Set to `null`
> for full dataset runs.

---

## Features (Table 5 equivalent)

28 lexical/structural features extracted from URL strings only (no network lookups):

| # | Feature | Description |
|---|---------|-------------|
| 1 | `url_length` | Total URL character count |
| 2 | `hostname_length` | Hostname character count |
| 3 | `path_length` | URL path length |
| 4 | `num_dots_in_url` | Count of `.` in full URL |
| 5 | `num_dots_in_hostname` | Count of `.` in hostname |
| 6 | `num_hyphens_in_url` | Count of `-` in URL |
| 7 | `num_hyphens_in_hostname` | Count of `-` in hostname |
| 8 | `num_underscores_in_url` | Count of `_` in URL |
| 9 | `num_slashes_in_path` | Count of `/` in path |
| 10 | `num_query_params` | Number of URL query parameters |
| 11 | `num_fragments` | 1 if URL has `#` fragment |
| 12 | `num_digits_in_url` | Count of digits in URL |
| 13 | `num_digits_in_hostname` | Count of digits in hostname |
| 14 | `ratio_digits_url` | Fraction of URL chars that are digits |
| 15 | `ratio_digits_hostname` | Fraction of hostname chars that are digits |
| 16 | `has_ip_address` | 1 if hostname is an IP address |
| 17 | `has_at_symbol` | 1 if URL contains `@` |
| 18 | `has_double_slash_redirect` | 1 if path contains `//` |
| 19 | `has_hex_encoding` | 1 if URL contains `%XX` encoding |
| 20 | `has_www` | 1 if hostname starts with `www.` |
| 21 | `has_https` | 1 if scheme is HTTPS |
| 22 | `url_entropy` | Shannon entropy of URL string |
| 23 | `hostname_entropy` | Shannon entropy of hostname |
| 24 | `longest_token_length` | Max token length (split on delimiters) |
| 25 | `avg_token_length` | Mean token length |
| 26 | `num_subdomains` | Number of subdomain components |
| 27 | `tld_length` | Character length of TLD |
| 28 | `path_depth` | Number of path components |

---

## Leakage Prevention

Two invariants are enforced and tested by `tests/test_no_leakage.py`:

1. **No exact URL** appears in more than one split.
2. **No registered domain** appears in both train and test.

Domain-based split: all URLs from a domain go entirely to train or test.  
Time-based split: train on older PhishTank entries, test on newer.

Run `make test-leakage` to verify after any data pipeline change.

---

## XAI Metrics (Quantitative Only)

No subjective 1–10 scoring. All metrics are objectively computable:

| Metric | Measures |
|--------|---------|
| Fidelity (deletion@k) | Prediction drop when top-k features removed |
| Fidelity (insertion@k) | Prediction rise when top-k features inserted |
| Stability (Jaccard@k) | Top-k feature set overlap across random seeds |
| Rank agreement (Kendall's τ) | SHAP vs LIME feature ranking correlation |
| Sparsity | Fraction of features with near-zero attribution |
| Runtime | Wall-clock time for SHAP vs LIME |

---

## Knowledge Graph (OWL + SPARQL)

**Ontology**: `src/graph/ontology.ttl`  
**Classes**: URL, Domain, IPAddress, ASN, Registrar, HostingProvider, Certificate, Campaign  
**Properties**: hasDomain, resolvesTo, hostedOn, registeredBy, usesCertificate, sharesInfrastructureWith, belongsToCampaign

Graph is queried with real SPARQL via rdflib. Campaign clustering uses Louvain
community detection (with connected-components fallback).

See `docs/ontology_design.md` for full design rationale.

---

## Output Files

All outputs are auto-generated in `results/`. Nothing in `results/` is hand-edited.

| File | Contents |
|------|----------|
| `results/summary.md` | All metrics from a single run |
| `results/training_results.json` | CV F1 mean±std per model |
| `results/evaluation_results.json` | Test set metrics + McNemar tests |
| `results/xai_results.json` | Raw SHAP/LIME attribution values |
| `results/xai_metrics.json` | Fidelity, stability, rank agreement, sparsity |
| `results/campaign_results.json` | Cluster sizes, graph stats |
| `results/cluster_evaluation.json` | Purity/NMI/ARI (if ground truth available) |
| `results/latency_benchmark.json` | p50/p95/throughput per pipeline stage |
| `results/graph/phishing_graph.ttl` | Full RDF graph in Turtle format |
| `results/figures/` | All plots (PNG) |

Every JSON file includes a `timestamp` and `git_commit` field so results
are traceable to the exact code version that produced them.

---

## Feature Importance Warning

If any single feature accounts for a disproportionate share of model importance
(>50% of total, or >3× the uniform baseline), the evaluation script flags it
with a warning in `results/evaluation_results.json` and in `results/summary.md`.
This is a finding to investigate and report honestly, not suppress.

---

## Enrichment Failures

If DNS/WHOIS/ASN/cert lookups fail for a URL, that URL is:
- Logged to stdout with the failure reason
- Excluded from graph-dependent analyses (clustering, correlation queries)
- NOT assigned placeholder or fabricated infrastructure data

Enrichment coverage statistics are reported in `results/graph/graph_stats.json`.

---

## Configuration

All paths, seeds, and hyperparameter grids are in `config.yaml`.  
Change `dev_sample_size: null` for full-dataset runs.

---

## Folder Structure

```
phishing-v2/
├── config.yaml                 # All configuration
├── requirements.txt            # Pinned dependencies
├── run_all.py                  # Top-level entrypoint
├── Makefile                    # make pipeline / make test
├── data/
│   ├── raw/                    # Downloaded data
│   ├── processed/              # Deduplicated + feature-extracted
│   ├── splits/                 # Train/val/test splits
│   └── cache/                  # DNS/WHOIS/ASN/cert caches
├── src/
│   ├── data/                   # Fetchers + splitter
│   ├── features/               # Feature extraction
│   ├── models/                 # Training + evaluation
│   ├── xai/                    # SHAP/LIME + metrics
│   ├── enrichment/             # DNS, WHOIS, ASN, cert lookups
│   ├── graph/                  # Ontology, RDF builder, SPARQL, clustering
│   ├── eval/                   # Cluster eval + latency benchmark
│   └── pipeline/               # Orchestrator
├── tests/
│   ├── test_no_leakage.py      # Leakage prevention tests
│   └── test_xai_metrics.py     # XAI metric unit tests
├── results/                    # Auto-generated outputs
├── docs/
│   └── ontology_design.md      # OWL design rationale
├── notebooks/                  # Exploratory only
└── paper/
    ├── response_to_reviewers.md
    └── tables/                 # LaTeX tables (auto-generated)
```
