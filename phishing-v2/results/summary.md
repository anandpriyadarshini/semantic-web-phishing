# Pipeline Results Summary

**Generated**: 2026-10-10T15:52:46.241292+00:00  
**Git commit**: unknown  
**Python**: 3.9.6

---

## Classification Results

| Model | Split | Accuracy | Precision | Recall | F1 | AUC | n_test |
|-------|-------|----------|-----------|--------|-----|-----|--------|
| XGBoost | time | 0.9524 | 1.0000 | 0.8857 | 0.9394 | 0.9913 | 168 |
| RandomForest | domain | 0.9820 | 1.0000 | 0.9565 | 0.9778 | 0.9963 | 167 |
| LogisticRegression | time | 0.9524 | 1.0000 | 0.8857 | 0.9394 | 0.9895 | 168 |
| LogisticRegression | domain | 0.9760 | 0.9851 | 0.9565 | 0.9706 | 0.9945 | 167 |
| KNN | time | 0.9464 | 0.9841 | 0.8857 | 0.9323 | 0.9545 | 168 |
| RandomForest | time | 0.9464 | 0.9841 | 0.8857 | 0.9323 | 0.9911 | 168 |
| DecisionTree | time | 0.9405 | 1.0000 | 0.8571 | 0.9231 | 0.9643 | 168 |
| KNN | domain | 0.9581 | 0.9559 | 0.9420 | 0.9489 | 0.9827 | 167 |
| XGBoost | domain | 0.9760 | 1.0000 | 0.9420 | 0.9701 | 0.9959 | 167 |
| DecisionTree | domain | 0.9820 | 1.0000 | 0.9565 | 0.9778 | 0.9829 | 167 |

### McNemar Pairwise Tests

| Comparison | χ² statistic | p-value | Significant (α=0.05) |
|------------|-------------|---------|---------------------|
| XGBoost_vs_LogisticRegression_time | 0.000 | 1.0000 | No |
| XGBoost_vs_KNN_time | 0.000 | 1.0000 | No |
| XGBoost_vs_RandomForest_time | 0.000 | 1.0000 | No |
| XGBoost_vs_DecisionTree_time | 0.500 | 0.4795 | No |
| LogisticRegression_vs_KNN_time | 0.000 | 1.0000 | No |
| LogisticRegression_vs_RandomForest_time | 0.000 | 1.0000 | No |
| LogisticRegression_vs_DecisionTree_time | 0.500 | 0.4795 | No |
| KNN_vs_RandomForest_time | 0.500 | 0.4795 | No |
| KNN_vs_DecisionTree_time | 0.000 | 1.0000 | No |
| RandomForest_vs_DecisionTree_time | 0.000 | 1.0000 | No |
| RandomForest_vs_LogisticRegression_domain | 0.000 | 1.0000 | No |
| RandomForest_vs_KNN_domain | 2.250 | 0.1336 | No |
| RandomForest_vs_XGBoost_domain | 0.000 | 1.0000 | No |
| RandomForest_vs_DecisionTree_domain | 0.000 | 1.0000 | No |
| LogisticRegression_vs_KNN_domain | 0.800 | 0.3711 | No |
| LogisticRegression_vs_XGBoost_domain | 0.500 | 0.4795 | No |
| LogisticRegression_vs_DecisionTree_domain | 0.000 | 1.0000 | No |
| KNN_vs_XGBoost_domain | 0.800 | 0.3711 | No |
| KNN_vs_DecisionTree_domain | 2.250 | 0.1336 | No |
| XGBoost_vs_DecisionTree_domain | 0.000 | 1.0000 | No |

### ⚠️ Feature Dominance Warnings

- **XGBoost_time**: Feature 'url_length' has 51.3% of total importance. This may indicate data leakage, a trivially-predictive proxy, or a dataset artifact. Investigate whether this feature genuinely captures phishing behavior or is simply an artifact of the data collection process.
- **XGBoost_time**: Feature 'url_length' importance (0.513) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **XGBoost_time**: Feature 'path_length' importance (0.136) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **XGBoost_time**: Feature 'num_subdomains' importance (0.121) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **RandomForest_domain**: Feature 'url_length' importance (0.196) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **RandomForest_domain**: Feature 'path_length' importance (0.158) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **RandomForest_domain**: Feature 'num_slashes_in_path' importance (0.159) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **LogisticRegression_time**: Feature 'num_slashes_in_path' importance (0.288) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **LogisticRegression_domain**: Feature 'num_slashes_in_path' importance (0.271) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **LogisticRegression_domain**: Feature 'longest_token_length' importance (0.143) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **LogisticRegression_domain**: Feature 'num_subdomains' importance (0.172) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **RandomForest_time**: Feature 'url_length' importance (0.176) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **RandomForest_time**: Feature 'path_length' importance (0.139) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **RandomForest_time**: Feature 'num_slashes_in_path' importance (0.173) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **DecisionTree_time**: Feature 'url_length' has 76.8% of total importance. This may indicate data leakage, a trivially-predictive proxy, or a dataset artifact. Investigate whether this feature genuinely captures phishing behavior or is simply an artifact of the data collection process.
- **DecisionTree_time**: Feature 'url_length' importance (0.768) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **XGBoost_domain**: Feature 'url_length' has 54.3% of total importance. This may indicate data leakage, a trivially-predictive proxy, or a dataset artifact. Investigate whether this feature genuinely captures phishing behavior or is simply an artifact of the data collection process.
- **XGBoost_domain**: Feature 'url_length' importance (0.543) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **XGBoost_domain**: Feature 'path_length' importance (0.150) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **XGBoost_domain**: Feature 'num_dots_in_url' importance (0.163) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **DecisionTree_domain**: Feature 'url_length' has 74.8% of total importance. This may indicate data leakage, a trivially-predictive proxy, or a dataset artifact. Investigate whether this feature genuinely captures phishing behavior or is simply an artifact of the data collection process.
- **DecisionTree_domain**: Feature 'url_length' importance (0.748) is >0.107 (3× uniform baseline of 0.036). Marked for review.
- **DecisionTree_domain**: Feature 'path_length' importance (0.127) is >0.107 (3× uniform baseline of 0.036). Marked for review.

## Cross-Validation Results

| Model | Split | CV F1 (mean ± std) | n_train |
|-------|-------|---------------------|---------|
| KNN | domain | 0.9671 ± 0.0061 | 669 |
| DecisionTree | domain | 0.9875 ± 0.0063 | 669 |
| RandomForest | domain | 0.9874 ± 0.0096 | 669 |
| LogisticRegression | domain | 0.9844 ± 0.0088 | 669 |
| XGBoost | domain | 0.9859 ± 0.0127 | 669 |
| KNN | time | 0.9762 ± 0.0205 | 755 |
| DecisionTree | time | 0.9903 ± 0.0034 | 755 |
| RandomForest | time | 0.9917 ± 0.0052 | 755 |
| LogisticRegression | time | 0.9917 ± 0.0052 | 755 |
| XGBoost | time | 0.9917 ± 0.0052 | 755 |

## Explainability (XAI) Metrics

All metrics are quantitative and objectively computable — no subjective scoring.

### RandomForest_domain

**SHAP–LIME Rank Agreement (Kendall's τ)**: -0.0739 ± 0.0388 (n=100)

**SHAP Sparsity**: 0.3654 ± 0.0756
**LIME Sparsity**: 1.0000 ± 0.0000
**Runtime**: SHAP=0.0s, LIME=3.0s
**SHAP Fidelity (deletion)**: deletion@3: 0.5825, deletion@5: 0.6140, deletion@10: 0.5718
**LIME Fidelity (deletion)**: deletion@3: 0.0182, deletion@5: 0.2248, deletion@10: 0.5190

### RandomForest_time

**SHAP–LIME Rank Agreement (Kendall's τ)**: -0.0716 ± 0.0367 (n=100)

**SHAP Sparsity**: 0.3789 ± 0.0739
**LIME Sparsity**: 1.0000 ± 0.0000
**Runtime**: SHAP=0.0s, LIME=2.9s
**SHAP Fidelity (deletion)**: deletion@3: 0.5874, deletion@5: 0.5969, deletion@10: 0.5746
**LIME Fidelity (deletion)**: deletion@3: 0.0342, deletion@5: 0.2399, deletion@10: 0.5372

## Campaign Clustering (Infrastructure Correlation)

- **Method**: louvain
- **Graph nodes**: 435
- **Graph edges**: 5168
- **Graph density**: 0.0547486625350919
- **Total campaigns**: 233
- **Significant campaigns (size ≥ 2)**: 24
- **Campaign sizes**: mean=9.4, median=3.0, max=88

### Cluster Evaluation

> **Note**: NO ground-truth campaign labels available. Cannot compute purity, NMI, or ARI. Reporting only descriptive cluster statistics. To enable supervised evaluation, provide campaign_ground_truth.csv in data/raw/ with columns: url, campaign_id

## Latency Benchmark

- **URLs benchmarked**: 100
- **Measurements**: 300
- **Throughput**: 1.8 URLs/s

| Stage | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|-------|----------|----------|----------|----------|
| feature_extraction | 0.1 | 0.2 | 0.2 | 0.2 |
| enrichment_dns | 0.0 | 2006.1 | 4004.2 | 4005.0 |
| enrichment_whois | 0.0 | 0.0 | 0.0 | 0.0 |
| enrichment_asn | 0.0 | 336.3 | 789.5 | 2003.0 |
| enrichment_cert | 0.0 | 1234.5 | 5743.5 | 6007.5 |
| enrichment_total | 0.0 | 3773.5 | 8016.4 | 9905.1 |
| inference | 4.6 | 9.1 | 9.9 | 13.8 |
| graph_insertion | 0.1 | 0.3 | 0.3 | 0.4 |
| correlation_query | 6.4 | 11.2 | 12.4 | 13.1 |
| end_to_end | 11.2 | 3793.9 | 8033.4 | 9923.6 |

> Latency measurements are real wall-clock times. Enrichment latency includes actual DNS/WHOIS/ASN/cert lookups (cached after first resolution). Network conditions affect enrichment timing.

## Knowledge Graph Statistics

- **Total RDF triples**: 3837
- **URLs processed**: 435
- **URLs with DNS**: 244
- **URLs with WHOIS**: 0
- **URLs with ASN**: 224
- **URLs with Certificates**: 0

---

## Caveats and Limitations

- Classification metrics are from k-fold CV (mean ± std). Single-split results on held-out test sets may differ.
- XAI stability depends on the number of perturbation runs and sample size. Results with <50 samples should be interpreted cautiously.
- Campaign clustering uses infrastructure correlation only. Without ground-truth campaign labels, these are **candidate clusters**, not confirmed attributions.
- Latency measurements include real network lookups (cached after first run). First-run latency will be significantly higher than cached-run latency.
- Enrichment coverage depends on DNS/WHOIS/crt.sh availability at runtime.
