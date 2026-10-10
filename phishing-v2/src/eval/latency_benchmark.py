from __future__ import annotations
"""
Near-real-time latency benchmark.

Measures wall-clock timings of the full pipeline:
  feature extraction → enrichment → inference → graph insertion → correlation query

Reports p50, p95, max, and throughput for each stage and end-to-end,
with enrichment latency broken out separately.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdflib import Graph as RDFGraph, Literal, Namespace, RDF, XSD, URIRef
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import load_config, setup_logging, get_project_root, save_result_json
from src.features.extract_features import extract_single_url, get_feature_names
from src.enrichment.dns_lookup import DNSResolver
from src.enrichment.whois_lookup import WhoisLookup
from src.enrichment.asn_lookup import ASNLookup
from src.enrichment.cert_lookup import CertLookup

logger = setup_logging("latency_benchmark")

PHISH = Namespace("http://example.org/phishing-ontology#")
INST = Namespace("http://example.org/phishing-instance#")


def _benchmark_single_url(
    url: str,
    model,
    feature_names: list[str],
    rdf_graph: RDFGraph,
    dns_resolver: DNSResolver,
    whois_lookup: WhoisLookup,
    asn_lookup: ASNLookup,
    cert_lookup: CertLookup,
) -> dict:
    """
    Benchmark the full pipeline for a single URL.

    Returns timing dict with per-stage wall-clock measurements.
    """
    timings = {}

    # Stage 1: Feature extraction
    t0 = time.perf_counter()
    features = extract_single_url(url)
    timings["feature_extraction"] = time.perf_counter() - t0

    # Stage 2: Enrichment (broken out)
    t0 = time.perf_counter()

    t_dns = time.perf_counter()
    dns_result = dns_resolver.resolve_url(url)
    timings["enrichment_dns"] = time.perf_counter() - t_dns

    t_whois = time.perf_counter()
    if whois_lookup is not None:
        whois_result = whois_lookup.lookup_url(url)
    else:
        whois_result = None
    timings["enrichment_whois"] = time.perf_counter() - t_whois

    ips = dns_result.get("ips", []) if dns_result else []
    t_asn = time.perf_counter()
    asn_results = [asn_lookup.lookup_ip(ip) for ip in ips[:1]]  # Limit to first IP
    timings["enrichment_asn"] = time.perf_counter() - t_asn

    t_cert = time.perf_counter()
    cert_result = cert_lookup.lookup_url(url)
    timings["enrichment_cert"] = time.perf_counter() - t_cert

    timings["enrichment_total"] = time.perf_counter() - t0

    # Stage 3: ML inference
    t0 = time.perf_counter()
    feature_vector = np.array([features.get(f, 0.0) for f in feature_names]).reshape(1, -1)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    prediction = model.predict(feature_vector)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feature_vector)[0, 1]
    else:
        proba = float(prediction)
    timings["inference"] = time.perf_counter() - t0

    # Stage 4: Graph insertion
    t0 = time.perf_counter()
    url_uri = URIRef(f"{INST}url/{quote(url, safe='')}")
    rdf_graph.add((url_uri, RDF.type, PHISH.URL))
    rdf_graph.add((url_uri, PHISH.urlString, Literal(url)))
    rdf_graph.add((url_uri, PHISH.label, Literal(int(prediction), datatype=XSD.integer)))
    rdf_graph.add((url_uri, PHISH.predictionScore, Literal(float(proba), datatype=XSD.float)))
    timings["graph_insertion"] = time.perf_counter() - t0

    # Stage 5: Correlation query
    t0 = time.perf_counter()
    correlation_query = f"""
    PREFIX phish: <http://example.org/phishing-ontology#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT (COUNT(?related) AS ?n_related)
    WHERE {{
        <{str(url_uri)}> phish:hasDomain ?d .
        ?d phish:resolvesTo ?ip .
        ?otherDomain phish:resolvesTo ?ip .
        ?related phish:hasDomain ?otherDomain .
        FILTER(?related != <{str(url_uri)}>)
    }}
    """
    try:
        list(rdf_graph.query(correlation_query))
    except Exception:
        pass  # Query may fail if no domain data; timing still valid
    timings["correlation_query"] = time.perf_counter() - t0

    # End-to-end
    timings["end_to_end"] = sum(timings[k] for k in [
        "feature_extraction", "enrichment_total", "inference",
        "graph_insertion", "correlation_query"
    ])

    return timings


def run_latency_benchmark(config: dict | None = None) -> dict:
    """
    Run the latency benchmark on a sample of URLs.

    Returns p50/p95/max/throughput for each pipeline stage.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    models_dir = root / config["paths"]["models"]
    processed_dir = root / config["paths"]["processed_data"]
    results_dir = root / config["paths"]["results"]

    n_urls = config["latency"]["n_stream_urls"]
    warmup = config["latency"]["warmup_urls"]
    repetitions = config["latency"]["repetitions"]
    feature_names = get_feature_names(config)

    # Load a model
    model = None
    for clf_name in ["RandomForest", "DecisionTree", "KNN", "LogisticRegression", "XGBoost"]:
        model_path = models_dir / f"{clf_name}_domain.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info("Using %s for latency benchmark.", clf_name)
            break

    if model is None:
        logger.error("No trained model found for latency benchmark.")
        return {}

    # Load URLs
    combined_path = processed_dir / "combined.csv"
    if not combined_path.exists():
        logger.error("Combined data not found. Run data pipeline first.")
        return {}

    df = pd.read_csv(combined_path)
    urls = df["url"].tolist()

    # Limit to configured size
    total_needed = warmup + n_urls
    if len(urls) > total_needed:
        rng = np.random.RandomState(config["random_seed"])
        indices = rng.choice(len(urls), size=total_needed, replace=False)
        urls = [urls[i] for i in indices]

    # Initialize services
    dns_resolver = DNSResolver(config)
    whois_enabled = config["enrichment"].get("whois_enabled", True)
    whois_lookup = WhoisLookup(config) if whois_enabled else None
    if not whois_enabled:
        logger.info("WHOIS disabled — enrichment_whois will be 0ms in benchmark.")
    asn_lookup = ASNLookup(config)
    cert_lookup = CertLookup(config)

    # Create a fresh graph for benchmarking
    rdf_graph = RDFGraph()
    rdf_graph.bind("phish", PHISH)
    rdf_graph.bind("inst", INST)

    # Warmup
    logger.info("Warming up with %d URLs…", min(warmup, len(urls)))
    for url in urls[:warmup]:
        try:
            _benchmark_single_url(
                url, model, feature_names, rdf_graph,
                dns_resolver, whois_lookup, asn_lookup, cert_lookup,
            )
        except Exception as e:
            logger.debug("Warmup error for %s: %s", url, e)

    # Benchmark
    all_timings = []
    benchmark_urls = urls[warmup:warmup + n_urls]
    logger.info("Benchmarking %d URLs × %d repetitions…", len(benchmark_urls), repetitions)

    for rep in range(repetitions):
        for url in benchmark_urls:
            try:
                timings = _benchmark_single_url(
                    url, model, feature_names, rdf_graph,
                    dns_resolver, whois_lookup, asn_lookup, cert_lookup,
                )
                timings["url"] = url
                timings["repetition"] = rep
                all_timings.append(timings)
            except Exception as e:
                logger.debug("Benchmark error for %s: %s", url, e)

    if not all_timings:
        logger.error("No successful benchmark measurements.")
        return {}

    # Aggregate results
    stages = [
        "feature_extraction", "enrichment_dns", "enrichment_whois",
        "enrichment_asn", "enrichment_cert", "enrichment_total",
        "inference", "graph_insertion", "correlation_query", "end_to_end",
    ]

    result = {
        "n_urls_benchmarked": len(benchmark_urls),
        "n_measurements": len(all_timings),
        "n_repetitions": repetitions,
        "stages": {},
    }

    for stage in stages:
        values = [t[stage] for t in all_timings if stage in t]
        if values:
            values_ms = [v * 1000 for v in values]  # Convert to ms
            result["stages"][stage] = {
                "p50_ms": float(np.percentile(values_ms, 50)),
                "p95_ms": float(np.percentile(values_ms, 95)),
                "p99_ms": float(np.percentile(values_ms, 99)),
                "max_ms": float(max(values_ms)),
                "mean_ms": float(np.mean(values_ms)),
                "std_ms": float(np.std(values_ms)),
                "min_ms": float(min(values_ms)),
            }

    # Throughput
    total_e2e = sum(t.get("end_to_end", 0) for t in all_timings)
    if total_e2e > 0:
        result["throughput_urls_per_second"] = float(len(all_timings) / total_e2e)
    else:
        result["throughput_urls_per_second"] = 0.0

    result["note"] = (
        "Latency measurements are real wall-clock times. "
        "Enrichment latency includes actual DNS/WHOIS/ASN/cert lookups "
        "(cached after first resolution). Network conditions affect enrichment timing."
    )

    save_result_json(result, results_dir / "latency_benchmark.json")
    logger.info(
        "Benchmark complete: e2e p50=%.1fms, p95=%.1fms, throughput=%.1f URLs/s",
        result["stages"].get("end_to_end", {}).get("p50_ms", 0),
        result["stages"].get("end_to_end", {}).get("p95_ms", 0),
        result["throughput_urls_per_second"],
    )

    return result


if __name__ == "__main__":
    run_latency_benchmark()
