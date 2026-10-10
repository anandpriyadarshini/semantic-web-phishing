from __future__ import annotations
"""
Build RDF knowledge graph from enriched data.

Populates an RDF graph using the phishing ontology with real data from
DNS, ASN, and certificate lookups. Only includes data that was
actually resolved — never fabricates triples.

Enrichment is parallelised across 20 threads so DNS timeouts on dead
phishing domains don't serialise the entire build.
"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import tldextract
from rdflib import Graph, Literal, Namespace, RDF, XSD, URIRef

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import load_config, setup_logging, get_project_root, save_result_json
from src.enrichment.dns_lookup import DNSResolver
from src.enrichment.whois_lookup import WhoisLookup
from src.enrichment.asn_lookup import ASNLookup
from src.enrichment.cert_lookup import CertLookup

logger = setup_logging("build_graph")

PHISH = Namespace("http://example.org/phishing-ontology#")
INST = Namespace("http://example.org/phishing-instance#")


def _safe_uri(base: str, value: str) -> URIRef:
    """Create a safe URI from a value, encoding special characters."""
    safe = quote(str(value), safe="")
    return URIRef(f"{base}{safe}")


def build_graph(config: dict | None = None, phishing_only: bool = True) -> Graph:
    """
    Build the RDF knowledge graph from enriched data.

    Enrichment is parallelised with a thread pool so dead phishing domains
    (DNS timeout) don't block the entire build.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_data"]
    graph_dir = root / config["paths"]["graph_output"]
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Load the ontology
    g = Graph()
    ontology_path = root / config["paths"]["ontology"]
    g.parse(ontology_path, format="turtle")
    g.bind("phish", PHISH)
    g.bind("inst", INST)
    logger.info("Loaded ontology with %d triples.", len(g))

    # Load processed data
    combined_path = processed_dir / "combined.csv"
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined data not found at {combined_path}")

    df = pd.read_csv(combined_path)
    if phishing_only:
        df = df[df["label"] == 1]
    logger.info("Building graph for %d URLs.", len(df))

    # Initialize enrichment services
    dns_resolver = DNSResolver(config)
    whois_enabled = config["enrichment"].get("whois_enabled", True)
    whois_lookup = WhoisLookup(config) if whois_enabled else None
    if not whois_enabled:
        logger.info("WHOIS disabled in config — graph uses DNS/ASN/cert signals only.")
    asn_lookup = ASNLookup(config)
    cert_lookup = CertLookup(config)

    stats = {
        "urls_processed": 0,
        "urls_with_dns": 0,
        "urls_with_whois": 0,
        "urls_with_asn": 0,
        "urls_with_certs": 0,
        "urls_skipped": 0,
        "total_triples": 0,
    }

    graph_lock = threading.Lock()
    stats_lock = threading.Lock()

    def _enrich_url(url: str, label: int):
        """Enrich one URL and return RDF triples + local stats (thread-safe)."""
        triples = []
        local = {k: 0 for k in stats}
        local["urls_processed"] = 1

        url_uri = _safe_uri(str(INST), f"url/{url}")
        triples += [
            (url_uri, RDF.type, PHISH.URL),
            (url_uri, PHISH.urlString, Literal(url)),
            (url_uri, PHISH.label, Literal(label, datatype=XSD.integer)),
        ]

        ext = tldextract.extract(url)
        domain_name = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
        if not domain_name or domain_name == ".":
            local["urls_skipped"] = 1
            return triples, local

        domain_uri = _safe_uri(str(INST), f"domain/{domain_name}")
        triples += [
            (domain_uri, RDF.type, PHISH.Domain),
            (domain_uri, PHISH.domainName, Literal(domain_name)),
            (url_uri, PHISH.hasDomain, domain_uri),
        ]

        # DNS + ASN
        dns_result = dns_resolver.resolve_url(url)
        if dns_result and dns_result.get("ips"):
            local["urls_with_dns"] = 1
            for ip in dns_result["ips"]:
                ip_uri = _safe_uri(str(INST), f"ip/{ip}")
                triples += [
                    (ip_uri, RDF.type, PHISH.IPAddress),
                    (ip_uri, PHISH.ipValue, Literal(ip)),
                    (domain_uri, PHISH.resolvesTo, ip_uri),
                ]
                asn_result = asn_lookup.lookup_ip(ip)
                if asn_result:
                    local["urls_with_asn"] = 1
                    asn_uri = _safe_uri(str(INST), f"asn/{asn_result['asn']}")
                    triples.append((asn_uri, RDF.type, PHISH.ASN))
                    triples.append((asn_uri, PHISH.asnNumber, Literal(asn_result["asn"])))
                    if asn_result.get("asn_name"):
                        triples.append((asn_uri, PHISH.asnName, Literal(asn_result["asn_name"])))
                    triples.append((ip_uri, PHISH.hostedOn, asn_uri))

        # WHOIS (optional)
        if whois_lookup is not None:
            whois_result = whois_lookup.lookup_url(url)
            if whois_result and whois_result.get("registrar"):
                local["urls_with_whois"] = 1
                reg_name = whois_result["registrar"]
                reg_uri = _safe_uri(str(INST), f"registrar/{reg_name}")
                triples += [
                    (reg_uri, RDF.type, PHISH.Registrar),
                    (reg_uri, PHISH.registrarName, Literal(reg_name)),
                    (domain_uri, PHISH.registeredBy, reg_uri),
                ]

        # Certificates
        certs = cert_lookup.lookup_url(url)
        if certs:
            local["urls_with_certs"] = 1
            for cert in certs[:5]:
                serial = cert.get("serial_number", "")
                if not serial:
                    continue
                cert_uri = _safe_uri(str(INST), f"cert/{serial}")
                triples.append((cert_uri, RDF.type, PHISH.Certificate))
                triples.append((cert_uri, PHISH.certSerialNumber, Literal(serial)))
                if cert.get("issuer_name"):
                    triples.append((cert_uri, PHISH.certIssuer, Literal(cert["issuer_name"])))
                triples.append((domain_uri, PHISH.usesCertificate, cert_uri))

        return triples, local

    # Parallel enrichment
    rows = [(str(r["url"]), int(r["label"])) for _, r in df.iterrows()]
    n_workers = min(20, len(rows))
    done = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_map = {executor.submit(_enrich_url, url, lbl): url for url, lbl in rows}
        for future in as_completed(future_map):
            try:
                triples, local = future.result()
                with graph_lock:
                    for triple in triples:
                        g.add(triple)
                with stats_lock:
                    for k, v in local.items():
                        stats[k] += v
                    done += 1
                    if done % 50 == 0:
                        logger.info("Processed %d/%d URLs…", done, len(rows))
            except Exception as e:
                logger.debug("Enrichment error: %s", e)

    logger.info("Processed %d/%d URLs (final).", done, len(rows))

    # Save caches
    dns_resolver.save_cache()
    if whois_lookup is not None:
        whois_lookup.save_cache()
    asn_lookup.save_cache()
    cert_lookup.save_cache()

    stats["total_triples"] = len(g)
    logger.info("Graph built: %d triples from %d URLs.", stats["total_triples"], stats["urls_processed"])
    logger.info("Enrichment: DNS=%d  ASN=%d  WHOIS=%d  Certs=%d  Skipped=%d",
                stats["urls_with_dns"], stats["urls_with_asn"],
                stats["urls_with_whois"], stats["urls_with_certs"], stats["urls_skipped"])

    output_path = graph_dir / "phishing_graph.ttl"
    g.serialize(destination=str(output_path), format="turtle")
    logger.info("Graph saved to %s", output_path)

    save_result_json(stats, graph_dir / "graph_stats.json")
    return g


def load_graph(config: dict | None = None) -> Graph:
    """Load the pre-built graph from disk."""
    if config is None:
        config = load_config()
    root = get_project_root()
    graph_path = root / config["paths"]["graph_output"] / "phishing_graph.ttl"
    g = Graph()
    g.parse(graph_path, format="turtle")
    g.bind("phish", PHISH)
    g.bind("inst", INST)
    return g


if __name__ == "__main__":
    build_graph()
