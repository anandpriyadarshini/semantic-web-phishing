from __future__ import annotations
"""
Campaign clustering via infrastructure graph analysis.

Uses SPARQL to extract shared-infrastructure edges, builds a NetworkX
graph, and applies Louvain community detection or connected components
to identify candidate phishing campaigns.

Note: These are "candidate campaigns" based on infrastructure correlation,
NOT confirmed attacker attributions.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from rdflib import Graph as RDFGraph, Namespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import load_config, setup_logging, get_project_root, save_result_json

logger = setup_logging("cluster_campaigns")

PHISH = Namespace("http://example.org/phishing-ontology#")


def _build_infrastructure_graph(rdf_graph: RDFGraph) -> nx.Graph:
    """
    Build a NetworkX graph where nodes are phishing URLs and edges
    connect URLs that share infrastructure.

    Uses SPARQL queries to find shared IPs, ASNs, registrars, and certificates.
    """
    G = nx.Graph()

    # Query: find all phishing URLs sharing IP
    shared_ip_query = """
    PREFIX phish: <http://example.org/phishing-ontology#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?url1 ?url2
    WHERE {
        ?u1 a phish:URL ;
            phish:urlString ?url1 ;
            phish:label "1"^^xsd:integer ;
            phish:hasDomain ?d1 .
        ?u2 a phish:URL ;
            phish:urlString ?url2 ;
            phish:label "1"^^xsd:integer ;
            phish:hasDomain ?d2 .
        ?d1 phish:resolvesTo ?ipNode .
        ?d2 phish:resolvesTo ?ipNode .
        FILTER(STR(?u1) < STR(?u2))
    }
    """

    shared_asn_query = """
    PREFIX phish: <http://example.org/phishing-ontology#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?url1 ?url2
    WHERE {
        ?u1 a phish:URL ;
            phish:urlString ?url1 ;
            phish:label "1"^^xsd:integer ;
            phish:hasDomain ?d1 .
        ?u2 a phish:URL ;
            phish:urlString ?url2 ;
            phish:label "1"^^xsd:integer ;
            phish:hasDomain ?d2 .
        ?d1 phish:resolvesTo ?ip1 .
        ?d2 phish:resolvesTo ?ip2 .
        ?ip1 phish:hostedOn ?asnNode .
        ?ip2 phish:hostedOn ?asnNode .
        FILTER(STR(?u1) < STR(?u2))
        FILTER(?d1 != ?d2)
    }
    """

    shared_registrar_query = """
    PREFIX phish: <http://example.org/phishing-ontology#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?url1 ?url2
    WHERE {
        ?u1 a phish:URL ;
            phish:urlString ?url1 ;
            phish:label "1"^^xsd:integer ;
            phish:hasDomain ?d1 .
        ?u2 a phish:URL ;
            phish:urlString ?url2 ;
            phish:label "1"^^xsd:integer ;
            phish:hasDomain ?d2 .
        ?d1 phish:registeredBy ?regNode .
        ?d2 phish:registeredBy ?regNode .
        FILTER(STR(?u1) < STR(?u2))
        FILTER(?d1 != ?d2)
    }
    """

    shared_cert_query = """
    PREFIX phish: <http://example.org/phishing-ontology#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?url1 ?url2
    WHERE {
        ?u1 a phish:URL ;
            phish:urlString ?url1 ;
            phish:label "1"^^xsd:integer ;
            phish:hasDomain ?d1 .
        ?u2 a phish:URL ;
            phish:urlString ?url2 ;
            phish:label "1"^^xsd:integer ;
            phish:hasDomain ?d2 .
        ?d1 phish:usesCertificate ?certNode .
        ?d2 phish:usesCertificate ?certNode .
        FILTER(STR(?u1) < STR(?u2))
        FILTER(?d1 != ?d2)
    }
    """

    edge_types = {
        "shared_ip": shared_ip_query,
        "shared_asn": shared_asn_query,
        "shared_registrar": shared_registrar_query,
        "shared_certificate": shared_cert_query,
    }

    for edge_type, query in edge_types.items():
        try:
            results = rdf_graph.query(query)
            count = 0
            for row in results:
                url1 = str(row.url1)
                url2 = str(row.url2)
                G.add_node(url1)
                G.add_node(url2)
                if G.has_edge(url1, url2):
                    G[url1][url2]["weight"] += 1
                    G[url1][url2]["types"].add(edge_type)
                else:
                    G.add_edge(url1, url2, weight=1, types={edge_type})
                count += 1
            logger.info("Found %d %s edges.", count, edge_type)
        except Exception as e:
            logger.warning("SPARQL query for %s failed: %s", edge_type, e)

    # Also add isolated phishing URL nodes (no shared infra)
    all_phishing_query = """
    PREFIX phish: <http://example.org/phishing-ontology#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?url WHERE {
        ?u a phish:URL ;
            phish:urlString ?url ;
            phish:label "1"^^xsd:integer .
    }
    """
    try:
        for row in rdf_graph.query(all_phishing_query):
            url = str(row.url)
            if url not in G:
                G.add_node(url)
    except Exception as e:
        logger.warning("Failed to query all phishing URLs: %s", e)

    return G


def cluster_campaigns(config: dict | None = None) -> dict:
    """
    Run campaign clustering on the infrastructure graph.

    Returns clustering results including campaign assignments and
    descriptive statistics.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    graph_dir = root / config["paths"]["graph_output"]
    results_dir = root / config["paths"]["results"]
    figures_dir = root / config["paths"]["figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load RDF graph
    graph_path = graph_dir / "phishing_graph.ttl"
    if not graph_path.exists():
        logger.error("RDF graph not found at %s. Run build_graph.py first.", graph_path)
        return {}

    from src.graph.build_graph import load_graph
    rdf_graph = load_graph(config)

    # Build infrastructure graph
    G = _build_infrastructure_graph(rdf_graph)
    logger.info("Infrastructure graph: %d nodes, %d edges.", G.number_of_nodes(), G.number_of_edges())

    if G.number_of_nodes() == 0:
        logger.warning("Empty infrastructure graph. No URLs with shared infrastructure found.")
        result = {
            "n_nodes": 0,
            "n_edges": 0,
            "n_campaigns": 0,
            "note": "No shared infrastructure found. Graph-based clustering requires "
                    "URLs with overlapping IP, ASN, registrar, or certificate data."
        }
        save_result_json(result, results_dir / "campaign_results.json")
        return result

    min_campaign_size = config["graph"]["min_campaign_size"]
    resolution = config["graph"]["campaign_resolution"]

    # Try Louvain clustering
    campaigns = {}
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, resolution=resolution, random_state=config["random_seed"])
        for node, comm_id in partition.items():
            campaigns.setdefault(comm_id, []).append(node)
        method = "louvain"
    except (ImportError, Exception) as e:
        logger.warning("Louvain clustering failed (%s). Falling back to connected components.", e)
        for i, component in enumerate(nx.connected_components(G)):
            campaigns[i] = list(component)
        method = "connected_components"

    # Filter by minimum size
    significant_campaigns = {
        k: v for k, v in campaigns.items()
        if len(v) >= min_campaign_size
    }

    # Compute descriptive statistics
    all_sizes = [len(v) for v in campaigns.values()]
    significant_sizes = [len(v) for v in significant_campaigns.values()]

    result = {
        "method": method,
        "resolution": resolution if method == "louvain" else None,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_campaigns_total": len(campaigns),
        "n_campaigns_significant": len(significant_campaigns),
        "min_campaign_size_threshold": min_campaign_size,
        "campaign_size_stats": {
            "all": {
                "min": int(min(all_sizes)) if all_sizes else 0,
                "max": int(max(all_sizes)) if all_sizes else 0,
                "mean": float(np.mean(all_sizes)) if all_sizes else 0,
                "median": float(np.median(all_sizes)) if all_sizes else 0,
                "std": float(np.std(all_sizes)) if all_sizes else 0,
            },
            "significant": {
                "min": int(min(significant_sizes)) if significant_sizes else 0,
                "max": int(max(significant_sizes)) if significant_sizes else 0,
                "mean": float(np.mean(significant_sizes)) if significant_sizes else 0,
                "median": float(np.median(significant_sizes)) if significant_sizes else 0,
                "std": float(np.std(significant_sizes)) if significant_sizes else 0,
            },
        },
        "graph_density": float(nx.density(G)),
        "n_connected_components": nx.number_connected_components(G),
        "campaigns": {
            str(k): {
                "size": len(v),
                "urls": v[:10],  # First 10 for display
            }
            for k, v in significant_campaigns.items()
        },
    }

    # Plot campaign size distribution
    if all_sizes:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(all_sizes, bins=min(50, len(set(all_sizes))), edgecolor="black", alpha=0.7)
        ax.set_xlabel("Campaign Size (URLs)")
        ax.set_ylabel("Count")
        ax.set_title("Campaign Size Distribution")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(figures_dir / "campaign_size_distribution.png", dpi=150)
        plt.close(fig)

    save_result_json(result, results_dir / "campaign_results.json")
    logger.info(
        "Clustering complete: %d total campaigns, %d significant (size ≥ %d).",
        len(campaigns), len(significant_campaigns), min_campaign_size,
    )

    return result


if __name__ == "__main__":
    cluster_campaigns()
