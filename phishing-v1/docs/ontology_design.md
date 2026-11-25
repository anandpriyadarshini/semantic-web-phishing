# Ontology Design Notes

## Overview

This document explains the design rationale for each OWL class and property
in `src/graph/ontology.ttl`. The ontology models phishing URL infrastructure
for the purpose of **infrastructure correlation** and **candidate campaign
clustering** — not attacker attribution.

---

## Classes

### `phish:URL`
**Represents**: A URL that has been analyzed (phishing or legitimate).

**Rationale**: The URL is the primary unit of analysis. Every URL is a node
in the graph. Linking URLs to their infrastructure allows us to detect
structural similarity across what may appear to be independent phishing
incidents.

**Key properties**: `urlString`, `label` (1=phishing / 0=legitimate),
`predictionScore`, `hasDomain`, `sharesInfrastructureWith`.

---

### `phish:Domain`
**Represents**: The registered domain extracted from a URL (e.g., `evil.com`).

**Rationale**: Multiple distinct URLs can share a domain (e.g., `evil.com/login`,
`evil.com/verify`). The domain is a stable identifier for one level of
infrastructure. Grouping by registered domain (not full hostname) avoids
over-splitting on subdomains while still capturing shared hosting.

**Key properties**: `domainName`, `resolvesTo`, `registeredBy`, `usesCertificate`.

---

### `phish:IPAddress`
**Represents**: An IPv4 or IPv6 address a domain resolves to via DNS.

**Rationale**: IP addresses are a primary shared-infrastructure signal.
Attackers frequently host multiple phishing pages on the same IP (or a block
of IPs within the same ASN). A domain→IP edge is created only for IPs
actually returned by DNS resolution at pipeline run time — never inferred
or fabricated.

**Key properties**: `ipValue`, `hostedOn`.

---

### `phish:ASN`
**Represents**: An Autonomous System Number (e.g., AS14618 = Amazon AWS).

**Rationale**: Different IP addresses sharing the same ASN indicates the same
hosting provider or network operator. Phishing campaigns often concentrate
on cheap/permissive hosting providers within specific ASNs. ASN is a
coarser-grained signal than IP but more stable across CDN-based deployments.

**Key properties**: `asnNumber`, `asnName`.

---

### `phish:Registrar`
**Represents**: The domain name registrar (e.g., Namecheap, GoDaddy).

**Rationale**: Attackers tend to register multiple phishing domains through
the same registrar, often exploiting permissive registrars or those with
low KYC requirements. The registrar is extracted from real WHOIS data —
if WHOIS is unavailable, no registrar node is created.

**Key properties**: `registrarName`.

---

### `phish:HostingProvider`
**Represents**: The organization operating the ASN hosting the IP.

**Rationale**: Conceptually distinct from ASN (an ASN may span multiple
providers via reselling), but in practice derived from the ASN name field.
Modeled as a separate class to allow future disambiguation if richer data
(e.g., CAIDA AS rank) becomes available.

> **Note**: In the current implementation, HostingProvider is inferred from
> ASN name and uses the same node as ASN. A separate class is provided in
> the ontology for future use.

---

### `phish:Certificate`
**Represents**: A TLS/SSL certificate from Certificate Transparency logs.

**Rationale**: Phishing sites sometimes reuse certificates across domains
(especially when using Let's Encrypt wildcard certs or shared CDN certificates).
Certificate issuer and serial number allow linking domains that share a cert.
Data sourced from crt.sh — only certificates actually found are linked.

**Key properties**: `certSerialNumber`, `certIssuer`, `certNotBefore`,
`certNotAfter`.

---

### `phish:Campaign`
**Represents**: A cluster of phishing URLs sharing infrastructure.

**Rationale**: URLs connected by shared IPs, ASNs, registrars, or certificates
are grouped into candidate campaigns using Louvain community detection or
connected components. The term "campaign" denotes **infrastructure correlation**,
NOT confirmed attacker identity. This class exists to aggregate cluster
membership data in the graph.

> **Important**: This class must NOT be used to claim attacker attribution
> without external ground-truth validation. See `src/eval/cluster_eval.py`
> for how this caveat is enforced in evaluation.

**Key properties**: `campaignId`, `campaignSize`.

---

## Properties

### Object Properties

| Property | Domain | Range | Rationale |
|----------|--------|-------|-----------|
| `hasDomain` | URL | Domain | Structural link from URL to its registered domain |
| `resolvesTo` | Domain | IPAddress | Real DNS A/AAAA result at runtime |
| `hostedOn` | IPAddress | ASN | Team Cymru ASN lookup per IP |
| `registeredBy` | Domain | Registrar | WHOIS registrar field |
| `usesCertificate` | Domain | Certificate | crt.sh CT log lookup |
| `sharesInfrastructureWith` | URL | URL | Inferred symmetric property — two URLs share ≥1 infrastructure node |
| `belongsToCampaign` | URL | Campaign | Assigned by cluster_campaigns.py |

### `sharesInfrastructureWith` (Symmetric)

This is declared as `owl:SymmetricProperty` because shared infrastructure is
mutual: if URL A and URL B share an IP, B also shares that IP with A. This
allows SPARQL queries to use a single direction and OWL reasoners to infer
the inverse automatically.

### Datatype Properties

Datatype properties encode literal values (strings, integers, floats) for
each entity. All values come from real pipeline outputs — no defaults or
placeholders are ever inserted into the graph.

---

## Design Decisions

1. **Why not use FOAF/SCHEMA.ORG?** — The domain is too specific (phishing
   infrastructure) for general-purpose vocabularies. A domain-specific
   namespace avoids vocabulary misuse and makes SPARQL queries more readable.

2. **Why registered domain, not full hostname?** — Full hostnames fragment
   related phishing pages (e.g., `a.evil.com` vs `b.evil.com`). The
   registered domain is the unit of registration, which is what attackers
   control.

3. **Why Louvain over label propagation?** — Louvain's resolution parameter
   allows controlling granularity. Connected components are used as a fallback
   because they require no hyperparameters.

4. **Missing data policy** — If any enrichment lookup fails, no triple is
   added. The URL node exists in the graph but with fewer connections.
   This conservative approach prevents fabricated edges from polluting
   clustering results.
