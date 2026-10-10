from __future__ import annotations
"""
ASN (Autonomous System Number) lookup via DNS-based queries.

Uses Team Cymru's DNS-based ASN lookup service (free, no API key needed).
Falls back gracefully if unavailable.
"""
import json
import sys
from pathlib import Path

import dns.resolver
import dns.exception
import dns.reversename

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import load_config, setup_logging, get_project_root

logger = setup_logging("asn_lookup")


class ASNLookup:
    """ASN lookup via Team Cymru DNS service with caching."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        self.timeout = config["enrichment"]["dns_timeout"]

        cache_dir = get_project_root() / config["enrichment"]["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / "asn_cache.json"
        self.cache = self._load_cache()
        self.stats = {"total": 0, "cached": 0, "resolved": 0, "failed": 0}

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def save_cache(self) -> None:
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)

    def lookup_ip(self, ip: str) -> dict | None:
        """
        Look up ASN for an IP address using Team Cymru's DNS service.

        Format: reverse the IP octets and query origin.asn.cymru.com
        Returns dict with 'asn', 'asn_name', 'prefix', or None on failure.
        """
        self.stats["total"] += 1

        if ip in self.cache:
            self.stats["cached"] += 1
            return self.cache[ip]

        # Only handle IPv4 for now
        parts = ip.split(".")
        if len(parts) != 4:
            self.cache[ip] = None
            self.stats["failed"] += 1
            return None

        reversed_ip = ".".join(reversed(parts))
        query = f"{reversed_ip}.origin.asn.cymru.com"

        resolver = dns.resolver.Resolver()
        resolver.timeout = self.timeout
        resolver.lifetime = self.timeout

        try:
            answers = resolver.resolve(query, "TXT")
            for rdata in answers:
                txt = str(rdata).strip('"')
                # Format: "ASN | Prefix | CC | Registry | Date"
                fields = [f.strip() for f in txt.split("|")]
                if len(fields) >= 3:
                    asn = fields[0]
                    prefix = fields[1]
                    country = fields[2]

                    # Get ASN name via peer query
                    asn_name = self._get_asn_name(asn)

                    result = {
                        "ip": ip,
                        "asn": asn,
                        "prefix": prefix,
                        "country": country,
                        "asn_name": asn_name,
                    }
                    self.cache[ip] = result
                    self.stats["resolved"] += 1
                    return result

        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer,
                dns.resolver.NoNameservers, dns.exception.Timeout) as e:
            logger.debug("ASN lookup failed for %s: %s", ip, e)
        except Exception as e:
            logger.debug("ASN lookup error for %s: %s", ip, e)

        self.cache[ip] = None
        self.stats["failed"] += 1
        return None

    def _get_asn_name(self, asn: str) -> str | None:
        """Look up the name of an ASN via Team Cymru."""
        query = f"AS{asn}.asn.cymru.com"
        resolver = dns.resolver.Resolver()
        resolver.timeout = self.timeout
        resolver.lifetime = self.timeout

        try:
            answers = resolver.resolve(query, "TXT")
            for rdata in answers:
                txt = str(rdata).strip('"')
                fields = [f.strip() for f in txt.split("|")]
                if len(fields) >= 5:
                    return fields[4]
        except Exception:
            pass
        return None


def lookup_ips(ips: list[str], config: dict | None = None) -> dict:
    """
    Look up ASN for a list of IP addresses.

    Returns dict mapping IP -> ASN result (or None for failures).
    """
    asn = ASNLookup(config)
    results = {}

    for ip in ips:
        results[ip] = asn.lookup_ip(ip)

    asn.save_cache()
    logger.info(
        "ASN: %d total, %d cached, %d resolved, %d failed",
        asn.stats["total"], asn.stats["cached"],
        asn.stats["resolved"], asn.stats["failed"],
    )
    return results


if __name__ == "__main__":
    results = lookup_ips(["8.8.8.8", "1.1.1.1"])
    for ip, r in results.items():
        print(f"{ip}: {r}")
