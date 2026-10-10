from __future__ import annotations
"""
DNS resolution for URL hostnames.

Resolves A/AAAA records with caching and timeout handling.
Failed lookups are logged and excluded, never fabricated.
"""
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import dns.resolver
import dns.exception

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import load_config, setup_logging, get_project_root

logger = setup_logging("dns_lookup")


class DNSResolver:
    """Cached DNS resolver with timeout and rate limiting."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        self.timeout = config["enrichment"]["dns_timeout"]
        self.max_retries = config["enrichment"]["max_retries"]

        cache_dir = get_project_root() / config["enrichment"]["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / "dns_cache.json"
        self.cache = self._load_cache()
        self.stats = {"total": 0, "cached": 0, "resolved": 0, "failed": 0}

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                return json.load(f)
        return {}

    def save_cache(self) -> None:
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)

    def resolve(self, hostname: str) -> dict | None:
        """
        Resolve a hostname to IP addresses.

        Returns dict with 'ips' list and 'hostname', or None if resolution fails.
        Never fabricates data.
        """
        self.stats["total"] += 1

        if hostname in self.cache:
            self.stats["cached"] += 1
            return self.cache[hostname]

        resolver = dns.resolver.Resolver()
        resolver.timeout = self.timeout
        resolver.lifetime = self.timeout

        ips = []

        for rdtype in ["A", "AAAA"]:
            for attempt in range(self.max_retries + 1):
                try:
                    answers = resolver.resolve(hostname, rdtype)
                    ips.extend([str(rdata) for rdata in answers])
                    break
                except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer,
                        dns.resolver.NoNameservers):
                    break  # Domain doesn't exist or has no records of this type
                except dns.exception.Timeout:
                    if attempt == self.max_retries:
                        logger.debug("DNS timeout for %s (%s), attempt %d/%d",
                                     hostname, rdtype, attempt + 1, self.max_retries + 1)
                except Exception as e:
                    logger.debug("DNS error for %s (%s): %s", hostname, rdtype, e)
                    break

        if ips:
            result = {"hostname": hostname, "ips": list(set(ips))}
            self.cache[hostname] = result
            self.stats["resolved"] += 1
            return result
        else:
            self.cache[hostname] = None
            self.stats["failed"] += 1
            logger.debug("DNS resolution failed for %s — excluded from graph.", hostname)
            return None

    def resolve_url(self, url: str) -> dict | None:
        """Extract hostname from URL and resolve it."""
        try:
            parsed = urlparse(url if "://" in url else f"http://{url}")
            hostname = parsed.hostname
            if not hostname:
                return None
            return self.resolve(hostname)
        except Exception as e:
            logger.debug("Failed to parse URL for DNS: %s — %s", url, e)
            return None


def resolve_urls(urls: list[str], config: dict | None = None) -> dict:
    """
    Resolve DNS for a list of URLs.

    Returns dict mapping URL -> DNS result (or None for failures).
    """
    resolver = DNSResolver(config)
    results = {}

    for url in urls:
        results[url] = resolver.resolve_url(url)

    resolver.save_cache()
    logger.info(
        "DNS resolution: %d total, %d cached, %d resolved, %d failed",
        resolver.stats["total"], resolver.stats["cached"],
        resolver.stats["resolved"], resolver.stats["failed"],
    )
    return results


if __name__ == "__main__":
    config = load_config()
    import pandas as pd

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_data"]
    df = pd.read_csv(processed_dir / "combined.csv")
    results = resolve_urls(df["url"].tolist()[:50], config)  # Small sample for testing
    print(f"Resolved {sum(1 for v in results.values() if v is not None)}/{len(results)} URLs")
