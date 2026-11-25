from __future__ import annotations
"""
Certificate transparency lookup via crt.sh API.

Queries the public crt.sh API to find certificates associated with domains.
Includes rate limiting and caching. Failed lookups are logged, never fabricated.
"""
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import load_config, setup_logging, get_project_root

logger = setup_logging("cert_lookup")


class CertLookup:
    """Certificate transparency lookup via crt.sh with caching."""

    CRT_SH_URL = "https://crt.sh/"

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        self.timeout = config["enrichment"]["cert_timeout"]
        self.rate_limit = config["enrichment"].get("whois_rate_limit_seconds", 2.0)

        cache_dir = get_project_root() / config["enrichment"]["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / "cert_cache.json"
        self.cache = self._load_cache()
        self.last_request_time = 0.0
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
            json.dump(self.cache, f, indent=2, default=str)

    def _rate_limit_wait(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def lookup(self, domain: str) -> list[dict] | None:
        """
        Look up certificates for a domain via crt.sh.

        Returns a list of certificate dicts, or None on failure.
        Each dict contains: issuer_name, common_name, serial_number, not_before, not_after.
        """
        self.stats["total"] += 1

        if domain in self.cache:
            self.stats["cached"] += 1
            return self.cache[domain]

        self._rate_limit_wait()

        try:
            resp = requests.get(
                self.CRT_SH_URL,
                params={"q": domain, "output": "json"},
                timeout=self.timeout,
                headers={"User-Agent": "phishing-v2-research"},
            )

            if resp.status_code != 200:
                logger.debug("crt.sh returned %d for %s", resp.status_code, domain)
                self.cache[domain] = None
                self.stats["failed"] += 1
                return None

            data = resp.json()
            if not data:
                self.cache[domain] = None
                self.stats["failed"] += 1
                return None

            # Extract relevant fields, deduplicate by serial number
            seen_serials = set()
            certs = []
            for entry in data[:20]:  # Limit to 20 most recent
                serial = entry.get("serial_number", "")
                if serial in seen_serials:
                    continue
                seen_serials.add(serial)

                cert = {
                    "issuer_name": entry.get("issuer_name", ""),
                    "common_name": entry.get("common_name", ""),
                    "serial_number": serial,
                    "not_before": entry.get("not_before", ""),
                    "not_after": entry.get("not_after", ""),
                    "name_value": entry.get("name_value", ""),
                }
                certs.append(cert)

            self.cache[domain] = certs
            self.stats["resolved"] += 1
            return certs

        except requests.RequestException as e:
            logger.debug("crt.sh request failed for %s: %s", domain, e)
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug("crt.sh response parse error for %s: %s", domain, e)
        except Exception as e:
            logger.debug("Unexpected error in cert lookup for %s: %s", domain, e)

        self.cache[domain] = None
        self.stats["failed"] += 1
        return None

    def lookup_url(self, url: str) -> list[dict] | None:
        """Extract domain from URL and look up certificates."""
        try:
            import tldextract
            ext = tldextract.extract(url)
            domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
            if not domain or domain == ".":
                return None
            return self.lookup(domain)
        except Exception as e:
            logger.debug("Failed to extract domain from %s for cert lookup: %s", url, e)
            return None


def lookup_urls(urls: list[str], config: dict | None = None) -> dict:
    """
    Look up certificates for a list of URLs.

    Returns dict mapping URL -> cert list (or None for failures).
    """
    cert = CertLookup(config)
    results = {}

    for url in urls:
        results[url] = cert.lookup_url(url)

    cert.save_cache()
    logger.info(
        "Certs: %d total, %d cached, %d resolved, %d failed",
        cert.stats["total"], cert.stats["cached"],
        cert.stats["resolved"], cert.stats["failed"],
    )
    return results


if __name__ == "__main__":
    results = lookup_urls(["https://google.com", "https://example.com"])
    for url, r in results.items():
        n = len(r) if r else 0
        print(f"{url}: {n} certificates found")
