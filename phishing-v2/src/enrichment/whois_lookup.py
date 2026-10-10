from __future__ import annotations
"""
WHOIS lookups with caching and rate limiting.

If a lookup fails or is unavailable, the URL is logged and excluded
from graph-dependent analyses. WHOIS data is NEVER fabricated.
"""
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import load_config, setup_logging, get_project_root

logger = setup_logging("whois_lookup")


class WhoisLookup:
    """Cached WHOIS lookup with rate limiting and graceful failure handling."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        self.timeout = config["enrichment"]["whois_timeout"]
        self.rate_limit = config["enrichment"]["whois_rate_limit_seconds"]
        self.max_retries = config["enrichment"]["max_retries"]

        cache_dir = get_project_root() / config["enrichment"]["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / "whois_cache.json"
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
        """Enforce rate limiting between WHOIS requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def lookup(self, domain: str) -> dict | None:
        """
        Perform WHOIS lookup for a domain.

        Returns dict with registrar, creation_date, expiration_date, name_servers,
        or None if lookup fails. NEVER fabricates data.

        Uses a hard wall-clock timeout (ThreadPoolExecutor) because python-whois
        ignores socket-level timeouts on many WHOIS servers.
        """
        self.stats["total"] += 1

        if domain in self.cache:
            self.stats["cached"] += 1
            return self.cache[domain]

        self._rate_limit_wait()

        def _do_whois(domain: str, timeout: float):
            import socket
            import whois
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(timeout)
            try:
                return whois.whois(domain)
            finally:
                socket.setdefaulttimeout(old_timeout)

        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        for attempt in range(self.max_retries + 1):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_do_whois, domain, self.timeout)
                    try:
                        w = future.result(timeout=self.timeout)
                    except FuturesTimeout:
                        future.cancel()
                        logger.debug(
                            "WHOIS timeout (%ds) for %s — excluding from graph-dependent analyses.",
                            self.timeout, domain,
                        )
                        self.cache[domain] = None
                        self.stats["failed"] += 1
                        return None

                if w is None or (hasattr(w, 'domain_name') and w.domain_name is None):
                    logger.debug("WHOIS returned empty for %s", domain)
                    self.cache[domain] = None
                    self.stats["failed"] += 1
                    return None

                result = {
                    "domain": domain,
                    "registrar": str(w.registrar) if w.registrar else None,
                    "creation_date": str(w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date) if w.creation_date else None,
                    "expiration_date": str(w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date) if w.expiration_date else None,
                    "name_servers": [str(ns).lower() for ns in (w.name_servers or [])] if w.name_servers else [],
                    "org": str(w.org) if hasattr(w, "org") and w.org else None,
                    "country": str(w.country) if hasattr(w, "country") and w.country else None,
                }

                self.cache[domain] = result
                self.stats["resolved"] += 1
                return result

            except Exception as e:
                if attempt == self.max_retries:
                    logger.debug(
                        "WHOIS lookup failed for %s after %d attempts: %s — "
                        "excluding from graph-dependent analyses.",
                        domain, attempt + 1, e,
                    )
                    self.cache[domain] = None
                    self.stats["failed"] += 1
                    return None
                time.sleep(1)  # Brief pause before retry

        return None

    def lookup_url(self, url: str) -> dict | None:
        """Extract domain from URL and perform WHOIS lookup."""
        try:
            import tldextract
            ext = tldextract.extract(url)
            domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
            if not domain or domain == ".":
                return None
            return self.lookup(domain)
        except Exception as e:
            logger.debug("Failed to extract domain from %s: %s", url, e)
            return None


def lookup_urls(urls: list[str], config: dict | None = None) -> dict:
    """
    Perform WHOIS lookups for a list of URLs.

    Returns dict mapping URL -> WHOIS result (or None for failures).
    """
    whois = WhoisLookup(config)
    results = {}

    for url in urls:
        results[url] = whois.lookup_url(url)

    whois.save_cache()
    logger.info(
        "WHOIS: %d total, %d cached, %d resolved, %d failed",
        whois.stats["total"], whois.stats["cached"],
        whois.stats["resolved"], whois.stats["failed"],
    )
    return results


if __name__ == "__main__":
    config = load_config()
    results = lookup_urls(["https://google.com", "https://example.com"], config)
    for url, r in results.items():
        print(f"{url}: {r}")
