from __future__ import annotations
"""
Fetch legitimate URLs from the Tranco top-sites list.

Tranco is an academically-curated list combining Alexa, Umbrella, Majestic,
and Chrome UX rankings.  See https://tranco-list.eu/ for methodology.
"""
import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import load_config, setup_logging, get_project_root

logger = setup_logging("fetch_legitimate")


def fetch_legitimate(config: dict | None = None, force: bool = False) -> Path:
    """
    Download Tranco top-1M list and produce data/raw/legitimate.csv.

    Returns the path to the saved file.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    raw_dir = root / config["paths"]["raw_data"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "legitimate.csv"

    if output_path.exists() and not force:
        logger.info("Legitimate data already present at %s — skipping.", output_path)
        return output_path

    url = config["data"]["tranco_url"]
    logger.info("Downloading Tranco list from %s …", url)

    try:
        resp = requests.get(url, timeout=120, headers={"User-Agent": "phishing-v2-research"})
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to download Tranco list: %s", exc)
        if output_path.exists():
            logger.warning("Using existing local copy at %s", output_path)
            return output_path
        # Fallback: generate legitimate URLs from well-known domains
        logger.warning(
            "Tranco download failed. Generating fallback legitimate URL list from "
            "well-known domains for dev/testing purposes."
        )
        return _generate_fallback_legitimate(output_path, config)

    # Tranco distributes as a zip
    try:
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f, header=None, names=["rank", "domain"])
    except (zipfile.BadZipFile, IndexError):
        # Maybe it's a plain CSV
        df = pd.read_csv(io.StringIO(resp.text), header=None, names=["rank", "domain"])

    logger.info("Downloaded %d domains from Tranco.", len(df))

    # Convert domains to URLs
    df["url"] = "https://" + df["domain"]
    df["label"] = 0  # legitimate

    # Match the sample size of phishing data for balance
    sample_size = config["data"].get("dev_sample_size")
    if sample_size is not None:
        df = df.head(sample_size)
        logger.info("Dev mode: limited to %d rows.", sample_size)

    df[["url", "domain", "label"]].to_csv(output_path, index=False)
    logger.info("Saved legitimate URLs to %s (%d rows).", output_path, len(df))
    return output_path


def _generate_fallback_legitimate(output_path: Path, config: dict) -> Path:
    """Generate a small fallback list of known legitimate domains for dev/testing."""
    # These are well-known, clearly legitimate domains for development purposes.
    # In production, the Tranco list should be used.
    domains = [
        "google.com", "youtube.com", "facebook.com", "amazon.com", "wikipedia.org",
        "twitter.com", "instagram.com", "linkedin.com", "reddit.com", "netflix.com",
        "microsoft.com", "apple.com", "github.com", "stackoverflow.com", "yahoo.com",
        "bing.com", "whatsapp.com", "zoom.us", "adobe.com", "salesforce.com",
        "paypal.com", "ebay.com", "cnn.com", "bbc.com", "nytimes.com",
        "washingtonpost.com", "forbes.com", "bloomberg.com", "reuters.com", "espn.com",
        "walmart.com", "target.com", "bestbuy.com", "costco.com", "homedepot.com",
        "ikea.com", "nike.com", "adidas.com", "spotify.com", "twitch.tv",
        "dropbox.com", "slack.com", "notion.so", "figma.com", "canva.com",
        "cloudflare.com", "digitalocean.com", "aws.amazon.com", "azure.microsoft.com",
        "oracle.com", "ibm.com", "intel.com", "nvidia.com", "amd.com", "cisco.com",
        "hp.com", "dell.com", "lenovo.com", "samsung.com", "sony.com",
        "harvard.edu", "mit.edu", "stanford.edu", "oxford.ac.uk", "cam.ac.uk",
        "nasa.gov", "nih.gov", "cdc.gov", "who.int", "un.org",
        "arxiv.org", "springer.com", "nature.com", "sciencedirect.com",
        "researchgate.net", "medium.com", "quora.com", "tumblr.com", "pinterest.com",
        "tiktok.com", "snapchat.com", "discord.com", "telegram.org", "signal.org",
        "shopify.com", "squarespace.com", "wordpress.com", "wix.com", "godaddy.com",
        "namecheap.com", "cloudflare.com", "akamai.com", "fastly.com",
        "stripe.com", "square.com", "visa.com", "mastercard.com", "chase.com",
        "bankofamerica.com", "wellsfargo.com", "citibank.com", "hsbc.com",
        "barclays.com",
        # Add more to reach sample size
        "zillow.com", "airbnb.com", "booking.com", "tripadvisor.com", "expedia.com",
        "uber.com", "lyft.com", "doordash.com", "grubhub.com", "instacart.com",
        "etsy.com", "wayfair.com", "chewy.com", "zappos.com", "nordstrom.com",
        "macys.com", "gap.com", "hm.com", "zara.com", "uniqlo.com",
        "craigslist.org", "yelp.com", "glassdoor.com", "indeed.com", "monster.com",
        "coursera.org", "udemy.com", "edx.org", "khanacademy.org", "duolingo.com",
        "mozilla.org", "brave.com", "opera.com", "vivaldi.com", "duckduckgo.com",
        "proton.me", "tutanota.com", "zoho.com", "freshworks.com", "hubspot.com",
        "mailchimp.com", "sendgrid.com", "twilio.com", "plaid.com", "braintreepayments.com",
        "intercom.com", "zendesk.com", "atlassian.com", "jira.atlassian.com",
        "bitbucket.org", "gitlab.com", "heroku.com", "vercel.com", "netlify.com",
        "render.com", "fly.io", "railway.app", "supabase.com", "firebase.google.com",
        "mongodb.com", "postgresql.org", "mysql.com", "redis.io", "elastic.co",
        "grafana.com", "datadog.com", "splunk.com", "newrelic.com", "pagerduty.com",
        "okta.com", "auth0.com", "1password.com", "lastpass.com", "bitwarden.com",
        "norton.com", "mcafee.com", "kaspersky.com", "malwarebytes.com", "crowdstrike.com",
        "sophos.com", "fortinet.com", "paloaltonetworks.com", "checkpoint.com",
        "vmware.com", "citrix.com", "nutanix.com", "rubrik.com", "veeam.com",
        "tableau.com", "powerbi.microsoft.com", "looker.com", "domo.com",
        "snowflake.com", "databricks.com", "palantir.com", "alteryx.com",
        "sap.com", "workday.com", "servicenow.com", "docusign.com", "box.com",
        "asana.com", "monday.com", "trello.com", "basecamp.com", "clickup.com",
        "loom.com", "calendly.com", "typeform.com", "surveymonkey.com", "qualtrics.com",
        "anaconda.com", "jupyter.org", "python.org", "rust-lang.org", "golang.org",
        "nodejs.org", "ruby-lang.org", "php.net", "swift.org", "kotlinlang.org",
        "dart.dev", "flutter.dev", "reactjs.org", "angular.io", "vuejs.org",
        "svelte.dev", "nextjs.org", "nuxt.com", "gatsby.com", "remix.run",
        "tailwindcss.com", "getbootstrap.com", "material.io", "ant.design",
        "chakra-ui.com",
        "starbucks.com", "mcdonalds.com", "subway.com", "dominos.com", "pizzahut.com",
        "cocacola.com", "pepsi.com", "nestle.com", "unilever.com", "pg.com",
        "toyota.com", "honda.com", "ford.com", "bmw.com", "mercedes-benz.com",
        "tesla.com", "spacex.com", "boeing.com", "airbus.com", "lockheedmartin.com",
        "ge.com", "siemens.com", "philips.com", "bosch.com", "honeywell.com",
        "3m.com", "dupont.com", "basf.com", "dow.com", "bayer.com",
        "pfizer.com", "johnson.com", "abbott.com", "merck.com", "novartis.com",
        "roche.com", "astrazeneca.com", "gsk.com", "sanofi.com", "lilly.com",
        "disney.com", "hbo.com", "hulu.com", "peacocktv.com", "paramountplus.com",
        "crunchyroll.com", "funimation.com", "deezer.com", "pandora.com",
        "soundcloud.com",
        "deviantart.com", "behance.net", "dribbble.com", "unsplash.com", "pexels.com",
        "shutterstock.com", "gettyimages.com", "istockphoto.com", "flickr.com",
        "500px.com",
        "imdb.com", "rottentomatoes.com", "metacritic.com", "tvtropes.org",
        "fandom.com",
        "weather.com", "accuweather.com", "timeanddate.com", "worldtimebuddy.com",
        "xe.com",
        "archive.org", "gutenberg.org", "openlibrary.org", "jstor.org",
        "pubmed.ncbi.nlm.nih.gov",
    ]

    sample_size = config["data"].get("dev_sample_size") or len(domains)
    domains = domains[:sample_size]

    df = pd.DataFrame({
        "url": ["https://" + d for d in domains],
        "domain": domains,
        "label": 0,
    })
    df.to_csv(output_path, index=False)
    logger.info("Generated fallback legitimate URL list with %d domains.", len(df))
    return output_path


if __name__ == "__main__":
    fetch_legitimate(force="--force" in sys.argv)
