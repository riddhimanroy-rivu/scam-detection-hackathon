"""
Turns a raw URL string into interpretable numeric features so that
LIME/Anchor explanations read as "has an IP address", "no HTTPS",
"too many subdomains" etc. instead of opaque character n-grams.
"""

import re
import math
import numpy as np
import pandas as pd

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "banking",
    "confirm", "signin", "password", "billing", "suspend", "urgent",
    "free", "bonus", "gift", "click", "limited",
]

IP_PATTERN = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}")
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "rb.gy", "cutt.ly",
}

FEATURE_NAMES = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "num_at_symbols",
    "num_digits",
    "num_subdomains",
    "has_ip_address",
    "has_https",
    "has_port",
    "num_suspicious_words",
    "has_shortener",
    "path_length",
    "num_query_params",
    "domain_entropy",
]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def extract_features(url: str) -> dict:
    url = str(url).strip()
    lower = url.lower()

    # crude URL parsing without relying on a fully-formed scheme
    no_scheme = re.sub(r"^https?://", "", lower)
    domain = no_scheme.split("/")[0].split("?")[0]
    path = no_scheme[len(domain):]
    query = url.split("?", 1)[1] if "?" in url else ""

    return {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_at_symbols": url.count("@"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_subdomains": max(domain.count(".") - 1, 0),
        "has_ip_address": 1 if IP_PATTERN.match(domain) else 0,
        "has_https": 1 if lower.startswith("https://") else 0,
        "has_port": 1 if re.search(r":\d{2,5}(/|$)", domain) else 0,
        "num_suspicious_words": sum(w in lower for w in SUSPICIOUS_WORDS),
        "has_shortener": 1 if any(s in domain for s in SHORTENERS) else 0,
        "path_length": len(path),
        "num_query_params": query.count("&") + (1 if query else 0),
        "domain_entropy": round(_entropy(domain), 3),
    }


def urls_to_feature_df(urls) -> pd.DataFrame:
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)