"""Shared filesystem layout. Runtime data can live outside the app tree via VOCALMATCH_DATA."""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("VOCALMATCH_DATA", BASE)
EXPERTS = os.environ.get("VOCALMATCH_EXPERTS", os.path.join(BASE, "experts"))
UPLOADS = os.path.join(DATA, "uploads")
RESULTS = os.path.join(DATA, "results")
CACHE = os.path.join(DATA, "cache")
DB_PATH = os.path.join(DATA, "vocalmatch.db")


def ensure_dirs():
    for d in (EXPERTS, UPLOADS, RESULTS, CACHE):
        os.makedirs(d, exist_ok=True)
