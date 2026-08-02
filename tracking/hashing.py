"""Generic file hashing, shared by dataset fingerprinting and code-version
provenance tags (e.g. "which exact fusion/rule_based.py produced this run").
"""
import hashlib


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
