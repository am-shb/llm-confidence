#!/usr/bin/env python3
"""Frozen constants and shared helpers for the Jev calibration study.

This module is the executable mirror of PROTOCOL.md. Every value the
pre-registration fixed lives here exactly once -- if a seed or epsilon appears
in two places they can drift, and the pre-registration stops being checkable.

Nothing here is Jev-specific; Jev is arm J of eight.
"""

import csv
import hashlib
import os
import statistics

import numpy as np

LABELS = ["cs.CV", "cs.LG", "cs.AI", "cs.RO", "cs.CL", "cs.CR", "cs.IR"]
K = len(LABELS)

# PROTOCOL.md section 5: floor at epsilon and renormalize, every arm.
EPSILON = 0.001

# PROTOCOL.md section 4: all derived from master 20260919.
SEEDS = {
    "master": 20260919,
    "evaluation": 20260920,
    "dev": 20260921,
    "shuffle": 20260922,
    "folds": 20260923,
    "bootstrap": 20260924,
}

# PROTOCOL.md section 2. Compared against `published`, the v1 submission date.
START_DATE = "2026-09-05"

CSV_PATH = "arxiv_last_30_days.csv"
CSV_SHA256 = "b625441d7802d7ec53d7093a2e9e729695f8faef619ec2a82afbd46232c4d9ad"

# Exact totals from PROTOCOL.md section 2. These gate the run: a mismatch means
# the filter disagrees with whatever produced the protocol's numbers.
EXPECTED = {"in_set": 9983, "post_start": 4280, "anchor": 5503}

POOL_SIZES = {"evaluation": 2000, "dev": 200}

# PROTOCOL.md sections 3 and 4, plus the pins recorded in DEVIATIONS.md.
# mechanism: "decisions" | "verbalized" | "letter"
ARMS = {
    "J": {
        "model": "typesafe/jev-1.13",
        "provider": "TypeSafe",
        "mechanism": "decisions",
        "params": {},  # empty supported_parameters: no seed, temperature or schema
    },
    "F1-V": {
        "model": "anthropic/claude-sonnet-5",
        "provider": "Anthropic",
        "mechanism": "verbalized",
        # Reasoning omitted so it runs as it ships. Neither seed nor temperature
        # is supported on this model.
        "params": {"structured_outputs": True},
    },
    "F2-V": {
        "model": "openai/gpt-5.6-terra",
        "provider": "OpenAI",
        "mechanism": "verbalized",
        "params": {"structured_outputs": True, "seed": SEEDS["master"]},
    },
    "Q-V": {
        "model": "qwen/qwen3.8-27b",
        "provider": "Parasail",
        "mechanism": "verbalized",
        # Section 4: "The Qwen arms stay non-thinking ... Q-V is matched to
        # Q-L." Non-thinking is NOT this endpoint's default -- reasoning must
        # be disabled explicitly or the model emits chain-of-thought tokens.
        "params": {"structured_outputs": True, "seed": SEEDS["master"],
                   "temperature": 0.0, "reasoning": {"enabled": False}},
    },
    "Q-L": {
        "model": "qwen/qwen3.8-27b",
        "provider": "Parasail",
        "mechanism": "letter",
        # No temperature, top-p or top-k (section 4). Logprobs on the first
        # generated token.
        # Section 4: "Q-L reads the first generated token's logprobs, which
        # reasoning tokens would displace." Non-thinking is NOT this
        # endpoint's default, so it must be sent explicitly; {"enabled":
        # False} is the only one of the tried settings that actually
        # suppresses reasoning tokens on the live pinned endpoint.
        "params": {"seed": SEEDS["master"], "logprobs": True, "top_logprobs": 20,
                   "reasoning": {"enabled": False}},
    },
}

LLM_ARMS = ["J", "F1-V", "F2-V", "Q-V", "Q-L"]

# Arm J's question key in the decisions payload and in the response's `answers`
# map. One name, used by both run.py and parse.py, so they cannot disagree.
JEV_QUESTION_KEY = "primary_category"


def floor_renorm(vec, eps=EPSILON):
    """PROTOCOL.md section 5 step 3: floor at epsilon, then renormalize.

    Applied to every arm including J, Q-L and A, and re-applied after
    temperature scaling so log-space operations are defined everywhere under
    one rule. An all-zero vector becomes uniform.
    """
    v = np.asarray(vec, dtype=float)
    if not np.isfinite(v).all() or (v < 0).any():
        raise ValueError("floor_renorm needs a finite, non-negative vector")
    total = v.sum()
    v = np.full(K, 1.0 / K) if total <= 0 else v / total
    v = np.maximum(v, eps)
    return v / v.sum()


def provider_block(arm):
    """The OpenRouter `provider` field for one arm.

    Section 4 requires pinning the provider with allow_fallbacks disabled on
    every arm: a silent reroute changes structured-output support and, on the
    Qwen arms, the quantization whose distribution Q-L reads. Centralised here
    so the rule is written once.
    """
    return {"order": [ARMS[arm]["provider"]], "allow_fallbacks": False}


csv.field_size_limit(10 ** 9)  # abstracts are long; the default limit trips


def csv_sha256(path=CSV_PATH):
    """Stream the file so a 50MB CSV does not land in memory twice."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path=CSV_PATH, verify=True):
    """Load the harvest CSV, refusing to proceed on an unexpected file.

    PROTOCOL.md section 2 records the SHA-256 so the counts stay checkable.
    Running against a different file silently invalidates everything
    downstream, so this fails loudly instead. `verify=False` is for fixtures.
    """
    if verify:
        got = csv_sha256(path)
        if got != CSV_SHA256:
            raise ValueError(
                f"SHA-256 mismatch for {path}: got {got}, "
                f"protocol records {CSV_SHA256}")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


ENV_PATH = ".env"


def read_env(path=ENV_PATH):
    """Minimal .env reader. No dependency needed for five keys."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def api_key():
    """The OpenRouter key, from the environment or .env.

    Never accepted as a command-line argument: run.py records request
    parameters per response and section 4 publishes those logs, so a key on
    the command line could reach shell history and the published JSONL.
    """
    key = os.environ.get("OPENROUTER_API_KEY") or read_env().get(
        "OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit(
            "STOP: no OPENROUTER_API_KEY. Copy .env.example to .env and fill "
            "it in.")
    return key


def redact(obj):
    """Recursively blank anything that could carry the credential."""
    key = os.environ.get("OPENROUTER_API_KEY") or read_env().get(
        "OPENROUTER_API_KEY", "")
    if isinstance(obj, dict):
        return {k: ("<redacted>" if k.lower() in
                    ("authorization", "api_key", "openrouter_api_key",
                     "user_id")
                    else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, str) and key and key in obj:
        return obj.replace(key, "<redacted>")
    return obj


def in_set(rows):
    """Papers whose author-chosen primary category is one of the 7 labels."""
    allowed = set(LABELS)
    return [r for r in rows if r["primary_category"] in allowed]


def integrity_report(rows):
    """PROTOCOL.md section 2 integrity line, recomputed rather than trusted."""
    ids = [r["id"] for r in rows]
    words = sorted(len(r["abstract"].split()) for r in rows)
    return {
        "n": len(rows),
        "duplicate_ids": len(ids) - len(set(ids)),
        "short_abstracts": sum(1 for r in rows if len(r["abstract"]) < 100),
        "median_words": statistics.median(words),
        "p95_words": words[int(0.95 * len(words))],
    }
