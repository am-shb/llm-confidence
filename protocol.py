#!/usr/bin/env python3
"""Frozen constants and shared helpers for the Jev calibration study.

This module is the executable mirror of PROTOCOL.md. Every value the
pre-registration fixed lives here exactly once -- if a seed or epsilon appears
in two places they can drift, and the pre-registration stops being checkable.

Nothing here is Jev-specific; Jev is arm J of eight.
"""

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
        "params": {"structured_outputs": True, "seed": SEEDS["master"],
                   "temperature": 0.0},
    },
    "Q-L": {
        "model": "qwen/qwen3.8-27b",
        "provider": "Parasail",
        "mechanism": "letter",
        # No temperature, top-p or top-k (section 4). Logprobs on the first
        # generated token.
        "params": {"seed": SEEDS["master"], "logprobs": True, "top_logprobs": 20},
    },
}

LLM_ARMS = ["J", "F1-V", "F2-V", "Q-V", "Q-L"]


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
