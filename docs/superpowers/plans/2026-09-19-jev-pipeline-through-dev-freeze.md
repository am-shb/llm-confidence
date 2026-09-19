# Jev Calibration Pipeline (Phases 0-6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pipeline that executes `PROTOCOL.md` up to the dev-set freeze, ending with a committed prompt/parser freeze and a priced go/no-go on the evaluation run.

**Architecture:** Small single-purpose scripts over one shared `protocol.py` that holds every frozen constant exactly once. The run phase (hours of network I/O, resumable, append-only JSONL) is separated from the analysis phase (reruns in seconds) by a file boundary. Nothing in the evaluation pool is read until after the phase 6 gate.

**Tech Stack:** Python 3.14, numpy, scipy, scikit-learn, matplotlib, pytest. HTTP on stdlib `urllib` + `ThreadPoolExecutor`, matching `harvest_arxiv.py`.

**Spec:** `PROTOCOL.md` (pre-registered, frozen) and `docs/superpowers/specs/2026-09-19-jev-calibration-pipeline-design.md`

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from `PROTOCOL.md`; if a value appears in code twice, that is a bug.

- **Labels (K=7), canonical order:** `cs.CV`, `cs.LG`, `cs.AI`, `cs.RO`, `cs.CL`, `cs.CR`, `cs.IR`
- **Seeds:** master `20260919`; evaluation sample `20260920`; dev sample `20260921`; option shuffle `20260922`; folds `20260923`; bootstrap `20260924`. API-level `seed` is `20260919` wherever the provider accepts one.
- **Epsilon:** `0.001`, floored and renormalized on every arm, re-applied after temperature scaling.
- **Start date:** `2026-09-05`, compared against the `published` field (the v1 submission date).
- **Pool sizes:** evaluation `N=2000`; dev `200`; anchor the remainder.
- **Exact gate totals:** in-set `9983`, on/after start `4280`, anchor `5503`. A mismatch is a stop condition, not a warning.
- **CSV SHA-256:** `b625441d7802d7ec53d7093a2e9e729695f8faef619ec2a82afbd46232c4d9ad` (verified reproducing as of 2026-09-19).
- **Provider pins**, all with `allow_fallbacks: false`: J -> `TypeSafe`; F1-V -> `Anthropic`; F2-V -> `OpenAI`; Q-V and Q-L -> `Parasail` (both arms, one identical backend).
- **Credential:** `OPENROUTER_API_KEY`, read from gitignored `.env`. Never passed on a command line and never serialized into run logs.
- **Deviations:** anything diverging from `PROTOCOL.md` goes in `DEVIATIONS.md` with a date and a reason. Never edit `PROTOCOL.md`.
- **Scope boundary:** this plan stops at the phase 6 gate. `anchor.py` and `analyze.py` are a separate plan, written after the gate and before the evaluation run.

---

### Task 1: Project scaffold and the two known deviations

**Files:**
- Create: `requirements.txt`, `pytest.ini`
- Modify: `DEVIATIONS.md`

**Interfaces:**
- Consumes: nothing.
- Produces: a working `.venv` and `pytest` invocation used by every later task.

- [ ] **Step 1: Create the venv and requirements**

```bash
cd /Users/amir/code/jev-test
python3 -m venv .venv
cat > requirements.txt <<'EOF'
numpy==2.3.3
scipy==1.16.2
scikit-learn==1.7.2
matplotlib==3.10.6
pytest==8.4.2
EOF
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q
```

If any pinned version fails to resolve on Python 3.14, install unpinned (`pip install numpy scipy scikit-learn matplotlib pytest`), then freeze the versions actually installed back into `requirements.txt` with `.venv/bin/pip freeze | grep -iE 'numpy|scipy|scikit-learn|matplotlib|pytest'`. Record the substitution in the commit message.

- [ ] **Step 2: Configure pytest**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 3: Verify the toolchain imports**

Run: `.venv/bin/python -c "import numpy, scipy, sklearn, matplotlib; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Record the two deviations found during endpoint verification**

Append these two rows to the table in `DEVIATIONS.md`:

```markdown
| 2026-09-19 | 4 | Qwen arms pinned to `Parasail` (fp8) instead of the declared first-party `Alibaba` endpoint. | `Alibaba` advertises `logprobs` but not `structured_outputs`, so it cannot serve Q-V under section 4's structured-outputs rule; it could serve only Q-L. No endpoint of the 16 offers unquantized + logprobs + structured outputs together (`DeepInfra` is the sole bf16 endpoint and has no logprobs). `Parasail` supports both mechanisms at 100% uptime, keeping Q-V and Q-L on one identical backend, which is the property C3 depends on. Consequence: section 9's quantization limitation sharpens — C3 is an fp8 estimate and the post must say so. |
| 2026-09-19 | 4 | F1-V pinned to first-party `Anthropic`; F2-V pinned to first-party `OpenAI`. | Section 4 requires structured outputs on every verbalized arm, but only 2 of 10 `claude-sonnet-5` endpoints support them (`Anthropic`, `Claude Platform on AWS`) and the Bedrock `gpt-5.6-terra` endpoint does not. Without an explicit pin, routing could silently drop constrained decoding and move parse failures into the primary metric. Section 3's claim that no frontier provider exposes answer-token logprobs was re-verified at the same time and holds: 0 of 10 Sonnet and 0 of 7 Terra endpoints list `logprobs`. |
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pytest.ini DEVIATIONS.md
git commit -m "Add toolchain and record the two endpoint deviations"
```

---

### Task 2: `protocol.py` constants and `floor_renorm`

**Files:**
- Create: `protocol.py`, `tests/test_protocol.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `LABELS: list[str]` (7, canonical order), `K: int`, `EPSILON: float`, `SEEDS: dict[str, int]`, `START_DATE: str`, `CSV_SHA256: str`, `EXPECTED: dict[str, int]`, `ARMS: dict[str, dict]`, `floor_renorm(vec: np.ndarray, eps: float = EPSILON) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_protocol.py
import numpy as np
import pytest
import protocol as P


def test_labels_are_the_seven_in_canonical_order():
    assert P.LABELS == ["cs.CV", "cs.LG", "cs.AI", "cs.RO", "cs.CL", "cs.CR", "cs.IR"]
    assert P.K == 7


def test_seeds_match_the_protocol():
    assert P.SEEDS == {
        "master": 20260919,
        "evaluation": 20260920,
        "dev": 20260921,
        "shuffle": 20260922,
        "folds": 20260923,
        "bootstrap": 20260924,
    }


def test_epsilon_and_start_date():
    assert P.EPSILON == 0.001
    assert P.START_DATE == "2026-09-05"


@pytest.mark.parametrize("eps", [0.0001, 0.001, 0.01])
def test_floor_renorm_sums_to_one_and_respects_the_exact_floor(eps):
    """Flooring adds at most K*eps before renormalizing, so the smallest
    entry lands at exactly eps/(1 + K*eps) -- just below eps, not above it.
    Section 7 recomputes the primary metric at all three of these epsilons.
    """
    tight_floor = eps / (1 + (P.K - 1) * eps)
    for raw in ([1.0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.5, 0, 0, 0, 0, 0],
                [1e-12, 1, 0, 0, 0, 0, 0]):
        out = P.floor_renorm(np.array(raw), eps=eps)
        assert out.sum() == pytest.approx(1.0)
        assert out.min() >= tight_floor * (1 - 1e-9)
        assert (out > 0).all(), "log-space operations must be defined everywhere"


def test_floor_renorm_handles_all_zero_vector():
    out = P.floor_renorm(np.zeros(7))
    assert out.sum() == pytest.approx(1.0)
    assert out == pytest.approx(np.full(7, 1 / 7))


def test_floor_renorm_converges_and_preserves_its_invariants():
    """Section 5 re-applies the floor after temperature scaling, so what
    matters is that re-application is well-defined and stable -- not that it
    is a no-op. Repeated application converges to a fixed point at min == eps.
    """
    once = P.floor_renorm(np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]))
    twice = P.floor_renorm(once)

    assert twice.sum() == pytest.approx(1.0)
    assert twice.min() >= P.EPSILON / (1 + (P.K - 1) * P.EPSILON) * (1 - 1e-9)
    assert np.abs(twice - once).max() < 1e-4, "re-application must be stable"

    x = np.array([1.0, 0, 0, 0, 0, 0, 0])
    for _ in range(8):
        x = P.floor_renorm(x)
    assert x.min() == pytest.approx(P.EPSILON, rel=1e-6)
    assert np.abs(P.floor_renorm(x) - x).max() < 1e-12, "fixed point reached"


def test_floor_renorm_preserves_argmax():
    out = P.floor_renorm(np.array([0.05, 0.60, 0.35, 0.0, 0.0, 0.0, 0.0]))
    assert int(out.argmax()) == 1


def test_arms_registry_pins_every_llm_arm():
    for arm in ["J", "F1-V", "F2-V", "Q-V", "Q-L"]:
        assert P.ARMS[arm]["provider"], f"{arm} must pin a provider"
    assert P.ARMS["Q-V"]["provider"] == P.ARMS["Q-L"]["provider"] == "Parasail"
    assert P.ARMS["Q-V"]["model"] == P.ARMS["Q-L"]["model"]
    assert P.ARMS["J"]["model"] == "typesafe/jev-1.13"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'protocol'`

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_protocol.py -v`
Expected: PASS, 11 tests (the epsilon test is parameterized three ways).

Note the floor bound is `eps / (1 + (K-1) * eps)`, not `eps`: a unit-sum vector always has an entry `>= 1/K > eps`, so at most `K-1` entries can be floored, and after renormalization those land just below `eps` (0.00099404 at the default, attained by a one-hot input). That is correct behaviour and not a bug to chase. `floor_renorm` is a contraction, not an involution — repeated application converges to a fixed point where the minimum equals `eps` exactly, so do not assert exact idempotency.

- [ ] **Step 5: Commit**

```bash
git add protocol.py tests/test_protocol.py
git commit -m "Add protocol.py frozen constants and the section 5 floor rule"
```

---

### Task 3: CSV loader with the SHA-256 guard and integrity checks

**Files:**
- Modify: `protocol.py`
- Modify: `tests/test_protocol.py`

**Interfaces:**
- Consumes: `LABELS`, `CSV_PATH`, `CSV_SHA256`, `EXPECTED`, `START_DATE` from Task 2.
- Produces: `csv_sha256(path: str) -> str`, `load_rows(path: str = CSV_PATH, verify: bool = True) -> list[dict]`, `in_set(rows: list[dict]) -> list[dict]`, `integrity_report(rows: list[dict]) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_protocol.py
import csv as _csv


def test_csv_sha256_matches_the_recorded_hash():
    assert P.csv_sha256(P.CSV_PATH) == P.CSV_SHA256


def test_load_rows_rejects_a_file_whose_hash_differs(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,primary_category,published,abstract,title\n1,cs.AI,2026-09-06T00:00:00Z,x,y\n")
    with pytest.raises(ValueError, match="SHA-256"):
        P.load_rows(str(bad))


def test_load_rows_can_skip_verification_for_fixtures(tmp_path):
    f = tmp_path / "f.csv"
    f.write_text("id,primary_category,published,abstract,title\n1,cs.AI,2026-09-06T00:00:00Z,x,y\n")
    assert len(P.load_rows(str(f), verify=False)) == 1


def test_in_set_keeps_only_the_seven_primary_categories():
    rows = [{"primary_category": c} for c in ["cs.AI", "cs.SE", "cs.CV", "stat.ML"]]
    assert [r["primary_category"] for r in P.in_set(rows)] == ["cs.AI", "cs.CV"]


def test_real_csv_reproduces_the_protocol_totals():
    rows = P.in_set(P.load_rows())
    assert len(rows) == P.EXPECTED["in_set"]
    post = [r for r in rows if r["published"][:10] >= P.START_DATE]
    assert len(post) == P.EXPECTED["post_start"]
    pre = [r for r in rows if r["published"][:10] < P.START_DATE]
    assert len(pre) - P.POOL_SIZES["dev"] == P.EXPECTED["anchor"]


def test_integrity_report_finds_no_duplicates_or_short_abstracts():
    rep = P.integrity_report(P.in_set(P.load_rows()))
    assert rep["duplicate_ids"] == 0
    assert rep["short_abstracts"] == 0
    assert 185 <= rep["median_words"] <= 195
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_protocol.py -v -k "csv or in_set or integrity or totals"`
Expected: FAIL — `AttributeError: module 'protocol' has no attribute 'csv_sha256'`

- [ ] **Step 3: Write the implementation**

Append to `protocol.py`:

```python
import csv
import hashlib
import statistics

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_protocol.py -v`
Expected: PASS, 14 tests. The real-CSV tests take a few seconds — hashing 50MB plus a full parse.

The median-words assertion is a range (185-195), not equality. Recomputation gives median 190 and p95 259 against the protocol's stated 191 and 261, a one-to-two-word difference from a different tokenization of the same abstracts. These are descriptive statistics in the protocol's integrity line, not gates, so they do not block. Do not "fix" the loader to chase them.

- [ ] **Step 5: Commit**

```bash
git add protocol.py tests/test_protocol.py
git commit -m "Add CSV loader with SHA-256 guard and integrity recomputation"
```

---

### Task 4: `pools.py` — derive the three pools and freeze them

**Files:**
- Create: `pools.py`, `tests/test_pools.py`
- Create (output, committed): `pools/evaluation.jsonl`, `pools/dev.jsonl`, `pools/anchor.jsonl`, `pools/counts.md`

**Interfaces:**
- Consumes: everything from Tasks 2-3.
- Produces: `derive_pools(rows: list[dict]) -> dict[str, list[dict]]` returning pool name -> list of `{"id", "label", "options"}` where `options` is a list of 7 label strings in that item's shuffled order; `permutation_for(item_id: str) -> list[str]`; `load_pool(name: str) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pools.py
import numpy as np
import pytest
import protocol as P
import pools


@pytest.fixture(scope="module")
def derived():
    return pools.derive_pools(P.in_set(P.load_rows()))


def test_pool_sizes_hit_the_protocol_exactly(derived):
    assert len(derived["evaluation"]) == 2000
    assert len(derived["dev"]) == 200
    assert len(derived["anchor"]) == P.EXPECTED["anchor"]


def test_pools_are_disjoint(derived):
    e = {r["id"] for r in derived["evaluation"]}
    d = {r["id"] for r in derived["dev"]}
    a = {r["id"] for r in derived["anchor"]}
    assert not (e & d) and not (e & a) and not (d & a)


def test_evaluation_is_all_post_start_and_others_all_pre_start(derived):
    assert all(r["published"][:10] >= P.START_DATE for r in derived["evaluation"])
    for name in ("dev", "anchor"):
        assert all(r["published"][:10] < P.START_DATE for r in derived[name])


def test_derivation_is_reproducible(derived):
    again = pools.derive_pools(P.in_set(P.load_rows()))
    for name in ("evaluation", "dev", "anchor"):
        assert [r["id"] for r in again[name]] == [r["id"] for r in derived[name]]


def test_every_item_carries_a_full_permutation_of_the_seven_labels(derived):
    for r in derived["evaluation"][:50]:
        assert sorted(r["options"]) == sorted(P.LABELS)
        assert len(r["options"]) == 7


def test_permutation_is_stable_per_item_and_varies_across_items(derived):
    first = derived["evaluation"][0]
    assert pools.permutation_for(first["id"]) == first["options"]
    orders = {tuple(r["options"]) for r in derived["evaluation"][:200]}
    assert len(orders) > 100, "shuffle should not collapse to a few orders"


def test_shuffle_round_trip_is_identity(derived):
    opts = derived["evaluation"][0]["options"]
    vec = np.arange(7, dtype=float)
    shuffled = pools.to_shuffled(vec, opts)
    assert pools.to_canonical(shuffled, opts) == pytest.approx(vec)


def test_gold_label_is_the_primary_category(derived):
    assert all(r["label"] in P.LABELS for r in derived["evaluation"])


def test_realized_class_counts_are_near_the_protocol_expectations(derived):
    from collections import Counter
    c = Counter(r["label"] for r in derived["evaluation"])
    expected = {"cs.CV": 442, "cs.LG": 429, "cs.AI": 343,
                "cs.RO": 292, "cs.CL": 291, "cs.CR": 160, "cs.IR": 44}
    for label, exp in expected.items():
        assert abs(c[label] - exp) <= 40, f"{label}: {c[label]} vs ~{exp}"
```

The last test uses a tolerance because section 2's per-class counts are marked "(rounded)" and are expectations derived from the 4,280-pool proportions, not realized draws. Confirmed with the user on 2026-09-19. Only the pool totals are exact gates.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pools'`

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
"""Derive and freeze the three pools from PROTOCOL.md section 2.

Run once; the manifests it writes are committed and are what every later stage
reads. Re-running must reproduce them byte for byte.

Usage:
    ./pools.py            # derive, validate against section 2, write manifests
    ./pools.py --check    # validate only, write nothing
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

import protocol as P

POOL_DIR = "pools"


def _rng(seed):
    return np.random.default_rng(seed)


def permutation_for(item_id):
    """This item's option order: stable per item, independent of pool membership.

    Section 4 requires one permutation per item applied identically to every
    arm including J. Deriving it from a hash of the id rather than from a
    sequential draw means an item's order does not depend on how many items
    were drawn before it, so the order survives any change in pool ordering.
    """
    digest = hashlib.sha256(
        f"{P.SEEDS['shuffle']}:{item_id}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    order = _rng(seed).permutation(P.K)
    return [P.LABELS[i] for i in order]


def to_shuffled(vec, options):
    """Canonical-order vector -> this item's shuffled option order."""
    idx = [P.LABELS.index(o) for o in options]
    return np.asarray(vec, dtype=float)[idx]


def to_canonical(vec, options):
    """Shuffled-order vector -> canonical LABELS order."""
    out = np.empty(P.K, dtype=float)
    for pos, opt in enumerate(options):
        out[P.LABELS.index(opt)] = vec[pos]
    return out


def _item(row):
    return {
        "id": row["id"],
        "label": row["primary_category"],
        "published": row["published"],
        "title": row["title"],
        "abstract": row["abstract"],
        "options": permutation_for(row["id"]),
    }


def derive_pools(rows):
    """Split in-set rows into evaluation, dev and anchor pools.

    Order matters and is fixed by section 2: the evaluation sample is drawn
    from post-start papers, then the dev sample from pre-start papers, then
    the anchor pool is every pre-start paper the dev draw did not take.
    """
    post = sorted((r for r in rows if r["published"][:10] >= P.START_DATE),
                  key=lambda r: r["id"])
    pre = sorted((r for r in rows if r["published"][:10] < P.START_DATE),
                 key=lambda r: r["id"])

    if len(rows) != P.EXPECTED["in_set"]:
        raise SystemExit(
            f"STOP: in-set count is {len(rows)}, protocol section 2 says "
            f"{P.EXPECTED['in_set']}. The filter disagrees with the "
            f"pre-registration; investigate before running anything.")
    if len(post) != P.EXPECTED["post_start"]:
        raise SystemExit(
            f"STOP: post-start count is {len(post)}, protocol says "
            f"{P.EXPECTED['post_start']}.")

    eval_idx = _rng(P.SEEDS["evaluation"]).choice(
        len(post), size=P.POOL_SIZES["evaluation"], replace=False)
    evaluation = [post[i] for i in sorted(eval_idx)]

    dev_idx = _rng(P.SEEDS["dev"]).choice(
        len(pre), size=P.POOL_SIZES["dev"], replace=False)
    dev_ids = {pre[i]["id"] for i in dev_idx}
    dev = [pre[i] for i in sorted(dev_idx)]
    anchor = [r for r in pre if r["id"] not in dev_ids]

    if len(anchor) != P.EXPECTED["anchor"]:
        raise SystemExit(
            f"STOP: anchor count is {len(anchor)}, protocol says "
            f"{P.EXPECTED['anchor']}.")

    return {name: [_item(r) for r in pool] for name, pool in
            (("evaluation", evaluation), ("dev", dev), ("anchor", anchor))}


def load_pool(name):
    path = os.path.join(POOL_DIR, f"{name}.jsonl")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write(pools_by_name):
    os.makedirs(POOL_DIR, exist_ok=True)
    for name, items in pools_by_name.items():
        with open(os.path.join(POOL_DIR, f"{name}.jsonl"), "w",
                  encoding="utf-8") as fh:
            for it in items:
                fh.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")


def _counts_md(pools_by_name, report):
    from collections import Counter
    lines = ["# Derived pool counts", "",
             "Generated by `pools.py`. Every number here is recomputed from "
             "`arxiv_last_30_days.csv`, whose SHA-256 is checked on load.", "",
             "| Pool | Count | Protocol section 2 |", "| --- | --- | --- |"]
    for name, expected in (("evaluation", 2000), ("dev", 200),
                           ("anchor", P.EXPECTED["anchor"])):
        lines.append(f"| {name} | {len(pools_by_name[name])} | {expected} |")
    lines += ["", "## Realized evaluation class counts", "",
              "Section 2's per-class numbers are expectations from the "
              "post-start pool proportions and are marked \"(rounded)\"; "
              "realized draws differ by a few items.", "",
              "| Label | Realized | Expected |", "| --- | --- | --- |"]
    c = Counter(r["label"] for r in pools_by_name["evaluation"])
    exp = {"cs.CV": 442, "cs.LG": 429, "cs.AI": 343, "cs.RO": 292,
           "cs.CL": 291, "cs.CR": 160, "cs.IR": 44}
    for label in P.LABELS:
        lines.append(f"| {label} | {c[label]} | {exp[label]} |")
    lines += ["", "## Integrity", "",
              f"- rows: {report['n']}",
              f"- duplicate ids: {report['duplicate_ids']}",
              f"- abstracts under 100 chars: {report['short_abstracts']}",
              f"- abstract words, median: {report['median_words']}, "
              f"p95: {report['p95_words']}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="validate only; write nothing")
    args = ap.parse_args()

    rows = P.in_set(P.load_rows())
    report = P.integrity_report(rows)
    pools_by_name = derive_pools(rows)

    for name, items in pools_by_name.items():
        print(f"  {name:11} {len(items):5}", file=sys.stderr)

    if args.check:
        print("check passed; nothing written", file=sys.stderr)
        return
    _write(pools_by_name)
    with open(os.path.join(POOL_DIR, "counts.md"), "w", encoding="utf-8") as fh:
        fh.write(_counts_md(pools_by_name, report))
    print(f"wrote manifests to {POOL_DIR}/", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pools.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Generate the manifests and confirm re-running is byte-identical**

```bash
chmod +x pools.py
.venv/bin/python pools.py
sha256sum pools/*.jsonl > /tmp/pools_first.txt
.venv/bin/python pools.py
sha256sum -c /tmp/pools_first.txt
```

Expected: three `OK` lines. If any differ, the derivation is not deterministic — stop and fix before committing.

- [ ] **Step 6: Commit the code and the frozen manifests together**

```bash
git add pools.py tests/test_pools.py pools/
git commit -m "Derive and freeze the three pools, validated against section 2"
```

---

### Task 5: Fetch the seven category descriptions verbatim

**Files:**
- Create: `taxonomy.py`, `tests/test_taxonomy.py`
- Create (output, committed): `category_descriptions.json`

**Interfaces:**
- Consumes: `LABELS` from Task 2.
- Produces: `category_descriptions.json` as `{label: {"name": str, "description": str}}`, and `load_descriptions() -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_taxonomy.py
import protocol as P
import taxonomy


def test_all_seven_labels_have_a_name_and_description():
    d = taxonomy.load_descriptions()
    assert set(d) == set(P.LABELS)
    for label, entry in d.items():
        assert entry["name"].strip()
        assert entry["description"].strip()


def test_descriptions_are_verbatim_single_line_no_markup():
    d = taxonomy.load_descriptions()
    for label, entry in d.items():
        assert "<" not in entry["description"]
        assert "\n" not in entry["description"]
        assert "  " not in entry["description"]


def test_known_values_are_exact():
    d = taxonomy.load_descriptions()
    assert d["cs.LG"]["name"] == "Machine Learning"
    assert d["cs.RO"]["name"] == "Robotics"
    # cs.RO's official description really is this short; do not pad it.
    assert d["cs.RO"]["description"] == (
        "Roughly includes material in ACM Subject Class I.2.9.")
    assert d["cs.CV"]["description"].startswith(
        "Covers image processing, computer vision, pattern recognition")


def test_no_base_rate_information_leaked_in():
    """Section 2: no arm gets base-rate information."""
    d = taxonomy.load_descriptions()
    blob = " ".join(e["description"] for e in d.values()).lower()
    for word in ["base rate", "frequency", "prior", "%", "most common"]:
        assert word not in blob
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_taxonomy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'taxonomy'`

- [ ] **Step 3: Write the implementation**

The parse pattern below was verified against the live page on 2026-09-19. The `<h4>` and its `<p>` sit in sibling `div.column` blocks with markup between them, so the pattern must span that gap — a naive `<h4>...</h4>\s*<p>` matches nothing.

```python
#!/usr/bin/env python3
"""Fetch arXiv's official category descriptions for the 7 study labels.

PROTOCOL.md section 3 requires every arm to get these descriptions verbatim,
so they are fetched once, committed as data, and never edited by hand.

Usage:
    ./taxonomy.py          # fetch and write category_descriptions.json
    ./taxonomy.py --show   # print what is stored
"""

import argparse
import html
import json
import os
import re
import sys
import urllib.request

import protocol as P

TAXONOMY_URL = "https://arxiv.org/category_taxonomy"
OUT_PATH = "category_descriptions.json"
USER_AGENT = "jev-calibration-study/1.0 (one-time taxonomy fetch)"

# <h4>cs.AI <span>(Artificial Intelligence)</span></h4> ... <p>description</p>
# with sibling divs in between, hence the non-greedy gap.
ENTRY_RE = re.compile(
    r'<h4>([a-zA-Z\-]+\.[a-zA-Z\-]+)\s*<span>\((.*?)\)</span></h4>.*?<p>(.*?)</p>',
    re.S)


def _clean(fragment):
    """Strip tags, unescape entities, collapse whitespace. No rewording."""
    text = re.sub(r"<.*?>", "", fragment)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def fetch_descriptions(url=TAXONOMY_URL):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        page = resp.read().decode("utf-8")

    found = {cat: {"name": _clean(name), "description": _clean(desc)}
             for cat, name, desc in ENTRY_RE.findall(page)}
    missing = [l for l in P.LABELS if l not in found]
    if missing:
        raise SystemExit(
            f"STOP: taxonomy parse missed {missing}. The page markup changed; "
            f"fix ENTRY_RE rather than hand-writing descriptions -- section 3 "
            f"requires them verbatim.")
    return {l: found[l] for l in P.LABELS}


def load_descriptions(path=OUT_PATH):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.show:
        for label, e in load_descriptions().items():
            print(f"\n{label} ({e['name']})\n  {e['description']}")
        return

    data = fetch_descriptions()
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    print(f"wrote {len(data)} descriptions to {OUT_PATH} "
          f"(source: {TAXONOMY_URL}, fetched once)", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Fetch, then run the tests**

```bash
chmod +x taxonomy.py
.venv/bin/python taxonomy.py
.venv/bin/pytest tests/test_taxonomy.py -v
```

Expected: `wrote 7 descriptions`, then PASS, 4 tests.

- [ ] **Step 5: Commit the fetched data with the code**

```bash
git add taxonomy.py tests/test_taxonomy.py category_descriptions.json
git commit -m "Fetch the seven arXiv category descriptions verbatim"
```

---

### Task 6: `prompt.py` — one shared body, three output shapes

**Files:**
- Create: `prompt.py`, `tests/test_prompt.py`

**Interfaces:**
- Consumes: `LABELS`, `ARMS` (Task 2); `load_descriptions` (Task 5); pool items with `options` (Task 4).
- Produces: `build_prompt(item: dict, mechanism: str) -> list[dict]` returning OpenAI-style messages; `json_schema(options: list[str]) -> dict`; `LETTERS: str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prompt.py
import json
import pytest
import protocol as P
import prompt


ITEM = {
    "id": "2609.00001",
    "label": "cs.LG",
    "title": "A Study of Things",
    "abstract": "We study things. " * 20,
    "options": ["cs.RO", "cs.IR", "cs.LG", "cs.CV", "cs.CR", "cs.AI", "cs.CL"],
}


def test_every_mechanism_includes_title_abstract_and_all_seven_options():
    for mech in ("verbalized", "letter", "decisions"):
        text = json.dumps(prompt.build_prompt(ITEM, mech))
        assert ITEM["title"] in text
        assert "We study things." in text
        for label in P.LABELS:
            assert label in text


def test_options_appear_in_the_items_shuffled_order():
    text = json.dumps(prompt.build_prompt(ITEM, "verbalized"))
    positions = [text.index(o) for o in ITEM["options"]]
    assert positions == sorted(positions), "options must follow the shuffle"


def test_descriptions_are_included_verbatim():
    import taxonomy
    desc = taxonomy.load_descriptions()
    text = json.dumps(prompt.build_prompt(ITEM, "verbalized"))
    assert desc["cs.RO"]["description"] in text
    assert desc["cs.RO"]["name"] in text


def test_no_base_rate_or_deadline_hints_anywhere():
    for mech in ("verbalized", "letter", "decisions"):
        text = json.dumps(prompt.build_prompt(ITEM, mech)).lower()
        for banned in ["base rate", "most common", "deadline", "prior probability",
                       "frequency", "usually"]:
            assert banned not in text


def test_all_arms_get_the_xml_tag_instruction():
    for mech in ("verbalized", "letter", "decisions"):
        text = json.dumps(prompt.build_prompt(ITEM, mech)).lower()
        assert "xml" in text


def test_no_anti_reasoning_instruction_anywhere():
    """Section 4: nothing tells a model not to think."""
    for mech in ("verbalized", "letter", "decisions"):
        text = json.dumps(prompt.build_prompt(ITEM, mech)).lower()
        for banned in ["do not think", "don't think", "no reasoning",
                       "without reasoning", "do not reason", "skip reasoning"]:
            assert banned not in text


def test_letter_mechanism_asks_for_one_letter_and_maps_a_to_g():
    text = json.dumps(prompt.build_prompt(ITEM, "letter"))
    assert "A" in text and "G" in text
    assert prompt.LETTERS == "ABCDEFG"
    assert len(prompt.LETTERS) == P.K


def test_json_schema_requires_all_seven_keys_in_shuffled_order():
    schema = prompt.json_schema(ITEM["options"])
    props = schema["schema"]["properties"]
    assert list(props) == ITEM["options"]
    assert set(schema["schema"]["required"]) == set(ITEM["options"])
    assert schema["schema"]["additionalProperties"] is False
    assert schema["strict"] is True


def test_prompt_body_is_identical_across_mechanisms_except_the_output_rule():
    bodies = {}
    for mech in ("verbalized", "letter", "decisions"):
        msgs = prompt.build_prompt(ITEM, mech)
        bodies[mech] = next(m["content"] for m in msgs if m["role"] == "user")
    shared = prompt.SHARED_BODY_MARKER
    assert all(shared in b for b in bodies.values())


def test_unknown_mechanism_raises():
    with pytest.raises(ValueError):
        prompt.build_prompt(ITEM, "telepathy")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_prompt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prompt'`

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
"""Prompt construction for every arm.

Section 4 requires the same prompt content for every LLM arm: same
instructions, same category descriptions. Only the final output instruction
differs, because the mechanisms differ. Keeping one shared body in one place is
what makes that claim true rather than aspirational.

Frozen by commit before the evaluation run.
"""

import taxonomy
import protocol as P

LETTERS = "ABCDEFG"

SHARED_BODY_MARKER = "Which arXiv primary category did the authors submit"

_TASK = (
    "You are given the title and abstract of an arXiv paper.\n\n"
    f"{SHARED_BODY_MARKER} this paper under?\n\n"
    "The primary category is the single category the submitting authors chose "
    "first. Papers are often cross-listed, but exactly one category is the "
    "primary one.\n"
)

# Section 4: goes to every arm for symmetry. Not an anti-reasoning instruction.
_TAGS = ("Do not include internal or system XML tags in your response.")


def _options_block(options):
    desc = taxonomy.load_descriptions()
    lines = []
    for i, label in enumerate(options):
        e = desc[label]
        lines.append(f"{LETTERS[i]}. {label} ({e['name']}): {e['description']}")
    return "\n".join(lines)


def _body(item):
    return (
        f"{_TASK}\n"
        f"Candidate categories:\n{_options_block(item['options'])}\n\n"
        f"Title: {item['title']}\n\n"
        f"Abstract: {item['abstract']}\n"
    )


# Arm J's `instructions` field. Derived from the SAME _TASK text the chat arms
# receive, plus section 4's universal XML-tag line, so section 4's "same prompt
# content" claim holds across the transport boundary rather than relying on two
# hand-kept copies agreeing. J emits no free text, so the tag line is inert for
# it -- it is present because section 4 says the instruction goes to every arm.
JEV_INSTRUCTIONS = f"{_TASK}\n{_TAGS}"


def json_schema(options):
    """Schema requiring all 7 keys, presented in this item's shuffled order.

    Section 4 puts structured outputs on every verbalized arm so that parse
    failures cannot differ across arms for reasons unrelated to calibration.
    """
    return {
        "name": "category_distribution",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                label: {
                    "type": "number",
                    "description": f"Probability the primary category is {label}",
                } for label in options
            },
            "required": list(options),
            "additionalProperties": False,
        },
    }


def build_prompt(item, mechanism):
    """Messages for one item under one mechanism."""
    if mechanism == "verbalized":
        tail = (
            "Respond with a JSON object giving your probability for each of the "
            "seven categories. Use the exact category keys shown above. The "
            "seven values must sum to 1."
        )
    elif mechanism == "letter":
        tail = (
            f"Respond with exactly one letter, {LETTERS[0]} through "
            f"{LETTERS[-1]}, naming the category you believe is the primary "
            "one. Output only that single letter and nothing else."
        )
    elif mechanism == "decisions":
        # J returns a typed choice; the option list is the choice set. The
        # transport shape is settled by the Task 7 probe, not here.
        tail = (
            "Choose the category you believe is the primary one."
        )
    else:
        raise ValueError(f"unknown mechanism: {mechanism!r}")

    return [{"role": "user", "content": f"{_body(item)}\n{tail}\n\n{_TAGS}"}]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_prompt.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add prompt.py tests/test_prompt.py
git commit -m "Add prompt construction with one shared body per section 4"
```

---

### Task 7: Probe Jev's wire format

**Files:**
- Create: `probe_jev.py`
- Create (output): `probes/jev_probe.json`
- Modify: `DEVIATIONS.md` (only if the probe contradicts the protocol)

**Interfaces:**
- Consumes: `.env` credential loader (written in this task, moved into `protocol.py`), `build_prompt(item, "decisions")` (Task 6), `load_pool("dev")` (Task 4).
- Produces: `protocol.read_env() -> dict[str, str]`, `protocol.api_key() -> str`; a documented record of Jev's request and response shape that Task 8's J adapter is written against.

This is a discovery task. Its deliverable is a recorded answer, not a finished adapter — the adapter is Task 8.

- [ ] **Step 1: Add the credential loader with a test that it never leaks**

```python
# append to tests/test_protocol.py
def test_read_env_parses_the_dotenv(tmp_path):
    f = tmp_path / ".env"
    f.write_text("# comment\nFOO=bar\nEMPTY=\nQUOTED=\"baz\"\n")
    env = P.read_env(str(f))
    assert env["FOO"] == "bar"
    assert env["EMPTY"] == ""
    assert env["QUOTED"] == "baz"


def test_api_key_is_present_and_looks_like_an_openrouter_key():
    key = P.api_key()
    assert key.startswith("sk-or-")
    assert len(key) > 40


def test_redact_removes_the_key_from_anything_serialized():
    key = P.api_key()
    payload = {"headers": {"Authorization": f"Bearer {key}"}, "model": "x"}
    assert key not in str(P.redact(payload))
```

Add to `protocol.py`:

```python
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
                    ("authorization", "api_key", "openrouter_api_key")
                    else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, str) and key and key in obj:
        return obj.replace(key, "<redacted>")
    return obj
```

`protocol.py` needs `import os` at the top for this.

- [ ] **Step 2: Run the credential tests**

Run: `.venv/bin/pytest tests/test_protocol.py -v -k "env or api_key or redact"`
Expected: PASS, 3 tests.

- [ ] **Step 3: Write the probe script**

```python
#!/usr/bin/env python3
"""Discover how typesafe/jev-1.13 accepts a choice set and returns probabilities.

Jev's OpenRouter record lists modality text->decisions, an empty
supported_parameters, and supports_tool_choice.function: true, and the model
description is truncated mid-sentence. So the wire format is unknown and this
script finds it out on dev-pool items -- which section 4 designates for exactly
this kind of debugging and which are disjoint from the evaluation pool.

Writes every attempt and its full response to probes/jev_probe.json.

Usage:
    ./probe_jev.py            # try each candidate shape on 2 dev items
"""

import json
import os
import sys
import urllib.error
import urllib.request

import pools
import prompt
import protocol as P
import taxonomy

URL = "https://openrouter.ai/api/v1/chat/completions"
OUT = "probes/jev_probe.json"


def post(payload):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {P.api_key()}",
        "Content-Type": "application/json",
        "X-Title": "jev-calibration-study",
    })
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def candidates(item):
    """Request shapes to try, cheapest and most likely first."""
    messages = prompt.build_prompt(item, "decisions")
    base = {"model": P.ARMS["J"]["model"],
            "provider": P.provider_block("J")}

    tool = {
        "type": "function",
        "function": {
            "name": "choose_category",
            "description": "Return the primary arXiv category.",
            "parameters": {
                "type": "object",
                "properties": {"category": {"type": "string",
                                            "enum": item["options"]}},
                "required": ["category"],
            },
        },
    }
    return [
        ("plain", {**base, "messages": messages}),
        ("tool_required", {**base, "messages": messages, "tools": [tool],
                           "tool_choice": "required"}),
        ("tool_named", {**base, "messages": messages, "tools": [tool],
                        "tool_choice": {"type": "function",
                                        "function": {"name": "choose_category"}}}),
        ("response_format_enum", {**base, "messages": messages,
                                  "response_format": {
                                      "type": "json_schema",
                                      "json_schema": prompt.json_schema(
                                          item["options"])}}),
    ]


def main():
    items = pools.load_pool("dev")[:2]
    results = []
    for item in items:
        for name, payload in candidates(item):
            status, body = post(payload)
            print(f"  {item['id']} {name:22} -> HTTP {status}", file=sys.stderr)
            results.append({"item": item["id"], "shape": name,
                            "status": status,
                            "request": P.redact(payload),
                            "response": body})
    os.makedirs("probes", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(results)} probe results to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the probe and read the results**

```bash
mkdir -p probes && chmod +x probe_jev.py
.venv/bin/python probe_jev.py
.venv/bin/python -c "
import json
for r in json.load(open('probes/jev_probe.json')):
    print(r['shape'], r['status'])
    print(json.dumps(r['response'])[:1200], '\n')
"
```

Read the output and answer these four questions explicitly before continuing:

1. Which request shape returns HTTP 200?
2. Where do the per-option probabilities appear — a `logprobs` block, a tool-call argument, a dedicated `decisions` field, or something else?
3. Are all 7 options scored, or only the chosen one? If only the chosen one, J cannot produce a K-vector and that is a protocol-level problem to raise with the user immediately, not to paper over.
4. Do the returned probabilities sum to 1 before flooring?

- [ ] **Step 5: Record the answer**

Write the answers as a comment block at the top of `probe_jev.py` under the heading `# FINDINGS (2026-09-19):`, naming the winning shape and the exact response path to the probabilities, e.g. `response["choices"][0]["..."]`. Task 8's J adapter is written directly against this.

If the probe contradicts anything in `PROTOCOL.md`, add a `DEVIATIONS.md` row. If question 3 comes back "only the chosen one", stop and raise it.

- [ ] **Step 6: Commit**

```bash
git add probe_jev.py probes/jev_probe.json protocol.py tests/test_protocol.py
git commit -m "Probe and record Jev's typed-decision wire format"
```

---

### Task 8: `run.py` — resumable arm runner

**Files:**
- Create: `run.py`, `tests/test_run.py`
- Modify: `prompt.py` (add `JEV_INSTRUCTIONS`), `tests/test_prompt.py` (one test for it)

**Interfaces:**
- Consumes: `ARMS`, `api_key`, `redact` (Tasks 2, 7); `build_prompt`, `json_schema` (Task 6); `load_pool` (Task 4); the Task 7 findings.
- Produces: `build_payload(arm: str, item: dict) -> dict`; `completed_ids(path: str) -> set[str]`; `runs/{arm}_{pool}.jsonl` with one JSON object per item carrying keys `id`, `arm`, `request`, `response`, `status`, `provider`, `latency_s`, `timestamp`.

- [ ] **Step 1: Write the failing tests**

These test payload construction and resume logic, which are pure. Network behaviour is exercised against the dev pool in Task 10 — that is what the dev pool is for.

```python
# tests/test_run.py
import json
import protocol as P
import run


ITEM = {
    "id": "2609.00002",
    "label": "cs.AI",
    "title": "T",
    "abstract": "A " * 60,
    "options": ["cs.RO", "cs.IR", "cs.LG", "cs.CV", "cs.CR", "cs.AI", "cs.CL"],
}


CHAT_ARMS = [a for a in P.LLM_ARMS if P.ARMS[a]["mechanism"] != "decisions"]


def test_every_chat_arm_pins_its_provider_and_forbids_fallbacks():
    for arm in CHAT_ARMS:
        pay = run.build_payload(arm, ITEM)
        assert pay["provider"]["allow_fallbacks"] is False
        assert pay["provider"]["order"] == [P.ARMS[arm]["provider"]]


def test_jev_payload_uses_the_decisions_schema_not_chat():
    """Arm J speaks TypeSafe's System One contract: no messages array, no
    response_format, no provider block. chat/completions returns HTTP 400 for
    this model, so a chat-shaped payload would fail every item.
    """
    pay = run.build_jev_payload(ITEM)
    assert "messages" not in pay
    assert "response_format" not in pay
    assert "provider" not in pay
    assert pay["model"] == "typesafe/jev-1.13"
    assert ITEM["title"] in pay["state"]
    assert ITEM["abstract"] in pay["state"]
    q = pay["questions"][P.JEV_QUESTION_KEY]
    assert q["type"] == "choice"
    assert set(q["criteria"]) == set(P.LABELS)


def test_jev_criteria_follow_the_items_shuffled_order():
    """Section 4 requires the same per-item permutation for every arm including
    J. Here the shuffle is carried by criteria insertion order, which Python
    preserves through json.dumps.
    """
    pay = run.build_jev_payload(ITEM)
    assert list(pay["questions"][P.JEV_QUESTION_KEY]["criteria"]) == ITEM["options"]


def test_jev_criteria_carry_the_verbatim_descriptions():
    import taxonomy
    desc = taxonomy.load_descriptions()
    crit = run.build_jev_payload(ITEM)["questions"][P.JEV_QUESTION_KEY]["criteria"]
    for label in P.LABELS:
        assert desc[label]["description"] in crit[label]
        assert desc[label]["name"] in crit[label]


def test_frontier_arms_send_no_reasoning_parameter():
    """Section 4: reasoning omitted so each model runs as it ships."""
    for arm in ("F1-V", "F2-V"):
        assert "reasoning" not in run.build_payload(arm, ITEM)


def test_f1v_sends_neither_seed_nor_temperature():
    pay = run.build_payload("F1-V", ITEM)
    assert "seed" not in pay and "temperature" not in pay


def test_f2v_sends_the_master_seed_but_no_temperature():
    pay = run.build_payload("F2-V", ITEM)
    assert pay["seed"] == 20260919
    assert "temperature" not in pay


def test_qv_is_temperature_zero_and_structured():
    pay = run.build_payload("Q-V", ITEM)
    assert pay["temperature"] == 0.0
    assert pay["response_format"]["type"] == "json_schema"


def test_ql_requests_logprobs_and_sets_no_sampling_params():
    pay = run.build_payload("Q-L", ITEM)
    assert pay["logprobs"] is True
    assert pay["top_logprobs"] >= 7
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in pay, f"section 4 forbids {banned} on Q-L"
    assert "response_format" not in pay


def test_ql_and_qv_share_model_and_provider():
    a, b = run.build_payload("Q-V", ITEM), run.build_payload("Q-L", ITEM)
    assert a["model"] == b["model"]
    assert a["provider"] == b["provider"]


def test_verbalized_schema_follows_the_items_shuffled_order():
    pay = run.build_payload("Q-V", ITEM)
    props = pay["response_format"]["json_schema"]["schema"]["properties"]
    assert list(props) == ITEM["options"]


def test_completed_ids_reads_back_what_was_written(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(
        json.dumps({"id": "a"}) + "\n" + json.dumps({"id": "b"}) + "\n")
    assert run.completed_ids(str(path)) == {"a", "b"}


def test_completed_ids_tolerates_a_truncated_final_line(tmp_path):
    """A crash mid-write must not make the whole run unresumable."""
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({"id": "a"}) + "\n" + '{"id": "b"')
    assert run.completed_ids(str(path)) == {"a"}


def test_completed_ids_on_missing_file_is_empty(tmp_path):
    assert run.completed_ids(str(tmp_path / "nope.jsonl")) == set()


def test_payload_never_contains_the_api_key():
    key = P.api_key()
    for arm in CHAT_ARMS:
        assert key not in json.dumps(run.build_payload(arm, ITEM))
    assert key not in json.dumps(run.build_jev_payload(ITEM))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_run.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'run'`

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
"""Run one arm over one pool, resumably.

Append-only JSONL keyed by item id. On start the existing ids are read and
skipped, so an interrupted run resumes instead of re-billing -- the full study
is roughly 14,000 calls and section 4 puts all arms inside one 48-hour window.

Retry and backoff follow harvest_arxiv.py: rate limits are backed off, not
retried hard. After 3 attempts the item is written with status "failure", which
parse.py scores as uniform per section 5.

Usage:
    ./run.py --arm F1-V --pool dev
    ./run.py --arm J --pool evaluation --run-id 2   # PC2 re-run
    ./run.py --arm Q-L --pool dev --sequential      # latency subsample
"""

import argparse
import datetime as dt
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pools
import prompt
import protocol as P
import taxonomy

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
# Arm J speaks TypeSafe's documented "System One" decisions contract
# (https://docs.typesafe.ai/introduction/quickstart), reached through
# OpenRouter's passthrough so the exact version pin from section 3 is kept.
# chat/completions rejects this model outright with HTTP 400.
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
RUN_DIR = "runs"
MAX_ATTEMPTS = 3
RATE_LIMIT_CODES = (408, 429, 500, 502, 503, 504)

_write_lock = threading.Lock()


def build_payload(arm, item):
    """The exact request body for this arm and item.

    Section 4 asymmetries live here and nowhere else: reasoning is omitted for
    the frontier arms so they run as they ship; Q-L sends no sampling
    parameters because reasoning or sampling tokens would displace the first
    generated token whose logprobs it reads; J exposes no parameters at all.
    """
    spec = P.ARMS[arm]
    mech = spec["mechanism"]
    payload = {
        "model": spec["model"],
        # Section 4's pin rule lives in protocol.provider_block, not here.
        "provider": P.provider_block(arm),
        "messages": prompt.build_prompt(item, mech),
    }
    params = spec["params"]

    if params.get("structured_outputs"):
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": prompt.json_schema(item["options"]),
        }
    if params.get("logprobs"):
        payload["logprobs"] = True
        payload["top_logprobs"] = params.get("top_logprobs", 20)
    if "seed" in params:
        payload["seed"] = params["seed"]
    if "temperature" in params:
        payload["temperature"] = params["temperature"]
    if "max_tokens" in params:
        payload["max_tokens"] = params["max_tokens"]

    return payload


def build_jev_payload(item):
    """Arm J's request. A different endpoint AND a different schema.

    Confirmed against TypeSafe's docs and probe_jev.py's FINDINGS: there is no
    `messages` array, no `response_format`, and no `provider`/`allow_fallbacks`
    field, so protocol.provider_block() does not apply -- TypeSafe is the only
    provider and the response echoes `provider` for checking instead.

    Section 4's per-item option permutation is carried by the INSERTION ORDER of
    the `criteria` mapping, which Python preserves through json.dumps. Section 4
    content parity is preserved: same question text, same verbatim descriptions.
    """
    desc = taxonomy.load_descriptions()
    criteria = {label: f"{desc[label]['name']}: {desc[label]['description']}"
                for label in item["options"]}   # shuffled order, per section 4
    return {
        "model": P.ARMS["J"]["model"],
        "state": f"Title: {item['title']}\n\nAbstract: {item['abstract']}",
        "questions": {
            P.JEV_QUESTION_KEY: {
                "type": "choice",
                "instructions": prompt.JEV_INSTRUCTIONS,
                "criteria": criteria,
            }
        },
    }


def completed_ids(path):
    """Ids already recorded. Tolerates a truncated final line from a crash."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _post(payload, url=CHAT_URL):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {P.api_key()}",
        "Content-Type": "application/json",
        "X-Title": "jev-calibration-study",
    })
    with urllib.request.urlopen(req, timeout=600) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def call(arm, item):
    """One item, up to MAX_ATTEMPTS. Returns the record to append."""
    is_jev = P.ARMS[arm]["mechanism"] == "decisions"
    payload = build_jev_payload(item) if is_jev else build_payload(arm, item)
    url = DECISIONS_URL if is_jev else CHAT_URL
    started = time.time()
    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            status, body = _post(payload, url)
            return {
                "id": item["id"], "arm": arm, "status": "ok",
                "http": status, "attempts": attempt + 1,
                "request": P.redact(payload), "response": body,
                "provider": body.get("provider"),
                "latency_s": round(time.time() - started, 3),
                "timestamp": dt.datetime.now(dt.UTC).isoformat(),
            }
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            last_error = f"HTTP {exc.code}: {raw[:500]}"
            if exc.code in RATE_LIMIT_CODES and attempt < MAX_ATTEMPTS - 1:
                time.sleep(min(20 * 2 ** attempt, 300))
            elif attempt == MAX_ATTEMPTS - 1:
                break
            else:
                time.sleep(5)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(10)

    # Section 5: a timeout after 3 retries is a failure, never a dropped item.
    return {
        "id": item["id"], "arm": arm, "status": "failure",
        "error": last_error, "attempts": MAX_ATTEMPTS,
        "request": P.redact(payload), "response": None,
        "latency_s": round(time.time() - started, 3),
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=P.LLM_ARMS)
    ap.add_argument("--pool", required=True, choices=["dev", "evaluation"])
    ap.add_argument("--run-id", type=int, default=1,
                    help="2 for PC2's J re-run (section 7)")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N items only, for smoke tests")
    ap.add_argument("--sequential", action="store_true",
                    help="one at a time, for the latency subsample")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    if args.pool == "evaluation" and not os.environ.get("JEV_ALLOW_EVAL"):
        raise SystemExit(
            "STOP: the evaluation pool is gated. Section 4 puts all arms in "
            "one 48-hour window and the run is priced at the phase 6 gate. "
            "Set JEV_ALLOW_EVAL=1 deliberately to proceed.")

    os.makedirs(RUN_DIR, exist_ok=True)
    suffix = "" if args.run_id == 1 else f"_{args.run_id}"
    out_path = os.path.join(RUN_DIR, f"{args.arm}_{args.pool}{suffix}.jsonl")

    items = pools.load_pool(args.pool)
    if args.limit:
        items = items[:args.limit]
    done = completed_ids(out_path)
    todo = [it for it in items if it["id"] not in done]
    print(f"{args.arm} on {args.pool}: {len(todo)} to do, "
          f"{len(done)} already recorded -> {out_path}", file=sys.stderr)
    if not todo:
        return

    counts = {"ok": 0, "failure": 0}
    fh = open(out_path, "a", encoding="utf-8")

    def handle(item):
        rec = call(args.arm, item)
        with _write_lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            counts[rec["status"]] += 1
            n = counts["ok"] + counts["failure"]
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  ok={counts['ok']} "
                      f"fail={counts['failure']}", file=sys.stderr)

    try:
        if args.sequential:
            for item in todo:
                handle(item)
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                list(pool.map(handle, todo))
    finally:
        fh.close()

    print(f"done: ok={counts['ok']} failure={counts['failure']}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

Add to `protocol.py`, filled in from the Task 7 findings (empty dict if the plain shape won):

```python
# Arm J's question key in the decisions payload and in the response's `answers`
# map. One name, used by both run.py and parse.py, so they cannot disagree.
JEV_QUESTION_KEY = "primary_category"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_run.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Snapshot the endpoint record at run start**

Section 4 requires recording provider **and quantization**, and says the
reasoning defaults in force "are recorded at run time". A chat completion
echoes the provider but not the quantization or the reasoning defaults, so
snapshot the registry record alongside the run.

Add to `run.py`, called once from `main()` before the items are dispatched:

```python
def snapshot_endpoint(arm, out_dir=RUN_DIR):
    """Record the pinned endpoint's registry entry beside the run.

    Section 4 requires provider and quantization on record. The completion
    response carries provider but not quantization, and nothing echoes the
    reasoning defaults, so the registry entry is the evidence.
    """
    spec = P.ARMS[arm]
    url = f"https://openrouter.ai/api/v1/models/{spec['model']}/endpoints"
    req = urllib.request.Request(url, headers={"User-Agent": "jev-study/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))["data"]
    match = [e for e in data["endpoints"]
             if e["provider_name"] == spec["provider"]]
    if not match:
        raise SystemExit(
            f"STOP: {spec['provider']} is not serving {spec['model']} any more. "
            f"The pin cannot be honoured; do not silently reroute.")
    snap = {
        "arm": arm, "model": spec["model"], "provider": spec["provider"],
        "quantization": match[0].get("quantization"),
        "supported_parameters": match[0].get("supported_parameters"),
        "context_length": match[0].get("context_length"),
        "pricing": match[0].get("pricing"),
        "captured": dt.datetime.now(dt.UTC).isoformat(),
    }
    path = os.path.join(out_dir, f"{arm}_endpoint.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=2)
    print(f"  endpoint: {spec['provider']} quant={snap['quantization']} "
          f"params={snap['supported_parameters']}", file=sys.stderr)
    return snap
```

Add this test to `tests/test_run.py`:

```python
def test_endpoint_snapshot_records_quantization_and_params(tmp_path):
    snap = run.snapshot_endpoint("Q-V", out_dir=str(tmp_path))
    assert snap["provider"] == "Parasail"
    assert "quantization" in snap
    assert "structured_outputs" in (snap["supported_parameters"] or [])


def test_endpoint_snapshot_confirms_ql_logprobs_are_advertised(tmp_path):
    snap = run.snapshot_endpoint("Q-L", out_dir=str(tmp_path))
    assert "logprobs" in (snap["supported_parameters"] or [])
```

Run: `.venv/bin/pytest tests/test_run.py -v -k endpoint`
Expected: PASS, 2 tests. These hit the network; if the pin has disappeared from
the registry the test fails loudly, which is the intended behaviour.

- [ ] **Step 6: Smoke-test against 3 dev items on the cheapest arm**

```bash
chmod +x run.py
.venv/bin/python run.py --arm Q-L --pool dev --limit 3
.venv/bin/python -c "
import json
for l in open('runs/Q-L_dev.jsonl'):
    r=json.loads(l); print(r['id'], r['status'], r.get('provider'))
"
```

Expected: 3 records, `status` `ok`, `provider` reported as `Parasail`. If the provider comes back as anything else, the pin is not being honoured — stop and fix before spending more.

- [ ] **Step 7: Verify resume does not re-bill**

Run: `.venv/bin/python run.py --arm Q-L --pool dev --limit 3`
Expected: `3 already recorded`, `0 to do`, and no new lines in the file.

- [ ] **Step 8: Commit**

```bash
git add run.py tests/test_run.py protocol.py runs/*_endpoint.json
git commit -m "Add resumable arm runner with per-arm section 4 parameter policy"
```

`runs/` is deliberately not committed yet; the dev run lands in Task 10 and is committed there.

---

### Task 9: `parse.py` — section 5 probability handling

**Files:**
- Create: `parse.py`, `tests/test_parse.py`

**Interfaces:**
- Consumes: `LABELS`, `K`, `EPSILON`, `floor_renorm` (Task 2); `to_canonical` (Task 4); `LETTERS` (Task 6).
- Produces: `parse_record(rec: dict, item: dict) -> tuple[np.ndarray, str]` returning a canonical-order K-vector and a status string from `FAILURE_KINDS | {"ok"}`; `parse_run(arm: str, pool: str) -> dict`; `FAILURE_KINDS: set[str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_parse.py
import json
import numpy as np
import pytest
import protocol as P
import parse


ITEM = {
    "id": "x",
    "label": "cs.LG",
    "options": ["cs.RO", "cs.IR", "cs.LG", "cs.CV", "cs.CR", "cs.AI", "cs.CL"],
}
UNIFORM = np.full(P.K, 1 / P.K)


def _verbalized(content, finish="stop"):
    return {"id": "x", "arm": "Q-V", "status": "ok",
            "response": {"choices": [{"finish_reason": finish,
                                      "message": {"content": content}}]}}


def test_verbalized_maps_keys_to_canonical_order():
    payload = {"cs.RO": 0.0, "cs.IR": 0.0, "cs.LG": 0.7, "cs.CV": 0.3,
               "cs.CR": 0.0, "cs.AI": 0.0, "cs.CL": 0.0}
    vec, status = parse.parse_record(_verbalized(json.dumps(payload)), ITEM)
    assert status == "ok"
    assert vec.sum() == pytest.approx(1.0)
    assert int(vec.argmax()) == P.LABELS.index("cs.LG")
    assert vec[P.LABELS.index("cs.CV")] > vec[P.LABELS.index("cs.AI")]


def test_verbalized_unnormalized_input_is_renormalized():
    payload = {l: 2.0 for l in P.LABELS}
    vec, status = parse.parse_record(_verbalized(json.dumps(payload)), ITEM)
    assert status == "ok"
    assert vec.sum() == pytest.approx(1.0)


def test_epsilon_floor_is_applied():
    payload = {l: 0.0 for l in P.LABELS}
    payload["cs.LG"] = 1.0
    vec, _ = parse.parse_record(_verbalized(json.dumps(payload)), ITEM)
    # The floor is applied ONCE, so a floored entry lands at eps/(1+m*eps)
    # where m is the number of floored entries (m <= K-1). A threshold of
    # eps*0.999 would fail correct code AND pass a double-floor bug, because
    # double-flooring pushes values back up toward eps.
    assert (vec >= P.EPSILON / (1 + (P.K - 1) * P.EPSILON) * (1 - 1e-9)).all()


def test_malformed_json_is_a_failure_scored_uniform():
    vec, status = parse.parse_record(_verbalized("not json at all"), ITEM)
    assert status == "parse_error"
    assert vec == pytest.approx(UNIFORM)


def test_missing_key_is_a_failure():
    payload = {l: 1 / 6 for l in P.LABELS if l != "cs.IR"}
    vec, status = parse.parse_record(_verbalized(json.dumps(payload)), ITEM)
    assert status == "parse_error"
    assert vec == pytest.approx(UNIFORM)


def test_negative_and_nonfinite_values_are_failures():
    for bad in (-0.5, float("inf"), float("nan")):
        payload = {l: 0.1 for l in P.LABELS}
        payload["cs.LG"] = bad
        vec, status = parse.parse_record(
            _verbalized(json.dumps(payload).replace("NaN", '"NaN"')), ITEM)
        assert status in ("parse_error", "bad_value")
        assert vec == pytest.approx(UNIFORM)


def test_truncation_is_a_failure():
    payload = {l: 1 / 7 for l in P.LABELS}
    vec, status = parse.parse_record(
        _verbalized(json.dumps(payload), finish="length"), ITEM)
    assert status == "truncated"
    assert vec == pytest.approx(UNIFORM)


def test_refusal_is_a_failure():
    rec = {"id": "x", "arm": "F1-V", "status": "ok",
           "response": {"choices": [{"finish_reason": "content_filter",
                                     "message": {"refusal": "no"}}]}}
    vec, status = parse.parse_record(rec, ITEM)
    assert status == "refusal"
    assert vec == pytest.approx(UNIFORM)


def test_runner_level_failure_is_scored_uniform():
    rec = {"id": "x", "arm": "J", "status": "failure", "response": None}
    vec, status = parse.parse_record(rec, ITEM)
    assert status == "timeout"
    assert vec == pytest.approx(UNIFORM)


def _letter(top):
    return {"id": "x", "arm": "Q-L", "status": "ok",
            "response": {"choices": [{
                "finish_reason": "stop",
                "message": {"content": top[0][0]},
                "logprobs": {"content": [{
                    "top_logprobs": [{"token": t, "logprob": lp}
                                     for t, lp in top]}]}}]}}


def test_letter_renormalizes_over_the_seven_label_tokens():
    top = [("C", -0.1), ("A", -2.0), ("D", -3.0), (" the", -5.0)]
    vec, status = parse.parse_record(_letter(top), ITEM)
    assert status == "ok"
    assert vec.sum() == pytest.approx(1.0)
    # C is position 2 in options -> cs.LG
    assert int(vec.argmax()) == P.LABELS.index("cs.LG")


def test_letter_labels_absent_from_topk_get_zero_before_the_floor():
    top = [("C", -0.1), ("A", -2.0)]
    vec, status = parse.parse_record(_letter(top), ITEM)
    assert status == "ok"
    # Only A and C were returned; the other five sit at the floor.
    floored = [v for v in vec if v <= P.EPSILON * 1.01]
    assert len(floored) == 5


def test_letter_outside_a_to_g_is_a_failure():
    vec, status = parse.parse_record(_letter([("Z", -0.1)]), ITEM)
    assert status == "bad_letter"
    assert vec == pytest.approx(UNIFORM)


def test_letter_ignores_case_and_surrounding_whitespace():
    vec, status = parse.parse_record(_letter([(" c ", -0.1), ("A", -2.0)]), ITEM)
    assert status == "ok"
    assert int(vec.argmax()) == P.LABELS.index("cs.LG")


def test_every_failure_kind_is_a_known_kind():
    assert "parse_error" in parse.FAILURE_KINDS
    assert "ok" not in parse.FAILURE_KINDS


def test_tag_leakage_is_detected_and_does_not_by_itself_fail_the_item():
    payload = {l: 1 / 7 for l in P.LABELS}
    content = "<thinking>hmm</thinking>" + json.dumps(payload)
    vec, status = parse.parse_record(_verbalized(content), ITEM)
    assert status == "ok"
    assert parse.has_tag_leakage(content)
    assert not parse.has_tag_leakage(json.dumps(payload))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_parse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'parse'`

Note: the module is `parse.py` in the project root. Python has no stdlib `parse`, so this does not shadow anything, but do not rename it to `parser` — that *is* a deprecated stdlib name.

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
"""PROTOCOL.md section 5, and nothing else.

1. Parse into a K-vector.
2. A failure is a parse error, refusal, truncation, non-finite or negative
   value, a letter outside A-G, or a timeout after 3 retries. Failures are
   scored as uniform and counted per arm. Never dropped.
3. Floor at epsilon and renormalize.

Vectors come out in canonical LABELS order regardless of the item's shuffle.

Usage:
    ./parse.py --arm Q-L --pool dev
"""

import argparse
import json
import os
import re
import sys

import numpy as np

import pools
import prompt
import protocol as P

PRED_DIR = "preds"
FAILURE_KINDS = {"parse_error", "bad_value", "truncated", "refusal",
                 "bad_letter", "timeout", "empty"}

TAG_RE = re.compile(r"</?(thinking|antml|system|internal|reasoning)\b",
                    re.IGNORECASE)


def has_tag_leakage(text):
    """Section 4 measures leakage; it is reported, not scored as a failure."""
    return bool(TAG_RE.search(text or ""))


def _uniform():
    return np.full(P.K, 1.0 / P.K)


def _parse_verbalized(choice, item):
    content = (choice.get("message") or {}).get("content")
    if (choice.get("message") or {}).get("refusal"):
        return None, "refusal"
    if choice.get("finish_reason") == "length":
        return None, "truncated"
    if choice.get("finish_reason") == "content_filter":
        return None, "refusal"
    if not content or not content.strip():
        return None, "empty"

    text = content.strip()
    match = re.search(r"\{.*\}", text, re.S)  # tolerate leaked prose around it
    if not match:
        return None, "parse_error"
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "parse_error"
    if not isinstance(obj, dict):
        return None, "parse_error"
    if set(obj) != set(P.LABELS):
        return None, "parse_error"

    vec = np.empty(P.K, dtype=float)
    for i, label in enumerate(P.LABELS):
        v = obj[label]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None, "bad_value"
        if not np.isfinite(v) or v < 0:
            return None, "bad_value"
        vec[i] = float(v)
    if vec.sum() <= 0:
        return None, "bad_value"
    return vec, "ok"


def _parse_letter(choice, item):
    """First generated token's logprobs, renormalized over the 7 label tokens.

    Any label missing from the returned top-k gets probability 0 before the
    floor (section 4).
    """
    if choice.get("finish_reason") == "length":
        return None, "truncated"
    lp = choice.get("logprobs") or {}
    content = lp.get("content") or []
    if not content:
        return None, "parse_error"
    top = content[0].get("top_logprobs") or []
    if not top:
        return None, "parse_error"

    shuffled = np.zeros(P.K, dtype=float)
    hit = False
    for entry in top:
        tok = (entry.get("token") or "").strip().upper()
        if len(tok) != 1 or tok not in prompt.LETTERS:
            continue
        pos = prompt.LETTERS.index(tok)
        shuffled[pos] += float(np.exp(entry["logprob"]))
        hit = True

    if not hit:
        # Nothing in top-k was a valid letter. Fall back to the emitted text so
        # a valid single letter outside top-k is not thrown away.
        emitted = ((choice.get("message") or {}).get("content") or "").strip().upper()
        if len(emitted) == 1 and emitted in prompt.LETTERS:
            shuffled[prompt.LETTERS.index(emitted)] = 1.0
        else:
            return None, "bad_letter"

    # shuffled is in option order; map back to canonical label order.
    return pools.to_canonical(shuffled, item["options"]), "ok"


def parse_record(rec, item):
    """One run record -> (canonical K-vector, status). Never raises."""
    if rec.get("status") == "failure":
        return _uniform(), "timeout"
    resp = rec.get("response") or {}
    mech = P.ARMS[rec["arm"]]["mechanism"]

    # J's response has no `choices` list -- it is a decisions-contract body.
    if mech == "decisions":
        vec, status = _parse_decisions(resp, item)
        if status != "ok":
            return _uniform(), status
        return P.floor_renorm(vec), "ok"

    choices = resp.get("choices") or []
    if not choices:
        return _uniform(), "empty"
    choice = choices[0]

    if mech == "letter":
        vec, status = _parse_letter(choice, item)
    else:
        vec, status = _parse_verbalized(choice, item)

    if status != "ok":
        return _uniform(), status
    return P.floor_renorm(vec), "ok"


def _parse_decisions(resp, item):
    """J's typed choice, per TypeSafe's System One response contract.

    resp["answers"][JEV_QUESTION_KEY] carries `choice`, `probabilities` (a dict
    keyed by label -- all 7 are scored, confirmed by probe_jev.py) and
    `confidence`. Unlike the chat arms there is no `choices` list, so this takes
    the whole response body rather than a choice element.
    """
    answer = (resp.get("answers") or {}).get(P.JEV_QUESTION_KEY)
    if not isinstance(answer, dict):
        return None, "parse_error"
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or set(probs) != set(P.LABELS):
        return None, "parse_error"
    vec = np.empty(P.K, dtype=float)
    for i, label in enumerate(P.LABELS):
        v = probs[label]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None, "bad_value"
        if not np.isfinite(v) or v < 0:
            return None, "bad_value"
        vec[i] = float(v)
    if vec.sum() <= 0:
        return None, "bad_value"
    return vec, "ok"


def parse_run(arm, pool):
    """Parse a whole run file into preds/{arm}_{pool}.npz plus a report."""
    items = {it["id"]: it for it in pools.load_pool(pool)}
    path = os.path.join("runs", f"{arm}_{pool}.jsonl")
    ids, vecs, statuses, leaks = [], [], [], 0

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            item = items[rec["id"]]
            vec, status = parse_record(rec, item)
            ids.append(rec["id"])
            vecs.append(vec)
            statuses.append(status)
            try:
                content = rec["response"]["choices"][0]["message"]["content"]
            except (TypeError, KeyError, IndexError):
                content = ""
            leaks += bool(has_tag_leakage(content))

    from collections import Counter
    counts = Counter(statuses)
    report = {
        "arm": arm, "pool": pool, "n": len(ids),
        "ok": counts.get("ok", 0),
        "failures": {k: v for k, v in counts.items() if k in FAILURE_KINDS},
        "failure_total": sum(v for k, v in counts.items() if k in FAILURE_KINDS),
        "tag_leakage": leaks,
    }

    os.makedirs(PRED_DIR, exist_ok=True)
    np.savez(os.path.join(PRED_DIR, f"{arm}_{pool}.npz"),
             ids=np.array(ids), probs=np.vstack(vecs),
             statuses=np.array(statuses),
             labels=np.array([items[i]["label"] for i in ids]))
    with open(os.path.join(PRED_DIR, f"{arm}_{pool}_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=P.LLM_ARMS)
    ap.add_argument("--pool", required=True, choices=["dev", "evaluation"])
    args = ap.parse_args()
    report = parse_run(args.arm, args.pool)
    print(json.dumps(report, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_parse.py -v`
Expected: PASS, 15 tests.

- [ ] **Step 5: Commit**

```bash
git add parse.py tests/test_parse.py
git commit -m "Add section 5 probability handling with the full failure taxonomy"
```

---

### Task 10: Dev runs for all five LLM arms

**Files:**
- Create (output, committed): `runs/*_dev.jsonl`, `preds/*_dev*`

`_parse_decisions` was already implemented and tested in Task 9, against the real
recorded responses in `probes/jev_probe.json`. Verify it rather than rewriting it.

**Interfaces:**
- Consumes: Tasks 4-9.
- Produces: real dev responses for all five LLM arms and a parse report per arm, which Task 11 reads.

- [ ] **Step 1: Verify the J parser against the recorded probe, then confirm the suite is green**

`_parse_decisions` already exists and reads `resp["answers"][JEV_QUESTION_KEY]["probabilities"]`. Confirm it still parses a real recorded Jev response and that the full suite passes before spending anything:

```bash
.venv/bin/pytest tests/ -q
.venv/bin/python -c "
import json, parse, pools
p = [r for r in json.load(open('probes/jev_probe.json')) if r.get('status') == 200][0]
item = next(i for i in pools.load_pool('dev') if i['id'] == p['item'])
vec, st = parse.parse_record({'id': item['id'], 'arm': 'J', 'status': 'ok',
                              'response': p['response']}, item)
print('status', st, '| sums to', round(float(vec.sum()), 10))
"
```
Expected: suite green, `status ok`, sums to 1.0. If either fails, STOP — do not spend money against a broken parser.

- [ ] **Step 2: Record the pre-spend cost expectation**

Before running anything, note the expected bill so a surprise is visible immediately. At the pinned prices (F1-V $2/M in, $10/M out; F2-V $1/M in, $6/M out; Qwen $0.24/M in, $2.20/M out; J $0.042/M in, free out) and ~1,050 prompt tokens per item, 200 items per arm comes to roughly: J $0.01, Q-L $0.05, Q-V $0.11, F2-V ~$1.35, F1-V ~$3.12 — about $4.64 total, with the two frontier figures dominated by reasoning tokens that are a GUESS until measured. If an arm's actual spend exceeds twice its estimate, stop and report rather than continuing.

- [ ] **Step 3: Run the cheap arms first**

```bash
for arm in Q-L Q-V J; do
  .venv/bin/python run.py --arm $arm --pool dev
done
```

Expected: 200 records each, failures in the low single digits or zero.

- [ ] **Step 4: Run the two expensive arms**

```bash
for arm in F2-V F1-V; do
  .venv/bin/python run.py --arm $arm --pool dev
done
```

These are the arms with reasoning at provider defaults, so they are slower and carry the real cost. Watch the failure counter; if truncations appear, stop — `max_tokens` needs raising before continuing, which is Task 11's job.

- [ ] **Step 5: Parse every arm**

```bash
for arm in J F1-V F2-V Q-V Q-L; do
  .venv/bin/python parse.py --arm $arm --pool dev
done
```

- [ ] **Step 6: Commit the dev artifacts**

```bash
git add runs/ preds/ parse.py tests/test_parse.py
git commit -m "Run and parse all five LLM arms on the dev pool"
```

---

### Task 11: The section 4 pre-freeze checks

**Files:**
- Create: `devcheck.py`
- Create (output, committed): `devcheck_report.md`
- Modify: `DEVIATIONS.md` (if any check forces a change)

**Interfaces:**
- Consumes: `preds/*_dev*` and `runs/*_dev.jsonl` (Task 10).
- Produces: `devcheck_report.md` carrying every number the phase 6 gate needs.

Section 4 requires all of these before the prompts and parsers are frozen.

- [ ] **Step 1: Write the check script**

```python
#!/usr/bin/env python3
"""The checks PROTOCOL.md section 4 requires before freezing prompts/parsers.

Every number in devcheck_report.md is computed here, not typed by hand.

Usage:
    ./devcheck.py
"""

import collections
import json
import os
import sys

import numpy as np

import pools
import prompt
import protocol as P

ARMS = P.LLM_ARMS
COST_PER_M = {  # USD per million tokens, from the OpenRouter endpoint records
    "J": (0.042, 0.0),
}


def letter_tokenization():
    """Each letter A-G must be a single token under Qwen's tokenizer.

    Section 4 requires a different alphabet if not. We do not have the
    tokenizer locally, so this is checked behaviourally: across the Q-L dev
    responses, every letter that appears in a top_logprobs list must appear as
    a bare single character, never glued to neighbouring text.
    """
    seen = collections.Counter()
    glued = collections.Counter()
    with open("runs/Q-L_dev.jsonl", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            try:
                top = rec["response"]["choices"][0]["logprobs"]["content"][0][
                    "top_logprobs"]
            except (TypeError, KeyError, IndexError):
                continue
            for e in top:
                tok = e.get("token") or ""
                stripped = tok.strip().upper()
                if stripped and stripped[0] in prompt.LETTERS:
                    if len(stripped) == 1:
                        seen[stripped] += 1
                    else:
                        glued[tok] += 1
    return seen, glued


def per_arm_stats():
    rows = []
    for arm in ARMS:
        rep_path = f"preds/{arm}_dev_report.json"
        if not os.path.exists(rep_path):
            continue
        rep = json.load(open(rep_path))
        prompt_toks = compl_toks = reason_toks = 0
        latencies = []
        providers = collections.Counter()
        quants = collections.Counter()
        with open(f"runs/{arm}_dev.jsonl", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                latencies.append(rec.get("latency_s", 0))
                resp = rec.get("response") or {}
                providers[resp.get("provider") or "?"] += 1
                u = resp.get("usage") or {}
                prompt_toks += u.get("prompt_tokens", 0) or 0
                compl_toks += u.get("completion_tokens", 0) or 0
                det = u.get("completion_tokens_details") or {}
                reason_toks += det.get("reasoning_tokens", 0) or 0
        n = max(rep["n"], 1)
        rows.append({
            "arm": arm, "n": rep["n"], "ok": rep["ok"],
            "failures": rep["failure_total"],
            "failure_kinds": rep["failures"],
            "tag_leakage": rep["tag_leakage"],
            "prompt_per_item": prompt_toks / n,
            "completion_per_item": compl_toks / n,
            "reasoning_per_item": reason_toks / n,
            "latency_p50": float(np.percentile(latencies, 50)) if latencies else 0,
            "latency_p95": float(np.percentile(latencies, 95)) if latencies else 0,
            "providers": dict(providers),
        })
    return rows


def pc1_dev_accuracy():
    """Section 7: PC1's accuracy spread is checked on dev first."""
    acc = {}
    for arm in ARMS:
        path = f"preds/{arm}_dev.npz"
        if not os.path.exists(path):
            continue
        d = np.load(path, allow_pickle=True)
        pred = [P.LABELS[i] for i in d["probs"].argmax(axis=1)]
        acc[arm] = float(np.mean([p == g for p, g in zip(pred, d["labels"])]))
    tiered = {a: v for a, v in acc.items() if a in ("J", "F1-V", "F2-V")}
    spread = (max(tiered.values()) - min(tiered.values())) * 100 if len(tiered) > 1 else 0
    return acc, spread


def main():
    rows = per_arm_stats()
    seen, glued = letter_tokenization()
    acc, spread = pc1_dev_accuracy()

    out = ["# Dev-set pre-freeze checks", "",
           "Generated by `devcheck.py`. Every number is computed, not typed.",
           "", "## Per-arm dev results (200 items)", "",
           "| Arm | ok | failures | kinds | leakage | prompt tok/item | "
           "completion tok/item | reasoning tok/item | p50 s | p95 s | provider |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(
            f"| {r['arm']} | {r['ok']} | {r['failures']} | "
            f"{r['failure_kinds'] or '-'} | {r['tag_leakage']} | "
            f"{r['prompt_per_item']:.0f} | {r['completion_per_item']:.0f} | "
            f"{r['reasoning_per_item']:.0f} | {r['latency_p50']:.1f} | "
            f"{r['latency_p95']:.1f} | {r['providers']} |")

    out += ["", "## Letter tokenization (Q-L)", "",
            f"- single-character letter tokens observed: {dict(seen)}",
            f"- letters glued to other text: {dict(glued) or 'none'}", ""]
    if glued:
        out.append("**Section 4 action required:** letters are not clean single "
                   "tokens. Substitute a different alphabet and re-run the dev "
                   "pass, recording the substitution in DEVIATIONS.md.")
    else:
        out.append("All observed letters are clean single tokens; A-G stands.")

    out += ["", "## PC1 accuracy check on dev (section 7)", "",
            "| Arm | dev accuracy |", "| --- | --- |"]
    for arm, v in acc.items():
        out.append(f"| {arm} | {v:.3f} |")
    out += ["", f"Spread across J, F1-V, F2-V: **{spread:.1f} points** "
            f"(PC1 threshold: 8).", ""]
    out.append("Exceeds the threshold: the tier-matched framing is dropped and "
               "the comparison is reported through the Brier decomposition."
               if spread > 8 else
               "Within the threshold on dev. Re-checked on the evaluation set, "
               "which is what section 7 reports.")

    out += ["", "## Extrapolated evaluation cost (2,000 items)", "",
            "| Arm | prompt tok | completion tok | note |",
            "| --- | --- | --- | --- |"]
    for r in rows:
        out.append(f"| {r['arm']} | {r['prompt_per_item'] * 2000:,.0f} | "
                   f"{r['completion_per_item'] * 2000:,.0f} | "
                   f"reasoning included in completion |")
    out += ["", "Multiply by each endpoint's posted price to get the bill. "
            "J's completion tokens are zero-priced against $0.042/M prompt. "
            "PC2 re-runs J once more on all 2,000 items.", ""]

    with open("devcheck_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
chmod +x devcheck.py
.venv/bin/python devcheck.py
```

- [ ] **Step 3: Act on each finding**

Work through the report and resolve each, adding a `DEVIATIONS.md` row for anything that changes:

- **Truncations above zero on F1-V or F2-V** → raise `max_tokens` in `ARMS[...]["params"]` to clear observed completion plus reasoning tokens with margin, re-run those arms on dev, re-run `devcheck.py`.
- **Letters glued to other text** → substitute an alphabet, re-run Q-L on dev.
- **Tag leakage above a couple of items** → record the count; section 4 expects near zero with thinking on and structured outputs in force.
- **Any provider other than the pin** → the pin is not holding; stop and fix.
- **Q-L records with no `logprobs` block** → Parasail is not returning logprobs in practice. This reopens local vLLM serving per design doc section 2.1; raise it rather than working around it.
- **Parse failures concentrated in one arm** → confirm `structured_outputs` was actually honoured on that endpoint.
- **Cost dominated by F1-V** → section 4 says to use "batch endpoints where available". Check whether OpenRouter exposes one for the pinned `Anthropic` endpoint at the time of the run. If it does, using it is in-protocol and needs no deviation; if it does not, record that in `FREEZE.md` so the choice is visible rather than assumed.

- [ ] **Step 4: Confirm structured outputs and logprobs were honoured live**

```bash
.venv/bin/python - <<'EOF'
import json
for arm in ["F1-V", "F2-V", "Q-V"]:
    n = ok = 0
    for line in open(f"runs/{arm}_dev.jsonl"):
        r = json.loads(line); n += 1
        try:
            json.loads(r["response"]["choices"][0]["message"]["content"])
            ok += 1
        except Exception:
            pass
    print(f"{arm}: {ok}/{n} responses are valid standalone JSON")
lp = sum(1 for l in open("runs/Q-L_dev.jsonl")
         if (json.loads(l).get("response") or {}).get("choices", [{}])[0].get("logprobs"))
print(f"Q-L: {lp} responses carry a logprobs block")
EOF
```

Expected: near 200/200 JSON for each verbalized arm, and near 200 logprobs blocks for Q-L. Anything well below that is a stop condition.

- [ ] **Step 5: Commit**

```bash
git add devcheck.py devcheck_report.md DEVIATIONS.md
git commit -m "Add and run the section 4 pre-freeze dev checks"
```

---

### Task 12: Freeze and price the evaluation run

**Files:**
- Create: `FREEZE.md`
- Modify: `DEVIATIONS.md` (if anything changed since the protocol commit)

**Interfaces:**
- Consumes: `devcheck_report.md` (Task 11).
- Produces: a commit that freezes prompts and parsers, and a priced go/no-go for the user.

- [ ] **Step 1: Confirm the full test suite passes**

Run: `.venv/bin/pytest -v`
Expected: PASS, roughly 60 tests, zero failures. Do not proceed on a red suite.

- [ ] **Step 2: Confirm the pools still reproduce**

Run: `.venv/bin/python pools.py --check`
Expected: the three counts printed and `check passed`. This proves the frozen manifests still match the CSV and the seeds.

- [ ] **Step 3: Write `FREEZE.md`**

Record: the commit SHA being frozen; the five arms with their exact model IDs and provider pins; the `max_tokens` finally set per arm; the letter alphabet in force; the resolution of every phase 5 check; the extrapolated cost per arm for 2,000 items plus the PC2 re-run; and the total. State explicitly that no evaluation item has been sent to any model.

- [ ] **Step 4: Commit the freeze**

```bash
git add FREEZE.md DEVIATIONS.md
git commit -m "Freeze prompts and parsers ahead of the evaluation run"
git log --oneline | head -15
```

- [ ] **Step 5: Stop and report to the user**

Report: the total extrapolated cost with the per-arm breakdown; every phase 5 finding and how it was resolved; any new `DEVIATIONS.md` rows; and the dev accuracy spread against PC1's 8-point threshold. Then ask for an explicit go/no-go on the evaluation spend.

Do not run any arm against the evaluation pool. `run.py` gates it behind `JEV_ALLOW_EVAL=1`, and section 4's 48-hour window means the run is a deliberate launch. `anchor.py` and `analyze.py` are the next plan, written after this gate and before the evaluation run, so the analysis code is frozen before any evaluation result exists.
