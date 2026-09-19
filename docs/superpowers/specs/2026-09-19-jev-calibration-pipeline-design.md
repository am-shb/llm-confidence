# Design: evaluation pipeline for the Jev calibration study

Design for the code that executes `PROTOCOL.md`. The protocol is the spec for
*what* to measure and is already committed and frozen; this document covers
*how* the code is structured to run it without drifting from it.

Written 2026-09-19, before any evaluation item has been sent to any model.

## 1. Constraints the design answers to

The protocol is pre-registered. That turns three ordinary engineering
preferences into requirements:

1. **One source of truth for every frozen constant.** Six seeds, the 7-label
   set, epsilon, the start date and the arm registry all appear in
   `PROTOCOL.md`. If any of them is written twice in the code, the two copies
   can drift and the pre-registration stops being checkable.
2. **Discoveries go in `DEVIATIONS.md`, never into `PROTOCOL.md`.** Section 4
   instructs us to confirm endpoint capabilities before the run, so finding a
   problem is an anticipated outcome, not a reason to quietly edit the spec.
3. **Nothing pre-registered is touched during debugging.** Section 4 restricts
   prompt and parser tuning to the dev pool. The evaluation pool is read only
   once, inside a single 48-hour window.

## 2. Findings from endpoint verification

Both were found on 2026-09-19 by querying the OpenRouter endpoint registry, and
both become dated `DEVIATIONS.md` rows.

### 2.1 The declared Qwen endpoint cannot satisfy section 4

Section 4 names the first-party `Alibaba` endpoint as the default for Q-V and
Q-L. That endpoint advertises `logprobs` but **not** `structured_outputs`.
Section 4 also mandates structured outputs on every verbalized arm, so Alibaba
can serve Q-L and cannot serve Q-V. Splitting the two across backends would
destroy the single property that makes the C3 contrast clean.

No endpoint of the 16 offers unquantized + logprobs + structured outputs
together. `DeepInfra` is the only bf16 endpoint and has no logprobs. The
endpoints supporting both mechanisms are:

| Endpoint   | Quantization | Uptime 30m | $/M prompt |
| ---------- | ------------ | ---------- | ---------- |
| DekaLLM    | unknown      | 99.9%      | 0.20       |
| Parasail   | fp8          | 100%       | 0.24       |
| CoreWeave  | fp8          | 100%       | 0.40       |
| Darkbloom  | fp4          | 99.4%      | 0.10       |

**Resolution: pin `Parasail` for both Qwen arms.** A known fp8 at 100% uptime
serving both mechanisms from one identical backend. The cost is that section 9's
quantization limitation gets sharper — C3 is now explicitly an fp8 estimate, and
the post must say so. fp4 was rejected as too lossy for a distribution Q-L reads
directly; `unknown` was rejected because the post could not then state what C3
was measured on.

Dropping structured outputs to keep Alibaba was rejected outright: parse
failures score as uniform under section 5, so they would land on the primary
metric, which is the exact contamination the structured-outputs rule exists to
prevent.

### 2.2 Frontier logprobs claim verifies; pinning matters more than stated

Section 3 claims no frontier provider exposes answer-token logprobs. Confirmed:
0 of 10 `claude-sonnet-5` endpoints and 0 of 7 `gpt-5.6-terra` endpoints list
`logprobs`. Section 4's determinism notes also verify — both models report
`temperature: false`, and Terra reports `seed: true`.

New constraint: only 2 of 10 Sonnet endpoints support `structured_outputs`
(`Anthropic` and `Claude Platform on AWS`). F1-V must therefore pin first-party
`Anthropic` or it silently loses constrained decoding. For Terra, the OpenAI and
Azure endpoints support it and Bedrock does not, so F2-V pins first-party
`OpenAI`.

### 2.3 Jev's record matches the protocol

`typesafe/jev-1.13` is absent from `/api/v1/models` but resolves on direct
lookup, exactly as section 3 states: empty `supported_parameters`, 32K context,
zero-priced completion against $0.042/M prompt, modality `text->decisions`,
single `TypeSafe` endpoint.

One detail section 3 does not record: the endpoint reports
`supports_tool_choice.function: true` despite the empty parameter list, and
OpenRouter truncates the model description mid-sentence. The wire format for a
typed-decision model is therefore still unknown and is resolved by a probe
(phase 3), not by guesswork.

## 3. Stack

A `.venv` with `requirements.txt`: numpy, scipy, scikit-learn, matplotlib.
Hand-rolling TF-IDF, multinomial logistic regression and a 2,000-resample
bootstrap would add error surface to exactly the code whose correctness the
study rests on, and a hand-rolled anchor arm is harder to defend in the post
than sklearn defaults.

HTTP stays on stdlib `urllib` plus `concurrent.futures.ThreadPoolExecutor`,
matching `harvest_arxiv.py`, whose retry and backoff shape is reused.

## 4. Module layout

Small single-purpose scripts over one shared module. The run phase is hours of
network I/O and must be resumable; the analysis phase reruns in seconds. Those
belong on opposite sides of a file boundary, with the frozen constants in
exactly one place.

| File          | Responsibility                                                    | Depends on           |
| ------------- | ----------------------------------------------------------------- | -------------------- |
| `protocol.py` | Frozen constants, arm registry, CSV loader, `floor_renorm`         | —                    |
| `pools.py`    | Derive the three pools and per-item option permutations            | `protocol.py`        |
| `prompt.py`   | Build per-arm prompts from one shared body                         | `protocol.py`        |
| `run.py`      | Execute one arm over one pool, resumably                           | `protocol`, `prompt` |
| `parse.py`    | Raw responses to K-vectors, with failure accounting                | `protocol.py`        |
| `anchor.py`   | Arms A, P and P\*                                                  | `protocol`, `pools`  |
| `analyze.py`  | Cross-fitting, metrics, bootstrap, contrasts, figures              | `protocol.py`        |

`protocol.py` is named for what it holds: the executable mirror of
`PROTOCOL.md`. It is not Jev-specific — Jev is one of eight arms.

### 4.1 `protocol.py`

Holds, each defined once: the 7 labels in canonical order; all six seeds (master
20260919, evaluation 20260920, dev 20260921, option shuffle 20260922, folds
20260923, bootstrap 20260924); epsilon 0.001; the 5 Sept 2026 start date; the
expected pool counts; output paths; and the arm registry mapping each arm ID to
its model, mechanism, provider pin and parameter policy.

Provides `floor_renorm(vec, eps)` — the single implementation of section 5's
floor-and-renormalize rule, used by every arm including A, P, P\* and after
temperature scaling — and a CSV loader that **asserts the file's SHA-256 matches
the value recorded in section 2** before returning rows. The protocol records
that hash so the numbers stay checkable; running against a different file
silently invalidates every count downstream, so it fails loudly at import.

### 4.2 `pools.py`

Filters to in-set papers on `primary_category`, runs section 2's integrity
checks (no duplicate IDs, no abstract under 100 characters, median and p95 word
counts), then splits on the 5 Sept 2026 start date using `published`, which is
the v1 submission date — the reason `harvest_arxiv.py` uses the search API
rather than OAI-PMH.

Draws the evaluation pool (N=2,000, seed 20260920) from post-start in-set
papers, the dev pool (200, seed 20260921) from pre-start papers, and assigns all
remaining pre-start papers to the anchor pool. Generates one label permutation
per item under seed 20260922, applied identically to every arm including J,
which also controls letter-position bias in Q-L.

Writes committed manifests to `pools/{evaluation,dev,anchor}.jsonl` carrying the
item ID, gold label and option order, plus a counts table.

**It fails hard if the exact totals do not reproduce:** 9,983 in-set, 4,280
post-start, 5,503 anchor. These cross-check (9,983 − 4,280 − 200 = 5,503), so a
mismatch means the filter disagrees with whatever produced section 2 and must be
investigated rather than accepted.

Section 2's per-class evaluation counts are marked "(rounded)", so they are read
as expectations derived from the 4,280-pool proportions rather than realized
draws. Realized counts differing by a few items is correct behaviour, not an
error, and only the exact pool totals gate the run.

### 4.3 `prompt.py`

One shared instruction body and one set of arXiv category descriptions, quoted
verbatim from the official taxonomy and committed as data, with no base-rate
information in any arm's prompt. Three output shapes:

- **Verbalized JSON** (F1-V, F2-V, Q-V): schema requiring all 7 keys, presented
  in the item's shuffled order, with structured outputs in force.
- **Letter** (Q-L): a single letter A–G indexing the shuffled options, with
  logprobs requested on the first generated token.
- **Typed decision** (J): shape fixed by the phase 3 probe.

Every arm additionally receives section 4's generic instruction not to emit
internal or system XML tags, for symmetry. No arm receives any instruction
against reasoning.

### 4.4 `run.py`

`./run.py --arm F1-V --pool dev`. Reads the arm's policy from the registry and
sends `allow_fallbacks: false` with the pinned provider, structured outputs or
logprobs as the mechanism requires, seed 20260919 wherever accepted, and section
4's temperature rules (0 for Q-V, unset for Q-L, unsettable for F1-V and J).

Reasoning parameters are **omitted** for F1-V and F2-V so each runs as it ships,
with the defaults in force recorded per response. Recorded alongside: provider,
quantization, full usage including reasoning tokens, latency, timestamp and the
exact request parameters.

Output is append-only JSONL at `runs/{arm}_{pool}.jsonl` keyed by item ID. On
start it reads the existing IDs and skips them, so an interrupted run resumes
instead of re-billing — the full study is roughly 14,000 calls. Three retries,
then the item is recorded as a failure per section 5. A `--sequential` flag
serves section 7's 200-item latency subsample; `--run-id 2` serves PC2's J
re-run.

### 4.5 `parse.py`

Implements section 5 and nothing else. Parses each raw response into a K-vector;
classifies parse errors, refusals, truncations, non-finite or negative values,
letters outside A–G, and post-retry timeouts as failures scored as uniform and
counted per arm, never dropped; applies the epsilon floor and renormalizes. For
Q-L, renormalizes over the 7 label tokens with any label missing from the
returned top-k set to 0 before the floor. Un-shuffles vectors back to canonical
label order.

Emits `preds/{arm}_{pool}.npz` plus a report carrying the failure, truncation
and tag-leakage counts section 7 requires as secondary outcomes.

### 4.6 `anchor.py`

Fits TF-IDF plus multinomial logistic regression on the anchor pool and predicts
the evaluation pool (arm A); emits anchor-pool base rates as a constant vector
(P) and evaluation-pool base rates as a constant vector (P\*, labelled an oracle
everywhere it appears). All three pass through the same epsilon floor and write
the same `preds/` format, so `analyze.py` treats every arm uniformly.

### 4.7 `analyze.py`

Splits the evaluation items into 5 folds stratified by label under seed
20260923. Per arm and fold, fits on the other four and applies frozen to the
held-out fold: temperature T over [0.05, 20] minimizing the NLL of q
proportional to p^(1/T), and the acceptance threshold tau for 5% error by
sorting training-fold items on top-label confidence and taking the largest k
whose top-k error is at most 5%, with tau infinite and coverage zero when k < 50.
Tau is fitted separately for raw and scaled probabilities. Epsilon is re-applied
after scaling so log-space operations stay defined.

Computes, raw and scaled: multiclass Brier (primary), calibration gap as raw
minus scaled Brier, accuracy, top-label ECE over 15 equal-mass bins, and
coverage at 5% error beside realized held-out error. Runs the paired bootstrap
over items, 2,000 resamples under seed 20260924, percentile 95% intervals on
each arm's difference from J, with out-of-fold predictions held fixed inside the
bootstrap. Evaluates PC1's 8-point accuracy-spread check across J, F1-V and
F2-V — checked on dev first per section 7, then reported on the evaluation set
whatever it shows — and PC2's Brier difference between J's two runs. Recomputes the primary
metric at epsilon 1e-4 and 1e-2. Renders per-arm reliability diagrams and
risk-coverage curves.

Reports the four contrasts of section 8 against the decision rules fixed there,
with A, P and P\* present in every table and excluded from every contrast.

## 5. Phasing

This round stops at the dev-set freeze. Section 4 requires prompts and parsers
to be tuned on dev and frozen by commit, and the evaluation spend is dominated
by F1-V running with adaptive thinking at effort `high` over 2,000 items, so the
cost extrapolation gates the run.

| Phase | Work                                                                                                                             |
| ----- | -------------------------------------------------------------------------------------------------------------------------------- |
| 0     | venv and `requirements.txt`; the two `DEVIATIONS.md` rows from section 2                                                          |
| 1     | `protocol.py` and `pools.py`; manifests committed, exact counts validated                                                        |
| 2     | Fetch the 7 official arXiv category descriptions verbatim; commit as data                                                        |
| 3     | Probe Jev's wire format on dev items; settle the J adapter                                                                       |
| 4     | `prompt.py`, `run.py`, `parse.py`; dev runs for all 6 LLM arms over 200 items                                                    |
| 5     | Section 4's mandated dev checks (below)                                                                                          |
| 6     | Freeze commit and cost report; **stop for go/no-go before any evaluation spend**                                                 |

Phase 5 checks, all required by section 4 before freezing: each letter A–G is a
single token under Qwen's tokenizer, with a different alphabet substituted if
not; tag-leakage counts near zero; parse-failure rates per arm;
`structured_outputs` confirmed live on each pinned endpoint; logprobs confirmed
live on Parasail; PC1's accuracy spread across J, F1-V and F2-V checked on dev,
since section 7 requires that check before the evaluation set; and dev token
usage extrapolated to a full bill, with `max_tokens` set to clear observed
thinking plus answer length with margin, since truncation scores as a failure.

`anchor.py` and `analyze.py` are built after the phase 6 gate and before the
evaluation run, so the analysis code is frozen before any evaluation result
exists.

## 6. Testing

Tests concentrate on the pure functions whose silent misbehaviour would corrupt
the primary metric without failing visibly:

- `floor_renorm`: sums to 1, respects epsilon, handles an all-zero vector, is
  idempotent under re-application.
- Section 5 failure taxonomy: each failure class maps to a uniform vector and
  increments the right counter.
- Q-L renormalization: labels absent from top-k go to 0 before the floor;
  letters outside A–G are failures.
- Shuffle round-trip: permute then un-permute is the identity, so a vector can
  never be silently misaligned to the wrong label.
- Temperature fitting: recovers a known T from synthetic data; T=1 is a no-op.
- Metrics: Brier and ECE against hand-computed values on small fixtures; a
  perfect predictor and a uniform predictor both score as expected.
- Tau fitting: k < 50 yields infinite tau and zero coverage.
- Pool derivation: seeded draws are reproducible across runs, and the three
  pools are disjoint.

Network calls are not unit-tested; `run.py` is exercised against the dev pool in
phase 4, which is what the dev pool is for.

## 7. Risks

- **Jev's wire format is unknown** until phase 3. The J adapter is the one
  component that cannot be fully specified in advance.
- **Parasail advertising `logprobs` is not proof it returns them.** Phase 5
  verifies live. Failure there reopens local vLLM serving, which would restore
  unquantized weights and remove the fp8 caveat at the cost of a second runner
  path.
- **F1-V dominates the bill** at effort `high` over 2,000 items. Phase 5 exists
  to price it before committing.
- **Section 2's exact totals may not reproduce.** Treated as a stop condition in
  phase 1, not something to accept and move past.
- **The 48-hour window** of section 4 makes the evaluation run a deliberate
  launch rather than something to drift into.
- **Jev's training cutoff remains unconfirmed**, as section 2 states. Nothing in
  this design changes that; if TypeSafe later reports a cutoff inside the
  evaluation window, results for J are void via `DEVIATIONS.md`.
