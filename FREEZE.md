# Freeze record: prompts and parsers, ahead of the evaluation run

**No evaluation-pool item has been sent to any model.** Every number below comes from the
200-item dev pool, which `PROTOCOL.md` section 4 designates for tuning and debugging and which is
disjoint from the 2,000-item evaluation pool.

Frozen at commit `45041af4b034aee527a88281f885b4dfe70ac4f9` on branch `jev-pipeline-dev-freeze`.
Suite: 109 tests passing. `PROTOCOL.md` is byte-identical to `main` — the pre-registration was
never edited; all divergences are in `DEVIATIONS.md` (7 rows).

## 1. Arms as frozen

| Arm | Model | Provider pin | Quantization | Reasoning, MEASURED | Mechanism |
| --- | --- | --- | --- | --- | --- |
| J | `typesafe/jev-1.13` | TypeSafe | unknown | 0 tok/item (no such parameter) | typed decision |
| F1-V | `anthropic/claude-sonnet-5` | Anthropic | unknown | **0 tok/item** | verbalized JSON |
| F2-V | `openai/gpt-5.6-terra` | OpenAI | unknown | 67 tok/item | verbalized JSON |
| Q-V | `qwen/qwen3.8-27b` | Parasail | **fp8** | 0 (forced off) | verbalized JSON |
| Q-L | `qwen/qwen3.8-27b` | Parasail | **fp8** | 0 (forced off) | first-token logprobs |

All arms pin their provider with `allow_fallbacks: false`. J is the exception in form only: its
decisions schema has no provider field, so its pin is implicit (TypeSafe is the sole provider) and is
checked against the `provider` field the response echoes.

**Q-V and Q-L share model, provider and quantization**, which is the property C3 depends on.
C3 is an **fp8** estimate — section 9's quantization limitation applies with that specificity.

`max_tokens` is NOT set on any arm. It was in the plan as a guard against truncation, but zero
truncations occurred on 1,000 dev calls, and an unnecessary cap risks penalising exactly the arms it
was meant to protect. Truncation remains a section 5 failure if it ever occurs.

Letter alphabet in force: **A-G**, unchanged. All seven verified as bare single-character tokens.

## 2. Section 4 pre-freeze checks

| Check | Result |
| --- | --- |
| Letter A-G single-token under Qwen's tokenizer | **PASS** — all seven seen as bare 1-char tokens (A 207 … G 205); top-1 was a bare letter on 200/200 |
| Tag leakage | **0** on every arm |
| Truncations | **0** on every arm |
| Parse failures | **0** on every arm, after re-running one HTTP 402 credit failure |
| `structured_outputs` honoured live | **200/200** valid standalone JSON on each verbalized arm |
| Q-L logprobs returned live | **200/200** — Parasail does return them; the local-vLLM fallback stays closed |
| Provider pin honoured | **200/200** on every arm; no reroutes |
| Qwen arms non-thinking | **enforced and verified** (0 reasoning tokens) — see DEVIATIONS |
| PC1 dev accuracy spread (J/F1-V/F2-V) | **2.0 points** vs an 8-point threshold → tier-matched framing HOLDS |
| Our own shuffle uniform | **verified**, label × letter-position over 49 cells, p ≈ 0.85 |

## 3. Dev results

| Arm | Accuracy | Raw Brier | Mean top prob | Dev cost |
| --- | --- | --- | --- | --- |
| J | 0.7600 | **0.3543** | 0.831 | $0.0099 |
| F1-V | 0.7800 | 0.3550 | 0.645 | $0.9243 |
| F2-V | 0.7800 | 0.3552 | 0.676 | $0.7329 |
| Q-V | 0.6650 | 0.5184 | 0.775 | $0.0685 |
| Q-L | 0.5600 | 0.6172 | 0.771 | $0.0423 |

## 4. Cost, billed from `usage.cost` (authoritative, not a price table)

Dev total **$1.778**. Projected evaluation run **$17.78**, plus PC2's J re-run $0.10 =
**$17.88**.

A price-table estimate gave $14.21 and was wrong: F2-V bills at exactly 2.00× the registry's
headline rates ($2/M prompt and $12/M completion against a listed $1/M and $6/M). Always bill
from the reported cost.

**BLOCKER: available credit is $1.72 against a $17.88 requirement.** A top-up of at least $16 is
needed, and ~$25 would leave headroom for retries and PC2. This is what caused the single F1-V
HTTP 402 mid-run.

## 5. What the dev data says about the study's claim

J leads on raw Brier by **0.0007** over F1-V and **0.0009** over F2-V. The projected PC2 noise floor
on 2,000 items is ~0.0005 Brier, so both margins sit at the floor. Section 8 requires J to beat both
"intervals excluding zero, margins above J's PC2 floor" — on dev evidence that is not met.

J is also the **most confident** arm (mean top probability 0.831) while both frontier arms hedge
(0.645, 0.676) at equal or better accuracy. Equal accuracy at higher confidence is overconfidence,
which the ECE and calibration-gap columns will quantify.

Dev is 200 items and is not the pre-registered comparison. The direction, however, is "no meaningful
difference" rather than a Jev advantage.

Two further observations worth carrying into the writeup:
- **J collapses on cs.AI**: per-class accuracy 0.353 against 0.765-0.944 everywhere else. Section 2
  predicted this — cs.AI is the top co-tag for all six other classes, making it the structurally
  ambiguous label.
- **C3's mechanism effect points the opposite way to the study's premise.** Holding the model fixed,
  verbalized (Q-V, Brier 0.5184) beats first-token logprobs (Q-L, 0.6172) by ~0.099. On this 27B
  model, asking for a distribution beat reading the logprobs.

## 6. Not yet built

`anchor.py` (arms A, P, P\*) and `analyze.py` (section 6 cross-fitting, section 7 metrics and
bootstrap, section 8 contrasts, figures) are a separate plan, deliberately written AFTER this gate and
BEFORE the evaluation run, so the analysis code is frozen before any evaluation result exists.
