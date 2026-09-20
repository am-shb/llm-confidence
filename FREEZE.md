# Freeze record: prompts and parsers, ahead of the evaluation run

**No evaluation-pool item has been sent to any model.** Every number below comes from the
200-item dev pool, which `PROTOCOL.md` section 4 designates for tuning and debugging and which is
disjoint from the 2,000-item evaluation pool.

Frozen at commit `fba0dbe` (the freeze commit itself; see the note on rewritten hashes below) on branch `jev-pipeline-dev-freeze`.
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

Caveat on how that is known: quantization parity is **evidenced, not enforced**. Both arms pin the
provider, but the request does not pin quantization (OpenRouter accepts `provider.quantizations`; it
is not used). The fp8 figures above come from two registry snapshots taken about ninety seconds apart.
`snapshot_endpoint` now fails the run if an arm's recorded quantization changes between runs, which
closes the gap going forward but does not retroactively harden the dev figures.

`max_tokens` is NOT set on any arm. It was in the plan as a guard against truncation, but zero
truncations occurred on 1,000 dev calls, and an unnecessary cap risks penalising exactly the arms it
was meant to protect. Truncation remains a section 5 failure if it ever occurs.

Letter alphabet in force: **A-G**, unchanged. All seven verified as bare single-character tokens.

## 2. Section 4 pre-freeze checks

| Check | Result |
| --- | --- |
| Letter A-G single-token under Qwen's tokenizer | **PASS** — all seven seen as bare 1-char tokens (A 207 … G 205); top-1 was a bare letter on 200/200 |
| Tag leakage | **0** on every arm — but see note: J's zero is structural, not measured |
| Truncations | **0** on every arm — but see note: J's zero is structural, not measured |
| Parse failures | **0** on every arm, after re-running one HTTP 402 credit failure |
| `structured_outputs` honoured live | **200/200** valid standalone JSON on each verbalized arm |
| Q-L logprobs returned live | **200/200** — Parasail does return them; the local-vLLM fallback stays closed |
| Provider pin honoured | **200/200** on every arm; no reroutes |
| Qwen arms non-thinking | **enforced and verified** (0 reasoning tokens) — see DEVIATIONS |
| PC1 dev accuracy spread (J/F1-V/F2-V) | **2.0 points** vs an 8-point threshold → tier-matched framing HOLDS |
| Our own shuffle uniform | **verified**, label × letter-position over 49 cells, p ≈ 0.85 |

Note on arm J's zeros: the leakage check reads `response.choices[0].message.content`, a path J's
decisions body has no analogue for, and `_parse_decisions` inspects no finish reason. So J's tag-leakage
and truncation counts are **structurally incapable of being non-zero** rather than measured as zero.
This is harmless in substance — J returns a typed decision and emits no free text in which a tag could
leak, and no text to truncate — but the two cells above should not be read as evidence about J.

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

J leads on raw Brier by **0.0007** over F1-V and **0.0009** over F2-V.

The PC2 noise floor is **~0.0005 Brier, extrapolated from five identical calls on a single dev item**
(per-item Brier sd 0.0129, divided by sqrt(2000) and scaled for a two-run difference). That is a
sanity estimate, NOT a measurement: per-item generation noise almost certainly scales with item
ambiguity, and five calls on one item is a thin basis for the number that decides whether a 0.0007
margin means anything. PC2 measures it properly on all 2,000 items, and section 8's rules are applied
to THAT figure, not to this one.

Taking the estimate at face value, both margins sit at the floor, and section 8's requirement that J
beat both "intervals excluding zero, margins above J's PC2 floor" is not met on dev evidence.

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

## 7. Note on commit hashes

Git history was rewritten once, on 2026-09-20, before this repository's first push, to remove an
OpenRouter account identifier from two intermediate commits' copies of `probes/jev_probe.json`. The
final content was already clean — `f205c90` had redacted it — so the rewrite altered only historical
blobs and left every tree's final state byte-identical.

The rewrite changed every commit hash from the probe commit onward. Hashes cited in commit messages
written before 2026-09-20 therefore refer to pre-rewrite commits and will not resolve. The mapping for
the ones referenced anywhere in this record:

| Cited (pre-rewrite) | Actual (post-rewrite) | Commit |
| --- | --- | --- |
| `65c0f7d` | `fba0dbe` | Freeze prompts and parsers |
| `b0bc052` | `8389753` | Re-run the F1-V credit failure |
| `e4b0110` | `493cc10` | Fix wave from the final review |
| `79583a0` | `074efc0` | Dedup prefers ok over failure |

`PROTOCOL.md` was not touched by the rewrite and remains byte-identical to its pre-registration commit,
which is the property that matters for the study: the pre-registration's content, not its hash, is what
the results are checked against.
