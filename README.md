# Are Jev's probabilities better than a frontier LLM's?

A pre-registered calibration study on arXiv category prediction. 2,000 papers,
eight arms, one question: when a model hands you a number and calls it a
confidence, does the number mean anything?

The protocol was committed before any evaluation item was sent to any model.
Every departure from it since is recorded in [`DEVIATIONS.md`](DEVIATIONS.md)
with a date and a reason.

Full write-up: **<https://blog.shahbandegan.me/llm-confidence/>**

## What it found

Both pre-registered claims failed, and the more interesting result is what
survived anyway.

| Claim ([PROTOCOL.md](PROTOCOL.md) section 8) | Requirement | Result |
| --- | --- | --- |
| "Better out of the box than typical LLM usage" | J beats both frontier arms on raw Brier | **Fails** — loses to both |
| "Calibrated" | J's raw ECE below 0.05 | **Fails** — 0.1296 |

Jev loses the headline metric by roughly 95x its own run-to-run noise floor.
But at a 5% error budget it auto-accepts **28.9%** of the set against GPT-5.6
Terra's 22.7% — about 28% more work cleared at the same error rate.

Both are true because Brier mixes two things. Jev has the **best resolution of
any arm tested** (0.0406, against 0.0342 and 0.0299): it ranks its own
uncertainty better than either frontier model. It also has the **worst
reliability** of the three (0.0215): the numbers it states are further from
what it achieves. Coverage inspects only the accepted set, where its ranking is
strongest. Brier integrates over every item, including the ones it got
confidently wrong.

The mechanism is visible in the raw output. Jev emits probabilities on a
two-decimal grid — 101 distinct values across 14,000 slots, 67% of them exactly
zero, nothing at all between zero and 0.01. When it rules a category out, it
rules it out absolutely — on 85 papers the true label got exactly zero, and on
146 it got 0.01 or less. There is no partial credit on any of them.

Details and figures in [`results/insights.md`](results/insights.md); the full
metric tables are in [`results/analysis.md`](results/analysis.md).

## The arms

| ID | Model | Mechanism | Role |
| --- | --- | --- | --- |
| J | `typesafe/jev-1.13` | Native `Choice` probabilities | Subject |
| F1-V | `anthropic/claude-sonnet-5` | Verbalized JSON distribution | Peer baseline |
| F2-V | `openai/gpt-5.6-terra` | Verbalized JSON distribution | Peer baseline |
| Q-V | `qwen/qwen3.8-27b` | Verbalized JSON distribution | Mechanism contrast |
| Q-L | `qwen/qwen3.8-27b`, same backend | Answer-token logprobs, renormalized | Mechanism contrast |
| A | TF-IDF + logistic regression | Supervised, local | Anchor, not a competitor |
| P | Anchor-pool base rates | Constant | Degenerate reference |
| P* | Evaluation-set base rates | Constant | Oracle — uses test information |

Q-V against Q-L is the within-model contrast: same weights, same backend, same
prompt but the final output instruction. It isolates what the readout mechanism
buys when model quality is held fixed.

## Reproducing it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env        # add an OpenRouter key
.venv/bin/python -m pytest  # 218 tests
```

All five LLM arms route through OpenRouter, so that key is the only credential
needed. Arms A, P and P* are local.

The committed data is enough to reproduce every number and figure without
spending anything — start at step 4. Steps 1–3 are what produced it.

```bash
# 1. Harvest arXiv metadata (~50 MB CSV; already committed)
.venv/bin/python harvest_arxiv.py

# 2. Draw the evaluation / dev / anchor pools from fixed seeds
.venv/bin/python pools.py --check

# 3. Run an LLM arm, then parse its responses into probability vectors
.venv/bin/python run.py   --arm J --pool evaluation
.venv/bin/python parse.py --arm J --pool evaluation

# 4. Fit the local reference arms
.venv/bin/python anchor.py --arm A --pool evaluation

# 5. Recalibrate, score, bootstrap; writes results/analysis.md
.venv/bin/python analyze.py

# 6. Regenerate the figures
.venv/bin/python figures.py
```

Step 3 cost **$18.80** across all five arms at n=2,000, most of it F1-V
($10.12) and F2-V ($7.30); Jev's full 2,000-item pass was $0.0988. Budget for
it before re-running. `run.py` appends to `runs/<arm>_<pool>.jsonl` and resumes
where it left off, so an interrupted run is safe to restart.

Pools are drawn from seeds fixed in `PROTOCOL.md` section 4, and
`protocol.py` verifies the source CSV against its recorded SHA-256 on load, so
steps 2–6 are deterministic. Step 1 is not: re-harvesting hits a moving arXiv
window and will produce a different CSV.

## What's in here

**The record**

| | |
| --- | --- |
| [`PROTOCOL.md`](PROTOCOL.md) | The pre-registration. Frozen before any evaluation item was sent; never edited since. |
| [`DEVIATIONS.md`](DEVIATIONS.md) | Every departure from it, dated, with reasons. |
| [`FREEZE.md`](FREEZE.md) | Prompt and parser state at the freeze, with the dev-set evidence behind it. |
| [`devcheck_report.md`](devcheck_report.md) | Pre-freeze checks on the 200-item dev pool. |

**Code**

| | |
| --- | --- |
| `protocol.py` | Labels, seeds, epsilon, the CSV hash check. Everything else imports it. |
| `harvest_arxiv.py` | arXiv API harvest into `arxiv_last_30_days.csv`. |
| `taxonomy.py`, `pools.py` | Category handling; the evaluation / dev / anchor split. |
| `prompt.py` | Prompt construction and the per-item option shuffle. |
| `run.py` | Calls one arm over one pool. Append-only log, resumable. |
| `parse.py` | Responses to probability vectors, with failure accounting. |
| `anchor.py` | Arms A, P and P*. |
| `analyze.py` | Recalibration, metrics, bootstrap contrasts; writes `results/analysis.md`. |
| `figures.py` | The post's six figures. Recomputes from `preds/` and `runs/`. |
| `devcheck.py` | Pre-freeze dev-pool diagnostics. |

**Data**

| | |
| --- | --- |
| `arxiv_last_30_days.csv` | 9,983 papers, harvested 19 Sept 2026. SHA-256 pinned in `PROTOCOL.md`. |
| `pools/` | The three drawn pools, as JSONL. |
| `runs/` | Every raw API response, one JSON per line, plus endpoint snapshots. |
| `preds/` | Parsed probability vectors (`.npz`) and per-arm parse reports. |
| `probes/` | Endpoint probes from before the run. |
| `results/` | Metrics, figures, and the written findings. |

## Caveats worth reading before citing this

**Jev's training cutoff is assumed, not confirmed.** The evaluation papers were
submitted 5–17 Sept 2026; `typesafe/jev-1.13` was listed 18 Sept 2026 and
resolves to a build stamped 2026-09-17. TypeSafe has not been asked. The study
runs on the assumption that the cutoff precedes September 2026 and says so
rather than burying it — it is the weakest load-bearing assumption here, and
`PROTOCOL.md` section 2 pre-commits that J's results are void if the cutoff
turns out to fall inside the evaluation window.

**The mechanism contrast is an fp8 estimate.** Q-V and Q-L share one quantized
27B backend, because no endpoint offers unquantized weights, logprobs and
structured outputs together. The matching frontier cell does not exist — no
frontier provider exposes answer-token logprobs.

**Latency numbers are not comparable across arms.** The arms ran concurrently
and F1-V was additionally throttled, so the recorded per-item latency reflects
queueing, not single-request response time. See `DEVIATIONS.md`.

**J's endpoint is alpha.** `/api/alpha/decisions` may change without notice.
The saved responses under `runs/` and `probes/` are the durable record of what
the contract was when the study ran.

## License and data

Code and prose are MIT — see [`LICENSE`](LICENSE).

`arxiv_last_30_days.csv` and `pools/*.jsonl` contain arXiv metadata. arXiv
distributes its metadata under CC0 1.0, but the abstracts within it remain
under whatever license each submitting author chose. Treat the corpus as
redistributable for replication, not as public-domain text.

`runs/*.jsonl` holds raw model responses from OpenRouter-routed endpoints,
published under `PROTOCOL.md` section 4's commitment to release them. They have
been scanned for account identifiers; see the 2026-09-20 entry in
`DEVIATIONS.md`.
