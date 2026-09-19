# Protocol: Are Jev's probabilities better than a frontier LLM's?

Pre-registration for a calibration comparison on arXiv category prediction. Committed before any evaluation item is sent to any model. Changes after that go in `DEVIATIONS.md` with a date and a reason.

**Standing assumption:** Jev's training cutoff precedes September 2026. TypeSafe has not confirmed it and we are not waiting on them. See section 2.

## 1. Question

On zero-shot classification with real label ambiguity, are Jev's native probabilities better calibrated and more useful than a frontier LLM?

## 2. Data

**Task.** Title plus abstract, plain text. _"Which arXiv primary category did the authors submit this paper under?"_ Label is the author-chosen primary category.

**Labels (K = 7).** cs.CV, cs.LG, cs.AI, cs.RO, cs.CL, cs.CR, cs.IR. Every arm gets arXiv's official category descriptions verbatim and no base-rate information.

**Source.** `arxiv_last_30_days.csv`, harvested 19 Sept 2026, covering v1 submissions 20 Aug – 17 Sept 2026. 9,983 in-set papers. SHA-256 `b625441d7802d7ec53d7093a2e9e729695f8faef619ec2a82afbd46232c4d9ad`.

**Pools.** Start date 5 Sept 2026.

| Pool            | Rule                                                                                              | Count |
| --------------- | ------------------------------------------------------------------------------------------------- | ----- |
| Evaluation      | N = 2,000, uniform random (seed 20260920) from the 4,280 in-set papers on or after the start date | 2,000 |
| Dev             | 200 in-set papers before the start date (seed 20260921), for prompt and parser debugging          | 200   |
| Anchor training | All remaining in-set papers before the start date                                                 | 5,503 |

Expected evaluation counts: cs.CV 442, cs.LG 429, cs.AI 343, cs.RO 292, cs.CL 291, cs.CR 160, cs.IR 44 (rounded).

**Ambiguity.** 34.3% of evaluation papers cross-list two or more of the 7 labels. cs.AI is the top co-tag for all six other classes (cs.LG 29%, cs.CL 29%, cs.CR 24%, cs.IR 24%, cs.CV 20%, cs.RO 14%). This is the aleatoric uncertainty the study needs.

**Class mix is not stationary.** cs.RO runs ~8% through August and reaches 23% on 16–17 Sept — 8.1% of the anchor pool against 14.6% of the evaluation pool, a 1.80x increase. No other class moves comparably. Probably a submission deadline; not established. Consequences, pre-declared: arms A and P inherit a stale prior, so P's top-label ECE is 0.023 by construction and A under-predicts cs.RO. Arm P\* separates the two. The zero-shot arms are affected equally — no model's priors anticipate a deadline ramp — so this is shift that calibration should absorb, not a confound between arms.

**Contamination.** Not measured, argued. F1-V, F2-V and the Qwen arms were listed 30 June – 14 Aug 2026, months before the evaluation window. For J the argument is an assumption: `typesafe/jev-1.13` was listed 18 Sept 2026, a day before the harvest, and a thirteenth point release days after launch suggests rapid iteration that may include retraining. We assume the cutoff precedes September 2026 and run on that basis. It is unconfirmed, it is the weakest load-bearing assumption in the protocol, and the post states it as an assumption rather than a fact. If TypeSafe later reports a cutoff inside the evaluation window, results for J are void — recorded in `DEVIATIONS.md`, not softened into a caveat.

**Integrity.** No duplicate IDs, no abstract under 100 characters. Median 191 words, p95 261.

## 3. Arms

| ID   | Model                                    | Mechanism                                         | Role                          |
| ---- | ---------------------------------------- | ------------------------------------------------- | ----------------------------- |
| J    | `typesafe/jev-1.13` (listed 2026-09-18)  | Native `Choice` probabilities                     | Subject                       |
| F1-V | `anthropic/claude-sonnet-5` (2026-06-30) | Verbalized JSON distribution                      | Peer baseline                 |
| F2-V | `openai/gpt-5.6-terra` (2026-07-09)      | Verbalized JSON distribution                      | Peer baseline                 |
| Q-V  | `qwen/qwen3.8-27b` (2026-08-14)          | Verbalized JSON distribution                      | Mechanism contrast            |
| Q-L  | `qwen/qwen3.8-27b`, same backend as Q-V  | Answer-token logprobs, renormalized over 7 labels | Mechanism contrast            |
| A    | TF-IDF + multinomial logistic regression | Supervised, local                                 | Anchor, not a competitor      |
| P    | Anchor-pool base rates, constant         | —                                                 | Degenerate reference          |
| P\*  | **Evaluation-set** base rates, constant  | —                                                 | Oracle, uses test information |

**Q-V vs Q-L is within-model:** same model, same items, same prompt but the final output instruction. The matching frontier cell is not run — no frontier provider exposes answer-token logprobs (verified against the OpenRouter registry on 19 Sept 2026, all 7 Terra and all 10 Sonnet endpoints). So the mechanism effect is estimated on a 27B model only.

**Jev's pin.** `typesafe/jev-1.13` is unlisted in `/api/v1/models` but resolves on direct lookup; 1.12 and 1.14 both 404 and there is no `latest` alias, so the pin is exact. Its record has empty `supported_parameters` (no seed, temperature, reasoning, or response_format), output modality `decisions`, 32K context, and zero-priced completion tokens against $0.042/M prompt. That empty list is what a single-pass typed-decision model exposes, and it drives the asymmetries in section 4.

**Training cutoff: assumed before September 2026, unconfirmed.** OpenRouter reports only a listing date and TypeSafe has not been asked. See section 2.

**Terra Pro is not used.** `openai/gpt-5.6-terra-pro` is identical on price, context and parameters. The choice of the base model is arbitrary and recorded here so it isn't made after seeing results.

## 4. Settings and controls

- **Seeds**, all derived from master 20260919 and fixed here: evaluation sample 20260920, dev sample 20260921, option shuffle 20260922, folds 20260923, bootstrap 20260924. API-level `seed` is 20260919 wherever the provider accepts one.
- **Option order** is shuffled per item (seed 20260922), same permutation for every arm including J. This also controls letter-position bias in Q-L.
- **Reasoning at provider defaults for F1-V and F2-V.** The reasoning parameter is omitted so each model runs as it ships — on `claude-sonnet-5` that means adaptive thinking on at effort `high`. The defaults in force are recorded at run time. This is what C1's "as they ship" means, and it removes the obvious objection that the baselines were handicapped. The Qwen arms stay non-thinking: Q-L reads the first generated token's logprobs, which reasoning tokens would displace, and Q-V is matched to Q-L. J has no reasoning parameter to set.
- **Same prompt content** for every LLM arm: same instructions, same category descriptions.
- **Structured outputs** on every verbalized arm. Without constrained decoding, parse-failure rates differ across arms for reasons unrelated to calibration, and failures are scored as uniform (section 5), so that difference would hit the primary metric directly. J is exempt — its output is `decisions`, not text, so there is nothing to parse. The asymmetry favours J on failure counts and is reported, not corrected. Confirm `structured_outputs` is listed on each pinned endpoint before the run.
- **Pin the provider on every OpenRouter arm**, with `allow_fallbacks: false`, and record provider and quantization. Mandatory for the Qwen arms — quantization changes the distribution Q-L reads, and a silent reroute would draw Q-V and Q-L from different backends, destroying the one property that makes the contrast clean. Default is the first-party `Alibaba` endpoint (logprobs, unquantized); local serving is preferable where available. The frontier arms are pinned for the same reason in weaker form: routing changes structured-output support and run-to-run variance.
- **Determinism differs by arm.** J exposes no parameters and F1-V supports neither `temperature` nor `seed`, so neither can be made reproducible. PC2 measures what that costs on J. F2-V takes `seed`; Qwen takes both, temperature 0 for Q-V and unset for Q-L.
- **No anti-reasoning instructions.** Nothing in any prompt tells a model not to think or not to reason — on Claude models that makes internal-tag leakage into visible text worse. A generic "do not include internal or system XML tags" instruction goes to all arms for symmetry. Leakage is measured on the dev set before freezing; with thinking on and structured outputs in force it should be near zero.
- **Q-L extraction.** Single letter A–G indexing that item's shuffled options; probabilities from the first generated token's logprobs, renormalized over the 7 label tokens. Each letter must be a single token under the model's tokenizer, verified on the dev run, with a different alphabet substituted if not. No temperature, top-p or top-k. Any label missing from the returned top-k gets probability 0 before the floor.
- Prompts and parsers are tuned on the dev set only and frozen by commit. All arms run inside one 48-hour window. Raw responses saved as JSONL and published.
- **Token budget.** `max_tokens` is capped, but for F1-V and F2-V the cap must clear observed dev-run thinking plus answer length with margin. Truncation is scored as a failure (section 5), so a cap set for non-thinking output would silently penalise exactly the arms now running with reasoning on. Dev-run usage is extrapolated to a full bill before the evaluation run. Batch endpoints where available.

## 5. Probability handling (identical for every arm)

1. Parse into a K-vector. The JSON schema requires all 7 keys; Q-L uses the rule above.
2. A failure is a parse error, refusal, truncation, non-finite or negative value, a letter outside A–G, or timeout after 3 retries. Failures are scored as uniform and counted per arm. Never dropped.
3. Floor at epsilon = 0.001 and renormalize — every arm, including J, Q-L and A. The floor is re-applied after temperature scaling, so log-space operations are defined everywhere under the same rule.

## 6. Recalibration: 5-fold cross-fitting

Evaluation items split into 5 folds, stratified by label, seed 20260923. Per arm and fold, fit on the other four and apply frozen to the held-out fold:

- **Temperature T**, minimizing NLL of q proportional to p^(1/T), T in [0.05, 20].
- **Acceptance threshold tau for 5% error.** Sort training-fold items by top-label confidence, take the largest k whose top-k error is at most 5%, set tau to the k-th confidence. If k < 50, tau is infinite and coverage is zero. Fit separately for raw and scaled.

Every metric is computed on all evaluation items using out-of-fold parameters. Raw metrics involve no fitting.

## 7. Metrics

Each reported raw and temperature-scaled.

1. **Multiclass Brier score (primary).**
2. **Calibration gap** = raw Brier − scaled Brier. How much an arm lost to miscalibration that one scalar fixes.
3. **Accuracy**, for context.
4. **Top-label ECE**, 15 equal-mass bins, reliability diagram per arm. Illustrative.
5. **Coverage at 5% error** at the frozen tau, next to realized held-out error. Full risk-coverage curve plotted.

P is in the table because it should score near-zero ECE and near-zero calibration gap while being useless — no resolution, zero coverage, worst Brier. That is the concrete reason the primary metric is a proper scoring rule and not ECE.

**Inference.** Paired bootstrap over items, 2,000 resamples, seed 20260924, percentile 95% intervals on each arm's difference from J. Out-of-fold predictions held fixed within the bootstrap.

**PC1 — capability matching.** If accuracy spread across J, F1-V and F2-V exceeds 8 points, the tier-matched framing is dropped and the comparison is reported through the Brier decomposition instead. Checked on dev first, reported on the evaluation set whatever it shows. Q-V and Q-L are excluded — they are mechanism arms.

**PC2 — is J stable?** J is re-run once on all 2,000 evaluation items ($0.05). The Brier difference between the two runs is J's run-to-run noise floor; any cross-arm margin smaller than it supports no claim, whatever the bootstrap says. The bootstrap resamples items against one fixed set of predictions, so it does not see generation variance at all, and J exposes no `seed` or `temperature` to remove it. F1-V is not re-run — its floor is unmeasured and, with thinking on and no seed, presumably larger than J's; that is stated rather than measured.

**Secondary, descriptive.** Failure, truncation and tag-leakage counts per arm; reasoning tokens consumed per item by F1-V and F2-V; latency p50/p95 from Toronto on a 200-item sequential subsample; cost per 1,000 items; primary metric recomputed at epsilon = 0.0001 and 0.01.

## 8. Contrasts and how they will be read

| #   | Contrast                                  | Question                                                                |
| --- | ----------------------------------------- | ----------------------------------------------------------------------- |
| C1  | J vs F1-V, J vs F2-V, raw                 | Is Jev better than frontier LLMs as they ship, reasoning included?           |
| C2  | Same after scaling, plus calibration gaps | One-scalar fix, or a real difference in uncertainty ranking?            |
| C3  | Q-L vs Q-V                                | What does the mechanism buy, holding the model fixed?                   |
| C4  | J vs Q-L, raw                             | Does Jev beat a native-probability baseline, not just a verbalized one? |

Decision rules, fixed now:

- **"Better out of the box than typical LLM usage"** holds if J beats both F1-V and F2-V on raw Brier, intervals excluding zero, margins above J's PC2 floor.
- **"Calibrated"** holds if J's raw ECE is below 0.05 and its calibration gap interval includes values near zero. The cutoff is a convention and is described as one.
- **Attribution.** If C3's mechanism effect is comparable to J's C1 margin, C1 is reported as consistent with mechanism alone and no model-quality claim is made. If J's margin substantially exceeds it, the excess is the part not explained by mechanism — an estimate carrying the transfer caveat, not a clean decomposition.
- **C4 is asymmetric.** A J loss is decisive: reported as "Jev's advantage is its probability mechanism, not its model," in those words. A J win is weak — Q-L is a 27B model, so a win reintroduces the model-quality confound. Reported next to C1 with equal prominence either way.
- **A, P and P\* are excluded from all of the above.** Reference points, reported in every table, never competitors. P\* is labelled an oracle wherever it appears.

All four contrasts are reported whatever they show. No multiplicity correction; this is the complete set of tests.

## 9. Limitations, stated in the post

- **Mechanism confound reduced, not removed.** C3 is measured on a 27B open-weights model because no frontier provider exposes logprobs. Transfer is untested, so attribution of J's margin stays partly open.
- **Contamination assumed away for J.** Its cutoff is unknown and unchecked; we assume it predates September 2026. 1.13 was listed a day before the harvest. This is the weakest load-bearing assumption here and is presented as an assumption, not a finding.
- **Single-pass against reasoning.** F1-V and F2-V think; J cannot. This is deliberate — it is the comparison a buyer faces — but it means C1 is not a like-for-like test of probability mechanisms, and it favours the baselines. C4, against non-thinking Q-L, is the matched-conditions contrast.
- **Constrained decoding**, so the verbalized arms are not literally unconstrained "ask for JSON" usage. Traded to keep parse failures out of the primary metric.
- **Curated label set and one task.** Seven labels, not a random sample of arXiv's 147, with stat.ML excluded as undecidable. cs.IR lands at ~44 items, so no per-class claims. Nothing generalizes until a second dataset agrees.
- **Non-stationary class mix**, degrading the two fitted reference arms. Stated, not corrected away.
- **Reference-arm asymmetry.** A has in-domain training data and P and P\* see base rates the zero-shot arms are denied. Beating P is necessary for an arm to be useful, not evidence of quality.
- **Quantization.** C3's estimate is specific to one pinned backend as well as one model.
- **No capability ceiling.** Whether J's advantage survives a stronger model is not tested.

## 10. Part two

Same data rules, metrics and probability handling. Listed now so part two is an extension, not a reaction to part one's results.

- **Vendor adapter arms** — both frontier models through TypeSafe's System One adapter, run alongside the self-written prompts, never replacing them.
- **Capability ceiling** — one more capable frontier arm.
- **Frontier logprob arm**, if any provider starts exposing answer-token logprobs.
- **Reasoning sensitivity** — F1-V and F2-V re-run at minimum reasoning, to measure what thinking bought them in part one.
- **Unconstrained decoding** — one verbalized arm without structured outputs, to measure what the constraint cost.
- **Second task.**
