# What the experiment found

All figures: 2,000 evaluation items, pre-registered in `PROTOCOL.md` before any item was sent.
Cross-arm contrasts use paired bootstrap, 2,000 resamples, seed 20260924. J's measured run-to-run
noise floor (PC2) is **0.00046** Brier; every margin below is stated against it.

## 1. The headline: both pre-registered claims fail

| Claim (section 8) | Requirement | Result |
| --- | --- | --- |
| "Better out of the box than typical LLM usage" | J beats BOTH frontier arms on raw Brier | **FAILS** — J loses to both |
| "Calibrated" | J's raw ECE below 0.05 | **FAILS** — 0.1296, or 0.1089 on its own `confidence` field |

- C1 raw: J vs F1-V **−0.0441** [−0.0619, −0.0267]; J vs F2-V **−0.0457** [−0.0623, −0.0306].
  Both exclude zero; both ~95x the PC2 floor.
- C2 scaled: J vs F1-V **−0.0267** [−0.0396, −0.0129]; J vs F2-V **−0.0236** [−0.0354, −0.0121].
  One scalar halves the deficit; it does not close it.
- PC1 accuracy spread across J/F1-V/F2-V is **3.50 points**, under the 8-point threshold, so the
  tier-matched framing stands and the comparison is legitimate.

## 2. But J wins the metric a practitioner actually buys

Coverage at a 5% error budget — what fraction you can auto-accept:

| Arm | Coverage (raw) | Realized error | Coverage (scaled) |
| --- | --- | --- | --- |
| **J** | **28.9%** | 5.7% | **32.5%** |
| F1-V | 27.9% | 5.2% | 28.0% |
| F2-V | 22.7% | 5.3% | 22.1% |
| A (supervised) | 14.4% | 7.3% | 16.6% |
| Q-V | 5.6% | 7.1% | 3.3% |
| Q-L | 2.5% | 7.8% | 2.5% |

On J's own `confidence` field, coverage is **32.8%**. J accepts ~28% more work than F2-V at the same
error budget. Its realized error slightly overshoots (5.7% against a 5% target) where the frontier
arms land at 5.2% and 5.3%.

## 3. Why both things are true at once

Murphy decomposition of top-label Brier:

| Arm | Reliability (miscalibration, fixable) | Resolution (real skill, unfixable) |
| --- | --- | --- |
| **J** | **0.0212** (worst) | **0.0404** (best) |
| F1-V | 0.0088 | 0.0349 |
| F2-V | 0.0070 | 0.0302 |

**J ranks items better than either frontier model and calibrates worse.** Coverage inspects only the
accepted set, where J's ranking is strongest. Brier integrates over every item, including the ones it
got catastrophically wrong.

## 4. The mechanism: hard zeros

J emits probabilities on a **2-decimal grid** — 101 distinct values across 14,000 slots, smallest
non-zero exactly **0.0100**, **67% of slots exactly zero**, mean 4.7 zeros of 7 per item. F1-V reaches
0.0005, F2-V 0.0001, with 15.5% and 9.9% exact zeros.

J cannot express "unlikely but possible." Consequences:

- **146 items (7.3%) where J assigns <0.01 to the true label.** F1-V does this on **1** item, F2-V on 7.
- Those 146 items account for **~100% of J's deficit against F1-V** and ~77% against F2-V. On the other
  1,854 items J is level with F1-V (+0.0006) and slightly behind F2-V (−0.0115).
- **105 of the 146 are cs.AI**, the class section 2 identified as structurally ambiguous.
- **85 items where J assigned <0.01 to a category the authors themselves tagged on the paper.**

**No rescaling can fix this.** Temperature scaling is monotone: `p^(1/T)` of a floored zero stays near
zero. On those 146 items scaling lifts p(true) only from 0.0047 to 0.027. The information is destroyed
at emission time, which is exactly why J's resolution advantage never cashes in.

**And it is not an artifact of our epsilon.** Flooring J's zeros at 0.03 — thirty times the
pre-registered value, three times J's own granularity — still leaves J at 0.3947 against F1-V's 0.3613
and F2-V's 0.3597.

## 5. J is miscalibrated but eminently calibratable

| Arm | Raw ECE | Scaled ECE |
| --- | --- | --- |
| J | 0.1296 (fails) | **0.0386 (passes)** |
| F1-V | 0.0830 | 0.0219 |
| F2-V | 0.0660 | **0.0519 (fails)** |

A single temperature (T\* ≈ 1.64, stable across all five folds: 1.646, 1.649, 1.642, 1.632, 1.649)
brings J's ECE inside section 8's cutoff — and below F2-V's. The practical advice is not "don't use it",
it is "recalibrate it, and never consume its zeros as zeros."

## 6. Direction of error is opposite across model families

| Arm | Mean confidence | Accuracy | Gap | T\* |
| --- | --- | --- | --- | --- |
| J | 0.861 | 0.731 | **+0.130** | 1.64 (needs softening) |
| Q-L | 0.778 | 0.597 | +0.181 | 1.72 |
| Q-V | 0.781 | 0.671 | +0.110 | 1.23 |
| F1-V | 0.671 | 0.754 | **−0.083** | 0.75 (needs sharpening) |
| F2-V | 0.704 | 0.766 | −0.062 | 0.89 |

The frontier models are **under**confident; every non-frontier arm is **over**confident. J is
overconfident in every confidence band, worst in the middle (+0.205 at 0.80–0.95), and still +0.073 at
0.95+. **400 items at stated confidence 1.000 are wrong 3.5–4% of the time** (3.2% even crediting any
author-tagged category).

## 7. J beats a supervised in-domain classifier

Arm A is TF-IDF + multinomial logistic regression fit on 5,503 same-distribution arXiv papers. J,
zero-shot with no training data:

- accuracy **0.731 vs 0.708**
- Brier **0.4054 vs 0.4657** — J better by **0.0602**, *larger* than the margin by which it loses to the
  frontier models
- coverage **28.9% vs 14.4%** — double

This is a real capability result and belongs alongside the negative findings.

## 8. C3: the mechanism result inverts the study's premise

Same model, same backend, same prompt; only the output instruction differs:

- Q-V (verbalized JSON) Brier **0.5160**, accuracy 0.671, coverage 5.6%
- Q-L (first-token logprobs) Brier **0.5781**, accuracy 0.597, coverage 2.5%
- Difference **+0.0621** [+0.0439, +0.0797], excludes zero, replicating the dev estimate of +0.0988

**Asking a model to write a distribution beat reading its logprobs.** The intuition that native
probabilities are inherently better calibrated does not hold on this model. Because J has no positive
C1 margin, section 8's attribution logic does not apply — there is nothing to attribute.

## 9. C4: J beats the native-probability baseline

Q-L − J = **+0.1727** [+0.1468, +0.1963]. Section 8 pre-declared a J win here weak evidence, since
Q-L is a 27B model, so this reintroduces the model-quality confound rather than resolving it.

## 10. The reference arms vindicate the design

Section 7 predicted P would "score near-zero ECE and near-zero calibration gap while being useless."
Measured: P's raw **ECE 0.0306** — second best of any arm — with a **0.0014** calibration gap, a
constant argmax for all 2,000 items, **0% coverage**, and the worst Brier at 0.8315. The oracle P\*,
which cheats by using evaluation-set base rates, buys only 0.0050 Brier over the stale prior.

**Anyone evaluating on ECE alone would have ranked a constant vector second.** That is the concrete
argument for a proper scoring rule as the primary metric, and this data makes it.

## 11. Universal cs.AI aversion — a task property, not a J defect

Prediction/truth ratio (1.00 = unbiased), cs.AI base rate 0.186:

| Arm | cs.AI ratio | mean p(cs.AI) |
| --- | --- | --- |
| J | 0.33x | 0.072 |
| Q-V | 0.33x | 0.093 |
| Q-L | 0.33x | 0.086 |
| F1-V | 0.58x | 0.147 |
| F2-V | 0.63x | 0.127 |

**Every arm** under-predicts cs.AI. J is most extreme but this is the task, not the model — which
protects the comparison from an unfairness charge. J's *distinctive* error is over-predicting cs.CL at
**1.49x** where every other arm sits near 1.00: on true-cs.AI items J's top prediction is cs.CL (117)
then cs.LG (113), with cs.AI only third (93), while both frontier arms rank cs.AI first.

## 12. Honest limitations

- **No calibration claim located.** TypeSafe's published docs and the OpenRouter model record contain no
  claim of calibration, no definition of `confidence`, and no stated limitations. Any
  claim-refutation framing needs the actual claim quoted first.
- **Label ambiguity is material.** 28% of J's errors are on categories the authors themselves tagged.
  Crediting any tagged category: accuracy 0.731 → 0.805, confidence gap +0.109 → +0.035, ECE 0.1089 →
  **0.0670**. Still above 0.05, but narrowly. Publish all three numbers.
- **The task is plausibly outside Jev's design envelope.** TypeSafe's own examples are 3-option routing
  decisions ("which team should handle this"). This is a 7-way academic taxonomy where the ambiguous
  class is the top co-tag of all six others.
- **One task, seven curated labels, one model version.** Nothing generalizes until a second dataset agrees.
- **Prompt parity holds in what was sent, not what was read.** F1-V received ~1,900 prompt tokens against
  J's 1,175 for byte-identical message content — the provider injects a tool definition for structured
  output.
- **Jev's training cutoff is unverified**, as section 2 states, and the pin resolves to a build stamped
  inside the harvest window.
