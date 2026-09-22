# TypeSafe's launch post vs. our measurements

Source: https://typesafe.ai/blog/introducing-system-one-models-and-jev
Test: 2,000 arXiv papers, 7-way primary-category prediction, pre-registered in `PROTOCOL.md`.
Five of seven checkable claims hold. The two that fail are the two the post leads with.

## Claims that HOLD

**"The model never makes type errors."** 2,000 of 2,000 responses returned a complete, correctly-keyed
7-label distribution. Zero parse failures across 2,200 Jev calls including the dev pass. Meanwhile the
verbalized LLM arms needed constrained decoding to reach parity, and even then 131 of F1-V's and 30 of
F2-V's responses had raw probabilities that did not sum to 1 (minimum observed sum 0.15, meaning
renormalisation scaled that item's whole vector by ~6.7x). **This claim is not just true, it is the
post's strongest and it is under-sold.**

**"Jev achieves similar levels of intelligence on System One tasks compared to existing LLMs."**
Accuracy 0.731 against 0.754 (Claude Sonnet 5) and 0.766 (GPT-5.6 Terra) — a 3.5-point spread, well
inside the 8-point threshold our protocol pre-registered for treating the arms as tier-matched.

**Speed.** Jev's p50 is 0.21–0.22s, inside their stated "70ms–500ms". But the *ratio* is flattered: our
matched frontier arms run at 2.5–3.4s p50, the very bottom of their quoted "3 to 329 seconds", giving
roughly 11–15x rather than 193.6x. (Our latencies were measured under concurrency and are inflated for
all arms; see `DEVIATIONS.md`.)

**Cost.** $0.0988 for 2,000 items against $7.30 (F2-V) and $10.12 (F1-V) — **74x to 102x cheaper**.
Their headline 444.6x compares against different models and settings, but the order of magnitude is real
and it is the most robust economic claim in the post.

**"Calibrated: higher confidence means higher accuracy."** *As literally worded, this holds.* Accuracy
rises monotonically across confidence deciles — 0.380, 0.445, 0.605, 0.625, 0.710, 0.845, 0.875, 0.900,
0.965, 0.960 — with one trivial inversion in the top two. Spearman rho = 0.445, p = 7.5e-98. And on the
Murphy decomposition Jev has the **best resolution of any arm tested** (0.0406 against 0.0342 and
0.0299): it ranks its own uncertainty better than either frontier model. This is a real strength.

## Claims that FAIL

**"All answers are accompanied with calibrated probabilities and confidence scores."**
Not in the standard sense. On its own `confidence` field Jev is overconfident by **+0.109** (mean
confidence 0.840 against accuracy 0.731), ECE **0.1089**; on the probability vector, ECE **0.1296**.
Both exceed our pre-registered 0.05 threshold. Most concretely: **443 items carry a stated confidence of
1.000 and are wrong 3.8% of the time** (3.2% even crediting any category the authors themselves tagged).

The distinction matters and the post blurs it. "Higher confidence means higher accuracy" is a claim about
*ordering*, and Jev satisfies it better than the frontier models do. "Calibrated probabilities" is a
claim about *numbers*, and Jev does not satisfy it. A reader who acts on the second reading — thresholding
at 0.9 and expecting 10% errors — gets roughly 29% errors instead.

**"Even if prompted for a confidence estimate, models tend to be overconfident and inconsistent."**
**Reversed in our data, and the reversal points at Jev.**

| Model | Mean confidence | Accuracy | Gap |
| --- | --- | --- | --- |
| Jev | 0.840 | 0.731 | **+0.109 overconfident** |
| Claude Sonnet 5 | 0.671 | 0.754 | **−0.083 underconfident** |
| GPT-5.6 Terra | 0.704 | 0.766 | **−0.062 underconfident** |

Both frontier models are *under*confident — they hedge more than they need to. Jev is the only
tier-matched arm that overstates its own reliability. The premise motivating the product is, on this
task, backwards.

## The mechanism behind the failure

Jev emits probabilities on a **2-decimal grid**: 101 distinct values across 14,000 slots, smallest
non-zero exactly 0.0100, **67% of slots exactly zero**. It cannot express "unlikely but possible".

- **146 items (7.3%) where it assigns <0.01 to the true label.** Sonnet does this once; Terra 7 times.
- Those items account for ~100% of its deficit against Sonnet and ~77% against Terra.
- **85 of them are categories the authors themselves tagged on the paper.** Jev called an
  author-applied label impossible.
- No recalibration can undo it: temperature scaling is monotone, so a floored zero stays a zero. Even
  flooring at 0.03 — three times Jev's own granularity — leaves it behind both frontier models.

This is what makes the two findings consistent. Jev *ranks* uncertainty better than the frontier models
and *expresses* it worse, so it wins on selective prediction and loses on any proper scoring rule.

## What this means for a buyer

Jev auto-accepts **28.9%** of this workload at a 5% error budget, against 27.9% for Sonnet and 22.7% for
Terra — the best of any arm, and double the supervised in-domain classifier's 14.4%. It also beats that
supervised classifier (TF-IDF + logistic regression on 5,503 same-distribution papers) by 0.060 Brier
while using no training data at all.

So: **use it for routing and thresholded acceptance, where only the ordering matters. Do not consume its
probabilities as probabilities, and never treat its zeros as zeros.** One temperature (T* ≈ 1.64, stable
across all five folds) brings its ECE to 0.0386, inside the threshold and below Terra's post-scaling
0.0519. The model is miscalibrated but trivially calibratable.

## Caveats we hold ourselves to

- One task, seven curated labels, one model version, a single 48-hour window.
- The task sits outside the post's own examples, which are 3-option routing decisions. This is a 7-way
  academic taxonomy whose most ambiguous class is the top co-tag of all six others.
- Label ambiguity is material: crediting any author-tagged category moves accuracy 0.731 → 0.805 and
  ECE 0.1089 → 0.0670. Still failing, but narrowly.
- Every arm under-predicts cs.AI (Jev 0.33x, Sonnet 0.58x, Terra 0.63x), so the hardest class is hard
  for everyone — this is a property of the task, not a Jev defect.
- Jev's training cutoff is undisclosed in the post and unverified by us; the pin resolves to a build
  stamped inside our harvest window.
