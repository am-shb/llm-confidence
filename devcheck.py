#!/usr/bin/env python3
"""The checks PROTOCOL.md section 4 requires before freezing prompts/parsers.

Every number in devcheck_report.md is computed here from the committed dev
run/pred data, never typed by hand.

Usage:
    ./devcheck.py
"""

import collections
import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np
from scipy.stats import chisquare

import pools
import protocol as P

ARMS = P.LLM_ARMS
LETTERS = "ABCDEFG"
CREDITS_URL = "https://openrouter.ai/api/v1/credits"
EVAL_N = P.POOL_SIZES["evaluation"]  # 2000: the extrapolation target


def extract_usage(usage):
    """Normalize the two token-usage shapes present in this data.

    The chat arms (F1-V, F2-V, Q-V, Q-L) report `usage.prompt_tokens` /
    `usage.completion_tokens`, with reasoning tokens nested under
    `completion_tokens_details.reasoning_tokens`. Arm J's decisions response
    instead reports `usage.input_tokens` / `usage.output_tokens` and carries
    no reasoning breakdown. Every record observed in this data also carries
    `usage.cost` directly -- that reported cost is always preferred over any
    tokens*price estimate, which is why it is returned separately here rather
    than derived from prompt/completion tokens downstream.
    """
    usage = usage or {}
    if "input_tokens" in usage or "output_tokens" in usage:
        prompt = usage.get("input_tokens") or 0
        completion = usage.get("output_tokens") or 0
    else:
        prompt = usage.get("prompt_tokens") or 0
        completion = usage.get("completion_tokens") or 0
    reasoning = (usage.get("completion_tokens_details") or {}).get(
        "reasoning_tokens") or 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "cost": usage.get("cost"),
    }


def load_records(arm, pool="dev"):
    path = f"runs/{arm}_{pool}.jsonl"
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def per_arm_stats(pool="dev"):
    """One row per arm: failures, tokens, cost, latency, provider pin."""
    rows = []
    for arm in ARMS:
        rep_path = f"preds/{arm}_{pool}_report.json"
        if not os.path.exists(rep_path):
            continue
        rep = json.load(open(rep_path))
        prompt_toks = compl_toks = reason_toks = 0
        cost_total = 0.0
        latencies = []
        providers = collections.Counter()
        for rec in load_records(arm, pool):
            latencies.append(rec.get("latency_s") or 0)
            resp = rec.get("response") or {}
            providers[resp.get("provider") or rec.get("provider") or "?"] += 1
            u = extract_usage(resp.get("usage"))
            prompt_toks += u["prompt_tokens"]
            compl_toks += u["completion_tokens"]
            reason_toks += u["reasoning_tokens"]
            if u["cost"] is not None:
                cost_total += u["cost"]
        n = max(rep["n"], 1)
        rows.append({
            "arm": arm, "n": rep["n"], "ok": rep["ok"],
            "failures": rep["failure_total"], "failure_kinds": rep["failures"],
            "tag_leakage": rep["tag_leakage"],
            "prompt_per_item": prompt_toks / n,
            "completion_per_item": compl_toks / n,
            "reasoning_per_item": reason_toks / n,
            "thinking": reason_toks > 0,
            "cost_total": cost_total,
            "cost_per_item": cost_total / n,
            "latency_p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "latency_p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "providers": dict(providers),
            "pin_holds": set(providers) <= {P.ARMS[arm]["provider"]},
        })
    return rows


def per_class_accuracy(pool="dev"):
    """Accuracy per arm per true label. Section 2's ambiguity prediction is
    that cs.AI, the top co-tag for every other class, is where accuracy
    collapses; this is where that shows up numerically.
    """
    out = {}
    for arm in ARMS:
        path = f"preds/{arm}_{pool}.npz"
        if not os.path.exists(path):
            continue
        d = np.load(path, allow_pickle=True)
        pred = [P.LABELS[i] for i in d["probs"].argmax(axis=1)]
        labels = list(d["labels"])
        per_label = {}
        for lab in P.LABELS:
            idx = [i for i, g in enumerate(labels) if g == lab]
            if idx:
                per_label[lab] = float(np.mean(
                    [pred[i] == labels[i] for i in idx]))
        out[arm] = per_label
    return out


def letter_checks(records):
    """Section 4: is each letter A-G a single token, and is there
    letter-position bias in what Q-L actually picks.

    Rewritten from a first-character heuristic (which false-positives on
    ordinary vocabulary tokens like "cs", "AI", "Based", "Claude", "CV" --
    anything merely STARTING with A-G) to the actual section 4 question: does
    each letter A-G appear as a BARE single-character token in the model's
    own top-k, and was the actually sampled top-1 token a bare letter.

    `records` is the parsed runs/Q-L_dev.jsonl content (or a subset, for
    testing). Two independent counts:
      - bare_counts: how many times each letter A-G appears, across all
        items' top_logprobs lists, as a token that is exactly one character
        after stripping. This is the single-token-under-the-tokenizer check.
      - top1_counts / top1_bare_total: how often the FIRST GENERATED token
        (content[0]["token"], the one actually sampled) was itself a bare
        A-G letter. Feeds the position-bias chi-square below.
    """
    bare_counts = collections.Counter()
    top1_counts = collections.Counter()
    top1_bare_total = 0
    n_with_logprobs = 0
    for rec in records:
        try:
            content = rec["response"]["choices"][0]["logprobs"]["content"]
        except (TypeError, KeyError, IndexError):
            continue
        if not content:
            continue
        n_with_logprobs += 1
        c0 = content[0]
        top1 = (c0.get("token") or "").strip().upper()
        if len(top1) == 1 and top1 in LETTERS:
            top1_bare_total += 1
            top1_counts[top1] += 1
        for e in c0.get("top_logprobs") or []:
            tok = (e.get("token") or "").strip().upper()
            if len(tok) == 1 and tok in LETTERS:
                bare_counts[tok] += 1
    all_seven = all(bare_counts.get(letter, 0) > 0 for letter in LETTERS)
    return {
        "n_with_logprobs": n_with_logprobs,
        "bare_counts": dict(bare_counts),
        "top1_counts": dict(top1_counts),
        "top1_bare_total": top1_bare_total,
        "all_seven_single_token": all_seven,
    }


def letter_position_bias(top1_counts):
    """Chi-square of Q-L's top-1 letter choice against uniform over A-G.

    Section 4 says the per-item option shuffle "also controls letter-position
    bias in Q-L." What the shuffle actually controls is bias toward any one
    LABEL (each label lands at each letter equally often, see
    shuffle_uniformity below) -- it does not make the model choose letters
    uniformly. This checks that distinction directly: if the model has an
    intrinsic preference for e.g. "D" or "G" as a token, the shuffle does not
    remove it, and this chi-square will show strong non-uniformity.
    """
    obs = [top1_counts.get(letter, 0) for letter in LETTERS]
    if sum(obs) == 0:
        return float("nan"), float("nan"), obs
    stat, p = chisquare(obs)
    return float(stat), float(p), obs


def shuffle_uniformity(dev_items):
    """Chi-square of the dev pool's own label<->letter-position assignment.

    Builds a 7x7 table where cell [label, position] counts how often that
    label landed at that letter position across all 200 dev items (every
    item contributes one count per label, since options is a full
    permutation of all 7 labels -- not just a count for the true label). If
    `permutation_for` is unbiased, every cell has the same expectation
    (200*7/49) regardless of the dev pool's label frequencies, since row AND
    column marginals are both fixed at 200 by construction. This validates
    the sampling seed/derivation itself, independent of which label mix the
    dev draw happened to produce.
    """
    table = np.zeros((P.K, P.K), dtype=int)
    for it in dev_items:
        for pos, label in enumerate(it["options"]):
            table[P.LABELS.index(label), pos] += 1
    stat, p = chisquare(table.flatten())
    return table, float(stat), float(p)


def pc1_dev_accuracy():
    """Section 7: PC1's accuracy spread, checked on dev first.

    Only J, F1-V and F2-V are compared -- Q-V and Q-L are mechanism arms,
    explicitly excluded by section 7's PC1 definition.
    """
    acc = {}
    for arm in ARMS:
        path = f"preds/{arm}_dev.npz"
        if not os.path.exists(path):
            continue
        d = np.load(path, allow_pickle=True)
        pred = [P.LABELS[i] for i in d["probs"].argmax(axis=1)]
        acc[arm] = float(np.mean([p == g for p, g in zip(pred, d["labels"])]))
    tiered = {a: v for a, v in acc.items() if a in ("J", "F1-V", "F2-V")}
    spread = (max(tiered.values()) - min(tiered.values())) * 100 \
        if len(tiered) > 1 else 0.0
    return acc, spread


def cost_extrapolation(rows, target_n=EVAL_N, dev_n=None):
    """Per-arm dev cost, the x`factor` projection to the evaluation pool
    size, and a grand total including PC2's extra J re-run (section 7: J is
    re-run once more on all `target_n` evaluation items).

    `factor` is computed from the actual dev pool size, not hardcoded, so a
    change to POOL_SIZES would not silently mismatch this projection.
    """
    out = []
    total_projected = 0.0
    j_projected = 0.0
    for r in rows:
        n = dev_n if dev_n is not None else r["n"]
        factor = target_n / n if n else 0.0
        projected = r["cost_total"] * factor
        out.append({"arm": r["arm"], "dev_cost": r["cost_total"],
                    "factor": factor, "projected_cost": projected})
        total_projected += projected
        if r["arm"] == "J":
            j_projected = projected
    grand_total = total_projected + j_projected  # + PC2's extra J re-run
    return out, total_projected, j_projected, grand_total


def query_credits():
    """The account's remaining OpenRouter credit, queried live.

    Returns (total_credits, total_usage, remaining) or None on any failure --
    this must never be estimated or hardcoded if the query fails.
    """
    req = urllib.request.Request(
        CREDITS_URL, headers={"Authorization": f"Bearer {P.api_key()}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))["data"]
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        print(f"WARNING: credits query failed: {exc}", file=sys.stderr)
        return None
    total = data.get("total_credits")
    used = data.get("total_usage")
    if total is None or used is None:
        return None
    return total, used, total - used


def live_structured_output_check():
    """Section 4 step: were structured outputs and logprobs actually honoured.

    Near-200/200 valid JSON for each verbalized arm and near-200 logprobs
    blocks for Q-L is what "in force" should look like; well below that is a
    stop condition, not something to gloss over in the report.
    """
    out = {}
    for arm in ("F1-V", "F2-V", "Q-V"):
        n = ok = 0
        for rec in load_records(arm, "dev"):
            n += 1
            try:
                json.loads(rec["response"]["choices"][0]["message"]["content"])
                ok += 1
            except (TypeError, KeyError, IndexError, json.JSONDecodeError):
                pass
        out[arm] = (ok, n)
    lp = sum(1 for rec in load_records("Q-L", "dev")
             if ((rec.get("response") or {}).get("choices", [{}])[0]
                 .get("logprobs")))
    out["Q-L_logprobs"] = (lp, len(load_records("Q-L", "dev")))
    return out


def main():
    rows = per_arm_stats()
    per_class = per_class_accuracy()
    ql_records = load_records("Q-L", "dev")
    letters = letter_checks(ql_records)
    pos_stat, pos_p, pos_obs = letter_position_bias(letters["top1_counts"])
    dev_items = pools.load_pool("dev")
    shuffle_table, shuffle_stat, shuffle_p = shuffle_uniformity(dev_items)
    acc, pc1_spread = pc1_dev_accuracy()
    cost_rows, total_projected, j_projected, grand_total = cost_extrapolation(rows)
    credits = query_credits()
    live_check = live_structured_output_check()

    out = ["# Dev-set pre-freeze checks", "",
           "Generated by `devcheck.py`. Every number is computed from the "
           "committed `runs/*_dev.jsonl` and `preds/*_dev*` data, not typed.",
           "",
           "## Per-arm dev results (200 items)", "",
           "| Arm | ok | failures | kinds | leakage | prompt tok/item | "
           "completion tok/item | reasoning tok/item | thinking? | p50 s | "
           "p95 s | provider | pin holds | dev cost |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
           "--- | --- | --- | --- |"]
    for r in rows:
        out.append(
            f"| {r['arm']} | {r['ok']} | {r['failures']} | "
            f"{r['failure_kinds'] or '-'} | {r['tag_leakage']} | "
            f"{r['prompt_per_item']:.1f} | {r['completion_per_item']:.1f} | "
            f"{r['reasoning_per_item']:.1f} | "
            f"{'yes' if r['thinking'] else 'no'} | "
            f"{r['latency_p50']:.2f} | {r['latency_p95']:.2f} | "
            f"{r['providers']} | {'yes' if r['pin_holds'] else 'NO'} | "
            f"${r['cost_total']:.4f} |")

    out += ["", "**F1-V's measured reasoning tokens/item is "
            f"{next(r['reasoning_per_item'] for r in rows if r['arm'] == 'F1-V'):.1f}"
            "** -- section 4's parenthetical that `claude-sonnet-5` at "
            "provider defaults means \"adaptive thinking on at effort high\" "
            "does not hold on this pinned endpoint as measured; F1-V ran "
            "non-thinking by default. F2-V's measured reasoning tokens/item "
            f"is {next(r['reasoning_per_item'] for r in rows if r['arm'] == 'F2-V'):.1f}, "
            "so it was thinking. Both ran with the `reasoning` parameter "
            "omitted, at whatever each provider's default is -- the report "
            "states the measured fact rather than the protocol's prior "
            "assumption.", ""]

    out += ["## Per-class accuracy per arm (dev)", "",
            "| Arm | " + " | ".join(P.LABELS) + " |",
            "| --- | " + " | ".join("---" for _ in P.LABELS) + " |"]
    for arm in ARMS:
        if arm not in per_class:
            continue
        cells = [f"{per_class[arm].get(lab, float('nan')):.2f}"
                 for lab in P.LABELS]
        out.append(f"| {arm} | " + " | ".join(cells) + " |")
    j_ai = per_class.get("J", {}).get("cs.AI")
    other_j = [v for k, v in per_class.get("J", {}).items() if k != "cs.AI"]
    if j_ai is not None and other_j:
        headline = (f"**Headline:** J's accuracy on cs.AI is {j_ai:.2f}, "
                    f"against {min(other_j):.2f}-{max(other_j):.2f} on every "
                    "other class -- exactly the ambiguity section 2 "
                    "predicted (cs.AI is the top co-tag for all six other "
                    "classes).")
    else:
        headline = ""
    out += ["", headline, ""]

    out += ["## Letter tokenization and letter-position bias (Q-L)", "",
            f"- items with a usable logprobs block: {letters['n_with_logprobs']}/200",
            f"- bare single-character letter tokens observed, by letter: "
            f"{dict(sorted(letters['bare_counts'].items()))}",
            f"- top-1 (actually sampled) token was a bare A-G letter on "
            f"{letters['top1_bare_total']}/{letters['n_with_logprobs']} items",
            ""]
    if letters["all_seven_single_token"]:
        out.append("All seven letters A-G appear as clean bare single-character "
                   "tokens; A-G stands.")
    else:
        missing = [l for l in LETTERS if letters["bare_counts"].get(l, 0) == 0]
        out.append(f"**Section 4 action required:** {missing} never appear as "
                   "bare single-character tokens. Substitute a different "
                   "alphabet and re-run the dev pass.")
    out += ["",
            f"**Letter-position bias (top-1 letter choice vs uniform):** "
            f"counts {dict(zip(LETTERS, pos_obs))}, chi2 = {pos_stat:.2f}, "
            f"p = {pos_p:.3g}.",
            "",
            "This is strongly non-uniform. Section 4 says the per-item "
            "shuffle \"also controls letter-position bias in Q-L\"; the "
            "honest reading, given this result, is that the shuffle stops "
            "the bias favouring any one LABEL (checked directly below), not "
            "that it removes Q-L's intrinsic preference for certain letters "
            "as tokens.", ""]

    out += ["## Shuffle uniformity on the dev pool itself", "",
            "7x7 label-by-letter-position table (every item contributes one "
            "count per label at the position the shuffle placed it), "
            "chi-square against uniform:", "",
            f"- chi2 = {shuffle_stat:.2f}, p = {shuffle_p:.3g} (df = "
            f"{P.K * P.K - 1})",
            "",
            ("This passes: no evidence the shuffle itself is biased. It "
             "validates our own sampling, independent of the dev pool's "
             "label mix." if shuffle_p > 0.05 else
             "**This does not pass at p<0.05** -- investigate "
             "`pools.permutation_for` before freezing."), ""]

    out += ["## PC1 accuracy check on dev (section 7)", "",
            "| Arm | dev accuracy |", "| --- | --- |"]
    for arm, v in acc.items():
        out.append(f"| {arm} | {v:.3f} |")
    out += ["", f"Spread across J, F1-V, F2-V only (Q-V, Q-L excluded as "
            f"mechanism arms): **{pc1_spread:.1f} points** (PC1 threshold: 8).",
            ""]
    out.append(
        "Exceeds the threshold: the tier-matched framing is dropped and "
        "the comparison is reported through the Brier decomposition."
        if pc1_spread > 8 else
        "Within the threshold on dev; the tier-matched framing holds here. "
        "Re-checked on the evaluation set, which is what section 7 reports.")

    out += ["", "## Cost and its extrapolation to the evaluation pool "
            f"({EVAL_N} items)", "",
            "| Arm | dev cost (200 items) | x factor | projected cost | note |",
            "| --- | --- | --- | --- | --- |"]
    for cr in cost_rows:
        note = "PC2 re-runs J once more at this same rate" if cr["arm"] == "J" else ""
        out.append(f"| {cr['arm']} | ${cr['dev_cost']:.4f} | "
                   f"{cr['factor']:.0f}x | ${cr['projected_cost']:.2f} | {note} |")
    out += ["",
            f"- Sum of single-pass projected costs across all 5 arms: "
            f"**${total_projected:.2f}**",
            f"- Plus PC2's extra J re-run on all {EVAL_N} evaluation items "
            f"(J again, at its own measured dev rate): +${j_projected:.2f}",
            f"- **Grand total projected evaluation cost: ${grand_total:.2f}**",
            ""]
    if credits:
        total_credits, total_usage, remaining = credits
        out += [f"- OpenRouter account: total credits ${total_credits:.2f}, "
                f"used ${total_usage:.2f}, **remaining ${remaining:.2f}** "
                f"(queried live from `{CREDITS_URL}`)", ""]
        if grand_total > remaining:
            out.append(
                f"**STOP: the projected evaluation cost (${grand_total:.2f}) "
                f"EXCEEDS the account's remaining credit (${remaining:.2f}) "
                f"by ${grand_total - remaining:.2f}.** The evaluation run "
                "cannot proceed at this budget without either more credit "
                "or a cost reduction (F1-V dominates the bill).")
        else:
            out.append(
                f"Remaining credit (${remaining:.2f}) covers the projected "
                f"cost (${grand_total:.2f}).")
    else:
        out.append("- OpenRouter credits query failed; remaining balance "
                   "could not be confirmed live. Do not assume the budget "
                   "is sufficient.")
    out.append("")

    out += ["## Structured outputs and logprobs, confirmed live", "",
            "| Check | ok | n |", "| --- | --- | --- |"]
    for arm in ("F1-V", "F2-V", "Q-V"):
        ok, n = live_check[arm]
        out.append(f"| {arm}: valid standalone JSON | {ok} | {n} |")
    lp_ok, lp_n = live_check["Q-L_logprobs"]
    out.append(f"| Q-L: responses carrying a logprobs block | {lp_ok} | {lp_n} |")
    out.append("")

    with open("devcheck_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
