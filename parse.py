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
                 "bad_letter", "timeout", "empty", "no_logprobs"}

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
    seen_entries = set()
    for entry in top:
        tok = (entry.get("token") or "").strip().upper()
        if len(tok) != 1 or tok not in prompt.LETTERS:
            continue
        marker = (tok, entry.get("logprob"))
        if marker in seen_entries:      # exact repeat: provider artifact
            continue
        seen_entries.add(marker)
        pos = prompt.LETTERS.index(tok)
        shuffled[pos] += float(np.exp(entry["logprob"]))
        hit = True

    if not hit:
        # No usable letter in the returned top-k, so there is no distribution to
        # read and section 5 scores this uniform either way. Distinguish WHY,
        # because section 4's pre-freeze checks act on it: a letter outside A-G
        # means the model is not following the instruction (consider a different
        # alphabet), whereas no letter-like token at all means the provider is
        # not returning the sampled token. Classified from the top-k tokens only
        # -- never from the emitted text, which must never influence a number.
        saw_other_letter = any(
            len((e.get("token") or "").strip()) == 1
            and (e.get("token") or "").strip().isalpha()
            for e in top)
        return None, ("bad_letter" if saw_other_letter else "no_logprobs")

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
