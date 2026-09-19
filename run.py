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
        except Exception as exc:  # noqa: BLE001 - see below
            # Deliberately broad. Section 5 requires every item to produce a
            # record; an unhandled exception here would bypass the write lock
            # and drop the item entirely. Anything unexpected becomes a
            # recorded failure instead. Does not catch KeyboardInterrupt or
            # SystemExit, which derive from BaseException.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(10)

    # Section 5: a timeout after 3 retries is a failure, never a dropped item.
    return {
        "id": item["id"], "arm": arm, "status": "failure",
        "error": last_error, "attempts": MAX_ATTEMPTS,
        "request": P.redact(payload), "response": None,
        "provider": None,
        "latency_s": round(time.time() - started, 3),
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
    }


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

    snapshot_endpoint(args.arm)

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
        try:
            rec = call(args.arm, item)
        except BaseException as exc:      # last-resort guard
            rec = {"id": item["id"], "arm": args.arm, "status": "failure",
                   "error": f"unhandled in call(): {type(exc).__name__}: {exc}",
                   "attempts": MAX_ATTEMPTS, "request": None, "response": None,
                   "provider": None, "latency_s": None,
                   "timestamp": dt.datetime.now(dt.UTC).isoformat()}
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
