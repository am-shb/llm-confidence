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


class FatalRunError(RuntimeError):
    """A run-level infrastructure failure that must never become data.

    401/402/403 and any other non-408/429 4xx mean the request itself is
    broken (bad key, out of credit, forbidden) -- retrying burns nothing but
    time, and section 5's "failure" record would score it uniform forever,
    since completed_ids() would then treat it as done. This is raised instead
    so main() can abort loudly and leave the item unrecorded, resumable once
    the cause is fixed.
    """

    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


def build_payload(arm, item):
    """The exact request body for this arm and item.

    Section 4 asymmetries live here and nowhere else: reasoning is omitted for
    the frontier arms (F1-V, F2-V) so they run as they ship; the Qwen arms
    (Q-V, Q-L) explicitly disable reasoning via params, since non-thinking is
    not this endpoint's default and reasoning or sampling tokens would
    displace the first generated token whose logprobs Q-L reads; J exposes no
    parameters at all.
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
    if "reasoning" in params:
        # Section 4: only the Qwen arms send this, to explicitly disable
        # thinking (see protocol.ARMS). F1-V/F2-V never set "reasoning" in
        # params, so this branch never fires for them -- they run at
        # provider defaults with the key omitted entirely.
        payload["reasoning"] = params["reasoning"]

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
    """Ids already recorded, so a resume does not re-bill them.

    A `status: "failure"` record is a run-level infrastructure failure (rate
    limiting exhausted, network error, an unhandled exception) -- never a
    model-behaviour outcome, since 401/402/403 and other fatal HTTP statuses
    now abort the run instead of becoming a record (see FatalRunError). Such
    ids are excluded here so a resume retries them instead of leaving them
    stuck uniform forever. `status: "ok"` records, even ones parse.py will
    later classify as a parse error, refusal, truncation or bad letter, are
    legitimate section 5 outcomes and stay done.

    Tolerates a truncated final line from a crash.
    """
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("status") == "failure":
                    continue
                done.add(rec["id"])
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
    last_http_status = None

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
            redacted = P.redact_error_body(raw)
            # 401/402/403 and any other non-408/429 4xx are not transient:
            # the account or the request is broken, not the network. Retrying
            # wastes time and, worse, section 5 would otherwise score it
            # uniform as if the model had answered. Fatal, not a record.
            if 400 <= exc.code < 500 and exc.code not in (408, 429):
                raise FatalRunError(exc.code, redacted)
            last_error = f"HTTP {exc.code}: {redacted[:500]}"
            last_http_status = exc.code
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
            # SystemExit, which derive from BaseException. FatalRunError is
            # raised from inside the HTTPError branch above, a sibling of this
            # clause, so it propagates straight out and is never caught here.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(10)

    # Section 5: a timeout after 3 retries is a failure, never a dropped item.
    # http_status is null for genuine timeouts/network exhaustion and set for
    # retry-exhausted HTTP failures (429/5xx) -- parse.py uses it to tell them
    # apart ("timeout" vs "run_error").
    return {
        "id": item["id"], "arm": arm, "status": "failure",
        "error": last_error, "attempts": MAX_ATTEMPTS,
        "request": P.redact(payload), "response": None,
        "provider": None, "http_status": last_http_status,
        "latency_s": round(time.time() - started, 3),
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
    }


def snapshot_endpoint(arm, out_dir=RUN_DIR):
    """Record the pinned endpoint's registry entry beside the run.

    Section 4 requires provider and quantization on record, and confirming
    `structured_outputs` on each pinned endpoint before the run. The
    completion response carries provider but not quantization, and nothing
    echoes the reasoning defaults, so the registry entry is the evidence --
    and the only place a lost capability or a silent quantization change
    would show up before money is spent.
    """
    spec = P.ARMS[arm]
    params = spec["params"]
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
    supported_raw = match[0].get("supported_parameters")
    supported = supported_raw or []

    # Structured-outputs mechanism (F1-V/F2-V/Q-V). Arm J is exempt: its
    # decisions schema has no structured_outputs concept at all.
    if params.get("structured_outputs") and "structured_outputs" not in supported:
        raise SystemExit(
            f"STOP: {spec['provider']}'s pinned endpoint for {arm} no longer "
            f"lists structured_outputs. {arm}'s mechanism requires it; do not "
            f"proceed on a silently degraded endpoint.")

    # Q-L reads the first generated token's logprobs -- no logprobs, no run.
    if params.get("logprobs") and "logprobs" not in supported:
        raise SystemExit(
            f"STOP: {spec['provider']}'s pinned endpoint for {arm} no longer "
            f"lists logprobs. Q-L cannot score anything without it.")

    quantization = match[0].get("quantization")
    path = os.path.join(out_dir, f"{arm}_endpoint.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            prev = json.load(fh)
        prev_quant = prev.get("quantization")
        if prev_quant != quantization:
            raise SystemExit(
                f"STOP: {arm}'s pinned endpoint quantization changed from "
                f"{prev_quant!r} to {quantization!r} since the earlier run. "
                f"This can shift the distribution a run measures; do not "
                f"silently proceed on a changed endpoint.")

    snap = {
        "arm": arm, "model": spec["model"], "provider": spec["provider"],
        "quantization": quantization,
        "supported_parameters": supported_raw,
        "context_length": match[0].get("context_length"),
        "pricing": match[0].get("pricing"),
        "captured": dt.datetime.now(dt.UTC).isoformat(),
    }
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
        except FatalRunError:
            # Must not be swallowed into a failure record -- see FatalRunError
            # and the abort handling below. Let it propagate.
            raise
        except BaseException as exc:      # last-resort guard
            rec = {"id": item["id"], "arm": args.arm, "status": "failure",
                   "error": f"unhandled in call(): {type(exc).__name__}: {exc}",
                   "attempts": MAX_ATTEMPTS, "request": None, "response": None,
                   "provider": None, "http_status": None, "latency_s": None,
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
    except FatalRunError as exc:
        # Section 5 never scores infrastructure as data: abort loudly instead
        # of recording a failure for the item in flight. Everything already
        # written to out_path stays valid and completed_ids() will skip it on
        # resume; the item that hit this error was NOT written and will be
        # retried once the cause (e.g. an empty credit balance) is fixed.
        raise SystemExit(
            f"STOP: HTTP {exc.status} from the API -- treated as a fatal "
            f"infrastructure failure, not data. No record was written for "
            f"the item in flight. {out_path} is unchanged otherwise and the "
            f"run is resumable once the cause is fixed.\n{exc}")
    finally:
        fh.close()

    print(f"done: ok={counts['ok']} failure={counts['failure']}", file=sys.stderr)


if __name__ == "__main__":
    main()
