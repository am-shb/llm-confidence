#!/usr/bin/env python3
# FINDINGS (2026-09-19):
#
# Round 1 (chat/completions): all four candidate shapes (plain, tool_required,
# tool_named, response_format_enum) returned HTTP 400 from
# POST /api/v1/chat/completions, byte-identical body:
#   {"error": {"message": "typesafe/jev-1.13 is a decisions model and cannot
#   be used with the chat/completions endpoint. Use the /api/alpha/decisions
#   endpoint instead.", "code": 400}}
# One such attempt is kept in probes/jev_probe.json (shape "chat_completions_plain")
# as evidence of this. Jev is not reachable through chat/completions at all;
# this route is not one of the four candidates any adapter should attempt.
#
# Round 2 (POST /api/alpha/decisions): HTTP 200 on both dev items tried.
# Schema confirmed against TypeSafe's official docs:
# https://docs.typesafe.ai/introduction/quickstart (the "System One" decisions
# contract). We reach it through OpenRouter's /api/alpha/decisions passthrough
# rather than TypeSafe's first-party https://api.typesafe.ai/v1/systemone,
# because PROTOCOL.md section 3 pins the exact version typesafe/jev-1.13 while
# the first-party API documents its default model as jev-latest, an unpinned
# version; the exact pin is load-bearing for section 2's contamination
# argument. Section 10 defers TypeSafe's System One adapter itself to part two
# of the study -- this probe uses the same underlying contract only to reach
# the pinned model through OpenRouter, not that adapter.
#
# 1. Winning shape/endpoint: POST https://openrouter.ai/api/alpha/decisions
#    with body {"model": ..., "state": <title+abstract text>,
#    "questions": {"primary_category": {"type": "choice",
#    "instructions": <question sentence>, "criteria": {LABEL: "name:
#    description", ...}}}}. There is no `messages` array and no
#    `provider`/`allow_fallbacks` field -- protocol.provider_block() does not
#    apply to this endpoint; the response's own `"provider": "TypeSafe"`
#    field is the confirmation instead, and TypeSafe is J's only provider.
#    `criteria` is a dict built in the item's shuffled order
#    (item["options"]), which json.dumps preserves as insertion order, so the
#    per-item permutation required by section 4 is carried the same way here
#    as the lettered options are for the other arms.
# 2. Probability access path:
#    response["answers"]["primary_category"]["probabilities"]
#    -- a dict keyed by label (e.g. {"cs.CV": 0.83, "cs.LG": 0.05, ...}).
#    response["answers"]["primary_category"]["choice"] gives the argmax label
#    and ["confidence"] a separate scalar. usage/cost is in response["usage"].
# 3. All seven options are scored on both dev items probed -- observed
#    `probabilities` dicts have exactly 7 keys, matching item["options"].
#    The hard-stop on question 3 does NOT trigger.
# 4. Sums to 1: yes, exactly, on both dev items probed (1.0 and 1.0, no
#    floating-point residue observed). Distributions are not degenerate --
#    mass is spread over 2-3 of the 7 labels per item (e.g. cs.CL 0.85 /
#    cs.AI 0.14 / cs.LG 0.01 on one item), not a hard one-hot, though several
#    labels do come back exactly 0. See probes/jev_probe.json for the raw
#    numbers and the report for the full per-item breakdown.
"""Discover how typesafe/jev-1.13 accepts a choice set and returns probabilities.

Jev's OpenRouter record lists modality text->decisions, an empty
supported_parameters, and supports_tool_choice.function: true, and the model
description is truncated mid-sentence. Round 1 established that Jev refuses
the chat/completions endpoint outright; this script now targets the real
contract, POST /api/alpha/decisions, on dev-pool items -- which section 4
designates for exactly this kind of debugging and which are disjoint from the
evaluation pool.

Writes every attempt (one chat/completions probe for the record, then the
decisions-endpoint probes) and its full response to probes/jev_probe.json.

Usage:
    ./probe_jev.py            # probe chat/completions once, then decisions, on 2 dev items
"""

import json
import os
import sys
import urllib.error
import urllib.request

import pools
import prompt
import protocol as P
import taxonomy

URL_CHAT = "https://openrouter.ai/api/v1/chat/completions"
URL_DECISIONS = "https://openrouter.ai/api/alpha/decisions"
OUT = "probes/jev_probe.json"


def post(url, payload):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {P.api_key()}",
        "Content-Type": "application/json",
        "X-Title": "jev-calibration-study",
    })
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def chat_probe(item):
    """One chat/completions attempt, kept on record as evidence of the 400."""
    messages = prompt.build_prompt(item, "decisions")
    payload = {"model": P.ARMS["J"]["model"],
              "provider": P.provider_block("J"),
              "messages": messages}
    return "chat_completions_plain", URL_CHAT, payload


def decisions_payload(item):
    """The real contract: POST /api/alpha/decisions.

    No `messages`, no `provider`/`allow_fallbacks` -- provider_block() does
    not apply here. `criteria` carries the item's shuffled option order via
    dict insertion order, same permutation every other arm gets via its
    lettered options block.
    """
    desc = taxonomy.load_descriptions()
    criteria = {label: f"{desc[label]['name']}: {desc[label]['description']}"
                for label in item["options"]}
    instructions = f"{prompt.SHARED_BODY_MARKER} this paper under?"
    state = f"Title: {item['title']}\n\nAbstract: {item['abstract']}"
    payload = {
        "model": P.ARMS["J"]["model"],
        "state": state,
        "questions": {
            "primary_category": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        },
    }
    return "decisions", URL_DECISIONS, payload


def main():
    items = pools.load_pool("dev")[:2]
    results = []
    for item in items:
        for name, url, payload in (chat_probe(item), decisions_payload(item)):
            status, body = post(url, payload)
            print(f"  {item['id']} {name:22} -> HTTP {status}", file=sys.stderr)
            results.append({"item": item["id"], "shape": name,
                            "status": status,
                            "request": P.redact(payload),
                            "response": body})
    os.makedirs("probes", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(results)} probe results to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
