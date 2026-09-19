#!/usr/bin/env python3
# FINDINGS (2026-09-19):
#
# All four candidate shapes (plain, tool_required, tool_named,
# response_format_enum) were sent to POST /api/v1/chat/completions for 2
# dev-pool items (8 requests total). Every one of the 8 came back HTTP 400
# with the SAME body:
#
#   {"error": {"message": "typesafe/jev-1.13 is a decisions model and cannot
#   be used with the chat/completions endpoint. Use the /api/alpha/decisions
#   endpoint instead.", "code": 400}}
#
# 1. Winning shape: NONE. No candidate shape reaches HTTP 200. The blocker is
#    not the payload -- it is the endpoint. OpenRouter's chat/completions
#    route refuses this model outright, before looking at messages/tools/
#    response_format, and names the correct route itself:
#    POST /api/alpha/decisions.
# 2/3/4. Unanswered. No response body from a "decisions"-shaped call was ever
#    observed, so there is no access path to the probabilities, no answer to
#    whether all 7 options are scored, and no answer on whether they sum to 1.
#
# Per this task's constraints, the /api/alpha/decisions endpoint was NOT
# probed -- trying it would mean inventing a fifth shape beyond the four the
# brief specified, on a different endpoint the brief never named. That is a
# structural finding for a human to route (a new/updated task against
# /api/alpha/decisions), not something to guess around here. See
# probes/jev_probe.json for the full redacted request/response pairs (all 8
# attempts, identical shape of failure).
"""Discover how typesafe/jev-1.13 accepts a choice set and returns probabilities.

Jev's OpenRouter record lists modality text->decisions, an empty
supported_parameters, and supports_tool_choice.function: true, and the model
description is truncated mid-sentence. So the wire format is unknown and this
script finds it out on dev-pool items -- which section 4 designates for exactly
this kind of debugging and which are disjoint from the evaluation pool.

Writes every attempt and its full response to probes/jev_probe.json.

Usage:
    ./probe_jev.py            # try each candidate shape on 2 dev items
"""

import json
import os
import sys
import urllib.error
import urllib.request

import pools
import prompt
import protocol as P

URL = "https://openrouter.ai/api/v1/chat/completions"
OUT = "probes/jev_probe.json"


def post(payload):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {P.api_key()}",
        "Content-Type": "application/json",
        "X-Title": "jev-calibration-study",
    })
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def candidates(item):
    """Request shapes to try, cheapest and most likely first."""
    messages = prompt.build_prompt(item, "decisions")
    base = {"model": P.ARMS["J"]["model"],
            "provider": P.provider_block("J")}

    tool = {
        "type": "function",
        "function": {
            "name": "choose_category",
            "description": "Return the primary arXiv category.",
            "parameters": {
                "type": "object",
                "properties": {"category": {"type": "string",
                                            "enum": item["options"]}},
                "required": ["category"],
            },
        },
    }
    return [
        ("plain", {**base, "messages": messages}),
        ("tool_required", {**base, "messages": messages, "tools": [tool],
                           "tool_choice": "required"}),
        ("tool_named", {**base, "messages": messages, "tools": [tool],
                        "tool_choice": {"type": "function",
                                        "function": {"name": "choose_category"}}}),
        ("response_format_enum", {**base, "messages": messages,
                                  "response_format": {
                                      "type": "json_schema",
                                      "json_schema": prompt.json_schema(
                                          item["options"])}}),
    ]


def main():
    items = pools.load_pool("dev")[:2]
    results = []
    for item in items:
        for name, payload in candidates(item):
            status, body = post(payload)
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
