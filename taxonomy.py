#!/usr/bin/env python3
"""Fetch arXiv's official category descriptions for the 7 study labels.

PROTOCOL.md section 3 requires every arm to get these descriptions verbatim,
so they are fetched once, committed as data, and never edited by hand.

Usage:
    ./taxonomy.py          # fetch and write category_descriptions.json
    ./taxonomy.py --show   # print what is stored
"""

import argparse
import html
import json
import os
import re
import sys
import urllib.request

import protocol as P

TAXONOMY_URL = "https://arxiv.org/category_taxonomy"
OUT_PATH = "category_descriptions.json"
USER_AGENT = "jev-calibration-study/1.0 (one-time taxonomy fetch)"

# <h4>cs.AI <span>(Artificial Intelligence)</span></h4> ... <p>description</p>
# with sibling divs in between, hence the non-greedy gap.
ENTRY_RE = re.compile(
    r'<h4>([a-zA-Z\-]+\.[a-zA-Z\-]+)\s*<span>\((.*?)\)</span></h4>.*?<p>(.*?)</p>',
    re.S)


def _clean(fragment):
    """Strip tags, unescape entities, collapse whitespace. No rewording."""
    text = re.sub(r"<.*?>", "", fragment)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def fetch_descriptions(url=TAXONOMY_URL):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        page = resp.read().decode("utf-8")

    found = {cat: {"name": _clean(name), "description": _clean(desc)}
             for cat, name, desc in ENTRY_RE.findall(page)}
    missing = [l for l in P.LABELS if l not in found]
    if missing:
        raise SystemExit(
            f"STOP: taxonomy parse missed {missing}. The page markup changed; "
            f"fix ENTRY_RE rather than hand-writing descriptions -- section 3 "
            f"requires them verbatim.")
    return {l: found[l] for l in P.LABELS}


def load_descriptions(path=OUT_PATH):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.show:
        for label, e in load_descriptions().items():
            print(f"\n{label} ({e['name']})\n  {e['description']}")
        return

    data = fetch_descriptions()
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    print(f"wrote {len(data)} descriptions to {OUT_PATH} "
          f"(source: {TAXONOMY_URL}, fetched once)", file=sys.stderr)


if __name__ == "__main__":
    main()
