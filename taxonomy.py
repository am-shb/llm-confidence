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
import re
import sys
import urllib.request

import protocol as P

TAXONOMY_URL = "https://arxiv.org/category_taxonomy"
OUT_PATH = "category_descriptions.json"
USER_AGENT = "jev-calibration-study/1.0 (one-time taxonomy fetch)"

# A category heading. The name group is [^)]* rather than .*? so it cannot span
# across categories to a later ")</span></h4>" -- that leak is what allowed a
# description to be attributed to the wrong category.
HEADING_RE = re.compile(
    r'<h4>([a-zA-Z\-]+\.[a-zA-Z\-]+)\s*<span>\(([^)]*)\)</span></h4>')
PARAGRAPH_RE = re.compile(r'<p>(.*?)</p>', re.S)


def _clean(fragment):
    """Strip tags, unescape entities, collapse whitespace. No rewording."""
    text = re.sub(r"<.*?>", "", fragment)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def parse_page(page):
    """Map category code -> {"name", "description"} from the taxonomy page.

    The <h4> and its <p> sit in sibling divs with markup between them, so the
    description cannot be matched by a pattern anchored to the heading alone.
    Instead each description is read ONLY from the region strictly before the
    next heading. A category with no <p> of its own is omitted rather than
    silently paired with the following category's text -- section 3 requires
    these descriptions verbatim, so a wrong answer must become a missing one,
    which fetch_descriptions then turns into a hard stop.
    """
    found = {}
    marks = list(HEADING_RE.finditer(page))
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(page)
        para = PARAGRAPH_RE.search(page[mark.end():end])
        if para is None:
            continue
        found[mark.group(1)] = {"name": _clean(mark.group(2)),
                                "description": _clean(para.group(1))}
    return found


def fetch_descriptions(url=TAXONOMY_URL):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        page = resp.read().decode("utf-8")

    found = parse_page(page)
    missing = [l for l in P.LABELS if l not in found]
    if missing:
        raise SystemExit(
            f"STOP: taxonomy parse missed {missing}. The page markup changed; "
            f"fix the parser rather than hand-writing descriptions -- section 3 "
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
