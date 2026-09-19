#!/usr/bin/env python3
"""Harvest arXiv papers published in the last N days into a single CSV.

Queries arXiv's Atom search API one calendar day at a time. Two reasons for the
day-at-a-time shape:

  * The API refuses to page deep (it breaks somewhere around 30k results). A
    30-day window holds ~35k papers, so a single query cannot retrieve it --
    but each individual day is only ~1-1500 papers, comfortably inside limits.
  * `submittedDate` filters on the v1 submission date, which is what
    "published" should mean here.

That second point is the whole reason this script does not use OAI-PMH, which
is the more obvious choice for bulk metadata. OAI's `<created>` field is NOT the
v1 date: arXiv reports created=2026-09-16 for paper 1804.09359, whose real v1
date is 2018-04-25. Filtering an OAI harvest on `created` let ~26% old papers
through. The search API's `published` field reports 2018-04-25 correctly.

Usage:
    ./harvest_arxiv.py                       # last 30 days -> arxiv_last_30_days.csv
    ./harvest_arxiv.py --days 7 -o week.csv
"""

import argparse
import csv
import datetime as dt
import os
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

API_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARX = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"

USER_AGENT = "arxiv-csv-harvester/1.0 (single-user bulk metadata export)"
PAGE_SIZE = 1000       # arXiv permits 2000; 1000 is noticeably more reliable
DEFAULT_DELAY = 3.0    # arXiv asks for ~3s between API calls
MAX_RETRIES = 8
# arXiv signals "slow down" with 406 as well as the documented 429/503. A 406
# here is not content negotiation -- the identical request succeeds once the
# penalty window passes -- so it must be backed off, not retried hard.
RATE_LIMIT_CODES = (406, 429, 503)

COLUMNS = [
    "id", "version", "published", "updated", "title", "abstract", "authors",
    "primary_category", "categories", "doi", "journal_ref", "comments",
    "abs_url", "pdf_url",
]

ID_RE = re.compile(r"arxiv\.org/abs/(.+?)(?:v(\d+))?$")


def collapse(text):
    """arXiv hard-wraps titles and abstracts; make them one clean CSV cell."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def fetch(query, start, cache_dir, day, delay):
    """GET one page of results, backing off on rate limits.

    Returns (body, delay); delay grows if the server pushed back.
    """
    cache_path = (os.path.join(cache_dir, f"{day}_{start:05d}.xml")
                  if cache_dir else None)
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            return fh.read(), delay

    url = (f"{API_URL}?search_query={query}&start={start}"
           f"&max_results={PAGE_SIZE}&sortBy=submittedDate&sortOrder=ascending")
    body = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            if exc.code in RATE_LIMIT_CODES:
                retry_after = exc.headers.get("Retry-After")
                wait = (int(retry_after) if retry_after and retry_after.isdigit()
                        else min(20 * 2 ** attempt, 300))
                delay = min(delay * 1.5, 30.0)
                print(f"    rate limited (HTTP {exc.code}); waiting {wait}s, "
                      f"delay now {delay:.0f}s", file=sys.stderr)
                time.sleep(wait)
            elif attempt == MAX_RETRIES - 1:
                raise
            else:
                time.sleep(10)
        except (urllib.error.URLError, TimeoutError):
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(10)
    if body is None:
        raise RuntimeError(f"giving up on {url} after {MAX_RETRIES} attempts")

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as fh:
            fh.write(body)
    return body, delay


def parse_entry(entry):
    """Turn one Atom <entry> into a CSV row dict."""
    raw_id = (entry.findtext(ATOM + "id") or "").strip()
    match = ID_RE.search(raw_id)
    arxiv_id = match.group(1) if match else raw_id
    version = match.group(2) if match and match.group(2) else ""

    authors = [collapse(a.findtext(ATOM + "name"))
               for a in entry.findall(ATOM + "author")]
    cats = [c.get("term") for c in entry.findall(ATOM + "category") if c.get("term")]
    primary = entry.find(ARX + "primary_category")

    pdf_url = ""
    for link in entry.findall(ATOM + "link"):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")

    return {
        "id": arxiv_id,
        "version": version,
        "published": (entry.findtext(ATOM + "published") or "").strip(),
        "updated": (entry.findtext(ATOM + "updated") or "").strip(),
        "title": collapse(entry.findtext(ATOM + "title")),
        "abstract": collapse(entry.findtext(ATOM + "summary")),
        "authors": ";".join(a for a in authors if a),
        "primary_category": primary.get("term", "") if primary is not None else "",
        "categories": ";".join(cats),
        "doi": collapse(entry.findtext(ARX + "doi")),
        "journal_ref": collapse(entry.findtext(ARX + "journal_ref")),
        "comments": collapse(entry.findtext(ARX + "comment")),
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
    }


def harvest_day(day, cache_dir, delay, writer, seen):
    """Fetch every paper submitted on `day`. Returns (written, total, delay)."""
    stamp = day.strftime("%Y%m%d")
    query = f"submittedDate:%5B{stamp}0000%20TO%20{stamp}2359%5D"

    start = written = 0
    total = None
    while True:
        body, delay = fetch(query, start, cache_dir, stamp, delay)
        root = ET.fromstring(body)
        if total is None:
            total = int(root.findtext(OPENSEARCH + "totalResults") or 0)

        entries = root.findall(ATOM + "entry")
        if not entries:
            break
        for entry in entries:
            row = parse_entry(entry)
            if not row["id"] or row["id"] in seen:
                continue
            seen.add(row["id"])
            writer.writerow(row)
            written += 1

        start += len(entries)
        if start >= total:
            break
        time.sleep(delay)

    return written, total or 0, delay


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=30, help="window size (default: 30)")
    ap.add_argument("-o", "--output", default=None, help="output CSV path")
    ap.add_argument("--cache-dir", default="raw",
                    help="cache raw XML so reruns cost no network (default: raw)")
    ap.add_argument("--no-cache", action="store_true", help="disable the raw XML cache")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                    help=f"seconds between API calls (default: {DEFAULT_DELAY:.0f})")
    args = ap.parse_args()

    today = dt.date.today()
    days = [today - dt.timedelta(days=n) for n in range(args.days, 0, -1)]
    out_path = args.output or f"arxiv_last_{args.days}_days.csv"

    cache_dir = None if args.no_cache else args.cache_dir
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    print(f"harvesting {days[0]} .. {days[-1]} ({len(days)} days) -> {out_path}",
          file=sys.stderr)

    delay = args.delay
    seen = set()
    grand_total = 0
    start_time = time.time()

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for i, day in enumerate(days, 1):
            written, total, delay = harvest_day(day, cache_dir, delay, writer, seen)
            grand_total += written
            print(f"  [{i:2}/{len(days)}] {day}: {written} papers "
                  f"(api reported {total}); running total {grand_total}",
                  file=sys.stderr)
            if i < len(days):
                time.sleep(delay)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\ndone: {grand_total} papers -> {out_path} ({size_mb:.1f} MB) "
          f"in {time.time() - start_time:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
