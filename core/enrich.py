#!/usr/bin/env python3
"""
enrich_abstracts.py — Fill missing abstracts in a literature_explorer cache.

Checks every paper in the cache (corpus papers AND embedded refs/citations)
for a missing abstract, then fetches from three sources in order:

  1. Semantic Scholar batch API  — POST /paper/batch, 500 IDs per call
                                   (fast: ~6 calls for 3 000 papers)
  2. OpenAlex                    — by DOI or title; reconstructs abstract
                                   from OA's inverted-index format
  3. CrossRef                    — by DOI; not all records have abstracts

The original cache file is backed up as <cache>.bak before any writes.
Subsequent runs skip papers that already have an abstract (idempotent).

Usage:
    python enrich_abstracts.py
    python enrich_abstracts.py --cache my_cache.json
    python enrich_abstracts.py --limit 200          # cap API calls (for testing)
    python enrich_abstracts.py --dry-run            # report gaps, don't fetch
    python enrich_abstracts.py --sources ss oa      # restrict sources to try
"""

import re, json, math, time, sys, argparse, shutil
from pathlib import Path
from collections import defaultdict

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")

# ── Config ──────────────────────────────────────────────────────────────────────
SS_BASE    = "https://api.semanticscholar.org/graph/v1/paper"
OA_BASE    = "https://api.openalex.org/works"
CR_BASE    = "https://api.crossref.org/works"
SLEEP      = 1.1          # seconds between individual OA / CrossRef calls
SS_BATCH   = 500          # max IDs per SS batch request
OA_BATCH   = 100          # max DOIs per OA batch request (pipe-filter limit)


# ── HTTP helpers ────────────────────────────────────────────────────────────────
def _get(url, params=None, timeout=15):
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 2 ** (attempt + 2)))
                print(f"  ⏳ Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.ok:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.exceptions.Timeout:
            print(f"  ⚠  Timeout (attempt {attempt+1})")
        except Exception as e:
            print(f"  ⚠  Request error: {e}")
        time.sleep(SLEEP)
    return None

def _post(url, payload, params=None, timeout=20):
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, params=params, timeout=timeout)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 2 ** (attempt + 2)))
                print(f"  ⏳ Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.ok:
                return r.json()
        except requests.exceptions.Timeout:
            print(f"  ⚠  Timeout (attempt {attempt+1})")
        except Exception as e:
            print(f"  ⚠  Request error: {e}")
        time.sleep(SLEEP)
    return None


# ── Abstract sources ────────────────────────────────────────────────────────────
def fetch_ss_chunk(chunk):
    """
    POST /paper/batch for a single chunk (≤ SS_BATCH IDs).
    Returns {paperId -> abstract} for that chunk only.
    """
    data = _post(
        f"{SS_BASE}/batch",
        payload={"ids": chunk},
        params={"fields": "paperId,abstract"},
    )
    time.sleep(SLEEP)
    if not data:
        return {}
    return {
        entry["paperId"]: entry["abstract"]
        for entry in data
        if entry and entry.get("paperId") and entry.get("abstract")
    }


def _oa_invert_to_text(inv):
    """
    Reconstruct a plain-text abstract from OpenAlex's inverted index.
    The inverted index maps word → [position, ...].
    """
    if not inv:
        return ""
    positions = [(pos, word) for word, positions in inv.items() for pos in positions]
    return " ".join(w for _, w in sorted(positions))

def fetch_oa_batch(doi_list):
    """
    Fetch abstracts for up to OA_BATCH DOIs in a single OpenAlex API call.
    Returns {doi_normalised -> abstract_text} for hits only.
    doi_normalised is lowercase with no 'https://doi.org/' prefix.
    """
    if not doi_list:
        return {}
    filter_val = "|".join(f"https://doi.org/{d}" for d in doi_list)
    data = _get(OA_BASE, params={
        "filter": f"doi:{filter_val}",
        "per_page": OA_BATCH,
        "select": "doi,abstract_inverted_index",
        "mailto": "lit-explorer",
    })
    time.sleep(SLEEP)
    if not data or not data.get("results"):
        return {}
    out = {}
    for work in data["results"]:
        doi_raw = work.get("doi") or ""
        doi_key = doi_raw.replace("https://doi.org/", "").lower()
        abst = _oa_invert_to_text(work.get("abstract_inverted_index"))
        if doi_key and abst:
            out[doi_key] = abst
    return out

def fetch_oa_abstract_single(doi=None, title=None):
    """
    OpenAlex fallback for papers without a DOI — title search only.
    Also used as a one-off DOI lookup when not covered by a batch call.
    """
    if doi:
        data = _get(f"{OA_BASE}/https://doi.org/{doi}",
                    params={"select": "abstract_inverted_index", "mailto": "lit-explorer"})
        time.sleep(SLEEP)
        if data:
            return _oa_invert_to_text(data.get("abstract_inverted_index")) or None
    if title:
        data = _get(OA_BASE,
                    params={"search": title, "per_page": 1,
                            "select": "display_name,abstract_inverted_index",
                            "mailto": "lit-explorer"})
        time.sleep(SLEEP)
        if data and data.get("results"):
            return _oa_invert_to_text(data["results"][0].get("abstract_inverted_index")) or None
    return None

def fetch_crossref_abstract(doi):
    """CrossRef DOI lookup — many records lack abstracts."""
    if not doi:
        return None
    data = _get(f"{CR_BASE}/{doi}", params={"mailto": "lit-explorer"})
    time.sleep(SLEEP)
    if data:
        msg = data.get("message") or {}
        raw = msg.get("abstract") or ""
        # CrossRef abstracts often contain JATS XML tags — strip them
        return re.sub(r"<[^>]+>", "", raw).strip() or None
    return None


# ── Cache traversal ─────────────────────────────────────────────────────────────
def collect_missing(data):
    """
    Walk every paper in the cache (corpus + embedded refs/citations).
    Returns:
        need_abstract : dict  paperId -> {"doi": ..., "title": ..., "nodes": [list of dicts]}
        corpus_no_pid : list  of corpus-level paper dicts that have no paperId but need abstract
    """
    need_abstract = defaultdict(lambda: {"doi": None, "title": None, "nodes": []})
    corpus_no_pid = []

    def _visit(paper_dict):
        if not paper_dict or paper_dict.get("abstract"):
            return
        pid   = paper_dict.get("paperId")
        doi   = (paper_dict.get("externalIds") or {}).get("DOI") or paper_dict.get("doi")
        title = paper_dict.get("title") or ""
        if pid:
            entry = need_abstract[pid]
            entry["nodes"].append(paper_dict)
            if doi   and not entry["doi"]:   entry["doi"]   = doi
            if title and not entry["title"]: entry["title"] = title
        else:
            # No paperId — can only try OA / CR by DOI or title
            corpus_no_pid.append(paper_dict)

    for corpus_paper in data.get("papers", {}).values():
        _visit(corpus_paper)
        for ref in (corpus_paper.get("references") or []):
            _visit(ref)
        for cit in (corpus_paper.get("citations") or []):
            _visit(cit)

    return need_abstract, corpus_no_pid


def apply_abstracts(abstract_map, need_abstract):
    """Write fetched abstracts back into every node in the cache."""
    filled = 0
    for pid, abstract in abstract_map.items():
        if not abstract:
            continue
        for node in need_abstract[pid]["nodes"]:
            node["abstract"] = abstract
            filled += 1
    return filled


# ── Main ────────────────────────────────────────────────────────────────────────
def _make_parser():
    parser = argparse.ArgumentParser(
        description="Enrich a literature_explorer cache with missing abstracts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cache",   required=True,
                        help="Cache file to enrich (e.g. references_cache.json)")
    parser.add_argument("--limit",   type=int, default=0,
                        help="Max unique papers to fetch abstracts for (0 = no limit)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report gaps without fetching anything")
    parser.add_argument("--sources", nargs="+", choices=["ss", "oa", "cr"],
                        default=["ss", "oa", "cr"],
                        help="Sources to try, in order (default: ss oa cr)")
    return parser

def run(args):
    cache_path = Path(args.cache)
    if not cache_path.exists():
        sys.exit(f"Cache file not found: {cache_path}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n📦 Loading cache: {cache_path}")
    data = json.loads(cache_path.read_text(encoding="utf-8"))

    # ── Audit ─────────────────────────────────────────────────────────────────
    print("\n🔎 Auditing cache for missing abstracts...")
    need_abstract, corpus_no_pid = collect_missing(data)
    n_pids = len(need_abstract)
    n_nodes = sum(len(v["nodes"]) for v in need_abstract.values())

    print(f"  Papers with a paperId needing abstract : {n_pids:>6}  ({n_nodes} total nodes)")
    print(f"  Corpus papers without paperId          : {len(corpus_no_pid):>6}")

    if args.dry_run:
        print("\n  [dry-run] No changes made.")
        return

    if n_pids == 0 and not corpus_no_pid:
        print("\n✅ All papers already have abstracts — nothing to do.")
        return

    # ── Backup ────────────────────────────────────────────────────────────────
    bak = cache_path.with_suffix(".bak")
    shutil.copy2(cache_path, bak)
    print(f"\n💾 Backup saved: {bak}")

    # nodes_filled  : total nodes updated across all sources
    # oa_cr_attempted: individual (expensive) OA / CR calls made — governs --limit
    nodes_filled    = 0
    oa_cr_attempted = 0
    SAVE_EVERY      = 25   # write cache to disk every N papers during OA/CR phases

    def _save():
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Source 1: Semantic Scholar batch ──────────────────────────────────────
    # SS batch is very cheap (1 HTTP call per 500 IDs), so --limit does NOT apply.
    # We save after every chunk so a crash mid-run loses at most one chunk's work.
    if "ss" in args.sources and need_abstract:
        pids   = list(need_abstract.keys())
        chunks = [pids[i:i+SS_BATCH] for i in range(0, len(pids), SS_BATCH)]
        print(f"\n🔬 Source 1 — Semantic Scholar batch ({len(pids)} unique IDs, {len(chunks)} chunks)...")
        ss_nodes_filled = 0
        for k, chunk in enumerate(chunks):
            print(f"    chunk {k+1}/{len(chunks)} ({len(chunk)} IDs)...", end=" ", flush=True)
            chunk_map = fetch_ss_chunk(chunk)
            filled    = apply_abstracts(chunk_map, need_abstract)
            nodes_filled    += filled
            ss_nodes_filled += filled
            _save()
            print(f"→ {filled} nodes filled, saved")
        print(f"  ✓ SS total: {ss_nodes_filled} nodes filled")

        # Remove from need_abstract any paper now satisfied
        need_abstract = {pid: v for pid, v in need_abstract.items()
                         if not any(n.get("abstract") for n in v["nodes"])}

    def _remaining_budget():
        """Individual OA/CR calls left in the --limit budget (None = no cap)."""
        if not args.limit:
            return None
        return max(0, args.limit - oa_cr_attempted)

    # ── Source 2: OpenAlex (batch by DOI, then title-search fallback) ────────
    if "oa" in args.sources and (need_abstract or corpus_no_pid):
        remaining = list(need_abstract.items())
        budget    = _remaining_budget()
        if budget is not None:
            remaining = remaining[:budget]

        # Split into papers with a DOI (batchable) and without (title-search only)
        with_doi    = [(pid, info) for pid, info in remaining if info["doi"]]
        without_doi = [(pid, info) for pid, info in remaining if not info["doi"]]

        total_oa = len(remaining) + len(corpus_no_pid)
        n_batches = math.ceil(len(with_doi) / OA_BATCH) if with_doi else 0
        print(f"\n🌍 Source 2 — OpenAlex ({total_oa} papers, "
              f"{len(with_doi)} with DOI → {n_batches} batch calls, "
              f"{len(without_doi) + len(corpus_no_pid)} title-search fallbacks)...")
        oa_filled = 0

        # ── 2a. Batch calls for papers with a DOI (100 per request) ──────────
        doi_to_pids = defaultdict(list)   # normalised DOI → [pid, ...]
        for pid, info in with_doi:
            doi_key = info["doi"].lower()
            doi_to_pids[doi_key].append(pid)

        doi_keys  = list(doi_to_pids.keys())
        chunks    = [doi_keys[i:i+OA_BATCH] for i in range(0, len(doi_keys), OA_BATCH)]
        for k, chunk in enumerate(chunks):
            print(f"    OA batch {k+1}/{len(chunks)} ({len(chunk)} DOIs)...", end=" ", flush=True)
            batch_map = fetch_oa_batch(chunk)   # {doi_key -> abstract}
            oa_cr_attempted += 1
            chunk_filled = 0
            for doi_key, abst in batch_map.items():
                for pid in doi_to_pids.get(doi_key, []):
                    info = need_abstract.get(pid)
                    if not info:
                        continue
                    for node in info["nodes"]:
                        if not node.get("abstract"):
                            node["abstract"] = abst
                    chunk_filled += 1
                    nodes_filled += 1
                    oa_filled    += 1
            _save()
            print(f"→ {chunk_filled} papers filled, saved")

        # ── 2b. Title-search fallback for papers without a DOI ───────────────
        for i, (pid, info) in enumerate(without_doi):
            abst = fetch_oa_abstract_single(title=info["title"])
            oa_cr_attempted += 1
            if abst:
                for node in info["nodes"]:
                    if not node.get("abstract"):
                        node["abstract"] = abst
                oa_filled  += 1
                nodes_filled += 1
            if (i + 1) % SAVE_EVERY == 0:
                _save()
                print(f"    💾 Progress saved ({i+1} title-searches done)")

        # ── 2c. corpus_no_pid papers (no paperId — try DOI then title) ───────
        for paper in corpus_no_pid:
            doi   = (paper.get("externalIds") or {}).get("DOI") or paper.get("doi")
            title = paper.get("title") or ""
            abst  = fetch_oa_abstract_single(doi=doi, title=title)
            oa_cr_attempted += 1
            if abst:
                paper["abstract"] = abst
                oa_filled  += 1
                nodes_filled += 1

        print(f"  ✓ OA filled {oa_filled} papers")

        need_abstract = {pid: v for pid, v in need_abstract.items()
                         if not any(n.get("abstract") for n in v["nodes"])}

    # ── Source 3: CrossRef (DOI-only, last resort) ────────────────────────────
    if "cr" in args.sources and need_abstract:
        has_doi  = {pid: v for pid, v in need_abstract.items() if v["doi"]}
        to_try   = list(has_doi.items())
        budget   = _remaining_budget()
        if budget is not None:
            to_try = to_try[:budget]
        print(f"\n📋 Source 3 — CrossRef ({len(to_try)} papers with DOI)...")
        cr_filled = 0
        for i, (pid, info) in enumerate(to_try):
            abst = fetch_crossref_abstract(info["doi"])
            oa_cr_attempted += 1
            if abst:
                for node in info["nodes"]:
                    if not node.get("abstract"):
                        node["abstract"] = abst
                cr_filled    += 1
                nodes_filled += 1
            if (i + 1) % 20 == 0:
                print(f"    ...{i+1}/{len(to_try)} checked  ({cr_filled} filled so far)")
            if (i + 1) % SAVE_EVERY == 0:
                _save()
                print(f"    💾 Progress saved ({i+1} papers checked)")
        print(f"  ✓ CrossRef filled {cr_filled} papers")

    # ── Final audit & save ────────────────────────────────────────────────────
    remaining_missing, _ = collect_missing(data)
    still_n = len(remaining_missing)

    print(f"\n✍️  Writing enriched cache...")
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ Done!")
    print(f"   OA/CR calls made      : {oa_cr_attempted}")
    print(f"   Nodes filled this run : {nodes_filled}")
    print(f"   Papers still missing  : {still_n}")
    print(f"   Cache                 : {cache_path}")
    print(f"   Backup                : {bak}")

    return dict(nodes_filled=nodes_filled, still_missing=still_n, cache_path=cache_path)

def main():
    parser = _make_parser()
    args   = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
