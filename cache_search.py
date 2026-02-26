#!/usr/bin/env python3
"""
cache_search.py — Search and explore your literature_explorer cache.

Usage:
    python cache_search.py "data availability"
    python cache_search.py "open data" --top 20
    python cache_search.py "replication" --field psychology
    python cache_search.py "data sharing" --year-min 2018 --year-max 2023
    python cache_search.py "reproducibility" --sort citations
    python cache_search.py --stats
    python cache_search.py "open science" --export results.bib
"""

import re, json, math, argparse, sys
from pathlib import Path
from collections import defaultdict

CACHE_FILE = Path("literature_explorer_cache.json")

# ── Load ───────────────────────────────────────────────────────────────────────
def load(cache_path):
    path = Path(cache_path)
    if not path.exists():
        sys.exit(f"Cache file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        sys.exit(f"Could not parse cache: {e}")
    return data

def all_papers(data):
    """
    Flatten the cache into a single deduplicated dict of paperId → paper.
    Sources:
      - data["papers"]   : corpus papers (with embedded refs + citations)
      - refs/citations embedded in each corpus paper
    """
    pool = {}

    def add(p):
        if not p or not p.get("title"):
            return
        pid = p.get("paperId") or p.get("title","")[:60]
        if pid and pid not in pool:
            pool[pid] = p

    for entry in data.get("papers", {}).values():
        add(entry)
        for ref in (entry.get("references") or []):
            add(ref)
        for cit in (entry.get("citations") or []):
            add(cit)

    return pool

# ── Search ─────────────────────────────────────────────────────────────────────
def score_match(paper, keywords, mode="any"):
    """
    Returns a relevance score in [0, ∞) for a paper against a list of keywords.
    Checks title (weight 3×) and abstract (weight 1×).
    mode="any"  → paper must contain at least one keyword
    mode="all"  → paper must contain all keywords
    """
    title    = (paper.get("title")    or "").lower()
    abstract = (paper.get("abstract") or "").lower()

    hits_title    = 0
    hits_abstract = 0
    matched_kws   = set()

    for kw in keywords:
        kw_l = kw.lower()
        if kw_l in title:
            hits_title += 1
            matched_kws.add(kw)
        if kw_l in abstract:
            hits_abstract += 1
            matched_kws.add(kw)

    if mode == "all" and len(matched_kws) < len(keywords):
        return 0.0
    if mode == "any" and not matched_kws:
        return 0.0

    return hits_title * 3 + hits_abstract * 1

def search(pool, keywords, mode="any", field_filter=None,
           year_min=None, year_max=None, sort="relevance", top=30):
    results = []
    for pid, paper in pool.items():
        # Year filter
        year = paper.get("year")
        try:
            year = int(year) if year else None
        except (ValueError, TypeError):
            year = None
        if year_min and (year is None or year < year_min):
            continue
        if year_max and (year is None or year > year_max):
            continue

        # Field filter
        if field_filter:
            fields = [f.lower() for f in (paper.get("fieldsOfStudy") or [])]
            if not any(field_filter.lower() in f for f in fields):
                continue

        # Keyword match
        sc = score_match(paper, keywords, mode=mode)
        if sc > 0:
            results.append((pid, sc, paper))

    # Sort
    if sort == "citations":
        results.sort(key=lambda x: -(x[2].get("citationCount") or 0))
    elif sort == "year":
        results.sort(key=lambda x: -(int(x[2].get("year") or 0)))
    else:  # relevance
        results.sort(key=lambda x: -x[1])

    return results[:top]

# ── Display ────────────────────────────────────────────────────────────────────
def fmt_paper(rank, pid, score, paper, keywords, verbose=False):
    doi      = (paper.get("externalIds") or {}).get("DOI", "")
    url      = f"https://doi.org/{doi}" if doi else ""
    title    = paper.get("title", "(no title)")
    year     = paper.get("year", "?")
    cites    = paper.get("citationCount", "?")
    auths    = paper.get("authors") or []
    auth_str = ", ".join(a.get("name","") for a in auths[:2])
    if len(auths) > 2: auth_str += " et al."
    fields   = ", ".join(paper.get("fieldsOfStudy") or []) or "—"
    abstract = paper.get("abstract") or ""

    # Highlight keyword hits in title
    title_hl = title
    for kw in keywords:
        title_hl = re.sub(f'(?i)({re.escape(kw)})', r'[\1]', title_hl)

    lines = [
        f"\n{'─'*72}",
        f"  {rank:>3}. {title_hl}  ({year})",
        f"       {auth_str}",
        f"       Citations: {cites}  |  Fields: {fields}",
    ]
    if url:
        lines.append(f"       {url}")
    if verbose and abstract:
        # Show abstract snippet with keyword hits highlighted
        snip = abstract[:400] + ("..." if len(abstract) > 400 else "")
        for kw in keywords:
            snip = re.sub(f'(?i)({re.escape(kw)})', r'[\1]', snip)
        lines.append(f"\n       {snip}")
    return "\n".join(lines)

# ── Stats ──────────────────────────────────────────────────────────────────────
def show_stats(data, pool):
    years = defaultdict(int)
    fields_count = defaultdict(int)
    sources = defaultdict(int)
    total_cites = 0
    with_abstract = 0

    for p in pool.values():
        y = p.get("year")
        if y:
            try: years[int(y)] += 1
            except: pass
        for f in (p.get("fieldsOfStudy") or []):
            fields_count[f.lower()] += 1
        src = p.get("_source", "semanticscholar")
        sources[src] += 1
        c = p.get("citationCount") or 0
        total_cites += c
        if p.get("abstract"):
            with_abstract += 1

    corpus_n = len(data.get("papers", {}))
    print(f"\n{'═'*60}")
    print(f"  📊 Cache Statistics")
    print(f"{'═'*60}")
    print(f"  Corpus papers (your bib):       {corpus_n}")
    print(f"  Total papers in pool:           {len(pool)}")
    print(f"  Papers with abstracts:          {with_abstract} ({100*with_abstract//max(len(pool),1)}%)")
    print(f"  Total citation count (sum):     {total_cites:,}")

    print(f"\n  📅 Papers by decade:")
    for decade in sorted(set(y // 10 * 10 for y in years if y > 1900)):
        n = sum(v for k, v in years.items() if k // 10 * 10 == decade)
        bar = "█" * (n // 10)
        print(f"     {decade}s  {bar} {n}")

    print(f"\n  🏷  Top fields:")
    for field, n in sorted(fields_count.items(), key=lambda x: -x[1])[:10]:
        print(f"     {n:>5}  {field}")

    print(f"\n  🔌 Sources:")
    for src, n in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"     {n:>5}  {src}")
    print()

# ── BibTeX export ──────────────────────────────────────────────────────────────
def make_key(paper):
    auths = paper.get("authors") or []
    last  = auths[0]["name"].split()[-1] if auths else "Unknown"
    year  = paper.get("year", "")
    word  = ((paper.get("title") or "").split() or ["paper"])[0]
    return re.sub(r'[^a-zA-Z0-9]', '', f"{last}{year}{word}")[:30]

def to_bibtex(paper):
    key     = make_key(paper)
    authors = " and ".join(a.get("name","") for a in (paper.get("authors") or []))
    doi     = (paper.get("externalIds") or {}).get("DOI","")
    title   = (paper.get("title") or "").replace("{","").replace("}","")
    return (f"@article{{{key},\n"
            f"  title  = {{{title}}},\n"
            f"  author = {{{authors}}},\n"
            f"  year   = {{{paper.get('year','')}}},\n"
            f"  doi    = {{{doi}}},\n"
            f"  note   = {{citationCount={paper.get('citationCount','')}}}\n}}")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Search your literature_explorer cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("keywords", nargs="*",
                        help="Keywords to search for (space or quote-separated)")
    parser.add_argument("--cache",    default="literature_explorer_cache.json",
                        help="Cache file (default: literature_explorer_cache.json)")
    parser.add_argument("--top",      type=int, default=30,
                        help="Max results to show (default: 30)")
    parser.add_argument("--sort",     choices=["relevance","citations","year"],
                        default="relevance",
                        help="Sort order (default: relevance)")
    parser.add_argument("--mode",     choices=["any","all"], default="any",
                        help="Match any keyword (default) or require all keywords")
    parser.add_argument("--field",    metavar="FIELD",
                        help="Filter by field of study (e.g. psychology)")
    parser.add_argument("--year-min", type=int, metavar="YEAR",
                        help="Only papers published from this year")
    parser.add_argument("--year-max", type=int, metavar="YEAR",
                        help="Only papers published up to this year")
    parser.add_argument("--verbose",  action="store_true",
                        help="Show abstract snippets with hits highlighted")
    parser.add_argument("--export",   metavar="FILE",
                        help="Export results to a .bib file (default dir: outputs/)")
    parser.add_argument("--stats",    action="store_true",
                        help="Show cache statistics and exit")
    args = parser.parse_args()

    data = load(args.cache)
    pool = all_papers(data)

    if args.stats or not args.keywords:
        show_stats(data, pool)
        if not args.keywords:
            print("  Tip: add keywords to search, e.g.:")
            print('       python cache_search.py "data availability" --top 20\n')
            return

    keywords = args.keywords
    results = search(
        pool, keywords,
        mode=args.mode,
        field_filter=args.field,
        year_min=args.year_min,
        year_max=args.year_max,
        sort=args.sort,
        top=args.top,
    )

    kw_str = ", ".join(f'"{k}"' for k in keywords)
    filters = []
    if args.field:    filters.append(f"field={args.field}")
    if args.year_min: filters.append(f"from {args.year_min}")
    if args.year_max: filters.append(f"to {args.year_max}")
    filter_str = f"  [{', '.join(filters)}]" if filters else ""

    print(f"\n🔎 Search: {kw_str}{filter_str}  →  {len(results)} results "
          f"(sorted by {args.sort}, mode={args.mode})")

    if not results:
        print("\n  No matches found. Try broader keywords or --mode any.\n")
        return

    for rank, (pid, score, paper) in enumerate(results, 1):
        print(fmt_paper(rank, pid, score, paper, keywords, verbose=args.verbose))

    print(f"\n{'─'*72}")
    print(f"  {len(results)} papers shown")
    if len(results) == args.top:
        print(f"  (limit reached — use --top N for more)")

    if args.export:
        bib_path = Path(args.export)
        # If only a bare filename (no parent dir), place it in outputs/
        if bib_path.parent == Path("."):
            bib_path = Path("outputs") / bib_path
        bib_path.parent.mkdir(parents=True, exist_ok=True)
        bib_text = "\n\n".join(to_bibtex(p) for _, _, p in results)
        bib_path.write_text(bib_text, encoding="utf-8")
        print(f"\n  📚 Exported {len(results)} entries → {bib_path}")

    print()

if __name__ == "__main__":
    main()
