#!/usr/bin/env python3
"""
search.py — Search and explore your literature_explorer cache.

Usage:
    python -m core.search --cache references_cache.json "data availability"
    python -m core.search --cache references_cache.json "open data" --top 20
    python -m core.search --cache references_cache.json "replication" --field psychology
    python -m core.search --cache references_cache.json "data sharing" --year-min 2018 --year-max 2023
    python -m core.search --cache references_cache.json "reproducibility" --sort citations
    python -m core.search --cache references_cache.json "reproducibility" --sort trending
    python -m core.search --cache references_cache.json --author "Nosek" --min-citations 100
    python -m core.search --cache references_cache.json --regex "open.*(science|access)"
    python -m core.search --cache references_cache.json '"data availability"' --search-in title
    python -m core.search --cache references_cache.json --stats
    python -m core.search --cache references_cache.json "open science" --export results.bib
    python -m core.search --cache references_cache.json "data" --new-only
    python -m core.search --cache references_cache.json "open science" --exclude preprint blog
    python -m core.search --cache references_cache.json --type meta-analysis
    python -m core.search --cache references_cache.json --venue "Nature"
    python -m core.search --cache references_cache.json --semantic "open science reform practices"
    python -m core.search --cache references_cache.json "data" --export out.csv --export-format csv
"""

import re, json, math, argparse, sys, shlex
from datetime import date
from pathlib import Path
from collections import defaultdict

CACHE_FILE = Path("literature_explorer_cache.json")
CURRENT_YEAR = date.today().year

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

# ── Keyword parsing ────────────────────────────────────────────────────────────
def parse_keywords(raw_keywords, regex_mode=False):
    """
    Parse the raw keyword list into tokens, respecting quoted phrases.
    e.g. ['"data availability"', 'open', 'science']
      → ['data availability', 'open', 'science']
    Each token is either a literal phrase or (if regex_mode) a compiled pattern.
    Returns list of (token_str, compiled_regex_or_None).
    """
    joined = " ".join(raw_keywords)
    try:
        tokens = shlex.split(joined)
    except ValueError:
        tokens = raw_keywords

    parsed = []
    for tok in tokens:
        if regex_mode:
            try:
                compiled = re.compile(tok, re.IGNORECASE)
            except re.error as e:
                sys.exit(f"Invalid regex '{tok}': {e}")
            parsed.append((tok, compiled))
        else:
            parsed.append((tok, None))
    return parsed

def _token_matches(token_str, compiled_re, text):
    if compiled_re:
        return bool(compiled_re.search(text))
    return token_str.lower() in text.lower()

# ── Paper type detection ───────────────────────────────────────────────────────
_TYPE_PATTERNS = {
    "meta-analysis":  [r"meta.anal", r"systematic review", r"pooled effect"],
    "review":         [r"we review", r"literature review", r"narrative review", r"we summarize"],
    "methods":        [r"we present a", r"we introduce", r"new method", r"we propose", r"framework for"],
    "replication":    [r"replicat", r"direct replication", r"failed to replicate"],
    "empirical":      [r"\bN\s*=\s*\d", r"\bparticipants\b", r"we collected", r"\bexperiment\b",
                       r"\bsurvey\b", r"sample of"],
    "theoretical":    [r"we argue", r"\btheoretical\b", r"conceptual framework", r"we propose a theory"],
}

def detect_paper_type(abstract):
    """Heuristic paper type classifier. Returns str or None."""
    if not abstract:
        return None
    text = abstract.lower()
    # Priority order matters: meta-analysis before review, replication before empirical
    for ptype in ("meta-analysis", "review", "replication", "methods", "empirical", "theoretical"):
        for pat in _TYPE_PATTERNS[ptype]:
            if re.search(pat, text, re.IGNORECASE):
                return ptype
    return None

# ── Semantic search via TF-IDF ────────────────────────────────────────────────
def semantic_search(pool, query, top_n=None):
    """
    Rank all papers in pool by TF-IDF cosine similarity to query string.
    Returns dict pid -> float score in [0,1].
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        sys.exit("scikit-learn is required for --semantic. pip install scikit-learn")

    pids   = list(pool.keys())
    texts  = [(pool[p].get("abstract") or "") + " " + (pool[p].get("title") or "")
              for p in pids]

    vecs = TfidfVectorizer(stop_words="english", max_features=8000).fit_transform(
        [query] + texts
    )
    sims = cosine_similarity(vecs[0:1], vecs[1:]).flatten()
    return {pids[i]: float(sims[i]) for i in range(len(pids))}

# ── Search ─────────────────────────────────────────────────────────────────────
def score_match(paper, tokens, mode="any", search_in="all"):
    title    = paper.get("title")    or ""
    abstract = paper.get("abstract") or ""

    if search_in == "title":
        check_title    = title;    check_abstract = ""
    elif search_in == "abstract":
        check_title    = "";       check_abstract = abstract
    else:
        check_title    = title;    check_abstract = abstract

    hits_title    = 0
    hits_abstract = 0
    matched_tokens = set()

    for tok, compiled in tokens:
        hit_t = bool(check_title)    and _token_matches(tok, compiled, check_title)
        hit_a = bool(check_abstract) and _token_matches(tok, compiled, check_abstract)
        if hit_t:
            hits_title += 1
            matched_tokens.add(tok)
        if hit_a:
            hits_abstract += 1
            matched_tokens.add(tok)

    if mode == "all" and len(matched_tokens) < len(tokens):
        return 0.0
    if mode == "any" and not matched_tokens:
        return 0.0

    return hits_title * 3 + hits_abstract * 1

def search(pool, tokens, mode="any", field_filter=None,
           year_min=None, year_max=None,
           min_citations=None, max_citations=None,
           author_filter=None, search_in="all",
           sort="relevance", top=30, regex_mode=False,
           new_only=False, exclude_tokens=None,
           type_filter=None, venue_filter=None,
           sem_scores=None):
    results = []
    for pid, paper in pool.items():
        # ── New-only: skip corpus papers ──
        if new_only and paper.get("_in_corpus"):
            continue

        # ── Year filter ──
        year = paper.get("year")
        try:
            year = int(year) if year else None
        except (ValueError, TypeError):
            year = None
        if year_min and (year is None or year < year_min):
            continue
        if year_max and (year is None or year > year_max):
            continue

        # ── Citation count filter ──
        cites = paper.get("citationCount") or 0
        if min_citations is not None and cites < min_citations:
            continue
        if max_citations is not None and cites > max_citations:
            continue

        # ── Field filter ──
        if field_filter:
            fields = [f.lower() for f in (paper.get("fieldsOfStudy") or [])]
            if not any(field_filter.lower() in f for f in fields):
                continue

        # ── Author filter ──
        if author_filter:
            authors = paper.get("authors") or []
            if not any(author_filter.lower() in (a.get("name") or "").lower()
                       for a in authors):
                continue

        # ── Venue filter ──
        if venue_filter:
            venue = (paper.get("venue") or "").lower()
            if venue_filter.lower() not in venue:
                continue

        # ── Type filter ──
        if type_filter:
            detected = detect_paper_type(paper.get("abstract") or "")
            if detected != type_filter:
                continue

        # ── Keyword match ──
        if tokens:
            sc = score_match(paper, tokens, mode=mode, search_in=search_in)
            if sc == 0.0:
                continue
        else:
            sc = 0.0

        # ── Exclude terms ──
        if exclude_tokens:
            text = ((paper.get("title") or "") + " " + (paper.get("abstract") or "")).lower()
            if any(_token_matches(tok, compiled, text) for tok, compiled in exclude_tokens):
                continue

        results.append((pid, sc, paper))

    # ── Blend semantic scores if provided ──
    if sem_scores:
        if tokens:
            # Blend: 50% keyword + 50% semantic
            results = [(pid, 0.5 * sc + 0.5 * sem_scores.get(pid, 0.0), paper)
                       for pid, sc, paper in results]
        else:
            results = [(pid, sem_scores.get(pid, 0.0), paper)
                       for pid, sc, paper in results]
        # Filter zero semantic score if semantic-only
        if not tokens:
            results = [(pid, sc, paper) for pid, sc, paper in results if sc > 0]

    # Sort
    if sort == "citations":
        results.sort(key=lambda x: -(x[2].get("citationCount") or 0))
    elif sort == "year":
        results.sort(key=lambda x: -(int(x[2].get("year") or 0)))
    elif sort == "trending":
        def velocity(paper):
            cites = paper.get("citationCount") or 0
            yr    = paper.get("year")
            try: yr = int(yr) if yr else None
            except: yr = None
            if not yr:
                return 0.0
            return cites / max(1, CURRENT_YEAR - yr)
        results.sort(key=lambda x: -velocity(x[2]))
    else:  # relevance
        results.sort(key=lambda x: -x[1])

    return results[:top]

# ── Display ────────────────────────────────────────────────────────────────────
def _highlight(text, tokens, regex_mode=False):
    for tok, compiled in tokens:
        if regex_mode and compiled:
            text = compiled.sub(lambda m: f"[{m.group(0)}]", text)
        else:
            text = re.sub(f'(?i)({re.escape(tok)})', r'[\1]', text)
    return text

def fmt_paper(rank, pid, score, paper, tokens, verbose=False, regex_mode=False):
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
    venue    = paper.get("venue") or ""
    cluster  = paper.get("_cluster")
    ptype    = detect_paper_type(abstract) if abstract else None

    title_hl = _highlight(title, tokens, regex_mode=regex_mode)

    tags = []
    if cluster is not None:
        tags.append(f"[cluster {cluster+1}]")
    if ptype:
        tags.append(f"[{ptype}]")
    tag_str = "  " + "  ".join(tags) if tags else ""

    lines = [
        f"\n{'─'*72}",
        f"  {rank:>3}. {title_hl}  ({year}){tag_str}",
        f"       {auth_str}",
        f"       Citations: {cites}  |  Fields: {fields}",
    ]
    if venue:
        lines.append(f"       Venue: {venue}")
    if url:
        lines.append(f"       {url}")
    if verbose and abstract:
        snip = abstract[:400] + ("..." if len(abstract) > 400 else "")
        snip = _highlight(snip, tokens, regex_mode=regex_mode)
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

# ── Export ─────────────────────────────────────────────────────────────────────
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

def to_csv_row(paper):
    doi      = (paper.get("externalIds") or {}).get("DOI","")
    authors  = "; ".join(a.get("name","") for a in (paper.get("authors") or []))
    fields   = "; ".join(paper.get("fieldsOfStudy") or [])
    abstract = (paper.get("abstract") or "").replace('"', '""')
    title    = (paper.get("title") or "").replace('"', '""')
    url      = f"https://doi.org/{doi}" if doi else ""
    venue    = paper.get("venue") or ""
    return (f'"{title}","{authors}",{paper.get("year","")},"{doi}",'
            f'{paper.get("citationCount","")},"{venue}","{fields}","{abstract}","{url}"')

def to_ris(paper):
    doi     = (paper.get("externalIds") or {}).get("DOI","")
    authors = paper.get("authors") or []
    lines   = ["TY  - JOUR"]
    lines.append(f"TI  - {paper.get('title','')}")
    for a in authors:
        lines.append(f"AU  - {a.get('name','')}")
    if paper.get("year"):
        lines.append(f"PY  - {paper['year']}")
    if doi:
        lines.append(f"DO  - {doi}")
    if paper.get("abstract"):
        lines.append(f"AB  - {paper['abstract']}")
    lines.append("ER  -")
    return "\n".join(lines)

def export_obsidian(results, out_dir):
    obs_dir = out_dir / "obsidian"
    obs_dir.mkdir(parents=True, exist_ok=True)
    for _, _, paper in results:
        doi     = (paper.get("externalIds") or {}).get("DOI","")
        url     = f"https://doi.org/{doi}" if doi else ""
        authors = "; ".join(a.get("name","") for a in (paper.get("authors") or []))
        title   = paper.get("title") or "untitled"
        safe    = re.sub(r'[<>:"/\\|?*]', '-', title)[:80]
        fields  = paper.get("fieldsOfStudy") or []
        cluster = paper.get("_cluster")
        ptype   = detect_paper_type(paper.get("abstract") or "")
        abstract = paper.get("abstract") or ""
        md = (
            f"---\n"
            f"title: \"{title}\"\n"
            f"authors: \"{authors}\"\n"
            f"year: {paper.get('year','')}\n"
            f"doi: \"{doi}\"\n"
            f"url: \"{url}\"\n"
            f"citations: {paper.get('citationCount','')}\n"
            f"venue: \"{paper.get('venue','')}\"\n"
            f"fields: {json.dumps(fields)}\n"
            + (f"cluster: {cluster}\n" if cluster is not None else "")
            + (f"type: {ptype}\n" if ptype else "")
            + f"---\n\n"
            f"## {title}\n\n"
            + (f"{abstract}\n" if abstract else "")
        )
        (obs_dir / f"{safe}.md").write_text(md, encoding="utf-8")
    print(f"  📝 Obsidian notes → {obs_dir}/  ({len(results)} files)")

def export_results(results, export_path, fmt, proj_out):
    path = Path(export_path)
    if path.parent == Path("."):
        path = proj_out / path.name

    if fmt == "obsidian":
        export_obsidian(results, path.parent / path.stem)
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        header = "title,authors,year,doi,citations,venue,fields,abstract,url"
        rows   = [header] + [to_csv_row(p) for _, _, p in results]
        path.write_text("\n".join(rows), encoding="utf-8")
        print(f"\n  📄 CSV exported ({len(results)} rows) → {path}")
    elif fmt == "ris":
        text = "\n\n".join(to_ris(p) for _, _, p in results)
        path.write_text(text, encoding="utf-8")
        print(f"\n  📄 RIS exported ({len(results)} entries) → {path}")
    else:  # bib
        text = "\n\n".join(to_bibtex(p) for _, _, p in results)
        path.write_text(text, encoding="utf-8")
        print(f"\n  📚 BibTeX exported ({len(results)} entries) → {path}")

# ── Main ───────────────────────────────────────────────────────────────────────
def _make_parser():
    parser = argparse.ArgumentParser(
        description="Search your literature_explorer cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("keywords", nargs="*",
                        help="Keywords to search for. Quote phrases: '\"data availability\"'")
    parser.add_argument("--cache",    required=True,
                        help="Cache file produced by literature_explorer.py")
    parser.add_argument("--top",      type=int, default=30,
                        help="Max results to show (default: 30)")
    parser.add_argument("--sort",     choices=["relevance","citations","year","trending"],
                        default="relevance",
                        help="Sort order (default: relevance; trending=citation velocity)")
    parser.add_argument("--mode",     choices=["any","all"], default="any",
                        help="Match any keyword (default) or require all keywords")
    parser.add_argument("--field",    metavar="FIELD",
                        help="Filter by field of study (e.g. psychology)")
    parser.add_argument("--year-min", type=int, metavar="YEAR")
    parser.add_argument("--year-max", type=int, metavar="YEAR")
    parser.add_argument("--min-citations", type=int, metavar="N")
    parser.add_argument("--max-citations", type=int, metavar="N")
    parser.add_argument("--author",   metavar="NAME",
                        help="Filter by author name (substring, case-insensitive)")
    parser.add_argument("--search-in", choices=["all","title","abstract"],
                        default="all", dest="search_in")
    parser.add_argument("--regex",    action="store_true",
                        help="Treat keywords as regular expressions")
    parser.add_argument("--verbose",  action="store_true",
                        help="Show abstract snippets with hits highlighted")
    parser.add_argument("--export",   metavar="FILE",
                        help="Export results to file")
    parser.add_argument("--export-format", choices=["bib","csv","ris","obsidian"],
                        default="bib", dest="export_format",
                        help="Export format (default: bib)")
    parser.add_argument("--stats",    action="store_true",
                        help="Show cache statistics and exit")
    parser.add_argument("--new-only", action="store_true", dest="new_only",
                        help="Exclude corpus papers (only show discovered candidates)")
    parser.add_argument("--exclude",  nargs="+", metavar="TERM",
                        help="Exclude papers containing any of these terms in title/abstract")
    parser.add_argument("--type",
                        choices=["review","meta-analysis","empirical","methods",
                                 "theoretical","replication"],
                        help="Filter by detected paper type (heuristic on abstract)")
    parser.add_argument("--venue",    metavar="VENUE",
                        help="Filter by venue/journal (substring, case-insensitive)")
    parser.add_argument("--semantic", metavar="QUERY",
                        help="Semantic (TF-IDF) search over abstracts+titles")
    parser.add_argument("--semantic-like", metavar="TITLE", dest="semantic_like",
                        help="Find similar papers to this title in the pool")
    return parser

def run(args):
    data      = load(args.cache)
    pool      = all_papers(data)
    proj_name = Path(args.cache).stem.removesuffix("_cache")
    proj_out  = Path("outputs") / proj_name

    if args.stats or (not args.keywords and not args.author
                      and args.min_citations is None and args.max_citations is None
                      and not getattr(args, "semantic", None)
                      and not getattr(args, "semantic_like", None)
                      and not getattr(args, "type", None)
                      and not getattr(args, "venue", None)):
        show_stats(data, pool)
        if not args.keywords:
            print("  Tip: add keywords or --semantic QUERY to search.\n")
            return

    # Keyword tokens
    tokens = parse_keywords(args.keywords, regex_mode=args.regex) if args.keywords else []

    # Exclude tokens
    exclude_tokens = None
    if getattr(args, "exclude", None):
        exclude_tokens = parse_keywords(args.exclude, regex_mode=args.regex)

    # Semantic scores
    sem_scores = None
    semantic_query = getattr(args, "semantic", None)
    semantic_like  = getattr(args, "semantic_like", None)
    if semantic_like:
        # Find best title match in pool
        best_pid, best_sim = None, 0.0
        query_low = semantic_like.lower()
        for pid, p in pool.items():
            t = (p.get("title") or "").lower()
            common = len(set(query_low.split()) & set(t.split()))
            if common > best_sim:
                best_sim, best_pid = common, pid
        if best_pid:
            anchor = pool[best_pid]
            semantic_query = (anchor.get("abstract") or "") + " " + (anchor.get("title") or "")
            print(f"  🔍 Semantic-like anchor: {anchor.get('title','?')[:70]}")
        else:
            print("  ⚠  Could not find matching paper for --semantic-like")

    if semantic_query:
        print("  🧮 Computing TF-IDF semantic scores...")
        sem_scores = semantic_search(pool, semantic_query)

    results = search(
        pool, tokens,
        mode=args.mode,
        field_filter=args.field,
        year_min=args.year_min,
        year_max=args.year_max,
        min_citations=args.min_citations,
        max_citations=args.max_citations,
        author_filter=args.author,
        search_in=args.search_in,
        sort=args.sort,
        top=args.top,
        regex_mode=args.regex,
        new_only=getattr(args, "new_only", False),
        exclude_tokens=exclude_tokens,
        type_filter=getattr(args, "type", None),
        venue_filter=getattr(args, "venue", None),
        sem_scores=sem_scores,
    )

    # ── Summary header ──
    kw_str = ", ".join(f'"{t}"' for t, _ in tokens) if tokens else "(no keywords)"
    filters = []
    if args.field:          filters.append(f"field={args.field}")
    if args.year_min:       filters.append(f"from {args.year_min}")
    if args.year_max:       filters.append(f"to {args.year_max}")
    if args.min_citations:  filters.append(f"cites≥{args.min_citations}")
    if args.max_citations:  filters.append(f"cites≤{args.max_citations}")
    if args.author:         filters.append(f"author={args.author}")
    if args.search_in != "all": filters.append(f"in={args.search_in}")
    if args.regex:          filters.append("regex")
    if getattr(args, "new_only", False):  filters.append("new-only")
    if getattr(args, "exclude", None):    filters.append(f"excl:{','.join(args.exclude)}")
    if getattr(args, "type", None):       filters.append(f"type={args.type}")
    if getattr(args, "venue", None):      filters.append(f"venue={args.venue}")
    if sem_scores:          filters.append("semantic")
    filter_str = f"  [{', '.join(filters)}]" if filters else ""

    print(f"\n🔎 Search: {kw_str}{filter_str}  →  {len(results)} results "
          f"(sorted by {args.sort}, mode={args.mode})")

    if not results:
        print("\n  No matches found. Try broader keywords, --mode any, or remove filters.\n")
        return []

    for rank, (pid, score, paper) in enumerate(results, 1):
        print(fmt_paper(rank, pid, score, paper, tokens,
                        verbose=args.verbose, regex_mode=args.regex))

    print(f"\n{'─'*72}")
    print(f"  {len(results)} papers shown")
    if len(results) == args.top:
        print(f"  (limit reached — use --top N for more)")

    if args.export:
        fmt = getattr(args, "export_format", "bib") or "bib"
        export_results(results, args.export, fmt, proj_out)

    print()
    return results

def main():
    parser = _make_parser()
    args   = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
