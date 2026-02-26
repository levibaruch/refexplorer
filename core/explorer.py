#!/usr/bin/env python3
"""
Literature Explorer
Finds papers you may have missed, given a .bib file or folder of PDFs.
Usage:
    python literature_explorer.py --bib refs.bib
    python literature_explorer.py --pdf-dir /path/to/pdfs
    python literature_explorer.py --bib refs.bib --out ./results --cache ~/my_cache.json
"""

import re, time, math, json, argparse, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")

# ── Config ─────────────────────────────────────────────────────────────────────
SS_BASE           = "https://api.semanticscholar.org/graph/v1/paper"
OA_BASE           = "https://api.openalex.org/works"
SLEEP             = 1.1
DEFAULT_CACHE     = Path("literature_explorer_cache.json")

# Fetch metadata + references + citations in ONE call per paper
SS_FIELDS = (
    "paperId,title,year,citationCount,authors,externalIds,abstract,fieldsOfStudy,"
    "venue,publicationVenue,"
    "references.paperId,references.title,references.year,references.citationCount,"
    "references.authors,references.externalIds,references.fieldsOfStudy,"
    "citations.paperId,citations.title,citations.year,citations.citationCount,"
    "citations.authors,citations.externalIds,citations.fieldsOfStudy"
)

WEIGHT_PRESETS = {
    "balanced":          dict(co_cite=0.35, citations=0.30, recency=0.20, field=0.15),
    "highly-cited":      dict(co_cite=0.20, citations=0.60, recency=0.10, field=0.10),
    "recent":            dict(co_cite=0.25, citations=0.15, recency=0.50, field=0.10),
    "interdisciplinary": dict(co_cite=0.35, citations=0.25, recency=0.15, field=0.25),
}

BONUS_FIELDS = {
    "psychology", "sociology", "political science", "social science",
    "education", "communication", "public health", "medicine", "neuroscience"
}

# ── Cache ──────────────────────────────────────────────────────────────────────
def load_cache(cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            n = len(data.get("papers", {}))
            print(f"  📦 Cache loaded: {n} papers already resolved")
            return data
        except Exception:
            print("  ⚠  Cache corrupted — starting fresh")
    else:
        print(f"  No cache yet — will create {cache_path.name}")
    return {"papers": {}}

def save_cache(cache, cache_path):
    Path(cache_path).write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

# ── Step 1: Parse corpus ───────────────────────────────────────────────────────
def _bib_field(pattern, entry):
    """
    Extract a BibTeX field value that may be delimited by {…} (possibly with
    one level of nested braces) or by "…".  Returns the cleaned string or None.
    """
    # Matches:  field = { content {nested} content }  OR  field = "content"
    pat = re.compile(
        pattern + r'\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|"([^"]*)")',
        re.IGNORECASE | re.DOTALL,
    )
    m = pat.search(entry)
    if not m:
        return None
    raw = (m.group(1) if m.group(1) is not None else m.group(2)) or ""
    # Strip any remaining inner braces used for capitalisation, e.g. {R}eplication
    return re.sub(r'\{([^}]*)\}', r'\1', raw).strip()

def parse_bib(bib_path):
    text = Path(bib_path).read_text(encoding="utf-8", errors="ignore")
    papers = []
    for entry in re.split(r'(?=@\w+\{)', text):
        title = _bib_field("title", entry) or ""
        year  = _bib_field("year",  entry) or ""
        doi   = _bib_field("doi",   entry)
        if doi:
            doi = doi.replace("https://doi.org/", "").strip().rstrip(".,)")
            if not doi.startswith("10."):
                doi = None
        papers.append({"title": title, "doi": doi, "year": year})
    return [p for p in papers if p["title"] or p["doi"]]

def parse_pdf_dir(pdf_dir):
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("Missing dependency: pip install pypdf")
    papers = []
    for pdf in Path(pdf_dir).glob("*.pdf"):
        try:
            reader = PdfReader(str(pdf))
            text = (reader.pages[0].extract_text() or "")[:2000]
            doi_m = re.search(r'10\.\d{4,}/\S+', text)
            doi = doi_m.group(0).rstrip(".,)") if doi_m else None
            lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 20]
            title = lines[0] if lines else pdf.stem
            papers.append({"title": title, "doi": doi, "year": ""})
        except Exception as e:
            print(f"  ⚠  Could not parse {pdf.name}: {e}")
    return papers

# ── Step 2: API helpers ────────────────────────────────────────────────────────
def ss_get(url, params=None):
    """GET from Semantic Scholar with retry + Retry-After header support."""
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", 2 ** (attempt + 2)))
                print(f"  ⏳ Rate limited — waiting {retry_after}s...")
                time.sleep(retry_after)
                continue
            if r.ok:
                return r.json()
            if r.status_code == 404:
                return None  # Not found — don't retry
        except requests.exceptions.Timeout:
            print(f"  ⚠  Timeout (attempt {attempt+1}), retrying...")
        except Exception as e:
            print(f"  ⚠  Request error: {e}")
        time.sleep(SLEEP)
    return None

def title_similarity(a, b):
    stopwords = {"the","a","an","of","in","and","to","for","on","with","is","are","at"}
    wa = set(re.sub(r'[^a-z ]', '', a.lower()).split()) - stopwords
    wb = set(re.sub(r'[^a-z ]', '', b.lower()).split()) - stopwords
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))

def oa_lookup(title=None, doi=None):
    """OpenAlex fallback — returns SS-shaped dict (no graph data) or None."""
    try:
        if doi:
            r = requests.get(f"{OA_BASE}/https://doi.org/{doi}",
                             params={"mailto": "literature-explorer"}, timeout=15)
            if r.ok:
                return _oa_to_ss(r.json())
        if title:
            r = requests.get(OA_BASE, params={"search": title, "per_page": 1,
                             "mailto": "literature-explorer"}, timeout=15)
            if r.ok and r.json().get("results"):
                oa = r.json()["results"][0]
                oa_title = oa.get("display_name") or ""
                if title_similarity(title, oa_title) < 0.3:
                    print(f"    ✗ OA mismatch: '{oa_title[:55]}' — skipping")
                    return None
                return _oa_to_ss(oa)
    except Exception:
        pass
    return None

def _oa_to_ss(oa):
    doi      = (oa.get("doi") or "").replace("https://doi.org/", "")
    authors  = [{"name": a.get("display_name", "?")} for a in (oa.get("authorships") or [])]
    concepts = [c.get("display_name", "") for c in (oa.get("concepts") or [])[:5]]
    return {
        "paperId":       oa.get("id", "").replace("https://openalex.org/", ""),
        "title":         oa.get("display_name") or oa.get("title", ""),
        "year":          oa.get("publication_year"),
        "citationCount": oa.get("cited_by_count", 0),
        "authors":       authors,
        "abstract":      oa.get("abstract", ""),
        "fieldsOfStudy": concepts,
        "externalIds":   {"DOI": doi},
        "references":    [],  # OA doesn't expose full reference lists cheaply
        "citations":     [],
        "_source":       "openalex",
    }

# ── Step 3: Resolve corpus (metadata + refs + citations in ONE call each) ──────
def resolve_all(corpus, cache, cache_path):
    """
    Resolves every corpus paper via a single Semantic Scholar request that
    returns metadata + full reference list + citation list together.
    Errors surface here, immediately, before any graph work begins.
    """
    resolved  = []
    failed    = []
    total     = len(corpus)
    api_calls = 0
    hits      = 0

    print(f"\n🔍 Resolving {total} papers (metadata + refs + citations in one pass)...")

    for i, paper in enumerate(corpus):
        doi       = paper.get("doi")
        title     = paper.get("title", "")
        cache_key = doi or title[:80]
        label     = (title or doi or "?")[:65]

        # Cache hit
        cached = cache["papers"].get(cache_key)
        if cached:
            resolved.append(cached)
            hits += 1
            refs_n = len(cached.get("references") or [])
            cits_n = len(cached.get("citations") or [])
            print(f"  [{i+1}/{total}] 📦 {label}  [refs:{refs_n} cits:{cits_n}]")
            continue

        # Semantic Scholar — single request gets everything
        result = None
        if doi:
            result = ss_get(f"{SS_BASE}/DOI:{doi}", {"fields": SS_FIELDS})
            api_calls += 1
            time.sleep(SLEEP)
        if not result and title:
            data = ss_get(f"{SS_BASE}/search", {"query": title, "fields": SS_FIELDS, "limit": 1})
            api_calls += 1
            time.sleep(SLEEP)
            if data and data.get("data"):
                result = data["data"][0]

        # OpenAlex fallback (corpus identity only — no graph data)
        if not result:
            print(f"  [{i+1}/{total}] ⚠  SS failed → trying OpenAlex...")
            result = oa_lookup(title=title, doi=doi)
            if result:
                print(f"    ✓ (OA, no graph) {result.get('title','?')[:60]}")
            else:
                print(f"  [{i+1}/{total}] ✗  Unresolvable: {label}")
                failed.append(paper)
                continue

        # If SS returned 0 references, try the dedicated /references endpoint.
        # This happens for some papers where the inline field is restricted.
        if not result.get("_source") and not result.get("references"):
            pid_ss = result.get("paperId")
            if pid_ss:
                ref_fields = ("paperId,title,year,citationCount,"
                              "authors,externalIds,fieldsOfStudy")
                ref_data = ss_get(
                    f"{SS_BASE}/{pid_ss}/references",
                    {"fields": ref_fields, "limit": 500},
                )
                api_calls += 1
                time.sleep(SLEEP)
                if ref_data and ref_data.get("data"):
                    result["references"] = [
                        r["citedPaper"] for r in ref_data["data"]
                        if r.get("citedPaper")
                    ]
                    print(f"    ↳ fetched {len(result['references'])} refs via /references endpoint")
                else:
                    print(f"    ↳ /references endpoint returned nothing — SS has no ref data for this paper")

        # Extract venue from publicationVenue (preferred) or venue field
        result["venue"] = (
            (result.get("publicationVenue") or {}).get("name")
            or result.get("venue")
            or ""
        )
        result["_in_corpus"] = True

        resolved.append(result)
        cache["papers"][cache_key] = result
        save_cache(cache, cache_path)

        src    = " (OA)" if result.get("_source") == "openalex" else ""
        refs_n = len(result.get("references") or [])
        cits_n = len(result.get("citations") or [])
        print(f"  [{i+1}/{total}] ✓{src} {result.get('title','?')[:55]}  "
              f"[refs:{refs_n} cits:{cits_n}]")

    print(f"\n  Summary: {len(resolved)} resolved · "
          f"{api_calls} API calls · {hits} cache hits · {len(failed)} failed")

    if failed:
        print("\n  ⚠  The following papers could not be resolved and will be skipped:")
        for p in failed:
            print(f"     • {(p.get('title') or p.get('doi') or '?')[:80]}")
        if len(resolved) < 3:
            sys.exit("\nToo few papers resolved to continue — aborting.")
        ans = input("\n  Continue with the papers that resolved? [Y/n] ").strip().lower()
        if ans == "n":
            sys.exit("Aborted.")

    return resolved

# ── Step 4: Build co-citation graph (pure in-memory — no more API calls) ───────
def build_cocitation_graph(resolved_corpus, corpus_ids):
    co_citation_count = defaultdict(int)
    candidate_meta    = {}

    for paper in resolved_corpus:
        if paper.get("_source") == "openalex":
            continue  # no graph data on OA records

        for ref in (paper.get("references") or []):
            pid = ref.get("paperId")
            if pid and pid not in corpus_ids and ref.get("title"):
                co_citation_count[pid] += 1
                # Keep richer metadata if we've seen this paper before
                if pid not in candidate_meta or (ref.get("abstract") and not candidate_meta[pid].get("abstract")):
                    candidate_meta[pid] = ref

        for cit in (paper.get("citations") or []):
            pid = cit.get("paperId")
            if pid and pid not in corpus_ids and cit.get("title"):
                co_citation_count[pid] += 1
                if pid not in candidate_meta or (cit.get("abstract") and not candidate_meta[pid].get("abstract")):
                    candidate_meta[pid] = cit

    return co_citation_count, candidate_meta

# ── Step 5: Score ──────────────────────────────────────────────────────────────
def score_paper(pid, meta, co_citation_count, current_year, weights=None):
    w         = weights or WEIGHT_PRESETS["balanced"]
    co_cite   = co_citation_count[pid]
    citations = meta.get("citationCount") or 0
    year      = int(meta.get("year") or 2000)
    recency   = max(0, 1 - (current_year - year) / 20)
    fields    = [f.lower() for f in (meta.get("fieldsOfStudy") or [])]
    bonus     = 1.5 if any(f in BONUS_FIELDS for f in fields) else 1.0
    base = (
        w["co_cite"]   * min(co_cite / 5, 1.0) +
        w["citations"] * min(math.log1p(citations) / 10, 1.0) +
        w["recency"]   * recency +
        w["field"]     * (bonus - 1.0)
    )
    return base * bonus

# ── Step 6: Semantic similarity ────────────────────────────────────────────────
def compute_similarity(resolved_corpus, top_candidates):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("  (scikit-learn not found — skipping semantic similarity)")
        return {pid: 0.0 for pid, _, _ in top_candidates}

    corpus_texts = [(p.get("abstract") or p.get("title", "")) for p in resolved_corpus]
    cand_texts   = [(m.get("abstract") or m.get("title", "")) for _, _, m in top_candidates]
    vecs = TfidfVectorizer(stop_words="english", max_features=5000).fit_transform(
        corpus_texts + cand_texts
    )
    sims = cosine_similarity(vecs[len(corpus_texts):], vecs[:len(corpus_texts)]).max(axis=1)
    return {pid: float(sims[i]) for i, (pid, _, _) in enumerate(top_candidates)}

# ── Step 7: Classify ───────────────────────────────────────────────────────────
def classify(co_cite, sim):
    return "core" if co_cite >= 2 and sim >= 0.2 else "periphery"

# ── Step 7b: Keyword scoring ───────────────────────────────────────────────────
def compute_keyword_scores(keywords, top_candidates):
    """
    Two-layer keyword relevance score in [0, 1]:

    Layer 1 — exact/stem match (fast, no dependencies):
        Checks title + abstract for each keyword and close variants.
        E.g. "data availability" also matches "data available", "available data".

    Layer 2 — TF-IDF semantic proximity (if scikit-learn present):
        Treats the keyword string as a pseudo-document and computes cosine
        similarity against each candidate's text. This catches synonyms like
        "open data", "data sharing", "data access" when you only typed
        "data availability".

    Final keyword score = 0.5 * exact + 0.5 * semantic
    (If scikit-learn is absent, falls back to exact-only.)
    """
    if not keywords:
        return {pid: 0.0 for pid, _, _ in top_candidates}

    kw_lower  = [k.lower().strip() for k in keywords]
    # Build simple stem variants: "availability" → also match "available", "availab"
    def variants(kw):
        parts = kw.split()
        stems = set()
        stems.add(kw)
        for p in parts:
            if len(p) > 5:
                stems.add(p[:5])   # crude prefix stem
        return stems

    all_variants = []
    for kw in kw_lower:
        all_variants.extend(variants(kw))

    # Layer 1: exact/stem match
    exact_scores = {}
    for pid, _, meta in top_candidates:
        text = ((meta.get("title") or "") + " " + (meta.get("abstract") or "")).lower()
        hits = sum(1 for v in all_variants if v in text)
        exact_scores[pid] = min(hits / max(len(kw_lower), 1), 1.0)

    # Layer 2: TF-IDF semantic proximity
    sem_scores = {pid: 0.0 for pid, _, _ in top_candidates}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        kw_doc       = " ".join(keywords)
        cand_texts   = [(meta.get("abstract") or meta.get("title") or "")
                        for _, _, meta in top_candidates]
        all_texts    = [kw_doc] + cand_texts
        vecs         = TfidfVectorizer(stop_words="english", max_features=5000).fit_transform(all_texts)
        sims         = cosine_similarity(vecs[1:], vecs[0:1]).flatten()
        for i, (pid, _, _) in enumerate(top_candidates):
            sem_scores[pid] = float(sims[i])
    except ImportError:
        pass  # stay with zeros, exact-only mode

    # Blend
    return {
        pid: 0.5 * exact_scores[pid] + 0.5 * sem_scores[pid]
        for pid, _, _ in top_candidates
    }

# ── Step 7c: Topic clustering ──────────────────────────────────────────────────
def cluster_candidates(top_candidates, n_clusters=None):
    """
    Cluster top_candidates by abstract+title TF-IDF.
    If n_clusters is None, auto-select k in [2..8] via silhouette score.
    Stores meta["_cluster"] = int on each candidate dict.
    Returns list of (cluster_id, top_terms, top_papers) for report.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("  (scikit-learn not found — skipping clustering)")
        return []

    texts = [(m.get("abstract") or "") + " " + (m.get("title") or "")
             for _, _, m in top_candidates]

    if len(top_candidates) < 4:
        return []

    vecs = TfidfVectorizer(stop_words="english", max_features=5000,
                           min_df=2).fit_transform(texts)

    if n_clusters:
        k = n_clusters
    else:
        best_k, best_score = 2, -1
        for k_try in range(2, min(9, len(top_candidates) // 3 + 1)):
            km = KMeans(n_clusters=k_try, random_state=42, n_init=10)
            labels = km.fit_predict(vecs)
            try:
                s = silhouette_score(vecs, labels, sample_size=min(500, len(texts)))
                if s > best_score:
                    best_k, best_score = k_try, s
            except Exception:
                pass
        k = best_k
        print(f"  🔍 Auto-selected k={k} clusters (silhouette={best_score:.3f})")

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(vecs)

    # Assign cluster labels back to meta dicts
    for i, (pid, sc, meta) in enumerate(top_candidates):
        meta["_cluster"] = int(labels[i])

    # Build cluster summaries: top TF-IDF terms + top 3 papers
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, min_df=2)
    vectorizer.fit(texts)
    feature_names = vectorizer.get_feature_names_out()

    summaries = []
    for c in range(k):
        idxs = [i for i, lbl in enumerate(labels) if lbl == c]
        if not idxs:
            continue
        cluster_vecs = vectorizer.transform([texts[i] for i in idxs])
        term_scores = cluster_vecs.mean(axis=0).A1
        top_term_idxs = term_scores.argsort()[::-1][:5]
        top_terms = [feature_names[j] for j in top_term_idxs]

        papers_in_cluster = [(top_candidates[i][1], top_candidates[i][2]) for i in idxs]
        papers_in_cluster.sort(key=lambda x: -x[0])
        top_papers = papers_in_cluster[:3]
        summaries.append((c, top_terms, top_papers))

    return summaries


# ── Step 8: Generate outputs ───────────────────────────────────────────────────
def make_bibtex_key(meta):
    authors = meta.get("authors") or []
    last    = authors[0]["name"].split()[-1] if authors else "Unknown"
    year    = meta.get("year", "")
    word    = ((meta.get("title") or "").split() or ["paper"])[0]
    return re.sub(r'[^a-zA-Z0-9]', '', f"{last}{year}{word}")[:30]

def to_bibtex(meta):
    key     = make_bibtex_key(meta)
    authors = " and ".join(a.get("name", "") for a in (meta.get("authors") or []))
    doi     = (meta.get("externalIds") or {}).get("DOI", "")
    title   = (meta.get("title") or "").replace("{", "").replace("}", "")
    return (f"@article{{{key},\n"
            f"  title  = {{{title}}},\n"
            f"  author = {{{authors}}},\n"
            f"  year   = {{{meta.get('year', '')}}},\n"
            f"  doi    = {{{doi}}},\n"
            f"  note   = {{citationCount={meta.get('citationCount', '')}}}\n}}")

def paper_block(rank, pid, score, meta, co_citation_count, sim_scores, abstract_len=300):
    doi       = (meta.get("externalIds") or {}).get("DOI", "")
    url       = f"https://doi.org/{doi}" if doi else ""
    title     = meta.get("title", "(no title)")
    link      = f"[{title}]({url})" if url else title
    auths     = meta.get("authors") or []
    auth_str  = ", ".join(a["name"] for a in auths[:3])
    if len(auths) > 3: auth_str += " et al."
    fields_str = ", ".join(meta.get("fieldsOfStudy") or []) or "N/A"
    co        = co_citation_count[pid]
    abstract  = meta.get("abstract") or ""
    snippet   = abstract[:abstract_len] + ("..." if len(abstract) > abstract_len else "")
    lines = [
        f"### {rank}. {link} — {meta.get('year', '')}",
        f"**Authors**: {auth_str}  |  **Citations**: {meta.get('citationCount', '?')}  "
        f"|  **Co-citation count**: {co}  |  **Score**: {score:.3f}",
        f"**Fields**: {fields_str}",
        f"**Why recommended**: Appears in references/citations of {co} papers in your corpus.",
    ]
    if snippet:
        lines.append(f"> {snippet}")
    lines.append("---\n")
    return "\n".join(lines)

def generate_report(scored_final, co_citation_count, sim_scores, corpus_size, today,
                    keywords=None, cluster_summaries=None):
    core = [(pid, sc, m) for pid, sc, m in scored_final
            if classify(co_citation_count[pid], sim_scores.get(pid, 0)) == "core"]
    peri = [(pid, sc, m) for pid, sc, m in scored_final
            if classify(co_citation_count[pid], sim_scores.get(pid, 0)) == "periphery"]

    lines = [
        "# Literature Explorer Report",
        f"**Corpus size**: {corpus_size} papers  |  "
        f"**Candidates evaluated**: {len(scored_final)}  |  **Date**: {today}",
    ]
    if keywords:
        lines.append(f"**Keyword focus**: {', '.join(f'`{k}`' for k in keywords)}")
    lines += ["\n---\n",
        "## 🔴 Core Papers You May Have Missed",
        "*Heavily co-cited with your corpus — likely foundational or highly influential works*\n",
    ]
    for rank, (pid, sc, m) in enumerate(core, 1):
        lines.append(paper_block(rank, pid, sc, m, co_citation_count, sim_scores, 300))

    lines += [
        "## 🟡 Periphery Papers Worth Exploring",
        "*Adjacent or emerging work — different angle, method, or subfield*\n",
    ]
    for rank, (pid, sc, m) in enumerate(peri, 1):
        lines.append(paper_block(rank, pid, sc, m, co_citation_count, sim_scores, 220))

    if cluster_summaries:
        lines += ["\n---\n", "## 🗂 Topic Clusters\n"]
        for c_id, terms, papers in cluster_summaries:
            lines.append(f"### Cluster {c_id+1} — Topics: {', '.join(terms)}\n")
            for i, (sc, m) in enumerate(papers, 1):
                title = m.get("title", "(no title)")
                doi   = (m.get("externalIds") or {}).get("DOI", "")
                url   = f"https://doi.org/{doi}" if doi else ""
                link  = f"[{title}]({url})" if url else title
                lines.append(f"{i}. {link} ({m.get('year', '?')}) — score {sc:.3f}")
            lines.append("")

    lines += [
        "## Methods Note",
        "**APIs**: Semantic Scholar (primary), OpenAlex (fallback, corpus identity only).  ",
        "**Scoring**: co-citation 35% · citation count 30% · recency 20% · field match 15% "
        "(×1.5 for psychology, sociology, political science, social science, education, "
        "communication, public health, medicine, neuroscience).  ",
        "**Classification**: Core = co-citation ≥ 2 AND TF-IDF similarity ≥ 0.20.  ",
        "No hard cap — list length reflects natural score distribution.",
    ]
    return "\n".join(lines), core, peri

# ── Main ───────────────────────────────────────────────────────────────────────
def _make_parser():
    parser = argparse.ArgumentParser(description="Literature Explorer — find papers you missed")
    parser.add_argument("--bib",     help="Path to .bib file")
    parser.add_argument("--pdf-dir", help="Path to folder of PDFs")
    parser.add_argument("--out",     default="outputs", help="Output directory (default: outputs/)")
    parser.add_argument("--cache",   default=None,
                        help="Cache file path (default: derived from --bib/--pdf-dir name)")
    parser.add_argument("--keywords", nargs="+", metavar="KEYWORD",
                        help="Focus keywords to re-weight scoring toward a topic pivot.")
    parser.add_argument("--preset", choices=list(WEIGHT_PRESETS.keys()),
                        default="balanced",
                        help="Scoring weight preset (default: balanced)")
    parser.add_argument("--cluster", type=int, default=None, metavar="N",
                        help="Cluster top candidates into N topic groups (omit to skip; 0=auto-detect k)")
    return parser

def resolve_paths(args):
    """Derive proj_name, cache_path, out_dir from args. Returns (proj_name, cache_path, out_dir)."""
    if args.bib:
        proj_name = Path(args.bib).stem
    else:
        proj_name = Path(args.pdf_dir.rstrip("/\\")).stem

    if args.cache:
        cache_path = Path(args.cache)
    elif args.bib:
        cache_path = Path(args.bib).parent / f"{Path(args.bib).stem}_cache.json"
    else:
        cache_path = Path(args.pdf_dir.rstrip("/\\")).parent / f"{proj_name}_cache.json"

    out_dir = Path(args.out) / proj_name
    return proj_name, cache_path, out_dir

def run(args):
    """Run the full exploration pipeline. args must have: bib/pdf_dir, out, cache, keywords."""
    if not args.bib and not getattr(args, "pdf_dir", None):
        sys.exit("Provide --bib or --pdf-dir")

    proj_name, cache_path, out_dir = resolve_paths(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Project : {proj_name}")
    print(f"  Cache   : {cache_path}")
    print(f"  Outputs : {out_dir}/")

    # 1. Parse
    print("\n📚 Parsing corpus...")
    corpus = parse_bib(args.bib) if args.bib else parse_pdf_dir(args.pdf_dir)
    print(f"  Found {len(corpus)} entries")

    # 2. Cache
    print("\n📦 Loading cache...")
    cache = load_cache(cache_path)

    # 3. Resolve
    resolved_corpus = resolve_all(corpus, cache, cache_path)
    corpus_ids = {p["paperId"] for p in resolved_corpus if p.get("paperId")}

    # 4. Graph
    print("\n🕸️  Building co-citation graph (no API calls — using cached data)...")
    co_citation_count, candidate_meta = build_cocitation_graph(resolved_corpus, corpus_ids)
    print(f"  Found {len(candidate_meta)} candidate papers")

    # 5. Score
    preset = getattr(args, "preset", "balanced") or "balanced"
    weights = WEIGHT_PRESETS.get(preset, WEIGHT_PRESETS["balanced"])
    if preset != "balanced":
        print(f"\n⚙️  Using preset: {preset}")
    print("\n📊 Scoring candidates...")
    current_year = datetime.now().year
    scored = sorted(
        [(pid, score_paper(pid, candidate_meta[pid], co_citation_count, current_year, weights),
          candidate_meta[pid])
         for pid in candidate_meta if candidate_meta[pid].get("title")],
        key=lambda x: -x[1]
    )
    top_candidates = scored[:200]

    # 6. Similarity
    print("\n🧮 Computing semantic similarity...")
    sim_scores = compute_similarity(resolved_corpus, top_candidates)

    # 6b. Keyword focus
    keywords = getattr(args, "keywords", None) or []
    if keywords:
        print(f"\n🎯 Applying keyword focus: {keywords}")
    kw_scores = compute_keyword_scores(keywords, top_candidates)

    if keywords:
        final_scored = sorted(
            [(pid, 0.60 * sc + 0.15 * sim_scores.get(pid, 0) + 0.25 * kw_scores.get(pid, 0), meta)
             for pid, sc, meta in top_candidates],
            key=lambda x: -x[1]
        )
    else:
        final_scored = sorted(
            [(pid, 0.80 * sc + 0.20 * sim_scores.get(pid, 0), meta)
             for pid, sc, meta in top_candidates],
            key=lambda x: -x[1]
        )
    n = max(10, min(50, len(final_scored)))
    mean_score   = sum(s for _, s, _ in final_scored[:n]) / n
    threshold    = mean_score * 0.4
    final_scored = [(pid, sc, m) for pid, sc, m in final_scored if sc >= threshold]
    print(f"  {len(final_scored)} papers above score threshold ({threshold:.3f})")

    # 6c. Clustering
    n_cluster = getattr(args, "cluster", None)
    cluster_summaries = None
    if n_cluster is not None:
        if n_cluster == 0:
            print("\n🗂  Auto-clustering candidates (k will be chosen by silhouette score)...")
            cluster_summaries = cluster_candidates(final_scored, n_clusters=None)
        elif n_cluster > 1:
            print(f"\n🗂  Clustering candidates into {n_cluster} topics...")
            cluster_summaries = cluster_candidates(final_scored, n_clusters=n_cluster)

    # 7. Output
    print("\n✍️  Generating report...")
    today = datetime.now().strftime("%Y-%m-%d")
    report_md, core_papers, peri_papers = generate_report(
        final_scored, co_citation_count, sim_scores, len(resolved_corpus), today,
        keywords=keywords, cluster_summaries=cluster_summaries
    )
    bib_text = "\n\n".join(to_bibtex(m) for _, _, m in final_scored)

    report_path = out_dir / f"{proj_name}_report_{today}.md"
    bib_path    = out_dir / f"{proj_name}_recommendations_{today}.bib"
    report_path.write_text(report_md, encoding="utf-8")
    bib_path.write_text(bib_text, encoding="utf-8")

    print(f"\n✅ Done!")
    print(f"   📄 Report  : {report_path}")
    print(f"   📚 BibTeX  : {bib_path}")
    print(f"   🔴 Core    : {len(core_papers)}")
    print(f"   🟡 Periphery: {len(peri_papers)}")

    return dict(proj_name=proj_name, cache_path=cache_path, out_dir=out_dir,
                core=len(core_papers), periphery=len(peri_papers))

def main():
    parser = _make_parser()
    args   = parser.parse_args()
    if not args.bib and not args.pdf_dir:
        parser.error("Provide --bib or --pdf-dir")
    run(args)

if __name__ == "__main__":
    main()