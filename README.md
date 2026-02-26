# Literature Explorer

Discover research papers you may have missed, starting from a `.bib` file or folder of PDFs. Built on Semantic Scholar and OpenAlex — both free, no API key required.

---

## Structure

```
core/
  explorer.py   — resolve corpus, build co-citation graph, score & cluster candidates
  search.py     — search and filter the cache interactively
  enrich.py     — fill in missing abstracts via OpenAlex / CrossRef
  viz.py        — generate interactive visualizations
run.py          — one-command pipeline (explore → enrich → visualize)
pipeline.ipynb  — step-by-step Jupyter notebook
requirements.txt
```

---

## Installation

Python 3.9+ required.

```bash
pip install -r requirements.txt

# Only needed if using --pdf-dir
pip install pypdf
```

---

## Quick start

```bash
# Run the full pipeline (explore → enrich abstracts → visualize)
python run.py --bib references.bib

# With keyword focus and clustering
python run.py --bib references.bib --keywords "open data" "data sharing" --cluster 0

# Use a scoring preset
python run.py --bib references.bib --preset recent
```

Outputs go to `outputs/{project_name}/`. The cache is saved next to the `.bib` file as `{name}_cache.json`.

---

## How it works

1. **Parse** your corpus from a `.bib` file or PDF folder
2. **Resolve** each paper via Semantic Scholar (OpenAlex fallback), fetching metadata + full reference/citation lists in one API call per paper
3. **Build** a co-citation graph — papers that recur across your corpus's reference and citation lists are strong candidates
4. **Score** candidates on co-citation frequency, citation count, recency, and field relevance
5. **Cluster** candidates into topic groups using TF-IDF + KMeans (optional)
6. **Output** a Markdown report split into Core and Periphery, a `.bib` file, and interactive visualizations

---

## `run.py` — full pipeline

```bash
python run.py --bib references.bib [options]
```

| Option | Description |
|--------|-------------|
| `--bib FILE` | Path to `.bib` file |
| `--pdf-dir DIR` | Path to folder of PDFs (alternative to `--bib`) |
| `--out DIR` | Output root directory (default: `outputs/`) |
| `--keywords KW ...` | Focus keywords to re-weight scoring |
| `--preset` | Scoring preset: `balanced` (default), `highly-cited`, `recent`, `interdisciplinary` |
| `--cluster N` | Cluster candidates into N topic groups; `0` = auto-detect best k |
| `--top N` | Top N candidates in visualizations (default: 50) |
| `--no-viz` | Skip visualization step |
| `--dry-run` | Skip abstract enrichment API calls |

---

## `core/explorer.py` — explore

Resolves your corpus and produces ranked recommendations.

```bash
python -m core.explorer --bib references.bib
python -m core.explorer --bib references.bib --keywords "replication" --preset recent
python -m core.explorer --bib references.bib --cluster 5
```

### Scoring presets

| Preset | co-cite | citations | recency | field |
|--------|---------|-----------|---------|-------|
| `balanced` (default) | 35% | 30% | 20% | 15% |
| `highly-cited` | 20% | 60% | 10% | 10% |
| `recent` | 25% | 15% | 50% | 10% |
| `interdisciplinary` | 35% | 25% | 15% | 25% |

### Clustering (`--cluster`)

Groups the top candidate papers into topic clusters using TF-IDF + KMeans on their abstracts. Each cluster is described by its top 5 terms and top 3 papers by score.

- `--cluster 0` — auto-detect the best number of clusters (via silhouette score, k in 2–8)
- `--cluster 5` — explicitly request 5 clusters

When clustering is active, the co-citation network colours nodes by cluster instead of score quartile, and a **Topic Clusters** section is appended to the Markdown report.

### Options

```
--bib FILE          Path to .bib file
--pdf-dir DIR       Folder of PDFs (alternative to --bib)
--out DIR           Output directory (default: outputs/)
--cache FILE        Cache file path (auto-derived from bib name by default)
--keywords KW ...   Focus keywords to re-weight scoring
--preset NAME       Scoring weight preset (balanced / highly-cited / recent / interdisciplinary)
--cluster N         Cluster candidates (0 = auto k, N > 1 = explicit k, omit = skip)
```

---

## `core/search.py` — search the cache

Search and filter the full paper pool (corpus + all referenced/citing papers) without re-running the pipeline.

```bash
python -m core.search --cache references_cache.json "data availability"
python -m core.search --cache references_cache.json "open science" --sort trending
python -m core.search --cache references_cache.json --type meta-analysis --field psychology
python -m core.search --cache references_cache.json --semantic "open science reform"
python -m core.search --cache references_cache.json --stats
```

### Filters

| Option | Description |
|--------|-------------|
| `keywords` | Keyword search (positional). Quote phrases: `'"data availability"'` |
| `--mode any\|all` | Match any keyword (default) or require all |
| `--search-in all\|title\|abstract` | Which field to search |
| `--regex` | Treat keywords as regular expressions |
| `--field FIELD` | Filter by field of study (substring) |
| `--author NAME` | Filter by author name (substring) |
| `--year-min / --year-max` | Publication year range |
| `--min-citations / --max-citations` | Citation count range |
| `--venue VENUE` | Filter by journal/venue name (substring) |
| `--type TYPE` | Heuristic paper type: `review`, `meta-analysis`, `empirical`, `methods`, `theoretical`, `replication` |
| `--new-only` | Exclude your original corpus papers — only show discovered candidates |
| `--exclude TERM ...` | Drop papers containing any of these terms |

### Sorting

```bash
--sort relevance   # keyword match score (default)
--sort citations   # total citation count
--sort year        # newest first
--sort trending    # citation velocity = citations / years since publication
```

### Semantic search

```bash
# Find papers conceptually related to a query (TF-IDF cosine similarity)
--semantic "open science reform practices"

# Find papers similar to a specific paper already in the cache
--semantic-like "The Role of Preregistration in Psychology"
```

### Export

```bash
--export results.bib                          # BibTeX (default)
--export results.csv --export-format csv      # CSV with title/authors/doi/abstract
--export results.ris --export-format ris      # RIS (Endnote/Mendeley/Zotero)
--export notes/ --export-format obsidian      # Markdown notes, one file per paper
```

---

## `core/enrich.py` — fill missing abstracts

Fetches missing abstracts for papers in the cache via Semantic Scholar batch API, then OpenAlex, then CrossRef.

```bash
python -m core.enrich --cache references_cache.json
python -m core.enrich --cache references_cache.json --dry-run   # preview only
python -m core.enrich --cache references_cache.json --limit 50  # cap API calls
```

---

## `core/viz.py` — visualizations

Generates four interactive outputs from the cache:

```bash
python -m core.viz --cache references_cache.json
python -m core.viz --cache references_cache.json --top 80 --no-table
```

| Output | Format | Description |
|--------|--------|-------------|
| `{proj}_network_{date}.html` | HTML | Force-directed co-citation network (pyvis). Corpus = blue dots, candidates = coloured squares. Colours by cluster if clustering was run, otherwise by score quartile. |
| `{proj}_timeline_{date}.html` | HTML | Interactive year × citation scatter (Plotly). Hover for abstract snippet; click to open DOI. |
| `{proj}_fields_{date}.png` | PNG | Field distribution bar chart. Green = overlap with your corpus. |
| `{proj}_scores_{date}.html` | HTML | Score breakdown table (Plotly). Columns: rank, title (clickable), year, score, co-citations, citations, cluster. Cells colour-coded by value. Includes CSV export button. |

Options:
```
--top N          Top N candidates (default: 50)
--no-network     Skip network
--no-timeline    Skip timeline
--no-fields      Skip field chart
--no-table       Skip score breakdown table
```

---

## `pipeline.ipynb` — Jupyter notebook

Step-by-step notebook. Set `BIB_FILE`, `KEYWORDS`, `PRESET`, `CLUSTER` in the config cell and run each step individually.

New cells for advanced search: type filter, semantic search, trending sort, venue filter, new-only mode, exclude terms.

---

## Caching

Each project gets its own cache file next to the `.bib` file: `{name}_cache.json`. On subsequent runs, cached papers are loaded instantly. If you interrupt a run, resolved papers are already saved — the next run skips them.

Running with a warm cache still re-runs graph building, scoring, and clustering (fast, in-memory). Only the Semantic Scholar API calls are skipped.

---

## Output layout

```
outputs/
  {project_name}/
    {project_name}_report_{date}.md
    {project_name}_recommendations_{date}.bib
    {project_name}_network_{date}.html
    {project_name}_timeline_{date}.html
    {project_name}_fields_{date}.png
    {project_name}_scores_{date}.html
```

---

## API notes

**Semantic Scholar**: ~100 requests/minute without a key. The script sleeps 1.1s between calls and backs off on rate limits. With a warm cache, subsequent runs make zero calls.

**OpenAlex**: Used as fallback when SS fails, and for batch abstract enrichment. No key needed; polite pool is used automatically.

**CrossRef**: Last-resort abstract fallback in `enrich.py`.
