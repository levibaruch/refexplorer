Small project to find related papers using some API's. Will make an actual readme soon.
Generated with Claude

Current AI generated README:

# Literature Explorer

A two-script toolkit for discovering research papers you may have missed, given a reading list you already have. Built around the Semantic Scholar and OpenAlex APIs — both free, no API key needed.

---

## Files

| File | Purpose |
|------|---------|
| `literature_explorer.py` | Main tool — takes your `.bib` file or a folder of PDFs, finds related papers you haven't read, and outputs a ranked report + BibTeX file |
| `cache_search.py` | Companion search tool — lets you query the cache that `literature_explorer` builds up over time |

---

## Installation

Python 3.8+ required.

```bash
pip install requests scikit-learn

# Only needed if using --pdf-dir
pip install pypdf
```

---

## literature_explorer.py

### What it does

1. Parses your corpus from a `.bib` file or a folder of PDFs
2. Resolves each paper via Semantic Scholar (with OpenAlex as fallback), fetching metadata, references, and citations in a single API call per paper
3. Builds a co-citation graph — papers that appear repeatedly in the reference lists or citation lists of your corpus are strong candidates
4. Scores candidates on co-citation strength, citation count, recency, and field relevance (psychology, sociology, social science, education, communication, public health, medicine, neuroscience all get a boost)
5. Optionally re-weights scores toward a keyword focus (e.g. if you're pivoting from reproducibility to data availability)
6. Outputs a Markdown report split into **Core** (papers you likely missed that are foundational) and **Periphery** (adjacent, interesting work), plus a `.bib` file of all recommendations

### Basic usage

```bash
# From a BibTeX file
python literature_explorer.py --bib references.bib

# From a folder of PDFs
python literature_explorer.py --pdf-dir ~/papers/

# Specify output folder
python literature_explorer.py --bib references.bib --out ./results
```

### Keyword focus

If you want to pivot your search toward a specific topic, pass keywords with `--keywords`. The scorer will up-weight papers whose titles and abstracts are relevant to those terms — using both exact matching and TF-IDF semantic proximity, so synonyms are also caught.

```bash
# Focus on data availability (also catches "open data", "data sharing", etc.)
python literature_explorer.py --bib references.bib \
  --keywords "data availability" "open data" "data sharing"

# Pivot back to reproducibility
python literature_explorer.py --bib references.bib \
  --keywords "reproducibility" "replication" "computational reproducibility"

# No keywords = neutral co-citation ranking
python literature_explorer.py --bib references.bib
```

When keywords are active, the scoring blend shifts from `80% base / 20% similarity` to `60% base / 15% similarity / 25% keyword relevance`. The report header shows which keywords were used.

### Caching

Every paper resolved from the API is saved to `literature_explorer_cache.json`. On subsequent runs, cached papers are loaded instantly with no API calls. The cache stores metadata, full reference lists, and citation lists together, so the co-citation graph step is always instant after the first run.

```bash
# Use a custom cache location (useful if working across multiple projects)
python literature_explorer.py --bib references.bib --cache ~/research/my_cache.json
```

If you interrupt a run, progress is saved periodically — the next run will pick up where it left off for already-resolved papers.

### All options

```
--bib FILE          Path to .bib file
--pdf-dir DIR       Path to folder of PDFs (alternative to --bib)
--out DIR           Output directory for report and .bib (default: current dir)
--cache FILE        Cache file path (default: literature_explorer_cache.json)
--keywords KW ...   One or more focus keywords to re-weight scoring
```

### Output files

| File | Contents |
|------|---------|
| `literature_report.md` | Ranked recommendations split into Core and Periphery sections, with authors, citation counts, co-citation counts, field tags, and abstract snippets |
| `literature_recommendations.bib` | BibTeX entries for all recommended papers, ready to import into Zotero or any reference manager |

---

## cache_search.py

### What it does

As `literature_explorer` runs over time, it builds up a large cache of papers — not just your corpus, but every paper that appeared in any reference or citation list it fetched. This can easily grow to 5,000–10,000 papers. `cache_search.py` lets you search and explore that pool without running the full pipeline again.

### Basic usage

```bash
# Search by keyword
python cache_search.py "data availability"

# Multiple keywords (matches any by default)
python cache_search.py "data availability" "open data" "data sharing"

# Require all keywords to be present
python cache_search.py "data" "reproducibility" --mode all

# Show cache statistics (paper count, field breakdown, year distribution)
python cache_search.py --stats
```

### Filtering and sorting

```bash
# Filter by field of study
python cache_search.py "open science" --field psychology

# Filter by year range
python cache_search.py "preregistration" --year-min 2015 --year-max 2022

# Sort by citation count instead of relevance
python cache_search.py "open data" --sort citations

# Sort by publication year (newest first)
python cache_search.py "replication" --sort year

# Show more results
python cache_search.py "transparency" --top 50
```

### Inspecting results

```bash
# Show abstract snippets with keyword hits highlighted in [brackets]
python cache_search.py "data sharing" --verbose
```

### Exporting

```bash
# Export search results to a .bib file
python cache_search.py "data availability" --top 50 --export data_availability.bib
```

### All options

```
keywords            Keywords to search for (positional, space or quote-separated)
--cache FILE        Cache file to search (default: literature_explorer_cache.json)
--top N             Max results to show (default: 30)
--sort              Sort by: relevance (default), citations, year
--mode              Match any keyword (default) or require all
--field FIELD       Filter by field of study
--year-min YEAR     Only papers from this year onward
--year-max YEAR     Only papers up to this year
--verbose           Show abstract snippets with keyword hits highlighted
--export FILE       Export results as a .bib file
--stats             Show cache statistics
```

---

## Typical workflow

```bash
# First run — resolves all papers and builds the cache (takes a few minutes)
python literature_explorer.py --bib references.bib --out results/

# Pivot to a new topic — instant, uses the cache
python literature_explorer.py --bib references.bib \
  --keywords "data availability" --out results/data_availability/

# Search the cache directly for something specific
python cache_search.py "data availability" --field psychology --sort citations

# Add new papers to your bib and re-run — only the new ones hit the API
python literature_explorer.py --bib references_v2.bib --out results/v2/
```

---

## Notes on API rate limits

Semantic Scholar allows roughly 100 requests per minute without an API key. The script sleeps ~1.1 seconds between calls and backs off automatically when rate limited. With a warm cache, subsequent runs make very few or zero API calls. If you have a Semantic Scholar API key, you can increase the `SLEEP` constant at the top of `literature_explorer.py` to a lower value.

## Notes on OpenAlex fallback

When a paper cannot be resolved via Semantic Scholar (e.g. due to rate limits or a missing record), the script tries OpenAlex. OpenAlex-resolved papers are included in your corpus identity set so they're excluded from recommendations, but they don't contribute to the co-citation graph since OpenAlex doesn't expose full reference lists in the same way. These are marked `(OA)` in the resolution log.
