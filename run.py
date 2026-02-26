#!/usr/bin/env python3
"""
run.py — Full literature discovery pipeline in one command.

Runs all four steps in sequence:
  1. literature_explorer  — resolve corpus, build co-citation graph, score candidates
  2. enrich_abstracts     — fill missing abstracts (SS batch → OpenAlex → CrossRef)
  3. visualize            — interactive co-citation network + timeline + field chart
  (search is interactive — use cache_search.py separately)

Usage:
    python run.py --bib references.bib
    python run.py --bib references.bib --keywords "open science" "data sharing"
    python run.py --bib references.bib --skip-enrich
    python run.py --bib references.bib --skip-viz
    python run.py --bib references.bib --top 80 --out results
"""

import argparse, sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(
        description="Full literature discovery pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── Input ──
    parser.add_argument("--bib",      help="Path to .bib file")
    parser.add_argument("--pdf-dir",  help="Path to folder of PDFs (alternative to --bib)")
    parser.add_argument("--cache",    default=None,
                        help="Cache file path (default: derived from bib name)")
    # ── Explorer ──
    parser.add_argument("--keywords", nargs="+", metavar="KW",
                        help="Focus keywords to re-weight scoring "
                             "(e.g. --keywords 'open data' 'data sharing')")
    parser.add_argument("--preset",
                        choices=["balanced","highly-cited","recent","interdisciplinary"],
                        default="balanced",
                        help="Scoring weight preset (default: balanced)")
    parser.add_argument("--cluster", type=int, default=None, metavar="N",
                        help="Cluster candidates into N topics (0=auto-detect k)")
    parser.add_argument("--out",      default="outputs",
                        help="Root output directory (default: outputs/)")
    # ── Enrich ──
    parser.add_argument("--skip-enrich", action="store_true",
                        help="Skip abstract enrichment step")
    parser.add_argument("--enrich-limit", type=int, default=0,
                        help="Cap OA/CR API calls during enrichment (0 = no limit)")
    # ── Visualize ──
    parser.add_argument("--skip-viz", action="store_true",
                        help="Skip visualization step")
    parser.add_argument("--top",      type=int, default=50,
                        help="Top N candidates to show in network (default: 50)")

    args = parser.parse_args()

    if not args.bib and not args.pdf_dir:
        parser.error("Provide --bib or --pdf-dir")

    # ── Derive cache path (shared across all steps) ──
    if args.cache:
        cache_path = Path(args.cache)
    elif args.bib:
        cache_path = Path(args.bib).parent / f"{Path(args.bib).stem}_cache.json"
    else:
        proj = Path(args.pdf_dir.rstrip("/\\")).stem
        cache_path = Path(args.pdf_dir).parent / f"{proj}_cache.json"

    print("\n" + "═" * 60)
    print("  🔭 Literature Discovery Pipeline")
    print("═" * 60)
    print(f"  Bib     : {args.bib or args.pdf_dir}")
    print(f"  Cache   : {cache_path}")
    print(f"  Outputs : {args.out}/")
    print("═" * 60 + "\n")

    # ─────────────────────────────────────────────────────────
    # Step 1: Explore
    # ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Step 1/3 — Resolving corpus & building co-citation graph")
    print("─" * 60)
    from core.explorer import run as run_explorer
    explorer_args = SimpleNamespace(
        bib=args.bib,
        pdf_dir=getattr(args, "pdf_dir", None),
        out=args.out,
        cache=str(cache_path),
        keywords=args.keywords,
        preset=args.preset,
        cluster=args.cluster,
    )
    result = run_explorer(explorer_args)

    # ─────────────────────────────────────────────────────────
    # Step 2: Enrich abstracts
    # ─────────────────────────────────────────────────────────
    if not args.skip_enrich:
        print("\n" + "─" * 60)
        print("  Step 2/3 — Enriching abstracts")
        print("─" * 60)
        from core.enrich import run as run_enrich
        enrich_args = SimpleNamespace(
            cache=str(cache_path),
            limit=args.enrich_limit,
            dry_run=False,
            sources=["ss", "oa", "cr"],
        )
        run_enrich(enrich_args)
    else:
        print("\n  ⏭  Skipping abstract enrichment (--skip-enrich)")

    # ─────────────────────────────────────────────────────────
    # Step 3: Visualize
    # ─────────────────────────────────────────────────────────
    if not args.skip_viz:
        print("\n" + "─" * 60)
        print("  Step 3/3 — Generating visualizations")
        print("─" * 60)
        from core.viz import run as run_viz
        viz_args = SimpleNamespace(
            cache=str(cache_path),
            top=args.top,
            out=args.out,
            no_network=False,
            no_timeline=False,
            no_fields=False,
            no_table=False,
        )
        out_paths = run_viz(viz_args)
    else:
        print("\n  ⏭  Skipping visualizations (--skip-viz)")
        out_paths = {}

    # ─────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────
    proj_name = result["proj_name"]
    out_dir   = Path(args.out) / proj_name

    print("\n" + "═" * 60)
    print("  ✅ Pipeline complete!")
    print("═" * 60)
    print(f"  Project   : {proj_name}")
    print(f"  Cache     : {cache_path}")
    print(f"  Outputs   : {out_dir}/")
    print(f"  Core      : {result['core']} papers")
    print(f"  Periphery : {result['periphery']} papers")
    if out_paths:
        for key, path in out_paths.items():
            print(f"  {key.capitalize():<10}: {path}")
    print()
    print("  Next: search your cache interactively:")
    print(f'    python cache_search.py --cache {cache_path} "your keywords"')
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
