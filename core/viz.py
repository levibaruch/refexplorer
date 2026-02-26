#!/usr/bin/env python3
"""
visualize.py — Generate visualizations from a literature_explorer cache.

Produces:
  outputs/cocitation_network_YYYY-MM-DD.html  — interactive force-directed graph
  outputs/timeline_YYYY-MM-DD.html            — interactive year × citation scatter
  outputs/fields_YYYY-MM-DD.png               — field distribution bar chart

Usage:
    python visualize.py --cache literature_explorer_cache.json
    python visualize.py --cache literature_explorer_cache.json --top 80
    python visualize.py --cache literature_explorer_cache.json --no-network
"""

import json, math, re, sys, argparse
from pathlib import Path
from collections import defaultdict
from datetime import date

# ── Optional imports — checked at runtime ─────────────────────────────────────
def _require(pkg, pip_name=None):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        name = pip_name or pkg
        sys.exit(f"Missing dependency: pip install {name}")

# ── Load cache ─────────────────────────────────────────────────────────────────
def load_cache(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"Cache not found: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        sys.exit(f"Could not parse cache: {e}")

# ── Derive co-citation counts from cache ──────────────────────────────────────
def build_cocitation(data):
    """
    Re-derive co-citation counts by walking corpus papers' references + citations.
    Returns:
        corpus_ids   : set of paperIds that are corpus papers
        cocit_count  : dict paperId -> int  (# corpus papers that mention it)
        all_papers   : dict paperId -> paper dict (corpus + all ref/cit papers)
    """
    corpus_ids  = set()
    cocit_count = defaultdict(int)
    all_papers  = {}

    corpus = data.get("papers", {})

    # Register corpus papers
    for paper in corpus.values():
        pid = paper.get("paperId")
        if pid:
            corpus_ids.add(pid)
            all_papers[pid] = paper

    # Walk refs + citations
    for paper in corpus.values():
        seen_this_corpus = set()
        for neighbour in list(paper.get("references") or []) + list(paper.get("citations") or []):
            if not neighbour:
                continue
            npid = neighbour.get("paperId")
            if not npid or npid in corpus_ids:
                continue
            all_papers.setdefault(npid, neighbour)
            if npid not in seen_this_corpus:
                cocit_count[npid] += 1
                seen_this_corpus.add(npid)

    return corpus_ids, cocit_count, all_papers

# ── Score candidates (simplified mirror of literature_explorer scoring) ────────
def score_candidates(corpus_ids, cocit_count, all_papers, top_n):
    """
    Score and rank candidate papers (non-corpus). Returns list of (pid, score, paper)
    sorted by score descending, limited to top_n.
    """
    max_cocit = max(cocit_count.values(), default=1)
    max_cites = max(
        (p.get("citationCount") or 0 for pid, p in all_papers.items() if pid not in corpus_ids),
        default=1
    )
    current_year = date.today().year

    scored = []
    for pid, paper in all_papers.items():
        if pid in corpus_ids:
            continue
        cc   = cocit_count.get(pid, 0)
        if cc == 0:
            continue
        cites = paper.get("citationCount") or 0
        year  = paper.get("year")
        try:    year = int(year) if year else None
        except: year = None
        recency = max(0, 1 - (current_year - year) / 30) if year else 0

        score = (
            0.35 * (cc   / max_cocit) +
            0.30 * math.log1p(cites) / math.log1p(max_cites) +
            0.20 * recency
        )
        scored.append((pid, score, paper))

    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]

# ── Helpers ────────────────────────────────────────────────────────────────────
def _short_title(paper, max_len=40):
    t = paper.get("title") or "(no title)"
    return t[:max_len] + "…" if len(t) > max_len else t

def _first_author(paper):
    auths = paper.get("authors") or []
    if auths:
        return auths[0].get("name", "")
    return ""

def _node_size(citations, lo=10, hi=40):
    """Map citation count to a node pixel size."""
    c = citations or 0
    if c <= 0:
        return lo
    log_c = math.log1p(c)
    log_hi = math.log1p(5000)  # ~saturates at 5000 citations
    return lo + (hi - lo) * min(log_c / log_hi, 1.0)

# ── Visualization 1: Interactive co-citation network (HTML) ───────────────────
def make_network(corpus_ids, cocit_count, all_papers, candidates, out_path):
    pyvis = _require("pyvis", "pyvis")
    Network = pyvis.network.Network

    net = Network(
        height="820px", width="100%",
        bgcolor="#1a1a2e", font_color="white",
        directed=False,
        notebook=False,
    )
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.003,
          "springLength": 120,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 150}
      },
      "interaction": {"hover": true, "navigationButtons": true},
      "edges": {"smooth": {"type": "continuous"}, "color": {"opacity": 0.5}}
    }
    """)

    candidate_pids = {pid for pid, _, _ in candidates}
    max_score      = candidates[0][1] if candidates else 1.0

    # 8-colour cluster palette
    _CLUSTER_PALETTE = [
        "#e63946", "#f4a261", "#2a9d8f", "#a8dadc",
        "#a786c9", "#f7c59f", "#43aa8b", "#f9c74f",
    ]

    def _candidate_colour(score, cluster=None):
        if cluster is not None:
            return _CLUSTER_PALETTE[cluster % len(_CLUSTER_PALETTE)]
        q = score / max(max_score, 1e-9)
        if q >= 0.75: return "#e63946"
        if q >= 0.50: return "#f4a261"
        if q >= 0.25: return "#2a9d8f"
        return "#a8dadc"

    has_clusters = any(all_papers.get(pid, {}).get("_cluster") is not None
                       for pid, _, _ in candidates)

    added_nodes = set()

    def add_corpus_node(pid, paper):
        if pid in added_nodes:
            return
        cites = paper.get("citationCount") or 0
        size  = _node_size(cites, lo=12, hi=35)
        label = _short_title(paper, 30)
        tip   = (f"<b>{paper.get('title','')}</b><br>"
                 f"Year: {paper.get('year','?')}<br>"
                 f"Author: {_first_author(paper)}<br>"
                 f"Citations: {cites}<br>"
                 f"<i>[CORPUS]</i>")
        net.add_node(pid, label=label, title=tip,
                     color="#4cc9f0", size=size, shape="dot",
                     borderWidth=2, borderWidthSelected=4)
        added_nodes.add(pid)

    def add_candidate_node(pid, score, paper):
        if pid in added_nodes:
            return
        cites   = paper.get("citationCount") or 0
        cc      = cocit_count.get(pid, 0)
        cluster = paper.get("_cluster")
        size    = _node_size(cites, lo=8, hi=30)
        label   = _short_title(paper, 30)
        cluster_tag = f"<br>Cluster: {cluster+1}" if cluster is not None else ""
        tip   = (f"<b>{paper.get('title','')}</b><br>"
                 f"Year: {paper.get('year','?')}<br>"
                 f"Author: {_first_author(paper)}<br>"
                 f"Citations: {cites}<br>"
                 f"Co-citations: {cc}<br>"
                 f"Score: {score:.3f}{cluster_tag}")
        colour = _candidate_colour(score, cluster if has_clusters else None)
        net.add_node(pid, label=label, title=tip,
                     color=colour, size=size, shape="square",
                     borderWidth=1)
        added_nodes.add(pid)

    # Build candidate pid → score lookup
    cand_score = {pid: sc for pid, sc, _ in candidates}

    # Add nodes + edges
    for cp_pid, corpus_paper in all_papers.items():
        if cp_pid not in corpus_ids:
            continue
        neighbours = (
            list(corpus_paper.get("references") or []) +
            list(corpus_paper.get("citations")  or [])
        )
        for nb in neighbours:
            if not nb:
                continue
            npid = nb.get("paperId")
            if not npid or npid not in candidate_pids:
                continue
            nb_paper = all_papers.get(npid, nb)
            sc = cand_score.get(npid, 0.0)

            add_corpus_node(cp_pid, corpus_paper)
            add_candidate_node(npid, sc, nb_paper)

            cc = cocit_count.get(npid, 1)
            net.add_edge(cp_pid, npid, value=cc,
                         color={"color": "#ffffff", "opacity": 0.15 + 0.15 * min(cc / 5, 1)})

    # Add any corpus nodes not yet added (isolated in the graph)
    for cp_pid, corpus_paper in all_papers.items():
        if cp_pid in corpus_ids and cp_pid not in added_nodes:
            add_corpus_node(cp_pid, corpus_paper)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(out_path))
    print(f"  📡 Network saved → {out_path}  ({len(added_nodes)} nodes)")

# ── Visualization 2: Interactive timeline scatter (HTML via Plotly) ────────────
def make_timeline(corpus_ids, cocit_count, all_papers, candidates, out_path):
    _require("plotly", "plotly")
    import plotly.graph_objects as go

    candidate_pids = {pid for pid, _, _ in candidates}
    cand_score     = {pid: sc for pid, sc, _ in candidates}
    max_cc         = max(cocit_count.values(), default=1)

    # Collect per-group data
    groups = {
        "corpus":    {"x":[], "y":[], "s":[], "text":[], "doi":[]},
        "core":      {"x":[], "y":[], "s":[], "text":[], "doi":[]},
        "periphery": {"x":[], "y":[], "s":[], "text":[], "doi":[]},
    }

    for pid, paper in all_papers.items():
        year  = paper.get("year")
        cites = paper.get("citationCount") or 0
        try:    year = int(year) if year else None
        except: year = None
        if not year or cites <= 0:
            continue

        cc      = cocit_count.get(pid, 0)
        authors = paper.get("authors") or []
        auth_str = ", ".join(a.get("name","") for a in authors[:3])
        if len(authors) > 3: auth_str += " et al."
        doi  = (paper.get("externalIds") or {}).get("DOI") or paper.get("doi") or ""
        url  = f"https://doi.org/{doi}" if doi else ""
        abst = (paper.get("abstract") or "")[:300]
        abst_snip = (abst + "…") if len(paper.get("abstract") or "") > 300 else abst

        tip = (
            f"<b>{paper.get('title','(no title)')}</b><br>"
            f"Authors: {auth_str}<br>"
            f"Year: {year}  |  Citations: {cites:,}<br>"
            f"Co-citations: {cc}<br>"
            + (f"DOI: {doi}<br>" if doi else "")
            + (f"<br><i>{abst_snip}</i>" if abst_snip else "")
        )

        dot_size = 8 + 22 * (cc / max_cc)  # 8–30px

        if pid in corpus_ids:
            g = groups["corpus"]; dot_size = 14
        elif pid in candidate_pids:
            sc = cand_score.get(pid, 0)
            g  = groups["core"] if sc >= 0.5 else groups["periphery"]
        else:
            continue

        g["x"].append(year); g["y"].append(cites)
        g["s"].append(dot_size); g["text"].append(tip); g["doi"].append(url)

    fig = go.Figure()

    STYLE = {
        "periphery": dict(color="rgba(168,218,220,0.45)", symbol="circle"),
        "core":      dict(color="rgba(244,162,97,0.85)",  symbol="circle"),
        "corpus":    dict(color="rgba(76,201,240,0.95)",  symbol="star"),
    }
    NAMES = {
        "periphery": "Periphery candidate",
        "core":      "Core candidate (score ≥ 0.5)",
        "corpus":    "Your corpus",
    }

    for key in ("periphery", "core", "corpus"):
        g = groups[key]
        if not g["x"]:
            continue
        fig.add_trace(go.Scatter(
            x=g["x"], y=g["y"],
            mode="markers",
            name=NAMES[key],
            marker=dict(
                size=g["s"],
                color=STYLE[key]["color"],
                symbol=STYLE[key]["symbol"],
                line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
            ),
            text=g["text"],
            hovertemplate="%{text}<extra></extra>",
            customdata=g["doi"],
        ))

    fig.update_layout(
        title=dict(
            text="Papers by Year and Citation Count<br>"
                 "<sup>Dot size ∝ co-citation frequency — hover for details</sup>",
            font=dict(color="white", size=16),
        ),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="white"),
        xaxis=dict(title="Year", gridcolor="#2a2a4a", color="white"),
        yaxis=dict(title="Citation count (log scale)", type="log",
                   gridcolor="#2a2a4a", color="white"),
        legend=dict(bgcolor="rgba(26,26,46,0.8)", bordercolor="#555",
                    borderwidth=1, font=dict(color="white")),
        hovermode="closest",
        hoverlabel=dict(bgcolor="#1a1a2e", bordercolor="#555",
                        font=dict(color="white", size=12)),
        margin=dict(l=60, r=30, t=80, b=60),
    )

    # Clicking a dot with a DOI opens the paper — inject via JS in the HTML
    click_js = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var plot = document.querySelector('.plotly-graph-div');
        plot.on('plotly_click', function(data) {
            var pt = data.points[0];
            var doi = pt.customdata;
            if (doi) { window.open(doi, '_blank'); }
        });
    });
    </script>
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    html = html.replace("</body>", click_js + "\n</body>")
    out_path.write_text(html, encoding="utf-8")
    print(f"  📈 Timeline saved → {out_path}  (click a dot to open the paper)")

# ── Visualization 3: Field distribution bar chart (PNG) ───────────────────────
def make_field_chart(corpus_ids, all_papers, candidates, out_path):
    plt = _require("matplotlib.pyplot", "matplotlib")
    import matplotlib.pyplot as plt

    candidate_pids = {pid for pid, _, _ in candidates}
    corpus_fields  = set()
    cand_fields    = defaultdict(int)

    for pid, paper in all_papers.items():
        fos = [f.lower() for f in (paper.get("fieldsOfStudy") or [])]
        if pid in corpus_ids:
            corpus_fields.update(fos)
        elif pid in candidate_pids:
            for f in fos:
                cand_fields[f] += 1

    if not cand_fields:
        print("  ⚠  No field data available — skipping field chart")
        return

    top_fields = sorted(cand_fields.items(), key=lambda x: -x[1])[:15]
    labels = [f for f, _ in top_fields]
    counts = [n for _, n in top_fields]
    colours = ["#2a9d8f" if lab in corpus_fields else "#e76f51" for lab in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars = ax.barh(labels[::-1], counts[::-1], color=colours[::-1], height=0.7)
    ax.set_xlabel("Number of candidate papers", color="white", fontsize=11)
    ax.set_title("Field Distribution of Candidate Papers\n(green = overlap with your corpus)",
                 color="white", fontsize=13)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # Value labels on bars
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", color="white", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Field chart saved → {out_path}")

# ── Visualization 4: Score breakdown table (Plotly HTML) ──────────────────────
def make_score_table(corpus_ids, cocit_count, all_papers, candidates, out_path):
    _require("plotly", "plotly")
    import plotly.graph_objects as go

    rows = []
    for rank, (pid, score, paper) in enumerate(candidates, 1):
        doi   = (paper.get("externalIds") or {}).get("DOI") or paper.get("doi") or ""
        title = paper.get("title") or "(no title)"
        url   = f"https://doi.org/{doi}" if doi else ""
        link  = f'<a href="{url}" target="_blank">{title[:70]}</a>' if url else title[:70]
        cc    = cocit_count.get(pid, 0)
        cites = paper.get("citationCount") or 0
        year  = paper.get("year") or ""
        cluster = paper.get("_cluster")
        cluster_str = str(cluster + 1) if cluster is not None else "—"
        rows.append((rank, link, year, f"{score:.3f}", cc, cites, cluster_str))

    if not rows:
        return

    ranks, links, years, scores, co_cites, citations_col, clusters = zip(*rows)

    # Color cells by value (green hue for high, white for low)
    def _cell_colors(vals, high_is_good=True):
        try:
            nums = [float(v) for v in vals]
        except (ValueError, TypeError):
            return ["rgba(255,255,255,0.05)"] * len(vals)
        mn, mx = min(nums), max(nums)
        rng = max(mx - mn, 1e-9)
        colors = []
        for n in nums:
            intensity = (n - mn) / rng if high_is_good else 1 - (n - mn) / rng
            g = int(80 + 120 * intensity)
            colors.append(f"rgba(0,{g},60,0.35)")
        return colors

    score_colors  = _cell_colors(scores)
    cocite_colors = _cell_colors(co_cites)
    cite_colors   = _cell_colors(citations_col)

    fig = go.Figure(data=[go.Table(
        columnwidth=[40, 400, 60, 80, 80, 100, 80],
        header=dict(
            values=["<b>Rank</b>", "<b>Title</b>", "<b>Year</b>",
                    "<b>Score</b>", "<b>Co-cites</b>", "<b>Citations</b>", "<b>Cluster</b>"],
            fill_color="#1a1a2e",
            font=dict(color="white", size=12),
            align="left",
            height=32,
        ),
        cells=dict(
            values=[list(ranks), list(links), list(years),
                    list(scores), list(co_cites), list(citations_col), list(clusters)],
            fill_color=[
                ["rgba(30,30,50,0.8)"] * len(ranks),
                ["rgba(30,30,50,0.8)"] * len(ranks),
                ["rgba(30,30,50,0.8)"] * len(ranks),
                score_colors,
                cocite_colors,
                cite_colors,
                ["rgba(30,30,50,0.8)"] * len(ranks),
            ],
            font=dict(color="white", size=11),
            align=["center", "left", "center", "center", "center", "center", "center"],
            height=28,
        ),
    )])

    fig.update_layout(
        title=dict(text="Candidate Score Breakdown", font=dict(color="white", size=16)),
        paper_bgcolor="#1a1a2e",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    # Add CSV export button via JS
    csv_js = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    var btn = document.createElement('button');
    btn.textContent = '⬇ Export CSV';
    btn.style = 'position:fixed;top:10px;right:10px;padding:8px 14px;'
               +'background:#2a9d8f;color:white;border:none;border-radius:4px;'
               +'cursor:pointer;font-size:13px;z-index:9999;';
    btn.onclick = function() {
        var rows = [['Rank','Title','Year','Score','Co-cites','Citations','Cluster']];
        var cells = document.querySelectorAll('.js-plotly-plot .cells .y-axis .y');
        // Simple fallback: use table data from page source is complex with Plotly;
        // instead trigger download from data embedded in script tag
        var data = window._scoreData;
        if (!data) { alert('No data available'); return; }
        var csv = data.map(function(r){ return r.join(','); }).join('\\n');
        var blob = new Blob([csv], {type:'text/csv'});
        var a = document.createElement('a'); a.href = URL.createObjectURL(blob);
        a.download = 'scores.csv'; a.click();
    };
    document.body.appendChild(btn);
});
</script>
"""
    # Embed data for CSV export
    csv_data = [["Rank","Title","Year","Score","Co-cites","Citations","Cluster"]]
    for r in rows:
        rank_v, link_v, year_v, score_v, cc_v, cites_v, cl_v = r
        # strip HTML from link
        plain_title = re.sub(r'<[^>]+>', '', link_v)
        csv_data.append([rank_v, plain_title, year_v, score_v, cc_v, cites_v, cl_v])

    data_js = f"<script>window._scoreData={json.dumps(csv_data)};</script>"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    html = html.replace("</body>", data_js + csv_js + "\n</body>")
    out_path.write_text(html, encoding="utf-8")
    print(f"  📋 Score table saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def _make_parser():
    parser = argparse.ArgumentParser(
        description="Visualize co-citation network and paper statistics from cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--cache", required=True,
                        help="Cache JSON file produced by literature_explorer.py")
    parser.add_argument("--top",   type=int, default=50,
                        help="Top N candidate papers to include in network (default: 50)")
    parser.add_argument("--out",   default="outputs",
                        help="Output directory (default: outputs/)")
    parser.add_argument("--no-network",   action="store_true",
                        help="Skip the interactive HTML network")
    parser.add_argument("--no-timeline",  action="store_true",
                        help="Skip the timeline HTML")
    parser.add_argument("--no-fields",    action="store_true",
                        help="Skip the field distribution PNG")
    parser.add_argument("--no-table",     action="store_true",
                        help="Skip the score breakdown HTML table")
    return parser

def run(args):
    today     = date.today().isoformat()
    proj_name = Path(args.cache).stem.removesuffix("_cache")
    out_dir   = Path(args.out) / proj_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Loading cache: {args.cache}")
    print(f"   Project : {proj_name}")
    print(f"   Outputs : {out_dir}/")
    data = load_cache(args.cache)

    print("🔗 Building co-citation graph...")
    corpus_ids, cocit_count, all_papers = build_cocitation(data)
    print(f"   Corpus papers: {len(corpus_ids)}")
    print(f"   Candidate papers with co-citations: {len(cocit_count)}")

    print(f"🏆 Scoring top {args.top} candidates...")
    candidates = score_candidates(corpus_ids, cocit_count, all_papers, args.top)
    print(f"   Selected {len(candidates)} candidates")

    if not candidates:
        sys.exit("No candidates found — is the cache populated with references/citations?")

    out_paths = {}
    print()
    if not getattr(args, "no_network", False):
        print("🌐 Generating interactive network (HTML)...")
        p = out_dir / f"{proj_name}_network_{today}.html"
        make_network(corpus_ids, cocit_count, all_papers, candidates, p)
        out_paths["network"] = p

    if not getattr(args, "no_timeline", False):
        print("📈 Generating interactive timeline (HTML)...")
        p = out_dir / f"{proj_name}_timeline_{today}.html"
        make_timeline(corpus_ids, cocit_count, all_papers, candidates, p)
        out_paths["timeline"] = p

    if not getattr(args, "no_fields", False):
        print("📊 Generating field distribution chart (PNG)...")
        p = out_dir / f"{proj_name}_fields_{today}.png"
        make_field_chart(corpus_ids, all_papers, candidates, p)
        out_paths["fields"] = p

    if not getattr(args, "no_table", False):
        print("📋 Generating score breakdown table (HTML)...")
        p = out_dir / f"{proj_name}_scores_{today}.html"
        make_score_table(corpus_ids, cocit_count, all_papers, candidates, p)
        out_paths["table"] = p

    print(f"\n✅ Done. All outputs in: {out_dir}/\n")
    return out_paths

def main():
    parser = _make_parser()
    args   = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
