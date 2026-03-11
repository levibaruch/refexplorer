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

# ── Visualization 1: Interactive co-citation network (HTML via cosmos.gl) ─────
def make_network(corpus_ids, cocit_count, all_papers, candidates, out_path):
    """GPU-accelerated graph using @cosmos.gl/graph (WebGL). No pyvis needed."""
    candidate_pids = {pid for pid, _, _ in candidates}
    max_score      = candidates[0][1] if candidates else 1.0
    cand_score     = {pid: sc for pid, sc, _ in candidates}

    _CLUSTER_PALETTE = [
        "#e63946", "#f4a261", "#2a9d8f", "#a8dadc",
        "#a786c9", "#f7c59f", "#43aa8b", "#f9c74f",
    ]
    has_clusters = any(all_papers.get(pid, {}).get("_cluster") is not None
                       for pid, _, _ in candidates)

    def _node_color(kind, score, cluster=None):
        if kind == "corpus":
            return "#4cc9f0"
        if cluster is not None and has_clusters:
            return _CLUSTER_PALETTE[cluster % len(_CLUSTER_PALETTE)]
        q = score / max(max_score, 1e-9)
        if q >= 0.75: return "#e63946"
        if q >= 0.50: return "#f4a261"
        if q >= 0.25: return "#2a9d8f"
        return "#a8dadc"

    # ── Node list ──────────────────────────────────────────────────────────
    node_list = []
    node_idx  = {}
    added     = set()

    def add_node(pid, paper, kind, score=0.0):
        if pid in added:
            return
        added.add(pid)
        cites   = paper.get("citationCount") or 0
        cc      = cocit_count.get(pid, 0)
        doi     = (paper.get("externalIds") or {}).get("DOI", "")
        cluster = paper.get("_cluster")
        size    = max(1, min(6, 1 + 5 * math.log1p(cites) / math.log1p(5000)))
        node_idx[pid] = len(node_list)
        node_list.append({
            "title":       paper.get("title", "(no title)"),
            "year":        paper.get("year", ""),
            "author":      _first_author(paper),
            "citations":   cites,
            "coCitations": cc,
            "score":       round(score, 3),
            "doi":         doi,
            "kind":        kind,
            "color":       _node_color(kind, score, cluster),
            "size":        size,
        })

    for pid in corpus_ids:
        paper = all_papers.get(pid)
        if paper:
            add_node(pid, paper, "corpus")

    for pid, sc, paper in candidates:
        add_node(pid, all_papers.get(pid, paper), "candidate", sc)

    # ── Edge list ──────────────────────────────────────────────────────────
    edge_list = []
    edge_set  = set()
    for cp_pid in corpus_ids:
        if cp_pid not in node_idx:
            continue
        cp_paper = all_papers.get(cp_pid, {})
        neighbours = (list(cp_paper.get("references") or []) +
                      list(cp_paper.get("citations")  or []))
        for nb in neighbours:
            if not nb:
                continue
            npid = nb.get("paperId")
            if not npid or npid not in node_idx:
                continue
            key = (min(node_idx[cp_pid], node_idx[npid]),
                   max(node_idx[cp_pid], node_idx[npid]))
            if key in edge_set:
                continue
            edge_set.add(key)
            edge_list.append({
                "s": node_idx[cp_pid],
                "t": node_idx[npid],
                "v": cocit_count.get(npid, 1),
            })

    graph_data = json.dumps({"nodes": node_list, "edges": edge_list},
                            ensure_ascii=False)
    html = _NETWORK_TEMPLATE \
        .replace("__GRAPH_DATA__", graph_data) \
        .replace("__N_NODES__",   str(len(node_list))) \
        .replace("__N_EDGES__",   str(len(edge_list)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  \U0001f4e1 Network saved \u2192 {out_path}  "
          f"({len(node_list)} nodes, {len(edge_list)} edges)")


# ── cosmos.gl HTML template ────────────────────────────────────────────────────
_NETWORK_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Co-citation Network</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;background:#1a1a2e;overflow:hidden;
  font-family:system-ui,-apple-system,sans-serif}
#graph{width:100vw;height:100vh}
#ui{position:fixed;top:12px;left:12px;z-index:10;display:flex;gap:8px;align-items:center}
#search{background:#16213e;color:#fff;border:1px solid #444;border-radius:4px;
  padding:6px 12px;font-size:13px;width:220px;outline:none}
#search:focus{border-color:#4cc9f0}
#stats{color:#aaa;font-size:12px}
#tip{position:fixed;pointer-events:none;background:rgba(20,22,35,0.92);color:#e0e0e0;
  border:1px solid rgba(255,255,255,0.12);border-radius:8px;padding:10px 14px;
  font-size:12px;line-height:1.8;max-width:320px;display:none;z-index:20;
  box-shadow:0 4px 20px rgba(0,0,0,0.6);backdrop-filter:blur(4px)}
#tip .tip-title{font-size:13px;font-weight:600;color:#fff;margin-bottom:4px;
  line-height:1.4;display:block}
#tip .tip-badge{display:inline-block;font-size:10px;font-weight:700;
  padding:1px 7px;border-radius:10px;margin-bottom:6px;letter-spacing:.5px}
#tip .tip-row{display:flex;justify-content:space-between;gap:16px;color:#aaa;font-size:11px}
#tip .tip-row span:last-child{color:#e0e0e0;font-weight:500}
#legend{position:fixed;bottom:12px;left:12px;color:#aaa;font-size:11px;z-index:10}
#legend span{display:inline-block;width:10px;height:10px;border-radius:50%;
  margin-right:4px;vertical-align:middle}
</style>
</head>
<body>
<div id="ui">
  <input id="search" type="text" placeholder="Search papers\u2026" autocomplete="off">
  <span id="stats">__N_NODES__ nodes &middot; __N_EDGES__ edges</span>
</div>
<div id="tip"></div>
<div id="legend">
  <span style="background:#4cc9f0"></span>Corpus &nbsp;
  <span style="background:#e63946"></span>High &nbsp;
  <span style="background:#f4a261"></span>Medium &nbsp;
  <span style="background:#2a9d8f"></span>Low
  &nbsp;<span style="color:#555;font-size:10px">scroll=zoom \u2022 drag=pan \u2022 click=DOI</span>
</div>
<div id="graph"></div>

<script type="module">
import { Graph } from 'https://esm.sh/@cosmos.gl/graph@2.6.4';

const RAW   = __GRAPH_DATA__;
const nodes = RAW.nodes;
const edges = RAW.edges;
const N = nodes.length, E = edges.length;

// ── Parse hex color \u2192 RGBA floats ────────────────────────────────────────
function hexRgba(hex, a) {
  return [
    parseInt(hex.slice(1,3),16)/255,
    parseInt(hex.slice(3,5),16)/255,
    parseInt(hex.slice(5,7),16)/255,
    a,
  ];
}

// ── Adjacency list for hover-highlight ───────────────────────────────────
const adj = Array.from({length:N}, ()=>[]);
edges.forEach(e=>{ adj[e.s].push(e.t); adj[e.t].push(e.s); });

// ── Float32Arrays ─────────────────────────────────────────────────────────
const positions  = new Float32Array(N * 2);
const colors     = new Float32Array(N * 4);
const sizes      = new Float32Array(N);
const linkArr    = new Float32Array(E * 2);
const linkColors = new Float32Array(E * 4);
const linkWidths = new Float32Array(E);

for (let i = 0; i < N; i++) {
  const a = Math.random() * 2 * Math.PI;
  const r = Math.sqrt(Math.random());
  positions[i*2]   = Math.cos(a) * r;
  positions[i*2+1] = Math.sin(a) * r;
}

nodes.forEach((n, i) => {
  const [r,g,b,a] = hexRgba(n.color, 1.0);
  colors[i*4]=r; colors[i*4+1]=g; colors[i*4+2]=b; colors[i*4+3]=a;
  sizes[i] = n.size;
});

edges.forEach((e, i) => {
  linkArr[i*2]=e.s; linkArr[i*2+1]=e.t;
  const op = 0.08 + 0.17 * Math.min(e.v / 5, 1);
  linkColors[i*4]=1; linkColors[i*4+1]=1; linkColors[i*4+2]=1; linkColors[i*4+3]=op;
  linkWidths[i] = Math.max(0.6, 0.4 + e.v * 0.3);
});

// ── Tooltip ───────────────────────────────────────────────────────────────
const tip = document.getElementById('tip');
function esc(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function row(label, val) {
  return '<div class="tip-row"><span>'+label+'</span><span>'+val+'</span></div>';
}
function showTip(i) {
  const n = nodes[i];
  const isCorpus = n.kind === 'corpus';
  const badgeColor = isCorpus ? '#4cc9f0' : n.color;
  const badgeText  = isCorpus ? 'CORPUS' : 'CANDIDATE';
  const title = n.title.length > 60 ? n.title.slice(0,60)+'\u2026' : n.title;
  tip.innerHTML =
    '<span class="tip-title">'+esc(title)+'</span>'
    + '<span class="tip-badge" style="background:'+badgeColor+'22;color:'+badgeColor+';border:1px solid '+badgeColor+'55">'+badgeText+'</span>'
    + row('Year',          n.year  || '—')
    + row('First author',  esc(n.author) || '—')
    + row('Total citations', (n.citations||0).toLocaleString())
    + (isCorpus ? '' : row('Co-citations',  n.coCitations || 0))
    + (isCorpus ? '' : row('Score',         n.score       || '—'));
  tip.style.display = 'block';
}
document.addEventListener('mousemove', e => {
  if (tip.style.display === 'block') {
    tip.style.left = Math.min(e.clientX+15, window.innerWidth-360)+'px';
    tip.style.top  = Math.min(e.clientY+15, window.innerHeight-220)+'px';
  }
});

// ── Hover highlight ───────────────────────────────────────────────────────
// Manual colour update — avoids selectPointsByIndices() which fires
// onHover(undefined) internally and immediately kills the tooltip.
let hoveredIdx = -1;
const origColors = new Float32Array(colors);   // snapshot of initial colours
const origLinks  = new Float32Array(linkColors);

function applyHighlight(idx) {
  const visible = new Set([idx]);
  adj[idx].forEach(nb => { if (nodes[nb].kind === 'corpus') visible.add(nb); });

  for (let i = 0; i < N; i++) {
    if (visible.has(i)) {
      colors[i*4]=origColors[i*4]; colors[i*4+1]=origColors[i*4+1];
      colors[i*4+2]=origColors[i*4+2]; colors[i*4+3]=1.0;
    } else {
      colors[i*4]=0.08; colors[i*4+1]=0.08; colors[i*4+2]=0.12; colors[i*4+3]=0.04;
    }
  }
  for (let i = 0; i < E; i++) {
    const s = linkArr[i*2], t = linkArr[i*2+1];
    linkColors[i*4+3] = (visible.has(s) && visible.has(t)) ? 0.7 : 0.01;
  }
  graph.setPointColors(colors);
  graph.setLinkColors(linkColors);
}

function clearHighlight() {
  colors.set(origColors);
  linkColors.set(origLinks);
  graph.setPointColors(colors);
  graph.setLinkColors(linkColors);
}

// ── Search ────────────────────────────────────────────────────────────────
const searchSizes = new Float32Array(sizes);
document.getElementById('search').addEventListener('input', function() {
  const term = this.value.toLowerCase().trim();
  for (let i = 0; i < N; i++) {
    searchSizes[i] = (!term || nodes[i].title.toLowerCase().includes(term)) ? sizes[i] : 0;
  }
  graph.setPointSizes(searchSizes);
});

// ── Track zoom level for label visibility ─────────────────────────────────
let   currentK     = 1;
const LABEL_SHOW_K = 3.0;

// ── Hover handler (called from both config callback and direct listener) ───
function handleHover(idx) {
  if (idx === undefined || idx === null) {
    if (hoveredIdx >= 0) { clearHighlight(); hoveredIdx = -1; }
    tip.style.display = 'none';
  } else {
    if (idx === hoveredIdx) return;   // already handled
    hoveredIdx = idx;
    applyHighlight(idx);
    showTip(idx);
  }
}

// ── Create cosmos graph ───────────────────────────────────────────────────
const graph = new Graph(document.getElementById('graph'), {
  // Rendering
  spaceSize:                4096,
  backgroundColor:          '#1a1a2e',
  scalePointsOnZoom:        true,
  pointGreyoutOpacity:      1,      // disable built-in greyout — handled manually
  linkGreyoutOpacity:       1,      // disable built-in greyout — handled manually
  renderHoveredPointRing:   true,
  hoveredPointCursor:       'pointer',

  // Fit view
  fitViewOnInit:            false,

  // Links
  linkDefaultWidth:         0.6,
  linkDefaultColor:         '#5F74C2',
  linkDefaultArrows:        false,

  // Simulation
  simulationGravity:        0.1,
  simulationLinkDistance:   1,
  simulationLinkSpring:     0.3,
  simulationRepulsion:      0.4,

  enableDrag:               true,
  curvedLinks:              false,

  // Track zoom so labels appear only when sufficiently zoomed in
  onZoom: (event) => {
    const k = event?.transform?.k;
    if (k) currentK = k;
  },
  onZoomEnd: (event) => {
    const k = event?.transform?.k;
    if (k) currentK = k;
  },

  // Config-based hover callback (may or may not fire depending on cosmos version)
  onHover: handleHover,

  onClick: (idx) => {
    if (idx !== undefined && idx !== null) {
      const doi = nodes[idx].doi;
      if (doi) window.open('https://doi.org/' + doi, '_blank');
    }
  },
});

// ── Load data and render ──────────────────────────────────────────────────
graph.setPointPositions(positions);
graph.setLinks(linkArr);
graph.setPointColors(colors);
graph.setPointSizes(sizes);
graph.setLinkColors(linkColors);
graph.setLinkWidths(linkWidths);
graph.render();

// ── Direct mousemove listener as guaranteed hover fallback ────────────────
// cosmos's config onHover may not fire in all versions; this catches the
// hover index via getPointIndexByCoordinates (available in cosmos v2.x).
document.getElementById('graph').addEventListener('mousemove', (e) => {
  if (typeof graph.getPointIndexByCoordinates === 'function') {
    const idx = graph.getPointIndexByCoordinates(e.offsetX, e.offsetY);
    handleHover(idx === -1 ? undefined : idx);
  }
});
document.getElementById('graph').addEventListener('mouseleave', () => {
  handleHover(undefined);
});

// ── Canvas label overlay ──────────────────────────────────────────────────
// Shows persistent labels for corpus + high-score (red) nodes when zoomed in,
// plus a label for whatever node is currently hovered.

// Identify which nodes get persistent labels: corpus or red (#e63946)
const labelIndices = [];
nodes.forEach((n, i) => {
  if (n.kind === 'corpus' || n.color === '#e63946') labelIndices.push(i);
});
graph.trackPointPositionsByIndices(labelIndices);

const labelCanvas = document.createElement('canvas');
labelCanvas.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:5';
document.body.appendChild(labelCanvas);
const ctx2d = labelCanvas.getContext('2d');
function resizeLabel() {
  labelCanvas.width  = window.innerWidth;
  labelCanvas.height = window.innerHeight;
}
resizeLabel();
window.addEventListener('resize', resizeLabel);

function drawOneLabel(graphPos, text, color) {
  if (!graphPos) return;
  const [x, y] = graph.spaceToScreenPosition(graphPos);
  if (x < 0 || y < 0 || x > labelCanvas.width || y > labelCanvas.height) return;
  ctx2d.shadowColor = 'rgba(0,0,0,0.8)';
  ctx2d.shadowBlur  = 4;
  ctx2d.fillStyle   = color;
  ctx2d.fillText(text, x + 6, y);
  ctx2d.shadowBlur  = 0;
}

function drawLabels() {
  ctx2d.clearRect(0, 0, labelCanvas.width, labelCanvas.height);
  ctx2d.font        = 'bold 10px system-ui,-apple-system,sans-serif';
  ctx2d.textBaseline = 'middle';

  // Persistent labels for corpus + red nodes (only when zoomed in enough)
  if (currentK >= LABEL_SHOW_K) {
    const posMap = graph.getTrackedPointPositionsMap();
    if (posMap) {
      posMap.forEach((graphPos, idx) => {
        const n = nodes[idx];
        const color = n.kind === 'corpus' ? '#4cc9f0' : '#ff6b6b';
        const title = n.title.length > 35 ? n.title.slice(0, 35) + '\u2026' : n.title;
        drawOneLabel(graphPos, title, color);
      });
    }
  }

  // Hovered node label — always visible regardless of zoom
  if (hoveredIdx >= 0) {
    const posMap = graph.getTrackedPointPositionsMap();
    // getTrackedPointPositionsMap only has labelIndices; for other nodes
    // fall back to the tooltip position which is always available
    const n = nodes[hoveredIdx];
    const title = n.title.length > 40 ? n.title.slice(0, 40) + '\u2026' : n.title;
    const cc    = n.coCitations ? ` · ${n.coCitations} co-cit.` : '';
    const text  = title + cc;

    // Try tracked position first; otherwise draw near tooltip
    const trackedPos = posMap && posMap.get(hoveredIdx);
    if (trackedPos) {
      ctx2d.font = 'bold 11px system-ui,-apple-system,sans-serif';
      drawOneLabel(trackedPos, text, '#ffffff');
      ctx2d.font = 'bold 10px system-ui,-apple-system,sans-serif';
    } else {
      // Draw near the tooltip element
      const tl = tip.style.left ? parseInt(tip.style.left) : -1;
      const tt = tip.style.top  ? parseInt(tip.style.top)  : -1;
      if (tl > 0 && tt > 0) {
        ctx2d.shadowColor = 'rgba(0,0,0,0.9)';
        ctx2d.shadowBlur  = 5;
        ctx2d.fillStyle   = '#ffffff';
        ctx2d.font        = 'bold 11px system-ui,-apple-system,sans-serif';
        ctx2d.fillText(text, tl, tt - 14);
        ctx2d.shadowBlur  = 0;
        ctx2d.font        = 'bold 10px system-ui,-apple-system,sans-serif';
      }
    }
  }

  requestAnimationFrame(drawLabels);
}
requestAnimationFrame(drawLabels);
</script>
</body></html>"""

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
