#!/usr/bin/env python3
"""
ULTRON Knowledge Graph Viewer
- Zero Python dependencies
- Zero JavaScript/CDN dependencies
- Canvas-based rendering (fast for thousands of nodes)
- Search + node details + neighborhood expansion
- Safe initial view: shows the largest connected component instead of accidentally
  rendering a blank canvas
- Automatically finds a free port if 8765 is already occupied

Run from the directory containing wiki/:
    python3 graph_viewer_fixed.py

Then open the URL printed by the program.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


# ============================================================
# CONFIG
# ============================================================

HOST = os.environ.get("ULTRON_GRAPH_HOST", "127.0.0.1")
START_PORT = int(os.environ.get("ULTRON_GRAPH_PORT", "8765"))
GRAPH_FILE = Path("wiki/graph/graph.json")


# ============================================================
# GRAPH LOADING / NORMALIZATION
# ============================================================

def load_graph():
    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"\nGraph file not found:\n  {GRAPH_FILE.resolve()}\n\n"
            "Run this script from your ULTRON/knowledgeBase directory."
        )

    with GRAPH_FILE.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    raw_nodes = raw.get("nodes", [])
    raw_edges = raw.get("edges", [])

    nodes = []
    node_ids = set()

    for i, n in enumerate(raw_nodes):
        if not isinstance(n, dict):
            continue

        node_id = str(n.get("id", i))
        if node_id in node_ids:
            continue

        node_ids.add(node_id)

        aliases = n.get("aliases", [])
        sources = n.get("sources", [])

        if not isinstance(aliases, list):
            aliases = [str(aliases)]
        if not isinstance(sources, list):
            sources = [str(sources)]

        nodes.append({
            "id": node_id,
            "name": str(n.get("name") or node_id),
            "type": str(n.get("type") or n.get("node_type") or "other"),
            "description": str(n.get("description") or ""),
            "aliases": [str(x) for x in aliases],
            "sources": [str(x) for x in sources],
        })

    edges = []
    seen_edges = set()

    for e in raw_edges:
        if not isinstance(e, dict):
            continue

        source = str(e.get("source", ""))
        target = str(e.get("target", ""))

        if source not in node_ids or target not in node_ids:
            continue

        # Keep parallel relations if their relation differs.
        key = (
            source,
            target,
            str(e.get("relation") or "related_to"),
            str(e.get("source_chunk") or ""),
        )
        if key in seen_edges:
            continue
        seen_edges.add(key)

        edges.append({
            "id": len(edges),
            "source": source,
            "target": target,
            "relation": str(e.get("relation") or "related_to"),
            "description": str(e.get("description") or ""),
            "source_chunk": str(e.get("source_chunk") or ""),
        })

    graph = {"nodes": nodes, "edges": edges}

    print()
    print("=" * 64)
    print("ULTRON GRAPH VIEWER")
    print("=" * 64)
    print(f"Graph file : {GRAPH_FILE.resolve()}")
    print(f"Nodes      : {len(nodes):,}")
    print(f"Edges      : {len(edges):,}")
    print("=" * 64)
    print()

    return graph


GRAPH = load_graph()


# ============================================================
# HTML / JAVASCRIPT
# ============================================================

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ULTRON Knowledge Graph</title>

<style>
:root {
    --bg: #070a0f;
    --panel: #0d121a;
    --panel2: #111824;
    --border: #202b3b;
    --text: #e8eef8;
    --muted: #8190a6;
    --accent: #5b9cff;
}

* { box-sizing: border-box; }

html, body {
    width: 100%;
    height: 100%;
    margin: 0;
    overflow: hidden;
    background: var(--bg);
    color: var(--text);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                 BlinkMacSystemFont, "Segoe UI", sans-serif;
}

#app {
    width: 100%;
    height: 100%;
    display: grid;
    grid-template-columns: 290px minmax(0, 1fr) 330px;
}

aside {
    background: var(--panel);
    overflow-y: auto;
    padding: 16px;
}

#left {
    border-right: 1px solid var(--border);
}

#right {
    border-left: 1px solid var(--border);
}

#stage {
    position: relative;
    min-width: 0;
    min-height: 0;
    background:
        radial-gradient(circle at 50% 45%, #121b2a 0%, #080c13 58%, #05070b 100%);
}

#canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    cursor: grab;
}

#canvas.dragging {
    cursor: grabbing;
}

.logo {
    font-size: 21px;
    font-weight: 850;
    letter-spacing: .7px;
}

.subtitle {
    color: var(--muted);
    font-size: 11px;
    margin-top: 2px;
    margin-bottom: 18px;
}

.section {
    margin-bottom: 18px;
}

.title {
    color: #69788f;
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 1.2px;
    font-weight: 750;
    margin-bottom: 8px;
}

input, select, button {
    width: 100%;
    border: 1px solid var(--border);
    background: var(--panel2);
    color: var(--text);
    border-radius: 8px;
    padding: 9px 10px;
    font-size: 12px;
    outline: none;
}

input:focus, select:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(91,156,255,.12);
}

button {
    cursor: pointer;
    margin-top: 6px;
}

button:hover {
    border-color: #3c4d68;
    background: #172131;
}

button.primary {
    background: #1e55c9;
    border-color: #3472ee;
}

button.primary:hover {
    background: #2865e5;
}

.stat {
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #182130;
    padding: 6px 0;
    font-size: 11px;
}

.stat span:first-child { color: var(--muted); }
.stat span:last-child { font-weight: 700; }

.checkbox {
    display: flex;
    align-items: center;
    gap: 7px;
    color: #aeb9ca;
    font-size: 11px;
    margin: 8px 0;
}

.checkbox input {
    width: auto;
}

#searchResults {
    margin-top: 7px;
    max-height: 250px;
    overflow-y: auto;
}

.result {
    border: 1px solid var(--border);
    background: #101722;
    border-radius: 7px;
    padding: 8px 9px;
    margin-bottom: 4px;
    cursor: pointer;
}

.result:hover {
    background: #182231;
}

.resultName {
    font-weight: 750;
    font-size: 12px;
}

.resultMeta {
    color: var(--muted);
    font-size: 10px;
    margin-top: 2px;
}

#status {
    position: absolute;
    left: 12px;
    bottom: 12px;
    z-index: 5;
    padding: 6px 9px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: rgba(8,12,18,.88);
    color: #8392a9;
    font-size: 10px;
    pointer-events: none;
}

#toolbar {
    position: absolute;
    top: 12px;
    left: 12px;
    right: 12px;
    z-index: 4;
    display: flex;
    gap: 6px;
    pointer-events: none;
}

.tool {
    width: auto;
    margin: 0;
    padding: 7px 9px;
    pointer-events: auto;
    background: rgba(13,18,26,.9);
}

#detailsTitle {
    color: #75849b;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 10px;
    font-weight: 800;
    margin-bottom: 12px;
}

#nodeName {
    font-size: 19px;
    font-weight: 850;
    line-height: 1.25;
    word-break: break-word;
}

.badge {
    display: inline-block;
    margin-top: 7px;
    padding: 4px 7px;
    border-radius: 999px;
    background: #18243a;
    color: #8db7ff;
    font-size: 10px;
}

.description {
    margin-top: 13px;
    color: #aab6c7;
    font-size: 12px;
    line-height: 1.55;
    white-space: pre-wrap;
}

.block {
    margin-top: 17px;
}

.blockTitle {
    color: #68768c;
    text-transform: uppercase;
    font-size: 9px;
    letter-spacing: 1px;
    font-weight: 800;
    margin-bottom: 7px;
}

.item {
    background: #111824;
    border: 1px solid #1d2838;
    border-radius: 7px;
    padding: 7px;
    margin-bottom: 5px;
    font-size: 10px;
    line-height: 1.45;
}

.item b { color: #8eb7ff; }
.item span { color: #77869d; }

.empty {
    color: #66758b;
    font-size: 12px;
    line-height: 1.5;
}

kbd {
    border: 1px solid #2a374b;
    background: #111824;
    border-radius: 4px;
    padding: 1px 4px;
    color: #a9b7ca;
    font-size: 9px;
}
</style>
</head>

<body>
<div id="app">

<aside id="left">
    <div class="logo">ULTRON</div>
    <div class="subtitle">Knowledge Graph Explorer</div>

    <div class="section">
        <div class="title">Search</div>
        <input id="search" placeholder="Search nodes..." autocomplete="off">
        <div id="searchResults"></div>
    </div>

    <div class="section">
        <div class="title">Node type</div>
        <select id="typeFilter">
            <option value="all">All types</option>
        </select>
    </div>

    <div class="section">
        <div class="title">View</div>
        <button class="primary" onclick="showCore()">Show connected graph</button>
        <button onclick="showAll()">Show all nodes</button>
        <button onclick="fitView()">Fit visible</button>
        <button onclick="resetView()">Reset zoom</button>
        <button onclick="hideIsolated()">Hide isolated</button>
        <button onclick="expandSelected()">Expand selected</button>
    </div>

    <div class="section">
        <div class="title">Graph statistics</div>
        <div class="stat"><span>Total nodes</span><span id="totalNodes">0</span></div>
        <div class="stat"><span>Total edges</span><span id="totalEdges">0</span></div>
        <div class="stat"><span>Visible nodes</span><span id="visibleNodes">0</span></div>
        <div class="stat"><span>Visible edges</span><span id="visibleEdges">0</span></div>
        <div class="stat"><span>Components</span><span id="components">0</span></div>
    </div>

    <div class="section">
        <div class="title">Options</div>

        <label class="checkbox">
            <input id="labels" type="checkbox" checked>
            Show labels
        </label>

        <label class="checkbox">
            <input id="edges" type="checkbox" checked>
            Show edges
        </label>

        <label class="checkbox">
            <input id="isolated" type="checkbox">
            Show isolated nodes
        </label>
    </div>

    <div class="section">
        <div class="title">Keyboard</div>
        <div class="empty">
            <kbd>/</kbd> search &nbsp;
            <kbd>Esc</kbd> clear<br><br>
            Mouse wheel = zoom<br>
            Drag background = pan<br>
            Click node = inspect<br>
            Double click = focus
        </div>
    </div>
</aside>


<main id="stage">
    <canvas id="canvas"></canvas>

    <div id="toolbar">
        <button class="tool" onclick="zoomIn()">＋</button>
        <button class="tool" onclick="zoomOut()">−</button>
        <button class="tool" onclick="fitView()">Fit</button>
    </div>

    <div id="status">Loading graph...</div>
</main>


<aside id="right">
    <div id="detailsTitle">Node details</div>

    <div id="nodeDetails" class="empty">
        Click a node to inspect it.
    </div>
</aside>

</div>

<script>
"use strict";

/* ============================================================
   DATA
   ============================================================ */

let graph = null;
let nodes = [];
let edges = [];

const nodeById = new Map();
const adjacency = new Map();
const components = [];
const degree = new Map();

let visible = new Set();
let selectedId = null;

let camera = {
    x: 0,
    y: 0,
    scale: 1
};

let canvas, ctx;
let width = 0;
let height = 0;
let dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));

let dragging = false;
let dragLastX = 0;
let dragLastY = 0;
let moved = false;


/* ============================================================
   COLORS
   ============================================================ */

const COLORS = {
    concept: "#4f8cff",
    technology: "#9b6cff",
    person: "#31c48d",
    organization: "#f59e0b",
    method: "#ec4899",
    algorithm: "#ef4444",
    dataset: "#22c1dc",
    model: "#7270ff",
    tool: "#84cc16",
    paper: "#06b6d4",
    topic: "#60a5fa",
    default: "#64748b"
};

function nodeColor(node) {
    return COLORS[node.type] || COLORS.default;
}


/* ============================================================
   STARTUP
   ============================================================ */

async function start() {
    canvas = document.getElementById("canvas");

    if (!canvas || !canvas.getContext) {
        setStatus("ERROR: canvas element not found");
        return;
    }

    ctx = canvas.getContext("2d");

    bindInteractions();

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    try {
        const response = await fetch("/graph", {cache: "no-store"});

        if (!response.ok) {
            throw new Error("HTTP " + response.status);
        }

        graph = await response.json();

        normalizeGraph();
        buildStructure();
        buildTypeFilter();

        // IMPORTANT:
        // Do NOT hide everything and wait for a physics engine.
        // We create deterministic positions immediately.
        buildInitialPositions();

        showCore(false);
        draw();
        updateStats();

        setStatus(
            `${nodes.length.toLocaleString()} nodes · ` +
            `${edges.length.toLocaleString()} edges`
        );

    } catch (error) {
        console.error(error);
        setStatus("ERROR: " + error.message);
        document.getElementById("nodeDetails").innerHTML =
            `<div class="empty">
                Could not load the graph.<br><br>
                ${escapeHtml(error.message)}
            </div>`;
    }
}


/* ============================================================
   NORMALIZATION
   ============================================================ */

function normalizeGraph() {
    nodes = (graph.nodes || []).map((n, i) => ({
        id: String(n.id ?? i),
        name: String(n.name || n.id || i),
        type: String(n.type || n.node_type || "other"),
        description: String(n.description || ""),
        aliases: Array.isArray(n.aliases) ? n.aliases : [],
        sources: Array.isArray(n.sources) ? n.sources : [],
        x: 0,
        y: 0,
        r: 6
    }));

    nodeById.clear();
    nodes.forEach(n => nodeById.set(n.id, n));

    edges = (graph.edges || [])
        .map((e, i) => ({
            id: i,
            source: String(e.source),
            target: String(e.target),
            relation: String(e.relation || "related_to"),
            description: String(e.description || ""),
            source_chunk: String(e.source_chunk || "")
        }))
        .filter(e => nodeById.has(e.source) && nodeById.has(e.target));
}


/* ============================================================
   STRUCTURE
   ============================================================ */

function buildStructure() {
    adjacency.clear();
    degree.clear();

    nodes.forEach(n => {
        adjacency.set(n.id, new Set());
        degree.set(n.id, 0);
    });

    edges.forEach(e => {
        if (!adjacency.has(e.source) || !adjacency.has(e.target)) return;

        adjacency.get(e.source).add(e.target);
        adjacency.get(e.target).add(e.source);

        degree.set(e.source, degree.get(e.source) + 1);
        degree.set(e.target, degree.get(e.target) + 1);
    });

    components.length = 0;

    const seen = new Set();

    for (const n of nodes) {
        if (seen.has(n.id)) continue;

        const component = [];
        const queue = [n.id];
        seen.add(n.id);

        while (queue.length) {
            const id = queue.pop();
            component.push(id);

            for (const next of adjacency.get(id)) {
                if (!seen.has(next)) {
                    seen.add(next);
                    queue.push(next);
                }
            }
        }

        components.push(component);
    }

    components.sort((a, b) => b.length - a.length);
}


/* ============================================================
   INITIAL POSITIONING
   ============================================================ */

function buildInitialPositions() {
    /*
       We deliberately avoid a force simulation.

       Your graph has many more nodes than edges. A force layout
       wastes time on thousands of isolated nodes and can produce
       a blank/unstable first frame.

       Connected components are arranged in rings.
       This is deterministic, instant, and always drawable.
    */

    const connected = components.filter(c => c.length > 1);

    let componentIndex = 0;

    for (const component of connected) {
        const cx = (componentIndex % 4) * 900;
        const cy = Math.floor(componentIndex / 4) * 700;

        const root = component
            .slice()
            .sort((a, b) => degree.get(b) - degree.get(a))[0];

        const dist = new Map([[root, 0]]);
        const queue = [root];

        while (queue.length) {
            const current = queue.shift();

            for (const next of adjacency.get(current)) {
                if (!dist.has(next)) {
                    dist.set(next, dist.get(current) + 1);
                    queue.push(next);
                }
            }
        }

        const layers = new Map();

        for (const id of component) {
            const d = dist.get(id) || 0;
            if (!layers.has(d)) layers.set(d, []);
            layers.get(d).push(id);
        }

        for (const [layer, ids] of layers) {
            const radius = layer === 0 ? 0 : 100 + layer * 80;
            const count = ids.length;

            ids.forEach((id, i) => {
                const n = nodeById.get(id);

                if (layer === 0) {
                    n.x = cx;
                    n.y = cy;
                } else {
                    const angle = (Math.PI * 2 * i / count) + layer * 0.35;
                    n.x = cx + Math.cos(angle) * radius;
                    n.y = cy + Math.sin(angle) * radius;
                }

                n.r = Math.min(11, 5 + Math.log2(1 + degree.get(id)) * 1.4);
            });
        }

        componentIndex++;
    }

    // Isolated nodes live in a compact grid far below the main graph.
    const isolated = nodes.filter(n => degree.get(n.id) === 0);
    const cols = Math.max(1, Math.ceil(Math.sqrt(isolated.length)));

    isolated.forEach((n, i) => {
        n.x = -1600 + (i % cols) * 28;
        n.y = 1800 + Math.floor(i / cols) * 28;
        n.r = 3.5;
    });
}


/* ============================================================
   TYPE FILTER
   ============================================================ */

function buildTypeFilter() {
    const select = document.getElementById("typeFilter");
    const types = [...new Set(nodes.map(n => n.type))].sort();

    for (const type of types) {
        const option = document.createElement("option");
        option.value = type;
        option.textContent = type;
        select.appendChild(option);
    }

    select.addEventListener("change", applyFilters);
}


/* ============================================================
   VISIBILITY
   ============================================================ */

function showCore(fit = true) {
    const core = components.find(c => c.length > 1) || [];

    visible = new Set(core);

    // If graph has no connected component, show first 100 nodes.
    if (!visible.size) {
        visible = new Set(nodes.slice(0, 100).map(n => n.id));
    }

    applyFilters(false);

    if (fit) fitView();
    updateStats();

    setStatus(
        `Connected graph: ${visible.size.toLocaleString()} nodes`
    );
}

function showAll() {
    visible = new Set(nodes.map(n => n.id));
    applyFilters(false);
    fitView();
    updateStats();
    setStatus("All nodes visible");
}

function hideIsolated() {
    visible = new Set(
        nodes
            .filter(n => degree.get(n.id) > 0)
            .map(n => n.id)
    );

    applyFilters(false);
    fitView();
    updateStats();

    setStatus(
        `${visible.size.toLocaleString()} connected nodes visible`
    );
}

function applyFilters(refit = true) {
    const type = document.getElementById("typeFilter").value;
    const showIsolated = document.getElementById("isolated").checked;

    if (type !== "all") {
        visible = new Set(
            [...visible].filter(id => nodeById.get(id)?.type === type)
        );
    }

    if (!showIsolated) {
        for (const id of [...visible]) {
            if (degree.get(id) === 0) visible.delete(id);
        }
    }

    draw();
    updateStats();

    if (refit) fitView();
}

document.getElementById("isolated").addEventListener(
    "change",
    () => {
        if (document.getElementById("isolated").checked) {
            showAll();
        } else {
            hideIsolated();
        }
    }
);


/* ============================================================
   SEARCH
   ============================================================ */

document.getElementById("search").addEventListener("input", function () {
    const q = this.value.trim().toLowerCase();
    const box = document.getElementById("searchResults");

    box.innerHTML = "";

    if (!q) {
        return;
    }

    const results = nodes
        .filter(n =>
            n.name.toLowerCase().includes(q) ||
            n.type.toLowerCase().includes(q) ||
            n.description.toLowerCase().includes(q) ||
            n.aliases.some(a => String(a).toLowerCase().includes(q))
        )
        .sort((a, b) => {
            const ax = a.name.toLowerCase().startsWith(q) ? 0 : 1;
            const bx = b.name.toLowerCase().startsWith(q) ? 0 : 1;
            return ax - bx || degree.get(b.id) - degree.get(a.id);
        })
        .slice(0, 40);

    for (const node of results) {
        const div = document.createElement("div");
        div.className = "result";

        div.innerHTML =
            `<div class="resultName">${escapeHtml(node.name)}</div>` +
            `<div class="resultMeta">${escapeHtml(node.type)} · ` +
            `${degree.get(node.id)} connections</div>`;

        div.onclick = () => {
            visible.add(node.id);

            for (const neighbor of adjacency.get(node.id)) {
                visible.add(neighbor);
            }

            selectedId = node.id;

            focusNode(node.id);
            showDetails(node.id);

            box.innerHTML = "";
        };

        box.appendChild(div);
    }
});


/* ============================================================
   NODE FOCUS / EXPAND
   ============================================================ */

function focusNode(id) {
    const n = nodeById.get(String(id));
    if (!n) return;

    visible.add(n.id);
    selectedId = n.id;

    camera.x = n.x;
    camera.y = n.y;
    camera.scale = Math.max(camera.scale, 1.2);

    draw();
    showDetails(n.id);
}

function expandSelected() {
    if (!selectedId) {
        setStatus("Select a node first");
        return;
    }

    const id = String(selectedId);
    visible.add(id);

    for (const neighbor of adjacency.get(id) || []) {
        visible.add(neighbor);
    }

    draw();
    fitView();
    updateStats();

    setStatus(
        `Expanded ${nodeById.get(id)?.name || id}`
    );
}


/* ============================================================
   DETAILS
   ============================================================ */

function showDetails(id) {
    const node = nodeById.get(String(id));
    if (!node) return;

    selectedId = node.id;

    const outgoing = edges.filter(e => e.source === node.id);
    const incoming = edges.filter(e => e.target === node.id);

    let html = "";

    html += `<div id="nodeName">${escapeHtml(node.name)}</div>`;
    html += `<div class="badge">${escapeHtml(node.type)}</div>`;

    if (node.description) {
        html += `<div class="description">${escapeHtml(node.description)}</div>`;
    }

    if (node.aliases.length) {
        html += `<div class="block">`;
        html += `<div class="blockTitle">Aliases</div>`;

        for (const alias of node.aliases.slice(0, 30)) {
            html += `<div class="item">${escapeHtml(alias)}</div>`;
        }

        html += `</div>`;
    }

    html += `<div class="block">`;
    html += `<div class="blockTitle">Connections (${outgoing.length + incoming.length})</div>`;

    for (const e of outgoing.slice(0, 80)) {
        const target = nodeById.get(e.target);

        html +=
            `<div class="item">` +
            `<b>→ ${escapeHtml(target?.name || e.target)}</b>` +
            `<br><span>${escapeHtml(e.relation)}</span>` +
            (e.description ? `<br><span>${escapeHtml(e.description)}</span>` : "") +
            `</div>`;
    }

    for (const e of incoming.slice(0, 80)) {
        const source = nodeById.get(e.source);

        html +=
            `<div class="item">` +
            `<b>← ${escapeHtml(source?.name || e.source)}</b>` +
            `<br><span>${escapeHtml(e.relation)}</span>` +
            (e.description ? `<br><span>${escapeHtml(e.description)}</span>` : "") +
            `</div>`;
    }

    html += `</div>`;

    if (node.sources.length) {
        html += `<div class="block">`;
        html += `<div class="blockTitle">Source chunks</div>`;

        for (const source of node.sources.slice(0, 20)) {
            html += `<div class="item">${escapeHtml(source)}</div>`;
        }

        html += `</div>`;
    }

    document.getElementById("nodeDetails").innerHTML = html;

    draw();
}


/* ============================================================
   CANVAS
   ============================================================ */

function resizeCanvas() {
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();

    width = rect.width;
    height = rect.height;

    dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));

    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    draw();
}

function worldToScreen(x, y) {
    return {
        x: (x - camera.x) * camera.scale + width / 2,
        y: (y - camera.y) * camera.scale + height / 2
    };
}

function screenToWorld(x, y) {
    return {
        x: (x - width / 2) / camera.scale + camera.x,
        y: (y - height / 2) / camera.scale + camera.y
    };
}

function draw() {
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    // Subtle grid.
    drawGrid();

    const showEdges = document.getElementById("edges").checked;
    const showLabels = document.getElementById("labels").checked;

    if (showEdges) {
        ctx.lineWidth = Math.max(0.5, Math.min(2, camera.scale));

        for (const e of edges) {
            if (!visible.has(e.source) || !visible.has(e.target)) continue;

            const a = nodeById.get(e.source);
            const b = nodeById.get(e.target);

            if (!a || !b) continue;

            const p1 = worldToScreen(a.x, a.y);
            const p2 = worldToScreen(b.x, b.y);

            if (!segmentVisible(p1.x, p1.y, p2.x, p2.y)) continue;

            ctx.strokeStyle = "#33445d";
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();

            if (camera.scale > 0.85) {
                drawArrow(p1, p2);
            }
        }
    }

    for (const n of nodes) {
        if (!visible.has(n.id)) continue;

        const p = worldToScreen(n.x, n.y);

        if (
            p.x < -40 || p.x > width + 40 ||
            p.y < -40 || p.y > height + 40
        ) {
            continue;
        }

        const radius = Math.max(2.5, n.r * camera.scale);

        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);

        ctx.fillStyle = nodeColor(n);
        ctx.fill();

        if (n.id === selectedId) {
            ctx.lineWidth = 3;
            ctx.strokeStyle = "#ffffff";
            ctx.stroke();
        } else {
            ctx.lineWidth = 1;
            ctx.strokeStyle = "rgba(255,255,255,.25)";
            ctx.stroke();
        }

        if (showLabels && camera.scale > 0.72 && radius > 3.5) {
            ctx.font = `${Math.max(9, Math.min(14, 11 * camera.scale))}px Inter, system-ui`;
            ctx.fillStyle = n.id === selectedId ? "#ffffff" : "#c9d4e4";
            ctx.fillText(
                truncate(n.name, camera.scale),
                p.x + radius + 4,
                p.y + 4
            );
        }
    }
}

function drawGrid() {
    const grid = 100 * camera.scale;

    if (grid < 20) return;

    const origin = worldToScreen(0, 0);

    ctx.strokeStyle = "rgba(255,255,255,.035)";
    ctx.lineWidth = 1;

    let startX = origin.x % grid;
    let startY = origin.y % grid;

    for (let x = startX; x < width; x += grid) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }

    for (let y = startY; y < height; y += grid) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }
}

function drawArrow(a, b) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy);

    if (len < 15) return;

    const ux = dx / len;
    const uy = dy / len;

    const size = 5;

    const x = b.x - ux * 7;
    const y = b.y - uy * 7;

    ctx.fillStyle = "#526783";
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(
        x - ux * size - uy * size * 0.6,
        y - uy * size + ux * size * 0.6
    );
    ctx.lineTo(
        x - ux * size + uy * size * 0.6,
        y - uy * size - ux * size * 0.6
    );
    ctx.closePath();
    ctx.fill();
}

function segmentVisible(x1, y1, x2, y2) {
    const minX = Math.min(x1, x2);
    const maxX = Math.max(x1, x2);
    const minY = Math.min(y1, y2);
    const maxY = Math.max(y1, y2);

    return !(
        maxX < -50 ||
        minX > width + 50 ||
        maxY < -50 ||
        minY > height + 50
    );
}


/* ============================================================
   HIT TEST
   ============================================================ */

function pickNode(screenX, screenY) {
    const world = screenToWorld(screenX, screenY);

    let best = null;
    let bestDist = Infinity;

    for (const n of nodes) {
        if (!visible.has(n.id)) continue;

        const dx = n.x - world.x;
        const dy = n.y - world.y;

        const distance = Math.hypot(dx, dy);
        const hitRadius = Math.max(10 / camera.scale, n.r + 5);

        if (distance <= hitRadius && distance < bestDist) {
            best = n;
            bestDist = distance;
        }
    }

    return best;
}


/* ============================================================
   MOUSE / TOUCH
   ============================================================ */

function bindInteractions() {
    canvas.addEventListener("mousedown", e => {
        dragging = true;
        moved = false;
        dragLastX = e.clientX;
        dragLastY = e.clientY;
        canvas.classList.add("dragging");
    });

    window.addEventListener("mousemove", e => {
        if (!dragging) return;

        const dx = e.clientX - dragLastX;
        const dy = e.clientY - dragLastY;

        if (Math.abs(dx) + Math.abs(dy) > 2) {
            moved = true;
        }

        camera.x -= dx / camera.scale;
        camera.y -= dy / camera.scale;

        dragLastX = e.clientX;
        dragLastY = e.clientY;

        draw();
    });

    window.addEventListener("mouseup", e => {
        if (!dragging) return;

        dragging = false;
        canvas.classList.remove("dragging");

        if (!moved) {
            const rect = canvas.getBoundingClientRect();
            const node = pickNode(
                e.clientX - rect.left,
                e.clientY - rect.top
            );

            if (node) {
                showDetails(node.id);
            } else {
                selectedId = null;
                draw();
            }
        }
    });

    canvas.addEventListener("dblclick", e => {
        const rect = canvas.getBoundingClientRect();

        const node = pickNode(
            e.clientX - rect.left,
            e.clientY - rect.top
        );

        if (node) {
            focusNode(node.id);

            // Automatically show its neighborhood.
            visible.add(node.id);

            for (const neighbor of adjacency.get(node.id) || []) {
                visible.add(neighbor);
            }

            fitView();
            updateStats();
        }
    });

    canvas.addEventListener("wheel", e => {
        e.preventDefault();

        const rect = canvas.getBoundingClientRect();

        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const before = screenToWorld(mouseX, mouseY);

        const factor = e.deltaY < 0 ? 1.16 : 0.86;

        camera.scale = Math.max(
            0.08,
            Math.min(8, camera.scale * factor)
        );

        const after = screenToWorld(mouseX, mouseY);

        camera.x += before.x - after.x;
        camera.y += before.y - after.y;

        draw();
    }, {passive: false});
}


/* ============================================================
   VIEW CONTROLS
   ============================================================ */

function fitView() {
    const selected = nodes.filter(n => visible.has(n.id));

    if (!selected.length) return;

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    for (const n of selected) {
        minX = Math.min(minX, n.x);
        minY = Math.min(minY, n.y);
        maxX = Math.max(maxX, n.x);
        maxY = Math.max(maxY, n.y);
    }

    const graphWidth = Math.max(100, maxX - minX);
    const graphHeight = Math.max(100, maxY - minY);

    const usableW = Math.max(200, width - 100);
    const usableH = Math.max(200, height - 100);

    camera.scale = Math.max(
        0.08,
        Math.min(
            3,
            Math.min(usableW / graphWidth, usableH / graphHeight)
        )
    );

    camera.x = (minX + maxX) / 2;
    camera.y = (minY + maxY) / 2;

    draw();
}

function resetView() {
    camera.x = 0;
    camera.y = 0;
    camera.scale = 1;
    draw();
}

function zoomIn() {
    camera.scale = Math.min(8, camera.scale * 1.3);
    draw();
}

function zoomOut() {
    camera.scale = Math.max(0.08, camera.scale / 1.3);
    draw();
}


/* ============================================================
   STATISTICS
   ============================================================ */

function updateStats() {
    const visibleEdges = edges.filter(
        e => visible.has(e.source) && visible.has(e.target)
    ).length;

    document.getElementById("totalNodes").textContent =
        nodes.length.toLocaleString();

    document.getElementById("totalEdges").textContent =
        edges.length.toLocaleString();

    document.getElementById("visibleNodes").textContent =
        visible.size.toLocaleString();

    document.getElementById("visibleEdges").textContent =
        visibleEdges.toLocaleString();

    document.getElementById("components").textContent =
        components.length.toLocaleString();
}


/* ============================================================
   UTILITIES
   ============================================================ */

function setStatus(text) {
    document.getElementById("status").textContent = text;
}

function truncate(text, scale) {
    if (scale > 1.3) return text;
    if (text.length <= 26) return text;
    return text.slice(0, 23) + "...";
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

document.getElementById("labels").addEventListener("change", draw);
document.getElementById("edges").addEventListener("change", draw);

document.addEventListener("keydown", e => {
    if (e.key === "/" && document.activeElement?.tagName !== "INPUT") {
        e.preventDefault();
        document.getElementById("search").focus();
    }

    if (e.key === "Escape") {
        document.getElementById("search").value = "";
        document.getElementById("searchResults").innerHTML = "";
    }

    if (e.key === "+" || e.key === "=") zoomIn();
    if (e.key === "-") zoomOut();
});


/* ============================================================
   BOOT
   ============================================================ */

start();
</script>
</body>
</html>
"""


# ============================================================
# HTTP SERVER
# ============================================================

class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class GraphHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def send_bytes(self, data: bytes, content_type: str, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            self.send_bytes(
                HTML.encode("utf-8"),
                "text/html; charset=utf-8"
            )
            return

        if path == "/graph":
            self.send_bytes(
                json.dumps(
                    GRAPH,
                    ensure_ascii=False,
                    separators=(",", ":")
                ).encode("utf-8"),
                "application/json; charset=utf-8"
            )
            return

        if path == "/health":
            self.send_bytes(
                b'{"status":"ok"}',
                "application/json"
            )
            return

        self.send_bytes(
            b"Not Found",
            "text/plain; charset=utf-8",
            404
        )


# ============================================================
# PORT HANDLING
# ============================================================

def find_free_port(host: str, start_port: int, attempts: int = 30):
    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            try:
                s.bind((host, port))
                return port
            except OSError:
                continue

    raise RuntimeError(
        f"Could not find a free port in {start_port}-{start_port + attempts - 1}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    port = find_free_port(HOST, START_PORT)

    if port != START_PORT:
        print(
            f"Port {START_PORT} is busy. "
            f"Using free port {port} instead."
        )

    server = ReusableThreadingHTTPServer(
        (HOST, port),
        GraphHandler
    )

    url = f"http://{HOST}:{port}"

    print("=" * 64)
    print("ULTRON GRAPH VIEWER READY")
    print("=" * 64)
    print(f"Open: {url}")
    print()
    print("If the browser was already open, refresh the page.")
    print("Press Ctrl+C to stop.")
    print("=" * 64)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping graph viewer...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
