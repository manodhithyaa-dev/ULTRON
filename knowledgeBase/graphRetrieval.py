#!/usr/bin/env python3
"""
LLM Wiki - Relevance-Guided Branch-and-Bound Graph Retrieval

Works with the wikiBuilder.py layout:

wiki/
├── chunks/
│   └── chunk_*.md
├── entities/
│   └── *.md
├── graph/
│   └── graph.json
└── index/
    ├── embeddings.json
    └── metadata.json

Retrieval strategy
------------------
1. Embed the user query with qwen3-embedding:8b.
2. Score graph entities using the embeddings of their source chunks.
3. Select promising starting entities.
4. Traverse the entity graph using a priority queue.
5. Use llama3.2:3b as a cheap relevance heuristic for candidate nodes.
6. Maintain the best relevance found so far (alpha).
7. Estimate an optimistic branch upper bound (UB).
8. Prune branches whose UB cannot beat the current retrieval frontier.
9. Resolve winning entities back to their source chunks.
10. Return ranked chunks with graph paths and evidence.

Important:
- The graph traversal itself is deterministic.
- llama3.2:3b only scores relevance.
- The UB is intentionally configurable because an LLM score is a heuristic,
  not a mathematically guaranteed bound.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import requests


# ============================================================
# CONFIGURATION
# ============================================================

WIKI_DIR = Path("wiki")
GRAPH_FILE = WIKI_DIR / "graph" / "graph.json"
EMBEDDINGS_FILE = WIKI_DIR / "index" / "embeddings.json"
METADATA_FILE = WIKI_DIR / "index" / "metadata.json"
CHUNKS_DIR = WIKI_DIR / "chunks"

OLLAMA_URL = "http://localhost:11434"

# User requested fast retrieval model.
LLM_MODEL = "llama3.2:3b"

# Must match the embedding model used by wikiBuilder.py.
EMBED_MODEL = "qwen3-embedding:8b"

REQUEST_TIMEOUT = 300

# Ollama defaults to a 2048 token context window and
# silently truncates longer prompts.
NUM_CTX = 16384

# ------------------------------------------------------------
# Retrieval controls
# ------------------------------------------------------------

DEFAULT_TOP_K = 8

# Number of graph entities used as initial roots.
START_NODES = 8

# Maximum graph traversal depth.
MAX_DEPTH = 4

# Maximum number of children expanded from one node.
MAX_NEIGHBORS = 12

# Maximum number of graph nodes actually expanded.
MAX_EXPANSIONS = 120

# Keep only this many promising frontier branches.
BEAM_WIDTH = 20

# LLM calls are expensive compared with cosine similarity.
# Only LLM-score candidates whose semantic score reaches this level.
LLM_GATE = 0.20

# Number of candidates scored by the LLM at each expansion.
LLM_BATCH_SIZE = 6

# Relevance weights.
SEMANTIC_WEIGHT = 0.55
LLM_WEIGHT = 0.35
STRUCTURAL_WEIGHT = 0.10

# Branch-and-bound upper-bound heuristic.
#
# UB(node) = current_score + MAX_GAIN_PER_HOP * remaining_depth
#
# This is deliberately conservative-ish rather than claiming to be a
# mathematically guaranteed bound. Increase it for better recall and less
# pruning; decrease it for more aggressive pruning.
MAX_GAIN_PER_HOP = 0.18

# Small epsilon for floating point comparisons.
EPSILON = 1e-9

# Don't return duplicate chunks.
MAX_CHUNKS_PER_ENTITY = 4


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Node:
    id: str
    name: str
    node_type: str
    description: str
    aliases: list[str]
    sources: list[str]


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    description: str
    source_chunk: str


@dataclass
class Candidate:
    node_id: str
    name: str
    depth: int
    semantic_score: float
    llm_score: float
    structural_score: float
    relevance: float
    upper_bound: float
    parent_id: Optional[str]
    relation: Optional[str]
    path: list[str]


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    entity_score: float
    document_id: str
    heading: str
    chunk_file: str
    word_count: int
    path: list[str]
    entity: str


# ============================================================
# LOGGING
# ============================================================

def log(message: str) -> None:
    print(f"[RETRIEVAL] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


# ============================================================
# OLLAMA
# ============================================================

class Ollama:
    def __init__(self, base_url: str = OLLAMA_URL):
        self.base_url = base_url.rstrip("/")

    def check(self) -> set[str]:
        response = requests.get(
            f"{self.base_url}/api/tags",
            timeout=10,
        )
        response.raise_for_status()

        models = response.json().get("models", [])
        names = {m.get("name") for m in models}

        required = {LLM_MODEL, EMBED_MODEL}
        missing = required - names

        if missing:
            raise RuntimeError(
                "Missing Ollama models: " + ", ".join(sorted(missing))
            )

        return names

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": EMBED_MODEL,
                "input": text,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings")

        if not embeddings:
            raise RuntimeError("Ollama returned no embedding.")

        return embeddings[0]

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_ctx": NUM_CTX,
                },
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        return response.json().get("response", "").strip()


# ============================================================
# JSON / TEXT
# ============================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def extract_json(text: str) -> Any:
    """
    Robustly extract JSON from an LLM response.
    """

    text = text.strip()

    # Direct JSON.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Remove code fences.
    cleaned = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            pass

    # Array.
    start = cleaned.find("[")
    end = cleaned.rfind("]")

    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            pass

    return None


# ============================================================
# VECTOR MATH
# ============================================================

def cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:

    if not a or not b:
        return 0.0

    n = min(len(a), len(b))

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for i in range(n):
        x = float(a[i])
        y = float(b[i])

        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    value = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

    # Cosine can theoretically have tiny numerical excursions.
    return max(-1.0, min(1.0, value))


def normalize_similarity(value: float) -> float:
    """
    Convert cosine similarity [-1, 1] to [0, 1].
    """

    return max(0.0, min(1.0, (value + 1.0) / 2.0))


# ============================================================
# GRAPH LOADING
# ============================================================

class WikiIndex:
    def __init__(self, wiki_dir: Path):
        self.wiki_dir = wiki_dir

        self.graph_file = wiki_dir / "graph" / "graph.json"
        self.embeddings_file = wiki_dir / "index" / "embeddings.json"
        self.metadata_file = wiki_dir / "index" / "metadata.json"
        self.chunks_dir = wiki_dir / "chunks"

        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

        self.outgoing: dict[str, list[Edge]] = {}
        self.incoming: dict[str, list[Edge]] = {}

        self.embeddings: dict[str, list[float]] = {}
        self.metadata: dict[str, dict[str, Any]] = {}

        # chunk_id -> entity IDs which cite the chunk
        self.chunk_to_entities: dict[str, list[str]] = {}

        # entity_id -> source chunk IDs
        self.entity_to_chunks: dict[str, list[str]] = {}

    def load(self) -> None:
        log("Loading graph...")

        graph = load_json(self.graph_file)

        for raw in graph.get("nodes", []):
            node = Node(
                id=str(raw.get("id", "")),
                name=str(raw.get("name", "")),
                node_type=str(raw.get("type", "other")),
                description=str(raw.get("description", "")),
                aliases=list(raw.get("aliases", []) or []),
                sources=list(raw.get("sources", []) or []),
            )

            if node.id:
                self.nodes[node.id] = node

        for raw in graph.get("edges", []):
            edge = Edge(
                source=str(raw.get("source", "")),
                target=str(raw.get("target", "")),
                relation=str(raw.get("relation", "related_to")),
                description=str(raw.get("description", "")),
                source_chunk=str(raw.get("source_chunk", "")),
            )

            if edge.source in self.nodes and edge.target in self.nodes:
                self.edges.append(edge)

                self.outgoing.setdefault(edge.source, []).append(edge)
                self.incoming.setdefault(edge.target, []).append(edge)

        log(
            f"Graph loaded: {len(self.nodes)} nodes, "
            f"{len(self.edges)} edges"
        )

        log("Loading embeddings...")

        self.embeddings = load_json(self.embeddings_file)
        self.metadata = load_json(self.metadata_file)

        log(
            f"Embeddings loaded: {len(self.embeddings)} chunks"
        )

        self._build_source_maps()

    def _build_source_maps(self) -> None:
        for node_id, node in self.nodes.items():
            valid_sources = []

            for chunk_id in node.sources:
                if chunk_id in self.embeddings:
                    valid_sources.append(chunk_id)

                    self.chunk_to_entities.setdefault(
                        chunk_id,
                        [],
                    ).append(node_id)

            self.entity_to_chunks[node_id] = valid_sources


# ============================================================
# ENTITY TEXT
# ============================================================

def node_text(node: Node) -> str:
    parts = [
        node.name,
        node.node_type,
        node.description,
    ]

    if node.aliases:
        parts.append(
            "Aliases: " + ", ".join(node.aliases)
        )

    return "\n".join(
        part for part in parts if part.strip()
    )


# ============================================================
# INITIAL SEMANTIC SEARCH
# ============================================================

def score_entity_semantically(
    node: Node,
    query_embedding: list[float],
    index: WikiIndex,
) -> float:

    best = 0.0

    # Entity descriptions are not embedded by the builder.
    # Therefore use the chunk embeddings referenced by the entity.
    for chunk_id in index.entity_to_chunks.get(node.id, []):
        embedding = index.embeddings.get(chunk_id)

        if embedding is None:
            continue

        cosine = cosine_similarity(
            query_embedding,
            embedding,
        )

        score = normalize_similarity(cosine)

        if score > best:
            best = score

    return best


def initial_candidates(
    query_embedding: list[float],
    index: WikiIndex,
    count: int,
) -> list[tuple[Node, float]]:

    scored = []

    for node in index.nodes.values():

        score = score_entity_semantically(
            node,
            query_embedding,
            index,
        )

        scored.append((node, score))

    scored.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored[:count]


# ============================================================
# STRUCTURAL SCORE
# ============================================================

def structural_score(
    node_id: str,
    depth: int,
    index: WikiIndex,
) -> float:

    degree = (
        len(index.outgoing.get(node_id, []))
        + len(index.incoming.get(node_id, []))
    )

    # A connected node is slightly more useful than an isolated node,
    # but structural information must not dominate semantic relevance.
    connectivity = min(1.0, math.log1p(degree) / math.log(11))

    # Small depth preference: shallow nodes are generally cheaper and
    # less likely to be noisy, while deeper nodes remain possible.
    depth_factor = 1.0 / (1.0 + 0.08 * depth)

    return connectivity * depth_factor


# ============================================================
# LLM RELEVANCE
# ============================================================

def llm_score_candidate(
    query: str,
    node: Node,
    semantic_score: float,
    parent: Optional[Node],
    relation: Optional[str],
    ollama: Ollama,
) -> tuple[float, str]:

    parent_text = ""

    if parent is not None:
        parent_text = (
            f"Parent concept: {parent.name}\n"
            f"Parent description: {parent.description}\n"
            f"Relation to candidate: {relation or 'unknown'}\n"
        )

    prompt = f"""
You are a fast relevance judge for a technical knowledge graph.

USER QUERY:
{query}

CANDIDATE NODE:
Name: {node.name}
Type: {node.node_type}
Description: {node.description}
Aliases: {", ".join(node.aliases) if node.aliases else "none"}

{parent_text}

Embedding similarity: {semantic_score:.4f}

Judge how relevant this candidate is to answering the query.

Scoring:
0.0 = completely irrelevant
0.25 = weakly related
0.50 = moderately relevant
0.75 = strongly relevant
1.00 = directly relevant / central

Do not reward a candidate merely because it is connected.
Judge its actual usefulness for answering the query.

Return ONLY valid JSON:

{{
  "score": 0.0,
  "reason": "short reason"
}}
"""

    try:
        raw = ollama.generate(prompt)
        data = extract_json(raw)

        if isinstance(data, dict):
            score = float(data.get("score", 0.0))
            reason = str(data.get("reason", ""))

            score = max(0.0, min(1.0, score))

            return score, reason

    except Exception as exc:
        warn(
            f"LLM scoring failed for {node.name}: {exc}"
        )

    # Safe fallback: semantic score.
    return semantic_score, "LLM fallback to semantic score"


# ============================================================
# BRANCH-AND-BOUND
# ============================================================

def calculate_relevance(
    semantic: float,
    llm: float,
    structural: float,
) -> float:

    return (
        SEMANTIC_WEIGHT * semantic
        + LLM_WEIGHT * llm
        + STRUCTURAL_WEIGHT * structural
    )


def calculate_upper_bound(
    current_relevance: float,
    depth: int,
) -> float:

    remaining_depth = max(
        0,
        MAX_DEPTH - depth,
    )

    upper_bound = (
        current_relevance
        + MAX_GAIN_PER_HOP * remaining_depth
    )

    return min(1.0, upper_bound)


def branch_and_bound(
    query: str,
    query_embedding: list[float],
    index: WikiIndex,
    ollama: Ollama,
) -> tuple[list[Candidate], dict[str, Any]]:

    # --------------------------------------------------------
    # Step 1: initial semantic roots
    # --------------------------------------------------------

    roots = initial_candidates(
        query_embedding,
        index,
        START_NODES,
    )

    if not roots:
        return [], {
            "expanded": 0,
            "pruned": 0,
            "llm_calls": 0,
        }

    log("Initial semantic roots:")

    for node, score in roots:
        log(
            f"  {node.name:<35} "
            f"semantic={score:.3f}"
        )

    # --------------------------------------------------------
    # Priority queue
    #
    # Python heapq is min-first, so use negative UB.
    # --------------------------------------------------------

    frontier = []

    sequence = 0

    visited: set[str] = set()

    best_candidates: dict[str, Candidate] = {}

    # Best exact relevance discovered.
    alpha = 0.0

    expanded = 0
    pruned = 0
    llm_calls = 0

    # Cache semantic scores because they can be reused.
    semantic_cache: dict[str, float] = {}

    # Cache LLM scores by query/node/parent relation.
    llm_cache: dict[
        tuple[str, str, str, str],
        tuple[float, str],
    ] = {}

    # --------------------------------------------------------
    # Seed roots.
    # --------------------------------------------------------

    for node, semantic in roots:

        semantic_cache[node.id] = semantic

        structural = structural_score(
            node.id,
            0,
            index,
        )

        # Roots don't need an LLM call immediately.
        # Their semantic score determines initial ordering.
        llm = semantic

        relevance = calculate_relevance(
            semantic,
            llm,
            structural,
        )

        ub = calculate_upper_bound(
            relevance,
            0,
        )

        candidate = Candidate(
            node_id=node.id,
            name=node.name,
            depth=0,
            semantic_score=semantic,
            llm_score=llm,
            structural_score=structural,
            relevance=relevance,
            upper_bound=ub,
            parent_id=None,
            relation=None,
            path=[node.name],
        )

        best_candidates[node.id] = candidate

        sequence += 1

        heapq.heappush(
            frontier,
            (
                -ub,
                -relevance,
                sequence,
                node.id,
            ),
        )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    while frontier and expanded < MAX_EXPANSIONS:

        (
            negative_ub,
            negative_score,
            _,
            node_id,
        ) = heapq.heappop(frontier)

        ub = -negative_ub

        if node_id in visited:
            continue

        # ----------------------------------------------------
        # Branch-and-bound pruning.
        # ----------------------------------------------------

        if ub + EPSILON < alpha:

            pruned += 1

            node = index.nodes.get(node_id)

            if node:
                log(
                    f"PRUNE {node.name} "
                    f"UB={ub:.3f} < alpha={alpha:.3f}"
                )

            continue

        node = index.nodes.get(node_id)

        if node is None:
            continue

        visited.add(node_id)
        expanded += 1

        candidate = best_candidates[node_id]

        # ----------------------------------------------------
        # Update alpha.
        # ----------------------------------------------------

        if candidate.relevance > alpha:
            alpha = candidate.relevance

        log(
            f"EXPAND {expanded:03d} | "
            f"depth={candidate.depth} | "
            f"score={candidate.relevance:.3f} | "
            f"UB={candidate.upper_bound:.3f} | "
            f"{node.name}"
        )

        # ----------------------------------------------------
        # Depth limit.
        # ----------------------------------------------------

        if candidate.depth >= MAX_DEPTH:
            continue

        # ----------------------------------------------------
        # Collect graph neighbors.
        # ----------------------------------------------------

        neighbors = index.outgoing.get(
            node_id,
            [],
        )

        # Include reverse edges too.
        reverse_edges = index.incoming.get(
            node_id,
            [],
        )

        neighbor_specs = []

        for edge in neighbors:
            neighbor_specs.append(
                (
                    edge.target,
                    edge.relation,
                    edge.description,
                )
            )

        for edge in reverse_edges:
            neighbor_specs.append(
                (
                    edge.source,
                    f"reverse:{edge.relation}",
                    edge.description,
                )
            )

        # Deduplicate neighbors.
        unique_neighbors = {}

        for target_id, relation, description in neighbor_specs:

            if target_id == node_id:
                continue

            if target_id in visited:
                continue

            unique_neighbors[target_id] = (
                relation,
                description,
            )

        # ----------------------------------------------------
        # Cheap semantic pre-ranking.
        # ----------------------------------------------------

        cheap_candidates = []

        for target_id, (
            relation,
            edge_description,
        ) in unique_neighbors.items():

            target = index.nodes.get(target_id)

            if target is None:
                continue

            semantic = semantic_cache.get(
                target_id
            )

            if semantic is None:
                semantic = score_entity_semantically(
                    target,
                    query_embedding,
                    index,
                )
                semantic_cache[target_id] = semantic

            if semantic < LLM_GATE:
                # Still allow a potentially useful connected node,
                # but don't spend an LLM call on it.
                llm = semantic
                structural = structural_score(
                    target_id,
                    candidate.depth + 1,
                    index,
                )

                relevance = calculate_relevance(
                    semantic,
                    llm,
                    structural,
                )

                cheap_candidates.append(
                    (
                        target_id,
                        relation,
                        edge_description,
                        semantic,
                        llm,
                        structural,
                        relevance,
                        False,
                    )
                )

                continue

            cheap_candidates.append(
                (
                    target_id,
                    relation,
                    edge_description,
                    semantic,
                    None,
                    None,
                    None,
                    True,
                )
            )

        # Sort by semantic similarity first.
        cheap_candidates.sort(
            key=lambda x: x[3],
            reverse=True,
        )

        # Prevent massive branching.
        cheap_candidates = cheap_candidates[
            :MAX_NEIGHBORS
        ]

        # ----------------------------------------------------
        # LLM reranking only for promising candidates.
        # ----------------------------------------------------

        scored_children = []

        llm_candidates = [
            item
            for item in cheap_candidates
            if item[7]
        ]

        llm_candidates = llm_candidates[
            :LLM_BATCH_SIZE
        ]

        llm_target_ids = {
            item[0]
            for item in llm_candidates
        }

        for item in cheap_candidates:

            (
                target_id,
                relation,
                edge_description,
                semantic,
                llm,
                structural,
                relevance,
                needs_llm,
            ) = item

            target = index.nodes[target_id]

            if needs_llm:

                cache_key = (
                    query,
                    target_id,
                    node_id,
                    relation,
                )

                cached = llm_cache.get(cache_key)

                if cached is not None:
                    llm, _reason = cached
                elif target_id in llm_target_ids:
                    llm, reason = llm_score_candidate(
                        query=query,
                        node=target,
                        semantic_score=semantic,
                        parent=node,
                        relation=relation,
                        ollama=ollama,
                    )

                    llm_cache[cache_key] = (
                        llm,
                        reason,
                    )

                    llm_calls += 1
                else:
                    llm = semantic

            if structural is None:
                structural = structural_score(
                    target_id,
                    candidate.depth + 1,
                    index,
                )

            if relevance is None:
                relevance = calculate_relevance(
                    semantic,
                    llm,
                    structural,
                )

            scored_children.append(
                (
                    target_id,
                    relation,
                    edge_description,
                    semantic,
                    llm,
                    structural,
                    relevance,
                )
            )

        # ----------------------------------------------------
        # Create child candidates.
        # ----------------------------------------------------

        for (
            target_id,
            relation,
            edge_description,
            semantic,
            llm,
            structural,
            relevance,
        ) in scored_children:

            target = index.nodes[target_id]

            child_depth = candidate.depth + 1

            ub = calculate_upper_bound(
                relevance,
                child_depth,
            )

            # Construct path.
            child_path = candidate.path + [
                f"--{relation}-->",
                target.name,
            ]

            child = Candidate(
                node_id=target_id,
                name=target.name,
                depth=child_depth,
                semantic_score=semantic,
                llm_score=llm,
                structural_score=structural,
                relevance=relevance,
                upper_bound=ub,
                parent_id=node_id,
                relation=relation,
                path=child_path,
            )

            # Keep only the best path to a node.
            previous = best_candidates.get(
                target_id
            )

            if (
                previous is not None
                and previous.relevance >= relevance
            ):
                continue

            best_candidates[target_id] = child

            # Update alpha with exact candidate relevance.
            if relevance > alpha:
                alpha = relevance

            # ------------------------------------------------
            # Branch-and-bound check BEFORE adding to frontier.
            # ------------------------------------------------

            if ub + EPSILON < alpha:
                pruned += 1

                log(
                    f"PRUNE child {target.name} "
                    f"UB={ub:.3f} < alpha={alpha:.3f}"
                )

                continue

            sequence += 1

            heapq.heappush(
                frontier,
                (
                    -ub,
                    -relevance,
                    sequence,
                    target_id,
                ),
            )

        # ----------------------------------------------------
        # Beam control.
        # ----------------------------------------------------

        if len(frontier) > BEAM_WIDTH:

            frontier.sort()

            frontier = frontier[
                :BEAM_WIDTH
            ]

            heapq.heapify(frontier)

    # --------------------------------------------------------
    # Return candidates.
    # --------------------------------------------------------

    results = sorted(
        best_candidates.values(),
        key=lambda x: (
            x.relevance,
            x.semantic_score,
            -x.depth,
        ),
        reverse=True,
    )

    stats = {
        "expanded": expanded,
        "pruned": pruned,
        "llm_calls": llm_calls,
        "visited": len(visited),
        "candidate_nodes": len(best_candidates),
        "alpha": alpha,
        "max_depth": MAX_DEPTH,
        "beam_width": BEAM_WIDTH,
    }

    return results, stats


# ============================================================
# RESOLVE ENTITIES -> CHUNKS
# ============================================================

def resolve_chunks(
    candidates: list[Candidate],
    index: WikiIndex,
    top_k: int,
) -> list[RetrievedChunk]:

    chunk_results: dict[str, RetrievedChunk] = {}

    for candidate in candidates:

        node_id = candidate.node_id

        chunk_ids = index.entity_to_chunks.get(
            node_id,
            [],
        )

        # Limit chunks per entity.
        chunk_ids = chunk_ids[
            :MAX_CHUNKS_PER_ENTITY
        ]

        for chunk_id in chunk_ids:

            metadata = index.metadata.get(
                chunk_id,
                {},
            )

            if not metadata:
                continue

            # Entity relevance is the main score.
            # Add a tiny chunk-specific semantic component
            # already represented by the entity candidate.
            score = candidate.relevance

            existing = chunk_results.get(
                chunk_id
            )

            result = RetrievedChunk(
                chunk_id=chunk_id,
                score=score,
                entity_score=candidate.relevance,
                document_id=str(
                    metadata.get(
                        "document_id",
                        "",
                    )
                ),
                heading=str(
                    metadata.get(
                        "heading",
                        "",
                    )
                ),
                chunk_file=str(
                    metadata.get(
                        "chunk_file",
                        "",
                    )
                ),
                word_count=int(
                    metadata.get(
                        "word_count",
                        0,
                    )
                ),
                path=candidate.path,
                entity=candidate.name,
            )

            if (
                existing is None
                or result.score > existing.score
            ):
                chunk_results[chunk_id] = result

    results = sorted(
        chunk_results.values(),
        key=lambda x: x.score,
        reverse=True,
    )

    return results[:top_k]


# ============================================================
# LOAD CHUNK CONTENT
# ============================================================

def load_chunk_content(
    result: RetrievedChunk,
    index: WikiIndex,
) -> str:

    path = index.chunks_dir / (
        f"{result.chunk_id}.md"
    )

    if not path.exists():
        return ""

    return read_text(path)


# ============================================================
# FINAL RESULT MARKDOWN
# ============================================================

def save_result_markdown(
    query: str,
    results: list[RetrievedChunk],
    index: WikiIndex,
    stats: dict[str, Any],
    output_path: Path,
) -> None:

    lines = [
        "# Graph Retrieval Result",
        "",
        f"**Query:** {query}",
        "",
        "## Retrieval Statistics",
        "",
        f"- Expanded nodes: {stats['expanded']}",
        f"- Pruned branches: {stats['pruned']}",
        f"- LLM calls: {stats['llm_calls']}",
        f"- Candidate nodes: {stats['candidate_nodes']}",
        f"- Final alpha: {stats['alpha']:.4f}",
        "",
        "## Retrieved Chunks",
        "",
    ]

    for i, result in enumerate(results, 1):

        lines.extend(
            [
                f"### {i}. {result.heading}",
                "",
                f"- Chunk: `{result.chunk_id}`",
                f"- Entity: **{result.entity}**",
                f"- Score: `{result.score:.4f}`",
                f"- Document: `{result.document_id}`",
                "",
                "#### Graph path",
                "",
                "```text",
                " ".join(result.path),
                "```",
                "",
                "#### Content",
                "",
                load_chunk_content(
                    result,
                    index,
                ),
                "",
                "---",
                "",
            ]
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# JSON RESULT
# ============================================================

def save_result_json(
    query: str,
    results: list[RetrievedChunk],
    stats: dict[str, Any],
    output_path: Path,
) -> None:

    payload = {
        "query": query,
        "algorithm": (
            "relevance-guided-"
            "branch-and-bound"
        ),
        "llm_model": LLM_MODEL,
        "embedding_model": EMBED_MODEL,
        "weights": {
            "semantic": SEMANTIC_WEIGHT,
            "llm": LLM_WEIGHT,
            "structural": STRUCTURAL_WEIGHT,
        },
        "branch_and_bound": {
            "max_depth": MAX_DEPTH,
            "beam_width": BEAM_WIDTH,
            "max_expansions": MAX_EXPANSIONS,
            "max_gain_per_hop": MAX_GAIN_PER_HOP,
        },
        "statistics": stats,
        "results": [
            asdict(result)
            for result in results
        ],
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# DISPLAY
# ============================================================

def print_results(
    query: str,
    results: list[RetrievedChunk],
    stats: dict[str, Any],
    index: WikiIndex,
) -> None:

    print()
    print("=" * 76)
    print("              GRAPH BRANCH-AND-BOUND RETRIEVAL")
    print("=" * 76)
    print()

    print(f"Query: {query}")
    print()

    print("Search statistics:")
    print(f"  Expanded nodes : {stats['expanded']}")
    print(f"  Pruned branches: {stats['pruned']}")
    print(f"  LLM calls      : {stats['llm_calls']}")
    print(f"  Candidate nodes: {stats['candidate_nodes']}")
    print(f"  Alpha          : {stats['alpha']:.4f}")
    print()

    if not results:
        print("No relevant chunks found.")
        return

    print("-" * 76)

    for i, result in enumerate(results, 1):

        print()
        print(
            f"[{i}] {result.heading}"
        )

        print(
            f"    score    : {result.score:.4f}"
        )

        print(
            f"    entity   : {result.entity}"
        )

        print(
            f"    chunk    : {result.chunk_id}"
        )

        print(
            f"    document : {result.document_id}"
        )

        print(
            "    path     : "
            + " ".join(result.path)
        )

        content = load_chunk_content(
            result,
            index,
        )

        # Display a compact preview.
        preview = re.sub(
            r"\s+",
            " ",
            content,
        ).strip()

        if len(preview) > 350:
            preview = preview[:350] + "..."

        print(
            f"    preview  : {preview}"
        )

        print("-" * 76)


# ============================================================
# RETRIEVAL PIPELINE
# ============================================================

def retrieve(
    query: str,
    wiki_dir: Path,
    top_k: int,
) -> tuple[list[RetrievedChunk], dict[str, Any]]:

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    ollama = Ollama()

    log("Checking Ollama...")
    ollama.check()

    index = WikiIndex(wiki_dir)
    index.load()

    log(
        f"Embedding query with {EMBED_MODEL}..."
    )

    query_embedding = ollama.embed(
        query.strip()
    )

    log(
        "Starting relevance-guided "
        "branch-and-bound..."
    )

    candidates, stats = branch_and_bound(
        query=query.strip(),
        query_embedding=query_embedding,
        index=index,
        ollama=ollama,
    )

    results = resolve_chunks(
        candidates=candidates,
        index=index,
        top_k=top_k,
    )

    # Save artifacts beside the wiki.
    output_json = wiki_dir / "retrieval_result.json"
    output_md = wiki_dir / "retrieval_result.md"

    save_result_json(
        query=query,
        results=results,
        stats=stats,
        output_path=output_json,
    )

    save_result_markdown(
        query=query,
        results=results,
        index=index,
        stats=stats,
        output_path=output_md,
    )

    print_results(
        query=query,
        results=results,
        stats=stats,
        index=index,
    )

    log(
        f"Saved: {output_json}"
    )

    log(
        f"Saved: {output_md}"
    )

    return results, stats


# ============================================================
# CLI
# ============================================================

def main() -> None:

    global MAX_DEPTH
    global BEAM_WIDTH
    global MAX_EXPANSIONS
    global START_NODES
    global MAX_GAIN_PER_HOP

    parser = argparse.ArgumentParser(
        description=(
            "LLM Wiki graph retrieval using "
            "relevance-guided branch-and-bound."
        )
    )

    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        help="Knowledge query",
    )

    parser.add_argument(
        "--wiki",
        type=str,
        default="wiki",
        help="Wiki directory",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to return",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=MAX_DEPTH,
        help="Maximum graph traversal depth",
    )

    parser.add_argument(
        "--beam-width",
        type=int,
        default=BEAM_WIDTH,
        help="Maximum frontier size",
    )

    parser.add_argument(
        "--max-expansions",
        type=int,
        default=MAX_EXPANSIONS,
        help="Maximum number of expanded graph nodes",
    )

    parser.add_argument(
        "--start-nodes",
        type=int,
        default=START_NODES,
        help="Number of semantic starting nodes",
    )

    parser.add_argument(
        "--max-gain-per-hop",
        type=float,
        default=MAX_GAIN_PER_HOP,
        help=(
            "Upper-bound gain allowed per remaining graph hop"
        ),
    )

    args = parser.parse_args()

    MAX_DEPTH = max(0, args.max_depth)
    BEAM_WIDTH = max(1, args.beam_width)
    MAX_EXPANSIONS = max(1, args.max_expansions)
    START_NODES = max(1, args.start_nodes)
    MAX_GAIN_PER_HOP = max(
        0.0,
        min(1.0, args.max_gain_per_hop),
    )

    query = args.query

    if not query:
        print()
        query = input(
            "Enter your knowledge query: "
        ).strip()

    if not query:
        print("Empty query.")
        sys.exit(1)

    start = time.time()

    try:

        retrieve(
            query=query,
            wiki_dir=Path(args.wiki),
            top_k=max(1, args.top_k),
        )

    except KeyboardInterrupt:
        print()
        warn("Retrieval interrupted.")
        sys.exit(130)

    except Exception as exc:
        print()
        warn(str(exc))
        sys.exit(1)

    elapsed = time.time() - start

    print()
    print(
        f"Completed in {elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
