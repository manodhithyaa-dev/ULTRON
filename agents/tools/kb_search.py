"""
KNOWLEDGE BASE TOOL
============================================================

Exposes the offline retrieval stack in knowledgeBase/ as an
agent-callable tool.

This reuses graphAnswer.py's pipeline but deliberately stops
short of answer generation:

    embed query
      -> branch_and_bound   graph traversal
      -> fuse_chunk_candidates  hybrid semantic + lexical + graph
      -> rerank_chunks      LLM rerank of the shortlist
      -> (return chunks)

graphAnswer.generate_answer and validate_answer are skipped.
ULTRON's agents are the answerer, so running a second
generator would produce a competing answer and waste the
agent's context window.

graphRetrieval.retrieve is never called because it writes
retrieval_result.json and .md into the wiki directory on
every invocation.

Two constraints shape the implementation:

- The index must be a singleton. WikiIndex.load parses a
  1.37MB graph plus a 118MB embeddings file; doing that per
  request would dominate response time.

- Output uses [KB n] citation labels, distinct from the
  [WEB n] labels used by the web search tool, because
  graphAnswer.build_context also labels passages "SOURCE i".
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import threading

from pathlib import Path
from typing import Any, Optional

from config import (
    KNOWLEDGE_BASE_DIR,
    KB_TOP_K,
    KB_SKIP_RERANK,
    KB_RERANK_K,
    KB_TIMEOUT,
    MODELS,
    OLLAMA_BASE_URL,
)

from tools.base import (
    ToolResult,
    ToolError,
)


logger = logging.getLogger(
    "ULTRON.tools.kb_search"
)


# ============================================================
# FRONTMATTER
# ============================================================

_FRONTMATTER = re.compile(
    r"\A---\s*\n.*?\n---\s*\n",
    re.DOTALL,
)


def strip_frontmatter(text: str) -> str:
    """
    Remove the YAML frontmatter from a chunk file.

    Every chunk in the wiki starts with id/document/heading
    frontmatter. The knowledge base pipeline injects that
    verbatim into its prompts, but it is metadata the agent
    already receives through the source list, so it is pure
    token waste here.
    """

    if not text:

        return ""

    return _FRONTMATTER.sub(
        "", text, count=1
    ).strip()


# ============================================================
# DOCUMENT LABELS
# ============================================================
#
# RetrievedChunk.document_id is an opaque hash like
# "doc_604e4dcb676808b0". The wiki's document pages carry a
# readable label in their frontmatter, typically the arXiv id
# ("2112.08782V2") and the originating PDF filename, so those
# are resolved once and cached to keep citations legible.
#
# Parsed with regex rather than a YAML library because some
# titles contain commas that break naive block mappings.

_DOC_FRONTMATTER = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)

_doc_labels: dict[str, dict[str, str]] = {}
_doc_lock = threading.Lock()


def _frontmatter_field(
    meta: str,
    key: str,
) -> str:
    match = re.search(
        rf"^{key}:\s*(.+)$",
        meta,
        re.M,
    )

    return match.group(1).strip() if match else ""


def get_doc_labels() -> dict[str, dict[str, str]]:
    """
    Map document ids to human-readable labels.
    """

    global _doc_labels

    if _doc_labels:

        return _doc_labels

    with _doc_lock:

        if _doc_labels:

            return _doc_labels

        labels: dict[str, dict[str, str]] = {}

        documents_dir = _KB_DIR / "documents"

        try:

            pages = sorted(
                documents_dir.glob("*.md")
            )

        except OSError:

            pages = []

        for page in pages:

            try:

                text = page.read_text(
                    encoding="utf-8",
                    errors="replace",
                )

            except OSError:

                continue

            match = _DOC_FRONTMATTER.match(text)

            if not match:

                continue

            meta = match.group(1)

            doc_id = _frontmatter_field(
                meta, "id"
            )

            if not doc_id:

                continue

            label = _frontmatter_field(
                meta, "title"
            )

            source_file = _frontmatter_field(
                meta, "source"
            )

            if not label or len(label) > 120:
                label = (
                    source_file
                    or doc_id
                )

            labels[doc_id] = {
                "label": label,
                "file": source_file,
            }

        _doc_labels = labels

        return _doc_labels


def document_label(
    document_id: str,
) -> str:
    """
    Return a readable label for a document id, falling back to
    the id itself when the wiki has no label for it.
    """

    return get_doc_labels().get(
        document_id,
        {},
    ).get("label") or document_id


# ============================================================
# GRAPH ANSWER MODULE LOADING
# ============================================================

_KB_DIR = Path(KNOWLEDGE_BASE_DIR)
_KB_PARENT = _KB_DIR.parent

_graph_answer: Any = None
_graph_load_lock = threading.Lock()


def _load_graph_answer():
    """
    Import graphAnswer once and register its directory on
    sys.path.

    graphAnswer resolves WIKI_DIR relative to the working
    directory, so the module has to be imported from within
    knowledgeBase/. Every function we call takes wiki_dir
    explicitly, which keeps us immune to that, but the import
    itself still needs the path.
    """

    global _graph_answer

    if _graph_answer is not None:

        return _graph_answer

    with _graph_load_lock:

        if _graph_answer is not None:

            return _graph_answer

        parent = str(_KB_PARENT)

        if parent not in sys.path:

            sys.path.insert(0, parent)

        try:

            import graphAnswer

        except ImportError as exc:

            raise ToolError(
                f"Could not import the knowledge "
                f"base module from {parent}: {exc}"
            ) from exc

        # graphAnswer logs to stdout unconditionally, which
        # would corrupt the FastAPI log stream.

        graphAnswer.log = (
            lambda message: None
        )

        _graph_answer = graphAnswer

        return _graph_answer


# ============================================================
# WIKI INDEX SINGLETON
# ============================================================

_wiki_index: Any = None
_index_lock = threading.Lock()
_index_error: Optional[str] = None


def get_wiki_index():
    """
    Return the shared WikiIndex, loading it on first use.

    The lock means concurrent first requests queue on one
    load rather than each parsing 118MB of embeddings.
    """

    global _wiki_index
    global _index_error

    if _wiki_index is not None:

        return _wiki_index

    with _index_lock:

        if _wiki_index is not None:

            return _wiki_index

        if _index_error:

            raise ToolError(
                _index_error
            )

        graph_answer = _load_graph_answer()

        try:

            index = graph_answer.WikiIndex(
                _KB_DIR
            )

            index.load()

        except FileNotFoundError as exc:

            _index_error = (
                f"Knowledge base is not built: {exc}. "
                f"Run wikiBuilder.py first."
            )

            raise ToolError(
                _index_error
            ) from exc

        except Exception as exc:

            _index_error = (
                f"Failed to load the knowledge base: "
                f"{type(exc).__name__}: {exc}"
            )

            raise ToolError(
                _index_error
            ) from exc

        logger.info(
            "Knowledge base index loaded: "
            "%d nodes, %d edges, %d embeddings",
            len(index.nodes),
            len(index.edges),
            len(index.embeddings),
        )

        _wiki_index = index

        return _wiki_index


def warm_index() -> bool:
    """
    Load the index ahead of the first request.

    Returns True on success. Used by the FastAPI lifespan so
    the first real query is not charged the load cost.
    """

    try:

        get_wiki_index()

        return True

    except ToolError as exc:

        logger.warning(
            "Knowledge base warmup failed: %s",
            exc,
        )

        return False


# ============================================================
# TOOL
# ============================================================

class KnowledgeBaseTool:
    """
    Semantic and graph search over the local knowledge base.
    """

    name = "kb_search"

    description = (
        "Search ULTRON's local knowledge base of "
        "reference documents. Uses both semantic similarity "
        "and knowledge-graph traversal, then reranks the "
        "results. Returns passages with their source "
        "document and heading."
    )

    def __init__(self):
        # Retrieval is CPU/IO heavy and blocks for seconds.

        self._lock = asyncio.Lock()

    async def run(
        self,
        query: str,
        top_k: int = KB_TOP_K,
        **_: Any,
    ) -> ToolResult:
        """
        Retrieve knowledge base passages for a query.
        """

        query = (query or "").strip()

        if not query:

            raise ToolError(
                "kb_search requires a query."
            )

        top_k = max(
            1,
            min(int(top_k), 10),
        )

        try:

            results, stats = (
                await asyncio.wait_for(
                    self._retrieve(
                        query,
                        top_k,
                    ),
                    timeout=KB_TIMEOUT,
                )
            )

        except asyncio.TimeoutError:

            return ToolResult(
                tool=self.name,
                query=query,
                ok=False,
                error=(
                    f"Knowledge base retrieval exceeded "
                    f"{KB_TIMEOUT:.0f}s and was abandoned."
                ),
            )

        except ToolError as exc:

            return ToolResult(
                tool=self.name,
                query=query,
                ok=False,
                error=str(exc),
            )

        except Exception as exc:

            return ToolResult(
                tool=self.name,
                query=query,
                ok=False,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        if not results:

            return ToolResult(
                tool=self.name,
                query=query,
                ok=False,
                error=(
                    "No knowledge base passages matched "
                    "this query."
                ),
            )

        return ToolResult(
            tool=self.name,
            query=query,
            ok=True,
            context=self._format(results),
            sources=[
                self._source(r, i)
                for i, r in enumerate(
                    results,
                    start=1,
                )
            ],
        )

    # --------------------------------------------------------
    # BLOCKING PIPELINE
    # --------------------------------------------------------

    async def _retrieve(
        self,
        query: str,
        top_k: int,
    ) -> tuple[list, dict]:
        """
        Run the retrieval pipeline off the event loop.
        """

        async with self._lock:

            return await asyncio.to_thread(
                self._retrieve_blocking,
                query,
                top_k,
            )

    def _retrieve_blocking(
        self,
        query: str,
        top_k: int,
    ) -> tuple[list, dict]:
        """
        Synchronous pipeline. Runs in a worker thread.

        graphAnswer.Ollama binds its base_url as a default
        argument evaluated at import time, so the ULTRON
        endpoint must be passed explicitly rather than by
        reassigning the module global.
        """

        ga = _load_graph_answer()

        index = get_wiki_index()

        ollama = ga.Ollama(
            base_url=OLLAMA_BASE_URL
        )

        ollama.check()

        query_embedding = ollama.embed(query)

        candidates, stats = ga.branch_and_bound(
            query=query,
            query_embedding=query_embedding,
            index=index,
            ollama=ollama,
            max_depth=ga.MAX_DEPTH,
            beam_width=ga.BEAM_WIDTH,
            max_expansions=ga.MAX_EXPANSIONS,
            start_nodes=ga.START_NODES,
            max_gain_per_hop=ga.MAX_GAIN_PER_HOP,
        )

        fused = ga.fuse_chunk_candidates(
            query=query,
            query_embedding=query_embedding,
            candidates=candidates,
            index=index,
            hybrid_k=ga.HYBRID_K,
            direct_k=ga.DIRECT_CHUNK_K,
        )

        stats["fused_candidates"] = len(fused)

        results = fused

        if (
            not KB_SKIP_RERANK
            and fused
        ):

            results, rerank_tokens = (
                ga.rerank_chunks(
                    query=query,
                    results=fused[:KB_RERANK_K],
                    index=index,
                    ollama=ollama,
                    final_k=top_k,
                    answer_model=MODELS[
                        "reranker"
                    ],
                )
            )

            stats["rerank_tokens"] = rerank_tokens

            for item in results:

                item.content = strip_frontmatter(
                    item.content
                )

        else:

            # rerank_chunks is what populates
            # RetrievedChunk.content, so when it is skipped
            # the chunk text has to be loaded explicitly.

            results = []

            for item in fused[:top_k]:

                content = strip_frontmatter(
                    index.chunk_text(item.chunk_id)
                )

                if not content:

                    continue

                item.content = content

                results.append(item)

        return results, stats

    # --------------------------------------------------------
    # FORMATTING
    # --------------------------------------------------------

    @staticmethod
    def _source(
        result,
        position: int,
    ) -> dict[str, Any]:
        """
        Citation metadata for a retrieved passage.

        title is the readable document label, location is the
        section inside it, and origin records which tool
        produced this source.
        """

        return {
            "id": f"KB{position}",
            "kind": "knowledge_base",
            "title": document_label(
                result.document_id
            ),
            "location": (
                result.heading
                or "(no heading)"
            ),
            "document_id": result.document_id,
            "score": round(
                result.score,
                3,
            ),
            "chunk_id": result.chunk_id,
        }

    def _format(
        self,
        results: list,
    ) -> str:
        """
        Render retrieved chunks as prompt-ready evidence.
        """

        ga = _load_graph_answer()

        blocks = []

        for position, result in enumerate(
            results,
            start=1,
        ):

            content = (
                result.content or ""
            ).strip()

            if not content:

                continue

            if len(content) > ga.MAX_CHUNK_CHARS:

                content = (
                    content[:ga.MAX_CHUNK_CHARS]
                    + "\n[truncated]"
                )

            heading = (
                result.heading
                or "(no heading)"
            )

            blocks.append(
                f"[KB{position}] "
                f"{document_label(result.document_id)}"
                f" — {heading}\n"
                f"relevance: "
                f"{result.score:.2f}\n"
                f"{content}"
            )

        return "\n\n".join(blocks)