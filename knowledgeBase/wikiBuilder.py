#!/usr/bin/env python3

"""
Local Markdown Wiki Builder
============================

Input:
    raw/
        *.md
        *.txt
        images/**/*.png
        images/**/*.jpg
        images/**/*.jpeg
        images/**/*.webp

Output:
    wiki/
        documents/
        chunks/
        entities/
        graph/
            graph.json
            nodes.md
            edges.md
        index/
            embeddings.json
            metadata.json
            images.json
            build_state.json
            chunks_cache.json
            graph_cache.json
            embed_cache.json
            image_cache.json

Ollama models:
    Semantic / graph:
        qwen3:4b

    Embeddings:
        qwen3-embedding:8b

    OCR:
        gemma3:4b

    Image understanding:
        qwen2.5vl:7b

The Markdown files remain the source of truth.
JSON files are derived indexes.

Incremental builds:
    The list of processed files and their content hashes
    is stored in index/build_state.json, so a rebuild only
    re-runs the LLM work for new or changed content and
    merges it into the existing graph.

    --status       show the saved state
    --full         ignore the state and reprocess everything
    --reset-state  delete the saved state
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import requests

# Document parsing extras
try:
    import pymupdf as fitz  # modern PyMuPDF API
except ImportError:
    try:
        import fitz  # legacy compatibility
    except ImportError:
        fitz = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None

try:
    from openpyxl import load_workbook
except ImportError:
    load_workbook = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import markdownify
except ImportError:
    markdownify = None

try:
    from ebooklib import epub
except ImportError:
    epub = None

try:
    import ebooklib
except ImportError:
    ebooklib = None

try:
    from striprtf.striprtf import rtf_to_text
except ImportError:
    try:
        import pypandoc
    except ImportError:
        pypandoc = None
    rtf_to_text = None

try:
    from odf.opendocument import OpenDocumentText
    from odf.text import P
except ImportError:
    OpenDocumentText = None
    P = None

# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = Path("raw")
WIKI_DIR = Path("wiki")

DOCUMENTS_DIR = WIKI_DIR / "documents"
CHUNKS_DIR = WIKI_DIR / "chunks"
ENTITIES_DIR = WIKI_DIR / "entities"
GRAPH_DIR = WIKI_DIR / "graph"
INDEX_DIR = WIKI_DIR / "index"

OLLAMA_URL = "http://localhost:11434"

# Fast local models
LLM_MODEL = "qwen3:4b"
EMBED_MODEL = "qwen3-embedding:8b"
OCR_MODEL = "gemma3:4b"
VISION_MODEL = "qwen2.5vl:7b"

SUPPORTED_TEXT = {
    ".md",
    ".markdown",
    ".txt",
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".csv",
    ".html",
    ".htm",
    ".epub",
    ".rtf",
    ".odt",
}

SUPPORTED_IMAGES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}

# Chunking
TARGET_CHUNK_WORDS = 350
MIN_CHUNK_WORDS = 80
MAX_CHUNK_WORDS = 700

# Semantic boundary threshold.
# Lower = fewer chunks.
# Higher = more aggressive splitting.
SEMANTIC_SIMILARITY_THRESHOLD = 0.55

# Only use LLM semantic validation around suspicious boundaries.
ENABLE_LLM_BOUNDARY_VALIDATION = True

# Image processing
ENABLE_IMAGE_PROCESSING = True
MAX_IMAGE_MB = 20

# Graph
MAX_ENTITIES_PER_CHUNK = 20
MAX_RELATIONS_PER_CHUNK = 30

# Incremental build
# Already processed files are tracked on disk so a rebuild
# only re-runs the expensive LLM work for new or changed
# content, then merges the result into the existing graph.
ENABLE_INCREMENTAL = True

# Delete generated files that no longer belong to the wiki.
PRUNE_STALE = True

# How often the incremental state is flushed to disk.
CHECKPOINT_EVERY = 10

# State file schema version.
STATE_VERSION = 1

# Images listed by --status before summarising.
IMAGE_STATUS_LIMIT = 10

# HTTP
REQUEST_TIMEOUT = 300


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Document:
    id: str
    source: str
    title: str
    content: str
    images: list[str]


@dataclass
class Chunk:
    id: str
    document_id: str
    heading: str
    content: str
    start_paragraph: int
    end_paragraph: int
    images: list[str]


@dataclass
class Entity:
    id: str
    name: str
    entity_type: str
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


# ============================================================
# GENERAL UTILITIES
# ============================================================

def log(message: str):
    print(f"[WIKI] {message}")


def warn(message: str):
    print(f"[WARN] {message}")


def error(message: str):
    print(f"[ERROR] {message}")


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text[:100] or "untitled"


def stable_id(*parts: str, prefix: str = "") -> str:
    raw = "||".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"


def ensure_directories():
    for directory in [
        WIKI_DIR,
        DOCUMENTS_DIR,
        CHUNKS_DIR,
        ENTITIES_DIR,
        GRAPH_DIR,
        INDEX_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# INCREMENTAL BUILD STATE
# ============================================================

def utc_now() -> str:
    return time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(),
    )


def content_fingerprint(
    text: str,
) -> str:

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def file_fingerprint(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for block in iter(
            lambda: handle.read(1 << 20),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json_file(
    path: Path,
    default: Any,
) -> Any:

    try:

        if not path.exists():
            return default

        return json.loads(
            path.read_text(encoding="utf-8")
        )

    except Exception as exc:

        warn(
            f"Ignoring unreadable cache "
            f"{path.name}: {exc}"
        )

        return default


class BuildCache:
    """
    Persistent record of everything already processed.

    Keeps:

        build_state.json   processed file list + hashes
        chunks_cache.json  chunks per source document
        graph_cache.json   extracted entities/relations per chunk
        embed_cache.json   embedding vectors per text
        image_cache.json   image analysis per image

    Every entry is content addressed, so a rebuild skips
    work whenever the underlying content is unchanged.
    """

    def __init__(
        self,
        enabled: bool = True,
    ):

        self.enabled = enabled

        self.state_file = (
            INDEX_DIR / "build_state.json"
        )

        self.chunks_file = (
            INDEX_DIR / "chunks_cache.json"
        )

        self.graph_file = (
            INDEX_DIR / "graph_cache.json"
        )

        self.embed_file = (
            INDEX_DIR / "embed_cache.json"
        )

        self.image_file = (
            INDEX_DIR / "image_cache.json"
        )

        self.state = self._load(
            self.state_file,
            self._empty_state(),
        )

        self.chunks = self._load(
            self.chunks_file,
            {},
        )

        self.graph = self._load(
            self.graph_file,
            {},
        )

        self.embed = self._load(
            self.embed_file,
            {},
        )

        self.images = self._load(
            self.image_file,
            {},
        )

        # Keys used by the current build, used to prune.
        self.live_chunks = set()
        self.live_graph = set()
        self.live_embed = set()
        self.live_images = set()

        # Images are only pruned when they were visited.
        self.images_touched = False

        self.hits = {
            "documents": 0,
            "images": 0,
            "graph": 0,
            "embeddings": 0,
        }

        self.misses = {
            "documents": 0,
            "images": 0,
            "graph": 0,
            "embeddings": 0,
        }

        self.failed = {
            "graph": 0,
            "embeddings": 0,
            "images": 0,
        }

    # --------------------------------------------------------
    # Internals
    # --------------------------------------------------------

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "updated_at": None,
            "documents": {},
            "images": {},
            "runs": 0,
            "stats": {},
        }

    def _load(
        self,
        path: Path,
        default: Any,
    ) -> Any:

        if not self.enabled:
            return default

        return load_json_file(
            path,
            default,
        )

    def _count(
        self,
        bucket: str,
        kind: str,
    ):

        target = (
            self.hits
            if kind == "hit"
            else self.misses
        )

        target[bucket] = (
            target.get(bucket, 0) + 1
        )

    def mark_failed(
        self,
        bucket: str,
    ):
        """
        Failed work is never cached, so the next build
        retries it.
        """

        self.failed[bucket] = (
            self.failed.get(bucket, 0) + 1
        )

    @staticmethod
    def _prune(
        cache: dict,
        live: set[str],
    ) -> dict:

        if not isinstance(cache, dict):
            return {}

        return {
            key: value
            for key, value in cache.items()
            if key in live
        }

    @staticmethod
    def _models() -> dict[str, str]:
        return {
            "llm": LLM_MODEL,
            "embedding": EMBED_MODEL,
            "ocr": OCR_MODEL,
            "vision": VISION_MODEL,
        }

    @staticmethod
    def chunk_config() -> str:

        return content_fingerprint(
            "|".join(
                [
                    LLM_MODEL,
                    str(TARGET_CHUNK_WORDS),
                    str(MIN_CHUNK_WORDS),
                    str(MAX_CHUNK_WORDS),
                    str(
                        SEMANTIC_SIMILARITY_THRESHOLD
                    ),
                    str(
                        ENABLE_LLM_BOUNDARY_VALIDATION
                    ),
                ]
            )
        )

    @staticmethod
    def _as_dicts(value: Any) -> list[dict]:

        if not isinstance(value, list):
            return []

        return [
            item
            for item in value
            if isinstance(item, dict)
        ]

    # --------------------------------------------------------
    # Documents
    # --------------------------------------------------------

    def get_document(
        self,
        source: str,
        digest: str,
    ) -> Optional[dict[str, Any]]:
        """
        Return the cached record for a source document
        when the file and chunking settings are unchanged.
        """

        if not self.enabled:
            return None

        record = self.chunks.get(source)

        if not isinstance(record, dict):
            return None

        if record.get("hash") != digest:
            return None

        if (
            record.get("config")
            != self.chunk_config()
        ):
            return None

        chunks = record.get("chunks")

        if not isinstance(chunks, list) or not chunks:
            return None

        # The generated markdown must still exist.
        document_file = DOCUMENTS_DIR / (
            slugify(str(record.get("title", "")))
            + ".md"
        )

        if not document_file.exists():
            return None

        self.live_chunks.add(source)

        self._count(
            "documents",
            "hit",
        )

        return record

    def set_document(
        self,
        source: str,
        digest: str,
        document: Document,
        chunks: list[Chunk],
    ):

        self._count(
            "documents",
            "miss",
        )

        self.adopt_document(
            source,
            digest,
            document,
            chunks,
        )

    def adopt_document(
        self,
        source: str,
        digest: str,
        document: Document,
        chunks: list[Chunk],
    ):
        """
        Store a document record without counting it as
        freshly processed work.
        """

        if not self.enabled:
            return

        record = {
            "hash": digest,
            "config": self.chunk_config(),
            "id": document.id,
            "title": document.title,
            "chunks": [
                asdict(chunk)
                for chunk in chunks
            ],
            "chunks_count": len(chunks),
            "processed_at": utc_now(),
        }

        self.chunks[source] = record
        self.live_chunks.add(source)

        self.state.setdefault(
            "documents",
            {},
        )[source] = {
            "hash": digest,
            "id": document.id,
            "title": document.title,
            "chunks": len(chunks),
            "processed_at": utc_now(),
        }

    def file_digest(
        self,
        path: Path,
    ) -> str:

        try:
            return file_fingerprint(path)

        except Exception as exc:

            warn(
                f"Cannot hash {path}: {exc}"
            )

            return ""

    # --------------------------------------------------------
    # Graph extraction
    # --------------------------------------------------------

    def get_graph(
        self,
        chunk: Chunk,
    ) -> Optional[tuple[list[dict], list[dict]]]:

        if not self.enabled:
            return None

        key = content_fingerprint(chunk.content)

        record = self.graph.get(key)

        self.live_graph.add(key)

        if not isinstance(record, dict):
            self._count(
                "graph",
                "miss",
            )
            return None

        self._count(
            "graph",
            "hit",
        )

        return (
            self._as_dicts(
                record.get("entities")
            ),
            self._as_dicts(
                record.get("relations")
            ),
        )

    def set_graph(
        self,
        chunk: Chunk,
        entities: list[dict],
        relations: list[dict],
    ):

        self._count(
            "graph",
            "miss",
        )

        self.adopt_graph(
            chunk,
            entities,
            relations,
        )

    def adopt_graph(
        self,
        chunk: Chunk,
        entities: list[dict],
        relations: list[dict],
    ):
        """
        Store an extraction without counting it as
        freshly processed work.
        """

        if not self.enabled:
            return

        key = content_fingerprint(chunk.content)

        self.graph[key] = {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "heading": chunk.heading,
            "entities": entities,
            "relations": relations,
            "processed_at": utc_now(),
        }

        self.live_graph.add(key)

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    def _embed_key(
        self,
        text: str,
    ) -> str:

        return content_fingerprint(
            EMBED_MODEL + "\n" + text
        )

    def get_embedding(
        self,
        text: str,
    ) -> Optional[list[float]]:

        key = self._embed_key(text)

        self.live_embed.add(key)

        if not self.enabled:
            return None

        vector = self.embed.get(key)

        if not isinstance(vector, list) or not vector:
            return None

        self._count(
            "embeddings",
            "hit",
        )

        return vector

    def set_embedding(
        self,
        text: str,
        vector: list[float],
    ):

        self._count(
            "embeddings",
            "miss",
        )

        self.adopt_embedding(
            text,
            vector,
        )

    def adopt_embedding(
        self,
        text: str,
        vector: list[float],
    ):
        """
        Store a vector without counting it as freshly
        computed work.
        """

        key = self._embed_key(text)

        self.live_embed.add(key)

        if self.enabled:
            self.embed[key] = vector

    def embed_text(
        self,
        text: str,
        ollama: "Ollama",
    ) -> Optional[list[float]]:
        """
        Embed text, reusing a cached vector when the exact
        text was already embedded with the same model.
        """

        cached = self.get_embedding(text)

        if cached is not None:
            return cached

        try:

            vector = ollama.embed(text)

        except Exception as exc:

            warn(
                f"Embedding failed: {exc}"
            )

            self.mark_failed(
                "embeddings"
            )

            return None

        self.set_embedding(text, vector)

        return vector

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    def get_image(
        self,
        source: str,
        digest: str,
    ) -> Optional[dict[str, Any]]:

        self.images_touched = True

        if not self.enabled:
            return None

        record = self.images.get(source)

        if not isinstance(record, dict):
            return None

        if record.get("hash") != digest:
            return None

        data = record.get("data")

        if not isinstance(data, dict):
            return None

        self.live_images.add(source)

        self._count(
            "images",
            "hit",
        )

        return data

    def set_image(
        self,
        source: str,
        digest: str,
        data: dict[str, Any],
    ):

        self._count(
            "images",
            "miss",
        )

        self.adopt_image(
            source,
            digest,
            data,
        )

    def adopt_image(
        self,
        source: str,
        digest: str,
        data: dict[str, Any],
    ):
        """
        Store image analysis without counting it as
        freshly processed work.
        """

        self.images_touched = True
        self.live_images.add(source)

        if not self.enabled:
            return

        self.images[source] = {
            "hash": digest,
            "data": data,
            "processed_at": utc_now(),
        }

        self.state.setdefault(
            "images",
            {},
        )[source] = {
            "hash": digest,
            "id": data.get("id"),
            "processed_at": utc_now(),
        }

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    def prune(self) -> None:

        self.chunks = self._prune(
            self.chunks,
            self.live_chunks,
        )

        self.graph = self._prune(
            self.graph,
            self.live_graph,
        )

        self.embed = self._prune(
            self.embed,
            self.live_embed,
        )

        self.images = self._prune(
            self.images,
            self.live_images,
        ) if self.images_touched else self.images

        self.state["documents"] = {
            key: value
            for key, value in (
                self.state.get("documents", {})
            ).items()
            if key in self.live_chunks
        }

        if self.images_touched:

            self.state["images"] = {
                key: value
                for key, value in (
                    self.state.get("images", {})
                ).items()
                if key in self.live_images
            }

    def summary(self) -> dict[str, Any]:

        return {
            "enabled": self.enabled,
            "cache_hits": dict(self.hits),
            "cache_misses": dict(self.misses),
            "failed": dict(self.failed),
            "cached_documents": self.hits[
                "documents"
            ],
            "cached_images": self.hits["images"],
            "cached_graph_chunks": self.hits["graph"],
            "cached_embeddings": self.hits[
                "embeddings"
            ],
            "reprocessed_documents": self.misses[
                "documents"
            ],
            "reprocessed_images": self.misses[
                "images"
            ],
            "reprocessed_graph_chunks": self.misses[
                "graph"
            ],
            "recomputed_embeddings": self.misses[
                "embeddings"
            ],
            "documents_tracked": len(
                self.live_chunks
            ),
            "images_tracked": len(
                self.live_images
            ),
            "graph_entries": len(self.live_graph),
        }

    def save(
        self,
        prune: bool = True,
    ) -> None:

        if not self.enabled:
            return

        if prune:
            self.prune()

            self.state["updated_at"] = utc_now()

            self.state["runs"] = (
                int(self.state.get("runs", 0)) + 1
            )

        self.state["version"] = STATE_VERSION
        self.state["models"] = self._models()
        self.state["stats"] = self.summary()

        write_json(
            self.chunks_file,
            self.chunks,
        )

        write_json(
            self.graph_file,
            self.graph,
        )

        write_json(
            self.embed_file,
            self.embed,
        )

        if self.images_touched:

            write_json(
                self.image_file,
                self.images,
            )

        write_json(
            self.state_file,
            self.state,
        )


def prune_generated(
    directory: Path,
    keep: set[str],
) -> int:
    """
    Remove generated files that are no longer part of the
    wiki, so incremental builds stay consistent.
    """

    if not directory.exists():
        return 0

    removed = 0

    for path in sorted(directory.glob("*.md")):

        if path.name in keep:
            continue

        try:

            path.unlink()
            removed += 1

        except Exception as exc:

            warn(
                f"Could not remove stale "
                f"{path}: {exc}"
            )

    return removed


def state_files() -> list[Path]:

    return [
        INDEX_DIR / "build_state.json",
        INDEX_DIR / "chunks_cache.json",
        INDEX_DIR / "graph_cache.json",
        INDEX_DIR / "embed_cache.json",
        INDEX_DIR / "image_cache.json",
    ]


def clear_build_state() -> int:

    removed = 0

    for path in state_files():

        if not path.exists():
            continue

        try:

            path.unlink()
            removed += 1

        except Exception as exc:

            warn(
                f"Could not remove "
                f"{path}: {exc}"
            )

    return removed


def show_state_status() -> None:

    state = load_json_file(
        INDEX_DIR / "build_state.json",
        {},
    )

    documents = state.get(
        "documents",
        {},
    )

    images = state.get(
        "images",
        {},
    )

    print()
    print("=" * 70)
    print("              INCREMENTAL BUILD STATE")
    print("=" * 70)
    print()

    if not state:

        warn(
            "No saved state found. "
            "The next build will process "
            "everything."
        )

        print()

        return

    print(f"Last build : {state.get('updated_at')}")
    print(f"Runs       : {state.get('runs', 0)}")
    print(f"Documents  : {len(documents)}")
    print(f"Images     : {len(images)}")

    stats = state.get("stats", {})

    if isinstance(stats, dict):

        print(
            f"Reused     : "
            f"{stats.get('cached_documents', 0)} documents, "
            f"{stats.get('cached_graph_chunks', 0)} graph chunks, "
            f"{stats.get('cached_embeddings', 0)} embeddings"
        )

    print()
    print("Processed files:")

    for source, record in sorted(
        documents.items()
    ):
        print(
            f"  - {source} "
            f"(chunks: {record.get('chunks', 0)}, "
            f"{str(record.get('hash', ''))[:12]})"
        )

    image_list = sorted(images.items())

    if not image_list:
        return

    print()
    print(f"Processed images: {len(image_list)}")

    for source, record in image_list[:IMAGE_STATUS_LIMIT]:

        print(
            f"  - {source} "
            f"({str(record.get('hash', ''))[:12]})"
        )

    if len(image_list) > IMAGE_STATUS_LIMIT:

        print(
            f"  ... and "
            f"{len(image_list) - IMAGE_STATUS_LIMIT} more"
        )

    print()


# ============================================================
# STATE ADOPTION
# ============================================================

FRONTMATTER_PATTERN = re.compile(
    r"^---\n(.*?)\n---\n\n(.*)$",
    re.DOTALL,
)


def read_markdown_file(
    path: Path,
) -> Optional[tuple[dict[str, str], str]]:
    """
    Split a generated markdown file into frontmatter
    and body.
    """

    try:

        text = path.read_text(encoding="utf-8")

    except Exception as exc:

        warn(f"Cannot read {path}: {exc}")

        return None

    match = FRONTMATTER_PATTERN.match(text)

    if not match:
        return None

    metadata: dict[str, str] = {}

    for line in match.group(1).splitlines():

        if ":" not in line:
            continue

        key, value = line.split(":", 1)

        metadata[key.strip()] = value.strip()

    return metadata, match.group(2)


def load_existing_documents() -> (
    dict[str, dict[str, Any]]
):

    documents: dict[str, dict[str, Any]] = {}

    if not DOCUMENTS_DIR.exists():
        return documents

    for path in sorted(DOCUMENTS_DIR.glob("*.md")):

        parsed = read_markdown_file(path)

        if parsed is None:
            continue

        metadata, body = parsed

        document_id = metadata.get("id", "")

        source = metadata.get("source", "")

        if not document_id or not source:
            continue

        documents[document_id] = {
            "id": document_id,
            "source": source,
            "title": metadata.get("title") or path.stem,
            "content": body,
        }

    return documents


def load_existing_chunks() -> (
    dict[str, list[dict[str, Any]]]
):
    """
    Return chunk records grouped by document id.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}

    if not CHUNKS_DIR.exists():
        return grouped

    for path in sorted(CHUNKS_DIR.glob("*.md")):

        parsed = read_markdown_file(path)

        if parsed is None:
            continue

        metadata, body = parsed

        heading = metadata.get("heading", "")

        prefix = f"# {heading}\n\n"

        if body.startswith(prefix):

            content = body[len(prefix):]

            # save_chunk appends a trailing newline.
            if content.endswith("\n"):
                content = content[:-1]

        else:

            content = body

        try:

            start = int(
                metadata.get(
                    "start_paragraph",
                    0,
                )
            )

            end = int(
                metadata.get(
                    "end_paragraph",
                    0,
                )
            )

        except ValueError:

            start = 0
            end = 0

        document_id = metadata.get("document", "")

        record = {
            "id": metadata.get("id") or path.stem,
            "document_id": document_id,
            "heading": heading,
            "content": content,
            "start_paragraph": start,
            "end_paragraph": end,
            "images": [],
        }

        grouped.setdefault(
            document_id,
            [],
        ).append(record)

    for records in grouped.values():

        records.sort(
            key=lambda item: (
                item["start_paragraph"],
                item["id"],
            )
        )

    return grouped


def load_existing_extractions() -> (
    tuple[
        dict[str, list[dict]],
        dict[str, list[dict]],
    ]
):
    """
    Rebuild per chunk entities/relations from the
    existing graph.json.

    The graph is merged, so this is an approximation:
    a chunk is given every entity that cites it and
    every edge attributed to it.
    """

    entities_by_chunk: dict[str, list[dict]] = {}

    relations_by_chunk: dict[str, list[dict]] = {}

    graph = load_json_file(
        GRAPH_DIR / "graph.json",
        {},
    )

    if not isinstance(graph, dict):
        return entities_by_chunk, relations_by_chunk

    name_by_id: dict[str, str] = {}

    nodes = graph.get("nodes") or []

    for node in nodes:

        if not isinstance(node, dict):
            continue

        node_id = str(node.get("id", ""))

        name_by_id[node_id] = str(
            node.get("name", "")
        )

    for node in nodes:

        if not isinstance(node, dict):
            continue

        name = str(node.get("name", ""))

        if not name:
            continue

        record = {
            "name": name,
            "type": node.get("type", "concept"),
            "description": node.get(
                "description",
                "",
            ),
            "aliases": node.get("aliases") or [],
        }

        sources = node.get("sources") or []

        for chunk_id in sources:

            chunk_id = str(chunk_id)

            entities_by_chunk.setdefault(
                chunk_id,
                [],
            ).append(record)

    for edge in graph.get("edges") or []:

        if not isinstance(edge, dict):
            continue

        source_name = name_by_id.get(
            str(edge.get("source", "")),
            "",
        )

        target_name = name_by_id.get(
            str(edge.get("target", "")),
            "",
        )

        if not source_name or not target_name:
            continue

        record = {
            "source": source_name,
            "target": target_name,
            "relation": edge.get(
                "relation",
                "related_to",
            ),
            "description": edge.get(
                "description",
                "",
            ),
        }

        chunk_id = str(edge.get("source_chunk", ""))

        if not chunk_id:
            continue

        relations_by_chunk.setdefault(
            chunk_id,
            [],
        ).append(record)

    return entities_by_chunk, relations_by_chunk


def adopt_existing(
    retry_empty: bool = False,
) -> dict[str, int]:
    """
    Seed the incremental state from wiki/ output that
    was produced before the state existed, so the next
    build only processes new or changed content.

    Nothing is sent to Ollama here.
    """

    stats = {
        "documents": 0,
        "chunks": 0,
        "graph_entries": 0,
        "embeddings": 0,
        "images": 0,
        "empty_graph": 0,
        "skipped": 0,
        "missing_sources": 0,
    }

    documents = load_existing_documents()

    chunks_by_document = load_existing_chunks()

    (
        entities_by_chunk,
        relations_by_chunk,
    ) = load_existing_extractions()

    embeddings = load_json_file(
        INDEX_DIR / "embeddings.json",
        {},
    )

    if not isinstance(embeddings, dict):
        embeddings = {}

    log(
        f"Found {len(documents)} documents, "
        f"{sum(len(v) for v in chunks_by_document.values())} "
        "chunks on disk"
    )

    cache = BuildCache(enabled=True)

    for document_id, document in sorted(
        documents.items(),
        key=lambda item: item[1]["source"],
    ):

        source = document["source"]

        raw_path = RAW_DIR / source

        if not raw_path.is_file():

            warn(
                f"Source not found, not adopting: "
                f"{source}"
            )

            stats["missing_sources"] += 1

            continue

        records = (
            chunks_by_document.get(document_id)
            or []
        )

        if not records:

            warn(
                f"No chunks on disk, not "
                f"adopting: {source}"
            )

            stats["skipped"] += 1

            continue

        chunks = [
            Chunk(**record)
            for record in records
        ]

        digest = file_fingerprint(raw_path)

        cache.adopt_document(
            source,
            digest,
            Document(
                id=document_id,
                source=source,
                title=document["title"],
                content=document["content"],
                images=[],
            ),
            chunks,
        )

        stats["documents"] += 1
        stats["chunks"] += len(chunks)

        for chunk in chunks:

            entities = (
                entities_by_chunk.get(chunk.id)
                or []
            )

            relations = (
                relations_by_chunk.get(chunk.id)
                or []
            )

            if not entities and not relations:

                stats["empty_graph"] += 1

                if retry_empty:
                    continue

            cache.adopt_graph(
                chunk,
                entities,
                relations,
            )

            stats["graph_entries"] += 1

            vector = embeddings.get(chunk.id)

            if (
                isinstance(vector, list)
                and vector
            ):

                cache.adopt_embedding(
                    chunk.content,
                    vector,
                )

                stats["embeddings"] += 1

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    image_records = load_json_file(
        INDEX_DIR / "images.json",
        [],
    )

    if isinstance(image_records, list):

        for record in image_records:

            if not isinstance(record, dict):
                continue

            source = str(
                record.get("source", "")
            )

            if not source:
                continue

            image_path = RAW_DIR / source

            if not image_path.is_file():
                continue

            cache.adopt_image(
                source,
                file_fingerprint(image_path),
                record,
            )

            stats["images"] += 1

    cache.save()

    print()
    print("=" * 70)
    print("        STATE ADOPTED FROM wiki/")
    print("=" * 70)
    print()
    print(f"Documents  : {stats['documents']}")
    print(f"Chunks     : {stats['chunks']}")
    print(f"Graph      : {stats['graph_entries']} entries")
    print(f"Embeddings : {stats['embeddings']}")
    print(f"Images     : {stats['images']}")
    print()

    if stats["empty_graph"]:

        message = (
            f"{stats['empty_graph']} chunk(s) produced "
            "no entities in the existing graph"
        )

        if retry_empty:

            message += (
                " and were left uncached, so they "
                "are re-extracted on the next build"
            )

        else:

            message += (
                " and were adopted as empty; run with "
                "--retry-empty to re-extract them"
            )

        warn(message)

    if (
        stats["missing_sources"]
        or stats["skipped"]
    ):

        warn(
            f"{stats['missing_sources']} missing source(s), "
            f"{stats['skipped']} skipped document(s)"
        )

    print()
    print(f"State      : {cache.state_file.resolve()}")
    print()

    return stats


# ============================================================
# OLLAMA CLIENT
# ============================================================

class Ollama:
    def __init__(self, base_url: str = OLLAMA_URL):
        self.base_url = base_url.rstrip("/")

    def check(self):
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )

            response.raise_for_status()

            models = response.json().get("models", [])

            names = {
                model.get("name")
                for model in models
            }

            log(f"Ollama available: {len(names)} models")

            required = {
                LLM_MODEL,
                EMBED_MODEL,
            }

            missing = required - names

            if missing:
                warn(
                    "Missing required models: "
                    + ", ".join(sorted(missing))
                )

            return names

        except Exception as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}: {exc}"
            )

    def generate(
        self,
        model: str,
        prompt: str,
        images: Optional[list[str]] = None,
        temperature: float = 0.0,
    ) -> str:

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if images:
            payload["images"] = images

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        if data.get("error"):
            raise RuntimeError(
                str(data["error"])
            )

        return data.get("response", "").strip()

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

        if data.get("error"):
            raise RuntimeError(
                str(data["error"])
            )

        embeddings = data.get("embeddings")

        if not embeddings:
            raise RuntimeError(
                "Ollama did not return an embedding."
            )

        return embeddings[0]


# ============================================================
# FILE DISCOVERY
# ============================================================

def discover_files():

    text_files = []
    image_files = []

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Missing raw directory: {RAW_DIR}"
        )

    for path in RAW_DIR.rglob("*"):

        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix in SUPPORTED_TEXT:
            text_files.append(path)

        elif suffix in SUPPORTED_IMAGES:
            image_files.append(path)

    return sorted(text_files), sorted(image_files)


# ============================================================
# MARKDOWN PARSING
# ============================================================

IMAGE_PATTERN = re.compile(
    r"!\[([^\]]*)\]\(([^)]+)\)"
)

HEADING_PATTERN = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$"
)


def extract_title(text: str, path: Path) -> str:

    for line in text.splitlines():

        match = HEADING_PATTERN.match(line)

        if match and len(match.group(1)) == 1:
            return match.group(2).strip()

    return path.stem.replace("_", " ").replace("-", " ").title()


def extract_images(text: str) -> list[str]:

    images = []

    for match in IMAGE_PATTERN.finditer(text):
        images.append(match.group(2))

    return images


def remove_image_markdown(text: str) -> str:

    return IMAGE_PATTERN.sub(
        lambda m: m.group(1),
        text,
    )


def clean_markdown(text: str) -> str:

    text = remove_image_markdown(text)

    # Remove HTML comments
    text = re.sub(
        r"<!--.*?-->",
        "",
        text,
        flags=re.DOTALL,
    )

    # Keep code content but remove fences
    text = re.sub(
        r"```[\w-]*",
        "",
        text,
    )

    # Markdown links → text
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text,
    )

    # Bold / italic
    text = re.sub(
        r"(\*\*|__)(.*?)\1",
        r"\2",
        text,
    )

    text = re.sub(
        r"(?<!\*)\*([^*]+)\*(?!\*)",
        r"\1",
        text,
    )

    # Inline code
    text = re.sub(
        r"`([^`]+)`",
        r"\1",
        text,
    )

    # Excessive whitespace
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()





def parse_document(path: Path) -> Document:

    suffix = path.suffix.lower()
    extracted_image_paths = []
    
    # Handle different file types
    if suffix == ".pdf":
        text, extracted_image_paths = parse_pdf(path)
    elif suffix in (".docx",):
        text, extracted_image_paths = parse_docx(path)
    elif suffix in (".doc",):
        # basic fallback
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            text = ""
        extracted_image_paths = []
    elif suffix in (".pptx",):
        text, extracted_image_paths = parse_pptx(path)
    elif suffix in (".ppt",):
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            text = ""
        extracted_image_paths = []
    elif suffix in (".xlsx", ".xls"):
        text, extracted_image_paths = parse_xlsx(path)
    elif suffix == ".csv":
        text, extracted_image_paths = parse_csv(path)
    elif suffix in (".html", ".htm"):
        text, extracted_image_paths = parse_html(path)
    elif suffix == ".epub":
        text, extracted_image_paths = parse_epub(path)
    elif suffix == ".rtf":
        text, extracted_image_paths = parse_rtf(path)
    elif suffix == ".odt":
        text, extracted_image_paths = parse_odt(path)
    else:
        # markdown/txt
        text = read_text(path)

    title = extract_title(
        text,
        path,
    )

    images = extract_images(text)
    # Add extracted image refs
    for img_p in extracted_image_paths:
        try:
            rel = str(img_p.relative_to(RAW_DIR))
            images.append(rel)
        except Exception:
            images.append(str(img_p))

    cleaned = clean_markdown(text) if suffix in (".md", ".markdown", ".txt") else clean_markdown(text)

    document_id = stable_id(
        str(path.relative_to(RAW_DIR)),
        cleaned,
        prefix="doc_",
    )

    return Document(
        id=document_id,
        source=str(path.relative_to(RAW_DIR)),
        title=title,
        content=cleaned,
        images=images,
    )


# ============================================================
# PARAGRAPH + SECTION PROCESSING
# ============================================================

def split_paragraphs(text: str) -> list[str]:

    blocks = re.split(
        r"\n\s*\n",
        text,
    )

    result = []

    for block in blocks:

        block = block.strip()

        if not block:
            continue

        result.append(block)

    return result


def find_heading_for_paragraph(
    paragraphs: list[str],
    index: int,
) -> str:

    heading = "Document"

    for i in range(index + 1):

        paragraph = paragraphs[i]

        lines = paragraph.splitlines()

        for line in lines:

            match = HEADING_PATTERN.match(line)

            if match:
                heading = match.group(2).strip()

    return heading


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:

    if not a or not b:
        return 0.0

    dot = sum(
        x * y
        for x, y in zip(a, b)
    )

    norm_a = math.sqrt(
        sum(x * x for x in a)
    )

    norm_b = math.sqrt(
        sum(x * x for x in b)
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# ============================================================
# SEMANTIC CHUNKING
# ============================================================

def semantic_chunk_document(
    document: Document,
    ollama: Ollama,
    cache: Optional[BuildCache] = None,
) -> list[Chunk]:

    paragraphs = split_paragraphs(
        document.content
    )

    if not paragraphs:
        return []

    log(
        f"Semantic chunking: "
        f"{document.source} "
        f"({len(paragraphs)} paragraphs)"
    )

    embeddings = []

    for i, paragraph in enumerate(paragraphs):

        # Don't embed tiny heading-only fragments separately.
        if len(paragraph.split()) < 5:
            embeddings.append(None)
            continue

        try:

            if cache is not None:

                embedding = cache.embed_text(
                    paragraph,
                    ollama,
                )

            else:

                embedding = ollama.embed(
                    paragraph
                )

            embeddings.append(embedding)

        except Exception as exc:

            warn(
                f"Embedding failed for paragraph {i}: {exc}"
            )

            embeddings.append(None)

    boundaries = []

    current_start = 0
    current_words = 0

    for i, paragraph in enumerate(paragraphs):

        words = len(paragraph.split())

        current_words += words

        boundary = False

        # ----------------------------------------------------
        # Heading boundary
        # ----------------------------------------------------

        if i > current_start:

            first_line = paragraph.splitlines()[0]

            if HEADING_PATTERN.match(first_line):

                boundary = True

        # ----------------------------------------------------
        # Semantic boundary
        # ----------------------------------------------------

        if (
            i > 0
            and embeddings[i - 1] is not None
            and embeddings[i] is not None
        ):

            similarity = cosine_similarity(
                embeddings[i - 1],
                embeddings[i],
            )

            if (
                similarity
                < SEMANTIC_SIMILARITY_THRESHOLD
                and current_words >= MIN_CHUNK_WORDS
            ):
                boundary = True

        # ----------------------------------------------------
        # Maximum size
        # ----------------------------------------------------

        if current_words >= MAX_CHUNK_WORDS:
            boundary = True

        if boundary and i > current_start:

            chunk_paragraphs = paragraphs[
                current_start:i
            ]

            create_chunk = True

            if sum(
                len(x.split())
                for x in chunk_paragraphs
            ) < MIN_CHUNK_WORDS:

                create_chunk = False

            if create_chunk:

                chunk = make_chunk(
                    document,
                    chunk_paragraphs,
                    current_start,
                    i - 1,
                )

                boundaries.append(chunk)

                current_start = i
                current_words = words

    # Remaining content
    if current_start < len(paragraphs):

        chunk_paragraphs = paragraphs[
            current_start:
        ]

        chunk = make_chunk(
            document,
            chunk_paragraphs,
            current_start,
            len(paragraphs) - 1,
        )

        boundaries.append(chunk)

    # Merge tiny trailing chunks
    boundaries = merge_small_chunks(
        boundaries
    )

    return boundaries


def make_chunk(
    document: Document,
    paragraphs: list[str],
    start: int,
    end: int,
) -> Chunk:

    content = "\n\n".join(paragraphs)

    heading = find_heading_for_paragraph(
        paragraphs,
        0,
    )

    # Better heading extraction
    for paragraph in paragraphs:

        for line in paragraph.splitlines():

            match = HEADING_PATTERN.match(line)

            if match:
                heading = match.group(2).strip()
                break

    chunk_id = stable_id(
        document.id,
        content,
        prefix="chunk_",
    )

    return Chunk(
        id=chunk_id,
        document_id=document.id,
        heading=heading,
        content=content,
        start_paragraph=start,
        end_paragraph=end,
        images=[],
    )


def merge_small_chunks(
    chunks: list[Chunk],
) -> list[Chunk]:

    if len(chunks) <= 1:
        return chunks

    result = []

    for chunk in chunks:

        words = len(
            chunk.content.split()
        )

        if (
            result
            and words < MIN_CHUNK_WORDS
        ):

            previous = result[-1]

            previous.content += (
                "\n\n"
                + chunk.content
            )

            previous.end_paragraph = (
                chunk.end_paragraph
            )

        else:
            result.append(chunk)

    return result


# ============================================================
# IMAGE PROCESSING
# ============================================================

def resolve_image(
    document_path: Path,
    image_reference: str,
) -> Optional[Path]:

    # Ignore remote URLs
    if image_reference.startswith(
        ("http://", "https://", "data:")
    ):
        return None

    image_reference = image_reference.split(
        "#"
    )[0]

    image_reference = image_reference.split(
        "?"
    )[0]

    candidates = [
        document_path.parent / image_reference,
        RAW_DIR / image_reference,
    ]

    for candidate in candidates:

        if candidate.exists():

            return candidate.resolve()

    return None


def image_to_base64(
    path: Path,
) -> str:

    size_mb = (
        path.stat().st_size
        / (1024 * 1024)
    )

    if size_mb > MAX_IMAGE_MB:
        raise ValueError(
            f"Image too large: "
            f"{path} ({size_mb:.1f} MB)"
        )

    return base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")


def process_image(
    path: Path,
    ollama: Ollama,
) -> dict[str, Any]:

    log(f"Processing image: {path}")

    encoded = image_to_base64(path)

    # First try OCR.
    try:

        ocr_prompt = """
Extract all meaningful text from this image.

Return ONLY the extracted text.

If there is no readable text, return:
NO_TEXT
"""

        ocr = ollama.generate(
            OCR_MODEL,
            ocr_prompt,
            images=[encoded],
        )

    except Exception as exc:

        warn(
            f"OCR failed for {path}: {exc}"
        )

        ocr = ""

    # Vision description
    try:

        vision_prompt = """
Analyze this image for a knowledge wiki.

Return JSON with exactly these fields:

{
  "description": "...",
  "objects": [],
  "concepts": [],
  "technical_information": []
}

Do not invent information.
Only describe what is actually visible.
"""

        raw = ollama.generate(
            VISION_MODEL,
            vision_prompt,
            images=[encoded],
        )

        vision = extract_json(
            raw
        )

        if not isinstance(vision, dict):
            vision = {}

    except Exception as exc:

        warn(
            f"Vision analysis failed for {path}: {exc}"
        )

        vision = {}

    image_id = stable_id(
        str(path.relative_to(RAW_DIR)),
        prefix="img_",
    )

    return {
        "id": image_id,
        "source": str(
            path.relative_to(RAW_DIR)
        ),
        "ocr": ocr,
        "description": vision.get(
            "description",
            "",
        ),
        "objects": vision.get(
            "objects",
            [],
        ),
        "concepts": vision.get(
            "concepts",
            [],
        ),
        "technical_information": vision.get(
            "technical_information",
            [],
        ),
    }


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(
    text: str,
) -> Any:

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"```$",
        "",
        text,
    )

    text = text.strip()

    try:
        return json.loads(text)

    except Exception:
        pass

    # Find object
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:

        try:
            return json.loads(
                text[start:end + 1]
            )

        except Exception:
            pass

    # Find array
    start = text.find("[")
    end = text.rfind("]")

    if start >= 0 and end > start:

        try:
            return json.loads(
                text[start:end + 1]
            )

        except Exception:
            pass

    return None


# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities_and_relations(
    chunk: Chunk,
    ollama: Ollama,
) -> tuple[list[dict], list[dict], bool]:
    """
    Returns entities, relations and whether the
    extraction produced a usable result.
    """

    prompt = f"""
You are building a technical knowledge graph.

Analyze the following wiki chunk.

CHUNK:
{chunk.content}

Extract:

1. Important entities/concepts.
2. Relationships between those entities.

Rules:

- Only extract concepts actually supported by the text.
- Do not hallucinate.
- Prefer meaningful technical concepts.
- Do not create an entity for generic words like "thing",
  "system", "method", "information".
- Merge obvious aliases conceptually.
- Relationships must be directional when appropriate.
- Keep the number of entities small.

Return ONLY valid JSON:

{{
  "entities": [
    {{
      "name": "...",
      "type": "concept|technology|person|organization|method|algorithm|hardware|software|dataset|other",
      "description": "...",
      "aliases": []
    }}
  ],
  "relations": [
    {{
      "source": "...",
      "target": "...",
      "relation": "uses|part_of|depends_on|related_to|implements|extends|produces|controls|measures|contains|similar_to",
      "description": "..."
    }}
  ]
}}
"""

    try:

        response = ollama.generate(
            LLM_MODEL,
            prompt,
        )

        data = extract_json(
            response
        )

        if not isinstance(data, dict):

            warn(
                f"Unusable graph extraction "
                f"for {chunk.id}"
            )

            return [], [], False

        entities = data.get(
            "entities",
            [],
        )

        relations = data.get(
            "relations",
            [],
        )

        if not isinstance(
            entities,
            list,
        ):
            entities = []

        if not isinstance(
            relations,
            list,
        ):
            relations = []

        return (
            entities[:MAX_ENTITIES_PER_CHUNK],
            relations[:MAX_RELATIONS_PER_CHUNK],
            True,
        )

    except Exception as exc:

        warn(
            f"Graph extraction failed "
            f"for {chunk.id}: {exc}"
        )

        return [], [], False


# ============================================================
# ENTITY NORMALIZATION
# ============================================================

def normalize_entity_name(
    name: str,
) -> str:

    name = str(name).strip()

    name = re.sub(
        r"\s+",
        " ",
        name,
    )

    return name


def entity_key(
    name: str,
) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "",
        name.lower(),
    )


def merge_entities(
    all_entities: dict[str, Entity],
    extracted: list[dict],
    chunk_id: str,
):

    for item in extracted:

        if not isinstance(item, dict):
            continue

        name = normalize_entity_name(
            item.get("name", "")
        )

        if not name:
            continue

        key = entity_key(name)

        if not key:
            continue

        entity_type = str(
            item.get(
                "type",
                "concept",
            )
        )

        description = str(
            item.get(
                "description",
                "",
            )
        )

        aliases = item.get(
            "aliases",
            [],
        )

        if not isinstance(
            aliases,
            list,
        ):
            aliases = []

        if key not in all_entities:

            all_entities[key] = Entity(
                id=stable_id(
                    key,
                    prefix="entity_",
                ),
                name=name,
                entity_type=entity_type,
                description=description,
                aliases=aliases,
                sources=[chunk_id],
            )

        else:

            existing = all_entities[key]

            if description and len(description) > len(
                existing.description
            ):
                existing.description = description

            existing.aliases.extend(
                aliases
            )

            if chunk_id not in existing.sources:
                existing.sources.append(
                    chunk_id
                )


# ============================================================
# EDGE PROCESSING
# ============================================================

def normalize_relation(
    relation: str,
) -> str:

    relation = str(
        relation
    ).strip().lower()

    relation = re.sub(
        r"\s+",
        "_",
        relation,
    )

    return relation


def make_edges(
    relations: list[dict],
    chunk_id: str,
    all_entities: dict[str, Entity],
) -> list[Edge]:

    edges = []

    for relation in relations:

        if not isinstance(
            relation,
            dict,
        ):
            continue

        source = normalize_entity_name(
            relation.get(
                "source",
                "",
            )
        )

        target = normalize_entity_name(
            relation.get(
                "target",
                "",
            )
        )

        if not source or not target:
            continue

        source_key = entity_key(
            source
        )

        target_key = entity_key(
            target
        )

        # Don't create edges to unknown concepts.
        if (
            source_key not in all_entities
            or target_key not in all_entities
        ):
            continue

        edges.append(
            Edge(
                source=all_entities[
                    source_key
                ].id,
                target=all_entities[
                    target_key
                ].id,
                relation=normalize_relation(
                    relation.get(
                        "relation",
                        "related_to",
                    )
                ),
                description=str(
                    relation.get(
                        "description",
                        "",
                    )
                ),
                source_chunk=chunk_id,
            )
        )

    return edges


# ============================================================
# MARKDOWN OUTPUT
# ============================================================

def save_document(
    document: Document,
):

    filename = (
        slugify(document.title)
        + ".md"
    )

    output = f"""---
id: {document.id}
title: {document.title}
source: {document.source}
type: document
---

# {document.title}

{document.content}
"""

    write_text(
        DOCUMENTS_DIR / filename,
        output,
    )


def save_chunk(
    chunk: Chunk,
):

    filename = (
        chunk.id
        + ".md"
    )

    output = f"""---
id: {chunk.id}
document: {chunk.document_id}
heading: {chunk.heading}
start_paragraph: {chunk.start_paragraph}
end_paragraph: {chunk.end_paragraph}
---

# {chunk.heading}

{chunk.content}
"""

    write_text(
        CHUNKS_DIR / filename,
        output,
    )


def save_entity(
    entity: Entity,
):

    filename = (
        slugify(entity.name)
        + ".md"
    )

    aliases = "\n".join(
        f"- {alias}"
        for alias in sorted(
            set(entity.aliases)
        )
    )

    sources = "\n".join(
        f"- [[{source}]]"
        for source in entity.sources
    )

    output = f"""---
id: {entity.id}
name: {entity.name}
type: {entity.entity_type}
---

# {entity.name}

{entity.description}

## Aliases

{aliases if aliases else "- None"}

## Sources

{sources if sources else "- None"}
"""

    write_text(
        ENTITIES_DIR / filename,
        output,
    )


def save_graph_markdown(
    entities: dict[str, Entity],
    edges: list[Edge],
):

    nodes = [
        "# Knowledge Graph Nodes",
        "",
    ]

    for entity in sorted(
        entities.values(),
        key=lambda x: x.name.lower(),
    ):

        nodes.append(
            f"- [[{slugify(entity.name)}]] "
            f"({entity.entity_type})"
        )

    write_text(
        GRAPH_DIR / "nodes.md",
        "\n".join(nodes),
    )

    edge_lines = [
        "# Knowledge Graph Edges",
        "",
    ]

    for edge in edges:

        source = next(
            (
                e
                for e in entities.values()
                if e.id == edge.source
            ),
            None,
        )

        target = next(
            (
                e
                for e in entities.values()
                if e.id == edge.target
            ),
            None,
        )

        if not source or not target:
            continue

        edge_lines.append(
            f"- [[{slugify(source.name)}]] "
            f"-- **{edge.relation}** --> "
            f"[[{slugify(target.name)}]]"
        )

    write_text(
        GRAPH_DIR / "edges.md",
        "\n".join(edge_lines),
    )


# ============================================================
# GRAPH JSON
# ============================================================

def save_graph_json(
    entities: dict[str, Entity],
    edges: list[Edge],
):

    nodes = []

    for entity in entities.values():

        nodes.append(
            {
                "id": entity.id,
                "name": entity.name,
                "type": entity.entity_type,
                "description": entity.description,
                "aliases": sorted(
                    set(entity.aliases)
                ),
                "sources": entity.sources,
            }
        )

    edge_data = [
        asdict(edge)
        for edge in edges
    ]

    graph = {
        "version": 1,
        "generated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "nodes": nodes,
        "edges": edge_data,
    }

    write_json(
        GRAPH_DIR / "graph.json",
        graph,
    )


# ============================================================
# EMBEDDING INDEX
# ============================================================



def build_embedding_index(
    chunks: list[Chunk],
    ollama: Ollama,
    cache: Optional[BuildCache] = None,
):

    embeddings = {}
    metadata = {}

    total = len(chunks)

    for i, chunk in enumerate(chunks, 1):

        log(
            f"Embedding "
            f"{i}/{total}: {chunk.id}"
        )

        try:

            if cache is not None:

                embedding = cache.embed_text(
                    chunk.content,
                    ollama,
                )

            else:

                embedding = ollama.embed(
                    chunk.content
                )

            if not embedding:
                continue

            embeddings[chunk.id] = embedding

            metadata[chunk.id] = {
                "document_id": chunk.document_id,
                "heading": chunk.heading,
                "chunk_file": (
                    f"../chunks/{chunk.id}.md"
                ),
                "word_count": len(
                    chunk.content.split()
                ),
            }

        except Exception as exc:

            warn(
                f"Embedding failed for "
                f"{chunk.id}: {exc}"
            )

    write_json(
        INDEX_DIR / "embeddings.json",
        embeddings,
    )

    write_json(
        INDEX_DIR / "metadata.json",
        metadata,
    )


# ============================================================
# IMAGE METADATA
# ============================================================

def save_image_metadata(
    image_data: list[dict],
):

    write_json(
        INDEX_DIR / "images.json",
        image_data,
    )


# ============================================================
# MAIN BUILD
# ============================================================

def build(
    force: bool = False,
    prune_stale: bool = True,
):

    start = time.time()

    print()
    print("=" * 70)
    print("        LOCAL MARKDOWN LLM WIKI BUILDER")
    print("=" * 70)
    print()

    ensure_directories()

    cache = BuildCache(
        enabled=(
            ENABLE_INCREMENTAL
            and not force
        )
    )

    if cache.enabled:

        log(
            "Incremental mode: reusing "
            f"{cache.state.get('runs', 0)} "
            "previous run(s)"
        )

    else:

        log(
            "Full rebuild: cache disabled"
        )

    ollama = Ollama()

    log("Checking Ollama...")

    installed_models = ollama.check()

    # --------------------------------------------------------
    # Discover files
    # --------------------------------------------------------

    text_files, image_files = discover_files()

    log(
        f"Found {len(text_files)} text documents"
    )

    log(
        f"Found {len(image_files)} images"
    )

    if not text_files:

        warn(
            "No Markdown/text files found in raw/"
        )

        return

    # --------------------------------------------------------
    # Ingest documents
    #
    # Unchanged files reuse their cached chunks and are
    # neither parsed nor chunked again.
    # --------------------------------------------------------

    documents = []

    document_queue = []

    for path in text_files:

        source = str(path.relative_to(RAW_DIR))

        digest = cache.file_digest(path)

        record = cache.get_document(
            source,
            digest,
        )

        if record is not None:

            log(
                f"Ingesting: {source} "
                "(unchanged)"
            )

            document = Document(
                id=str(record.get("id", "")),
                source=source,
                title=str(
                    record.get("title", "")
                ),
                content="",
                images=[],
            )

            chunks = [
                Chunk(**item)
                for item in record["chunks"]
                if isinstance(item, dict)
            ]

            documents.append(document)

            document_queue.append(
                (document, digest, chunks)
            )

            continue

        log(f"Ingesting: {source}")

        try:

            document = parse_document(
                path
            )

            documents.append(
                document
            )

            save_document(
                document
            )

            document_queue.append(
                (document, digest, None)
            )

        except Exception as exc:

            warn(
                f"Failed to parse {path}: {exc}"
            )

    # --------------------------------------------------------
    # Semantic chunking
    # --------------------------------------------------------

    all_chunks = []

    reused_chunks = 0

    for document, digest, chunks in (
        document_queue
    ):

        if chunks is None:

            chunks = semantic_chunk_document(
                document,
                ollama,
                cache,
            )

            cache.set_document(
                document.source,
                digest,
                document,
                chunks,
            )

            cache.save(prune=False)

        else:

            reused_chunks += len(chunks)

        for chunk in chunks:

            save_chunk(
                chunk
            )

        all_chunks.extend(
            chunks
        )

    log(
        f"{len(all_chunks)} semantic chunks "
        f"({len(all_chunks) - reused_chunks} new, "
        f"{reused_chunks} reused)"
    )

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    image_metadata = []

    if (
        ENABLE_IMAGE_PROCESSING
        and image_files
    ):

        for image in image_files:

            source = str(
                image.relative_to(RAW_DIR)
            )

            digest = cache.file_digest(
                image
            )

            data = cache.get_image(
                source,
                digest,
            )

            if data is not None:

                image_metadata.append(data)

                continue

            try:

                data = process_image(
                    image,
                    ollama,
                )

                image_metadata.append(
                    data
                )

                cache.set_image(
                    source,
                    digest,
                    data,
                )

            except Exception as exc:

                warn(
                    f"Image failed: "
                    f"{image}: {exc}"
                )

                cache.mark_failed(
                    "images"
                )

        cache.save(prune=False)

    elif (
        image_files
        and (INDEX_DIR / "images.json").exists()
    ):

        # Image processing is off, keep existing metadata.
        log(
            "Image processing disabled, "
            "keeping index/images.json"
        )

    if ENABLE_IMAGE_PROCESSING:

        save_image_metadata(
            image_metadata
        )

    # --------------------------------------------------------
    # Entity + relationship extraction
    #
    # Chunks already extracted by an earlier build are
    # loaded from cache instead of calling the LLM again.
    # --------------------------------------------------------

    entities: dict[str, Entity] = {}

    pending_relations = []

    total = len(all_chunks)

    for i, chunk in enumerate(
        all_chunks,
        1,
    ):

        cached = cache.get_graph(chunk)

        if cached is not None:

            log(
                f"Building graph "
                f"{i}/{total}: "
                f"{chunk.heading} "
                "(cached)"
            )

            extracted_entities, relations = cached

        else:

            log(
                f"Building graph "
                f"{i}/{total}: "
                f"{chunk.heading}"
            )

            extracted_entities, relations, ok = (
                extract_entities_and_relations(
                    chunk,
                    ollama,
                )
            )

            if ok:

                cache.set_graph(
                    chunk,
                    extracted_entities,
                    relations,
                )

            else:

                cache.mark_failed(
                    "graph"
                )

        merge_entities(
            entities,
            extracted_entities,
            chunk.id,
        )

        pending_relations.append(
            (
                chunk.id,
                relations,
            )
        )

        if i % CHECKPOINT_EVERY == 0:

            cache.save(prune=False)

    # --------------------------------------------------------
    # Create graph edges after all entities exist
    # --------------------------------------------------------

    edges = []

    for chunk_id, relations in (
        pending_relations
    ):

        edges.extend(
            make_edges(
                relations,
                chunk_id,
                entities,
            )
        )

    # Remove duplicate edges
    unique_edges = {}

    for edge in edges:

        key = (
            edge.source,
            edge.target,
            edge.relation,
        )

        unique_edges[key] = edge

    edges = list(
        unique_edges.values()
    )

    # --------------------------------------------------------
    # Save entities
    # --------------------------------------------------------

    for entity in entities.values():

        # Remove duplicate aliases/sources
        entity.aliases = sorted(
            set(entity.aliases)
        )

        entity.sources = sorted(
            set(entity.sources)
        )

        save_entity(
            entity
        )

    # --------------------------------------------------------
    # Save graph
    # --------------------------------------------------------

    save_graph_json(
        entities,
        edges,
    )

    save_graph_markdown(
        entities,
        edges,
    )

    # --------------------------------------------------------
    # Remove files of deleted/updated sources
    # --------------------------------------------------------

    removed = {
        "documents": 0,
        "chunks": 0,
        "entities": 0,
    }

    if prune_stale and PRUNE_STALE:

        removed["documents"] = prune_generated(
            DOCUMENTS_DIR,
            {
                slugify(document.title) + ".md"
                for document in documents
            },
        )

        removed["chunks"] = prune_generated(
            CHUNKS_DIR,
            {
                chunk.id + ".md"
                for chunk in all_chunks
            },
        )

        removed["entities"] = prune_generated(
            ENTITIES_DIR,
            {
                slugify(entity.name) + ".md"
                for entity in entities.values()
            },
        )

    # --------------------------------------------------------
    # Build embedding index
    # --------------------------------------------------------

    build_embedding_index(
        all_chunks,
        ollama,
        cache,
    )

    # --------------------------------------------------------
    # Persist incremental state
    # --------------------------------------------------------

    cache.save()

    cache_stats = cache.summary()

    # --------------------------------------------------------
    # Build build-info
    # --------------------------------------------------------

    build_info = {
        "documents": len(documents),
        "chunks": len(all_chunks),
        "entities": len(entities),
        "edges": len(edges),
        "images": len(image_metadata),
        "models": {
            "llm": LLM_MODEL,
            "embedding": EMBED_MODEL,
            "ocr": OCR_MODEL,
            "vision": VISION_MODEL,
        },
        "source": str(
            RAW_DIR.resolve()
        ),
        "output": str(
            WIKI_DIR.resolve()
        ),
        "generated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "elapsed_seconds": round(
            time.time() - start,
            2,
        ),
        "incremental": cache_stats,
        "removed_stale_files": removed,
    }

    write_json(
        WIKI_DIR / "build.json",
        build_info,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("                    BUILD COMPLETE")
    print("=" * 70)
    print()
    print(f"Documents : {len(documents)}")
    print(f"Chunks    : {len(all_chunks)}")
    print(f"Entities  : {len(entities)}")
    print(f"Edges     : {len(edges)}")
    print(f"Images    : {len(image_metadata)}")
    print(f"Time      : {elapsed:.1f}s")
    print()

    if cache.enabled:

        print(
            f"Reused    : "
            f"{cache_stats['cached_documents']} documents, "
            f"{cache_stats['cached_graph_chunks']} graph chunks, "
            f"{cache_stats['cached_embeddings']} embeddings, "
            f"{cache_stats['cached_images']} images"
        )

        print(
            f"Processed : "
            f"{cache_stats['reprocessed_documents']} documents, "
            f"{cache_stats['reprocessed_graph_chunks']} graph chunks, "
            f"{cache_stats['recomputed_embeddings']} embeddings, "
            f"{cache_stats['reprocessed_images']} images"
        )

        print(
            f"State     : {(cache.state_file).resolve()}"
        )

    failed_total = sum(cache.failed.values())

    if failed_total:

        warn(
            f"{failed_total} item(s) failed and "
            "were not cached; the next build "
            "retries them."
        )

    stale_removed = sum(removed.values())

    if stale_removed:

        print(
            f"Removed   : {stale_removed} stale files"
        )

    print()
    print(f"Wiki      : {WIKI_DIR.resolve()}")
    print(f"Graph     : {(GRAPH_DIR / 'graph.json').resolve()}")
    print()


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build a local Markdown knowledge wiki "
            "from raw/ using Ollama."
        )
    )

    parser.add_argument(
        "--raw",
        type=str,
        default="raw",
        help="Input directory",
    )

    parser.add_argument(
        "--wiki",
        type=str,
        default="wiki",
        help="Output directory",
    )

    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Disable image processing",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Ignore the saved build state and "
            "reprocess every file"
        ),
    )

    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help=(
            "Disable incremental mode for this run "
            "without deleting the saved state"
        ),
    )

    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Delete the saved build state and cache",
    )

    parser.add_argument(
        "--keep-stale",
        action="store_true",
        help=(
            "Keep generated files whose sources "
            "were deleted or changed"
        ),
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help=(
            "Show the saved build state "
            "and exit"
        ),
    )

    parser.add_argument(
        "--adopt-existing",
        action="store_true",
        help=(
            "Create the build state from the current "
            "wiki/ output and exit"
        ),
    )

    parser.add_argument(
        "--retry-empty",
        action="store_true",
        help=(
            "With --adopt-existing, do not treat chunks "
            "without entities as processed"
        ),
    )

    args = parser.parse_args()

    global RAW_DIR
    global WIKI_DIR
    global DOCUMENTS_DIR
    global CHUNKS_DIR
    global ENTITIES_DIR
    global GRAPH_DIR
    global INDEX_DIR
    global ENABLE_IMAGE_PROCESSING
    global ENABLE_INCREMENTAL
    global PRUNE_STALE

    RAW_DIR = Path(
        args.raw
    )

    WIKI_DIR = Path(
        args.wiki
    )

    DOCUMENTS_DIR = (
        WIKI_DIR / "documents"
    )

    CHUNKS_DIR = (
        WIKI_DIR / "chunks"
    )

    ENTITIES_DIR = (
        WIKI_DIR / "entities"
    )

    GRAPH_DIR = (
        WIKI_DIR / "graph"
    )

    INDEX_DIR = (
        WIKI_DIR / "index"
    )

    if args.no_images:
        ENABLE_IMAGE_PROCESSING = False

    if args.no_incremental:
        ENABLE_INCREMENTAL = False

    if args.keep_stale:
        PRUNE_STALE = False

    if args.reset_state:

        removed = clear_build_state()

        log(
            f"Removed {removed} state files"
        )

    if args.status:

        show_state_status()
        return

    if args.adopt_existing:

        adopt_existing(
            retry_empty=args.retry_empty,
        )

        return

    try:

        build(
            force=args.full,
            prune_stale=(
                not args.keep_stale
            ),
        )

    except KeyboardInterrupt:

        print()
        warn(
            "Build interrupted. "
            "Progress was saved, so the next "
            "run continues where it stopped."
        )
        sys.exit(130)

    except Exception as exc:

        print()
        error(str(exc))
        sys.exit(1)

def parse_pdf(path: Path) -> tuple[str, list[Path]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed")
    doc = fitz.open(path)
    text_parts = []
    extracted_images = []
    images_dir = RAW_DIR / "images" / "extracted"
    images_dir.mkdir(parents=True, exist_ok=True)
    for page_num, page in enumerate(doc, 1):
        text_parts.append("## Page " + str(page_num) + "\n\n")
        try:
            text = page.get_text("text") or ""
        except Exception:
            text = ""
        text_parts.append(text)
        try:
            for img in page.get_images(full=True):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha < 4:
                        img_path = images_dir / f"{path.stem}_p{page_num}_img{len(extracted_images)+1}.png"
                    else:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        img_path = images_dir / f"{path.stem}_p{page_num}_img{len(extracted_images)+1}.png"
                    pix.save(img_path)
                    extracted_images.append(img_path)
                    pix = None
                except Exception:
                    continue
        except Exception:
            pass
    try:
        doc.close()
    except Exception:
        pass
    full_text = "\n\n".join(text_parts)
    return full_text, extracted_images


def parse_docx(path: Path) -> tuple[str, list[Path]]:
    if DocxDocument is None:
        raise RuntimeError("python-docx not installed")
    doc = DocxDocument(path)
    lines = []
    for p in doc.paragraphs:
        lines.append(p.text)
    text = "\n\n".join(lines)
    extracted = []
    images_dir = RAW_DIR / "images" / "extracted"
    images_dir.mkdir(parents=True, exist_ok=True)
    try:
        rels = doc.part.rels
        for rel in rels.values():
            if "image" in rel.target_ref:
                try:
                    img_data = rel.target_part.blob
                    ext = ".png"
                    # guess
                    img_path = images_dir / f"{path.stem}_img{len(extracted)+1}{ext}"
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    extracted.append(img_path)
                except Exception:
                    pass
    except Exception:
        pass
    return text, extracted



def parse_pptx(path: Path) -> tuple[str, list[Path]]:
    if Presentation is None:
        raise RuntimeError("python-pptx not installed")
    prs = Presentation(path)
    lines = []
    for i, slide in enumerate(prs.slides, 1):
        lines.append(f"## Slide {i}")
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                t = shape.text.strip()
                if t:
                    lines.append(t)
    text = "\n\n".join(lines)
    extracted = []
    images_dir = RAW_DIR / "images" / "extracted"
    images_dir.mkdir(parents=True, exist_ok=True)
    try:
        for i, slide in enumerate(prs.slides):
            for shape in slide.shapes:
                if hasattr(shape, "image"):
                    img = shape.image
                    img_path = images_dir / f"{path.stem}_slide{i+1}_{len(extracted)+1}.png"
                    with open(img_path, "wb") as f:
                        f.write(img.blob)
                    extracted.append(img_path)
    except Exception:
        pass
    return text, extracted



def parse_xlsx(path: Path) -> tuple[str, list[Path]]:
    lines = []
    extracted = []
    if pd is not None:
        try:
            xls = pd.read_excel(path, sheet_name=None)
            for sheet, df in xls.items():
                lines.append(f"## Sheet: {sheet}")
                lines.append(df.to_markdown(index=False))
        except Exception as e:
            lines.append(str(e))
    elif load_workbook is not None:
        try:
            wb = load_workbook(path, data_only=True)
            for ws in wb.worksheets:
                lines.append(f"## Sheet: {ws.title}")
                for row in ws.iter_rows(values_only=True):
                    lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
        except Exception as e:
            lines.append(str(e))
    return "\n\n".join(lines), extracted


def parse_csv(path: Path) -> tuple[str, list[Path]]:
    lines = []
    try:
        if pd is not None:
            df = pd.read_csv(path)
            lines.append(df.to_markdown(index=False))
        else:
            import csv
            with open(path, encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    lines.append("| " + " | ".join(row) + " |")
    except Exception as e:
        lines.append(str(e))
    return "\n\n".join(lines), []



def parse_html(path: Path) -> tuple[str, list[Path]]:
    extracted = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            html = f.read()
        if markdownify is not None:
            md = markdownify.markdownify(html)
        elif BeautifulSoup is not None:
            soup = BeautifulSoup(html, "html.parser")
            # simple
            md = soup.get_text("\n\n")
        else:
            md = html
    except Exception as e:
        md = str(e)
    return md, extracted


def parse_epub(path: Path) -> tuple[str, list[Path]]:
    lines = []
    extracted = []
    if epub is None:
        raise RuntimeError("ebooklib not installed")
    try:
        book = epub.read_epub(path)
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                try:
                    content = item.get_content().decode("utf-8", errors="ignore")
                    if markdownify is not None:
                        lines.append(markdownify.markdownify(content))
                    else:
                        lines.append(content)
                except Exception:
                    pass
    except Exception as e:
        lines.append(str(e))
    return "\n\n".join(lines), extracted


def parse_rtf(path: Path) -> tuple[str, list[Path]]:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            data = f.read()
        if rtf_to_text is not None:
            text = rtf_to_text(data)
        else:
            text = data
    except Exception as e:
        text = str(e)
    return text, []


def parse_odt(path: Path) -> tuple[str, list[Path]]:
    lines = []
    if OpenDocumentText is None:
        raise RuntimeError("odfpy not installed")
    try:
        doc = OpenDocumentText(path)
        for elem in doc.getElementsByType(P):
            try:
                lines.append(str(elem))
            except Exception:
                pass
        text = "\n\n".join(lines)
    except Exception as e:
        text = str(e)
    return text, []


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
