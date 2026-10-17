"""
WEB SEARCH TOOL
============================================================

Exposes ULTRON's existing web_tool.py as an agent-callable
retrieval tool.

Latency is the whole problem here. InternetResearcher.research
fetches ten full pages with a 0.5s sleep between each, which
is far too slow to sit inside an agent loop. So this adapter
defaults to snippet-only search, which returns in roughly two
to six seconds, and only downloads page bodies when an agent
explicitly asks for a small number of them.

Output is formatted with [WEB n] citation labels, distinct
from the [KB n] labels used by the knowledge base tool.
"""

from __future__ import annotations

import asyncio

from typing import Any

import web_tool

from tools.base import (
    ToolResult,
    ToolError,
)


# ============================================================
# TOOL
# ============================================================

class WebSearchTool:
    """
    Live internet search returning titles, URLs and snippets.
    """

    name = "web_search"

    description = (
        "Search the live internet. Returns page titles, "
        "URLs and search snippets. Pass fetch_top_n to "
        "also download the full text of the top results."
    )

    def __init__(self):
        # The underlying researcher prints to stdout, which
        # would corrupt the FastAPI log stream.

        web_tool.QUIET = True

        self._researcher = (
            web_tool.InternetResearcher(
                max_results=10,
                search_timeout=8,
            )
        )

        # A single requests.Session is reused across calls,
        # but it is not thread safe, so the tool takes a lock
        # instead of relying on per-thread instances.

        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        """
        Whether the search backend can actually run.
        """

        return web_tool.DDGS_AVAILABLE

    async def run(
        self,
        query: str,
        max_results: int = 6,
        fetch_top_n: int = 0,
        **_: Any,
    ) -> ToolResult:
        """
        Search the web.

        max_results   How many results to return.
        fetch_top_n   How many of the top results to also
                      download in full. 0 means snippets only,
                      which is the default because fetching
                      page bodies is slow.
        """

        query = (query or "").strip()

        if not query:

            raise ToolError(
                "web_search requires a query."
            )

        if not self.available:

            raise ToolError(
                "Web search is unavailable: the "
                "'ddgs' package is not installed. Run "
                "pip install -r requirements.txt."
            )

        max_results = max(
            1,
            min(int(max_results), 10),
        )

        fetch_top_n = max(
            0,
            min(int(fetch_top_n), 5),
        )

        async with self._lock:

            results = await asyncio.to_thread(
                self._search_blocking,
                query,
                max_results,
                fetch_top_n,
            )

        if not results:

            return ToolResult(
                tool=self.name,
                query=query,
                ok=False,
                error=(
                    "The search backend returned no "
                    "results. It may be rate limited or "
                    "the network may be unavailable."
                ),
            )

        return ToolResult(
            tool=self.name,
            query=query,
            ok=True,
            context=self._format(results),
            sources=[
                self._source(r)
                for r in results
            ],
        )

    # --------------------------------------------------------
    # BLOCKING WORKER
    # --------------------------------------------------------

    def _search_blocking(
        self,
        query: str,
        max_results: int,
        fetch_top_n: int,
    ) -> list:
        """
        Runs in a worker thread.

        Uses search_only when no page bodies are requested,
        and research(fetch_top_n=...) otherwise.
        """

        if fetch_top_n <= 0:

            return self._researcher.search_only(
                query,
                max_results=max_results,
            )

        return self._researcher.research(
            query,
            fetch_top_n=fetch_top_n,
            delay=0.3,
        )

    # --------------------------------------------------------
    # FORMATTING
    # --------------------------------------------------------

    @staticmethod
    def _source(result) -> dict[str, Any]:
        """
        Citation metadata for a web result.
        """

        return {
            "id": f"WEB{result.rank}",
            "kind": "web",
            "title": result.title,
            "url": result.url,
            "domain": result.domain,
            "author": result.author,
            "date": result.date,
            "fetched": bool(result.content),
        }

    def _format(
        self,
        results: list,
    ) -> str:
        """
        Render results as prompt-ready evidence.
        """

        blocks = []

        for result in results:

            body = (
                result.content
                or result.snippet
            )

            if not body:

                continue

            block = (
                f"[WEB{result.rank}] "
                f"{result.title}\n"
                f"url: {result.url}\n"
                f"{body}"
            )

            blocks.append(block)

        return "\n\n".join(blocks)