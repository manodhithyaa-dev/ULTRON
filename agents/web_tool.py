# ============================================================
# INSTALL
# ============================================================

# !pip -q install -U ddgs requests beautifulsoup4 lxml


# ============================================================
# IMPORTS
# ============================================================

import re
import json
import time
import requests

from dataclasses import dataclass, asdict
from typing import List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

try:

    from ddgs import DDGS

    DDGS_AVAILABLE = True

except ImportError:

    DDGS = None
    DDGS_AVAILABLE = False

except Exception:

    DDGS = None
    DDGS_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_RESULTS = 10
PAGE_CHAR_LIMIT = 30000
REQUEST_TIMEOUT = 20
SEARCH_TIMEOUT = 8

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


# ============================================================
# LOGGING
# ============================================================
#
# This module is imported by the ULTRON server, so it must not
# write to stdout on every call. Server code sets QUIET to True
# and reads results programmatically instead.
# ============================================================

QUIET = False


def log(*args, **kwargs):

    if QUIET:

        return

    print(*args, **kwargs)


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class SearchResult:

    rank: int

    title: str

    url: str

    domain: str

    snippet: str

    content: str = ""

    headings: List[str] = None

    author: str = ""

    date: str = ""

    content_length: int = 0

    fetch_status: str = ""

    score: float = 0.0

    def __post_init__(self):

        if self.headings is None:
            self.headings = []


# ============================================================
# WEB SEARCH ENGINE
# ============================================================

class InternetResearcher:

    def __init__(
        self,
        max_results=DEFAULT_RESULTS,
        timeout=REQUEST_TIMEOUT,
        search_timeout=SEARCH_TIMEOUT,
    ):

        self.max_results = max_results
        self.timeout = timeout
        self.search_timeout = search_timeout

        self.session = requests.Session()

        self.session.headers.update({

            "User-Agent": USER_AGENT,

            "Accept":
                "text/html,"
                "application/xhtml+xml,"
                "application/xml;q=0.9,"
                "image/webp,"
                "*/*;q=0.8",

            "Accept-Language":
                "en-US,en;q=0.9",

            "DNT": "1",

            "Connection": "keep-alive",
        })


    # ========================================================
    # SEARCH
    # ========================================================

    def search(self, query, max_results=None):

        log("\n" + "=" * 80)
        log("🔎 INTERNET SEARCH")
        log("=" * 80)

        log(f"\nQuery:")
        log(query)

        results = []

        if not DDGS_AVAILABLE:

            log(
                "\n❌ Web search unavailable: "
                "the 'ddgs' package is not installed"
            )

            return results

        if max_results is None:

            max_results = self.max_results

        try:

            with DDGS(
                timeout=self.search_timeout
            ) as ddgs:

                raw = ddgs.text(
                    query,
                    max_results=max_results,
                    timeout=self.search_timeout
                )

                for index, item in enumerate(
                    raw,
                    start=1
                ):

                    title = item.get(
                        "title",
                        ""
                    ).strip()

                    url = item.get(
                        "href",
                        ""
                    ).strip()

                    snippet = item.get(
                        "body",
                        ""
                    ).strip()

                    if not url:
                        continue

                    results.append(

                        SearchResult(

                            rank=index,

                            title=title,

                            url=url,

                            domain=self.get_domain(
                                url
                            ),

                            snippet=snippet,

                            headings=[],
                        )
                    )

        except Exception as e:

            log(
                f"\n❌ Search failed: {e}"
            )

        log(
            f"\n✅ Search returned "
            f"{len(results)} results"
        )

        return results


    # ========================================================
    # FETCH PAGE
    # ========================================================

    def fetch(self, result):

        log(
            f"\n[{result.rank}] 🌐 "
            f"{result.domain}"
        )

        log(
            f"    {result.title}"
        )

        try:

            response = self.session.get(

                result.url,

                timeout=self.timeout,

                allow_redirects=True,
            )

            status = response.status_code

            content_type = response.headers.get(
                "Content-Type",
                ""
            ).lower()


            # ------------------------------------------------
            # HTTP ERROR
            # ------------------------------------------------

            if status >= 400:

                result.fetch_status = (
                    f"HTTP {status}"
                )

                log(
                    f"    ⚠️ {result.fetch_status}"
                )

                return result


            # ------------------------------------------------
            # NON HTML
            # ------------------------------------------------

            if "text/html" not in content_type:

                result.fetch_status = (
                    "NON_HTML"
                )

                log(
                    "    ⚠️ Non-HTML resource"
                )

                return result


            # ------------------------------------------------
            # PARSE
            # ------------------------------------------------

            soup = BeautifulSoup(
                response.text,
                "lxml"
            )

            self.extract_metadata(
                soup,
                result
            )

            self.extract_headings(
                soup,
                result
            )

            content = self.extract_content(
                soup
            )

            if len(content) > PAGE_CHAR_LIMIT:

                content = content[
                    :PAGE_CHAR_LIMIT
                ]

                content += (
                    "\n\n[CONTENT TRUNCATED]"
                )

            result.content = content

            result.content_length = len(
                content
            )

            result.fetch_status = "OK"

            log(
                f"    ✅ Extracted "
                f"{len(content):,} characters"
            )

        except requests.exceptions.RequestException as e:

            result.fetch_status = (
                f"REQUEST_ERROR: {e}"
            )

            log(
                f"    ⚠️ Request failed: {e}"
            )

        except Exception as e:

            result.fetch_status = (
                f"PARSE_ERROR: {e}"
            )

            log(
                f"    ⚠️ Parse failed: {e}"
            )

        return result


    # ========================================================
    # METADATA
    # ========================================================

    def extract_metadata(
        self,
        soup,
        result
    ):

        # ----------------------------------------------------
        # AUTHOR
        # ----------------------------------------------------

        author = ""

        author_meta = soup.find(
            "meta",
            attrs={
                "name": "author"
            }
        )

        if author_meta:

            author = author_meta.get(
                "content",
                ""
            )

        if not author:

            author_meta = soup.find(
                "meta",
                attrs={
                    "property":
                    "article:author"
                }
            )

            if author_meta:

                author = author_meta.get(
                    "content",
                    ""
                )

        result.author = (
            author.strip()
        )


        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        date = ""

        date_names = [
            "date",
            "publish-date",
            "publication-date",
            "article:published_time",
        ]

        for name in date_names:

            meta = soup.find(
                "meta",
                attrs={
                    "name": name
                }
            )

            if meta:

                date = meta.get(
                    "content",
                    ""
                )

                if date:
                    break


        if not date:

            meta = soup.find(
                "meta",
                attrs={
                    "property":
                    "article:published_time"
                }
            )

            if meta:

                date = meta.get(
                    "content",
                    ""
                )

        result.date = date.strip()


    # ========================================================
    # HEADINGS
    # ========================================================

    def extract_headings(
        self,
        soup,
        result
    ):

        headings = []

        for tag in soup.find_all(
            [
                "h1",
                "h2",
                "h3",
            ]
        ):

            text = tag.get_text(
                " ",
                strip=True
            )

            text = self.clean_text(
                text
            )

            if text and text not in headings:

                headings.append(
                    text
                )

        result.headings = headings[:50]


    # ========================================================
    # CONTENT EXTRACTION
    # ========================================================

    def extract_content(
        self,
        soup
    ):

        # ----------------------------------------------------
        # REMOVE NOISE
        # ----------------------------------------------------

        for tag in soup.find_all(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "iframe",
                "nav",
                "footer",
                "form",
                "aside",
                "menu",
            ]
        ):

            tag.decompose()


        # ----------------------------------------------------
        # FIND MAIN CONTENT
        # ----------------------------------------------------

        main = None

        for selector in [
            "article",
            "main",
            '[role="main"]',
            ".article-body",
            ".article-content",
            ".post-content",
            ".entry-content",
            "#content",
        ]:

            main = soup.select_one(
                selector
            )

            if main:
                break


        if main is None:

            main = soup.body


        if main is None:

            return ""


        # ----------------------------------------------------
        # EXTRACT PARAGRAPHS
        # ----------------------------------------------------

        blocks = []

        for element in main.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "p",
                "li",
                "blockquote",
            ]
        ):

            text = element.get_text(
                " ",
                strip=True
            )

            text = self.clean_text(
                text
            )

            if not text:
                continue

            # Ignore extremely short noise

            if len(text) < 20:

                continue

            blocks.append(
                text
            )


        # ----------------------------------------------------
        # REMOVE DUPLICATES
        # ----------------------------------------------------

        unique = []

        seen = set()

        for block in blocks:

            normalized = block.lower()

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            unique.append(
                block
            )


        return "\n\n".join(
            unique
        )


    # ========================================================
    # CLEAN TEXT
    # ========================================================

    @staticmethod
    def clean_text(text):

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()


    # ========================================================
    # DOMAIN
    # ========================================================

    @staticmethod
    def get_domain(url):

        try:

            domain = urlparse(
                url
            ).netloc

            if domain.startswith(
                "www."
            ):

                domain = domain[4:]

            return domain

        except:

            return ""


    # ========================================================
    # FETCH ALL
    # ========================================================

    def fetch_all(
        self,
        results,
        delay=0.5
    ):

        log("\n")
        log("=" * 80)
        log("📖 READING WEB PAGES")
        log("=" * 80)

        for result in results:

            self.fetch(
                result
            )

            # Don't hammer websites

            if delay > 0:

                time.sleep(
                    delay
                )

        return results


    # ========================================================
    # SEARCH ONLY
    # ========================================================

    def search_only(
        self,
        query,
        max_results=None
    ):
        """
        Run a search and return snippets without
        downloading any page bodies.

        This is the default mode used by ULTRON because
        fetching every page is far too slow to run inside
        an agent loop.
        """

        results = self.search(
            query,
            max_results=max_results
        )

        for result in results:

            result.fetch_status = (
                "SNIPPET_ONLY"
            )

        return results


    # ========================================================
    # SEARCH + FETCH
    # ========================================================

    def research(
        self,
        query,
        fetch_top_n=None,
        delay=0.5
    ):
        """
        Search, then optionally fetch the top N pages.

        fetch_top_n=None means fetch every result.
        fetch_top_n=0 means never fetch page bodies.
        """

        results = self.search(
            query
        )

        if not results:

            return []

        if fetch_top_n is None:

            return self.fetch_all(
                results,
                delay=delay
            )

        fetch_top_n = max(
            0,
            min(
                fetch_top_n,
                len(results)
            )
        )

        if fetch_top_n == 0:

            for result in results:

                result.fetch_status = (
                    "SNIPPET_ONLY"
                )

            return results

        return self.fetch_all(
            results[:fetch_top_n],
            delay=delay
        )


    # ========================================================
    # BUILD LLM CONTEXT
    # ========================================================

    def build_llm_context(
        self,
        results,
        max_chars=50000
    ):

        context = []

        total = 0

        for i, result in enumerate(
            results,
            start=1
        ):

            content = (
                result.content
                if result.content
                else result.snippet
            )

            block = f"""
============================================================
SOURCE {i}
============================================================

TITLE:
{result.title}

URL:
{result.url}

DOMAIN:
{result.domain}

AUTHOR:
{result.author or "Unknown"}

DATE:
{result.date or "Unknown"}

SEARCH SNIPPET:
{result.snippet}

HEADINGS:
{chr(10).join("- " + h for h in result.headings)}

CONTENT:
{content}

FETCH STATUS:
{result.fetch_status}

"""

            if total + len(block) > max_chars:

                break

            context.append(
                block
            )

            total += len(
                block
            )

        return "\n".join(
            context
        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    def save_json(
        self,
        results,
        filename="web_research.json"
    ):

        data = [
            asdict(r)
            for r in results
        ]

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False
            )

        log(
            f"\n💾 Saved to {filename}"
        )


# ============================================================
# DISPLAY DETAILED RESULTS
# ============================================================

def display_detailed_results(
    results
):

    log("\n")
    log("=" * 100)
    log("🌐 DETAILED INTERNET RESEARCH RESULTS")
    log("=" * 100)

    for i, result in enumerate(
        results,
        start=1
    ):

        log("\n")
        log("#" * 100)

        log(
            f"RESULT {i}"
        )

        log("#" * 100)

        log(
            f"\n📌 TITLE\n"
            f"{result.title}"
        )

        log(
            f"\n🌐 URL\n"
            f"{result.url}"
        )

        log(
            f"\n🏢 DOMAIN\n"
            f"{result.domain}"
        )

        log(
            f"\n👤 AUTHOR\n"
            f"{result.author or 'Unknown'}"
        )

        log(
            f"\n📅 DATE\n"
            f"{result.date or 'Unknown'}"
        )

        log(
            f"\n📊 FETCH STATUS\n"
            f"{result.fetch_status}"
        )

        log(
            f"\n📏 CONTENT LENGTH\n"
            f"{result.content_length:,} characters"
        )

        log(
            "\n📝 SEARCH SNIPPET\n"
            f"{result.snippet}"
        )

        if result.headings:

            log(
                "\n📚 HEADINGS"
            )

            for heading in result.headings:

                log(
                    f"   • {heading}"
                )

        if result.content:

            log(
                "\n📖 FULL EXTRACTED CONTENT"
            )

            log(
                "-" * 100
            )

            log(
                result.content
            )

        else:

            log(
                "\n⚠️ PAGE CONTENT UNAVAILABLE"
            )

            log(
                "Using search-engine snippet instead."
            )

        log("\n")


# ============================================================
# RUN
# ============================================================
# RUN
# ============================================================
#
# Everything below is CLI-only. The ULTRON server imports this
# module for InternetResearcher, so none of this may execute on
# import.
# ============================================================

def main():

    query = "latest artificial intelligence news"

    researcher = InternetResearcher(
        max_results=10
    )

    results = researcher.research(
        query
    )

    display_detailed_results(
        results
    )

    researcher.save_json(
        results,
        "ai_news_research.json"
    )


if __name__ == "__main__":

    main()