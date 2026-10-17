"""
ULTRON TOOLS
============================================================

Retrieval tools available to ULTRON agents.

Modules:

- base.py          Tool, ToolResult, ToolRegistry primitives
- web_search.py    Live internet search (web_tool.py)
- kb_search.py     Offline knowledge base (knowledgeBase/)
- pre_retrieve.py  Gated pre-retrieval before the agent loop

Tools are always async at the boundary and offload their
blocking work to a thread, because the FastAPI event loop
must stay responsive.

Every tool returns a ToolResult rather than raising. A
failing tool is a degraded answer, never a dead request.
"""

from tools.base import (
    Tool,
    ToolResult,
    ToolRegistry,
    get_tool_registry,
)

from tools.web_search import WebSearchTool
from tools.kb_search import KnowledgeBaseTool

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "get_tool_registry",
    "WebSearchTool",
    "KnowledgeBaseTool",
]