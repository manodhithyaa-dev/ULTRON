"""
ULTRON TOOL PRIMITIVES
============================================================

Defines the three pieces every ULTRON tool shares:

    Tool         Declares a capability and its contract.
    ToolResult   Carries retrieved evidence back to the agent.
    ToolRegistry Owns the tools and validates requests.

Design rules:

- Tools never raise. Failures are reported through
  ToolResult.ok / ToolResult.error so that a dead Ollama or
  an unreachable search backend degrades the answer instead
  of failing the request.

- Tools are async and offload blocking I/O to a thread.

- Retrieved evidence is plain text already formatted for
  prompt injection, plus a compact source list so agents can
  cite it.
"""

from __future__ import annotations

import time

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
)


# ============================================================
# ERRORS
# ============================================================

class ToolError(Exception):
    """
    Raised inside tool implementations. Tool wrappers catch
    this and convert it into a failed ToolResult.
    """


class ToolNotFoundError(ToolError):
    """
    Raised when an agent requests a tool that is not
    registered.
    """


class ToolDisabledError(ToolError):
    """
    Raised when an agent requests a tool it is not
    permitted to use, or that is globally disabled.
    """


# The caller used for ULTRON's own gated pre-retrieval, as
# opposed to an agent asking for something mid-task.
#
# allowed_agents exists to stop individual agents from
# requesting tools that do not suit their role. It must not
# block ULTRON's own retrieval, so requests carrying this
# caller skip the permission check and are governed instead
# by tools.pre_retrieve.plan_retrieval.

SYSTEM_AGENT = "system"


# ============================================================
# TOOL RESULT
# ============================================================

@dataclass
class ToolResult:
    """
    Evidence returned by a tool.
    """

    tool: str

    query: str

    ok: bool

    # Prompt-ready evidence text, or a human-readable
    # explanation of why nothing was retrieved.

    context: str = ""

    # [{"id": "KB1", "title": ..., "detail": ...}, ...]
    # Lets an agent cite without re-parsing the context.

    sources: list[dict[str, Any]] = field(
        default_factory=list
    )

    error: Optional[str] = None

    latency_ms: int = 0

    @property
    def source_count(self) -> int:

        return len(self.sources)

    def summary(self) -> str:
        """
        Compact one-line description used in transcripts and
        in the state JSON.
        """

        if not self.ok:

            return f"{self.tool}: failed ({self.error})"

        return (
            f"{self.tool}: "
            f"{self.source_count} source(s)"
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializable form for state and session memory.
        The evidence body is intentionally omitted here; it is
        already carried separately on AgentState.retrieval.
        """

        return {
            "tool": self.tool,
            "query": self.query,
            "ok": self.ok,
            "sources": self.sources,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


# ============================================================
# TOOL
# ============================================================

class Tool(Protocol):
    """
    The contract every ULTRON tool implements.
    """

    name: str

    description: str

    async def run(
        self,
        query: str,
        **kwargs: Any,
    ) -> ToolResult:
        ...


# ============================================================
# REGISTRY
# ============================================================

class ToolRegistry:
    """
    Owns the available tools and validates agent requests
    against them.

    A request is rejected when the tool is unknown, globally
    disabled, or not permitted for the calling agent. Rejection
    returns a failed ToolResult rather than raising, so the
    agent simply learns the tool was unavailable.
    """

    def __init__(self):
        self._tools: dict[str, Any] = {}

        # name -> set of agents permitted to request it

        self._permissions: dict[
            str,
            set[str],
        ] = {}

    def register(
        self,
        tool: Any,
        allowed_agents: Optional[
            set[str]
        ] = None,
    ) -> None:
        """
        Register a tool instance.
        """

        name = getattr(
            tool,
            "name",
            None,
        )

        if not name:

            raise ToolError(
                "Tool must expose a 'name' attribute."
            )

        self._tools[name] = tool

        self._permissions[name] = (
            set(allowed_agents)
            if allowed_agents
            else set()
        )

    def get(self, name: str) -> Any:
        """
        Return a registered tool.
        """

        if name not in self._tools:

            raise ToolNotFoundError(
                f"Unknown tool: {name}"
            )

        return self._tools[name]

    def has(self, name: str) -> bool:

        return name in self._tools

    def names(self) -> list[str]:

        return sorted(
            self._tools.keys()
        )

    def allowed_for(
        self,
        agent: str,
    ) -> list[str]:
        """
        Return the tools a given agent may request.

        An empty permission set means every agent may use
        the tool.
        """

        allowed = []

        for name, tool in (
            self._tools.items()
        ):

            permitted = (
                self._permissions.get(name)
                or set()
            )

            if not permitted or (
                agent in permitted
            ):

                allowed.append(name)

        return sorted(allowed)

    def check(
        self,
        name: str,
        agent: str,
        globally_enabled: bool = True,
    ) -> Optional[str]:
        """
        Validate a request.

        Returns None if the request is allowed, otherwise a
        reason string explaining the rejection.
        """

        if not globally_enabled:

            return (
                "Tools are disabled for this "
                "ULTRON instance."
            )

        if not self.has(name):

            available = ", ".join(
                self.names()
            ) or "none"

            return (
                f"Unknown tool '{name}'. "
                f"Available: {available}"
            )

        # ULTRON's own gated pre-retrieval is not subject to
        # per-agent permissions; plan_retrieval governs it.

        if agent == SYSTEM_AGENT:

            return None

        permitted = (
            self._permissions.get(name) or set()
        )

        if permitted and (
            agent not in permitted
        ):

            allowed = ", ".join(
                sorted(permitted)
            )

            return (
                f"Agent '{agent}' may not use "
                f"'{name}'. Allowed agents: "
                f"{allowed}"
            )

        return None

    async def run(
        self,
        name: str,
        query: str,
        agent: str = "system",
        globally_enabled: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute a tool with full validation and error
        containment.
        """

        started = time.monotonic()

        rejection = self.check(
            name,
            agent,
            globally_enabled=globally_enabled,
        )

        if rejection:

            return ToolResult(
                tool=name,
                query=query,
                ok=False,
                error=rejection,
                latency_ms=self._elapsed_ms(
                    started
                ),
            )

        tool = self.get(name)

        try:

            result = await tool.run(
                query,
                **kwargs,
            )

        except ToolError as exc:

            result = ToolResult(
                tool=name,
                query=query,
                ok=False,
                error=str(exc),
            )

        except Exception as exc:

            result = ToolResult(
                tool=name,
                query=query,
                ok=False,
                error=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        result.latency_ms = self._elapsed_ms(
            started
        )

        return result

    @staticmethod
    def _elapsed_ms(
        started: float,
    ) -> int:

        return int(
            (time.monotonic() - started)
            * 1000
        )


# ============================================================
# GLOBAL REGISTRY
# ============================================================

_REGISTRY: Optional[ToolRegistry] = None


def build_registry() -> ToolRegistry:
    """
    Construct a registry with ULTRON's built-in tools.

    Imports are deferred so that a missing optional
    dependency disables one tool instead of breaking startup.
    """

    from config import TOOLS

    registry = ToolRegistry()

    # Imported here rather than at module scope because
    # kb_search touches the knowledge base and web_search
    # pulls in an HTTP stack.

    from tools.web_search import (
        WebSearchTool,
    )

    from tools.kb_search import (
        KnowledgeBaseTool,
    )

    registry.register(
        WebSearchTool(),
        allowed_agents=TOOLS[
            "web_search"
        ]["allowed_agents"],
    )

    registry.register(
        KnowledgeBaseTool(),
        allowed_agents=TOOLS[
            "kb_search"
        ]["allowed_agents"],
    )

    return registry


def get_tool_registry() -> ToolRegistry:
    """
    Return the process-wide tool registry, building it on
    first use.
    """

    global _REGISTRY

    if _REGISTRY is None:

        _REGISTRY = build_registry()

    return _REGISTRY


def reset_tool_registry() -> None:
    """
    Drop the cached registry. Used by tests.
    """

    global _REGISTRY

    _REGISTRY = None