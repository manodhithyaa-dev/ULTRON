"""
PRE-RETRIEVAL
============================================================

Runs knowledge base and web search once per query, before the
agent loop starts, and stores the evidence on the state so
every agent prompt receives it.

The effort level that main.classify_effort already assigned to
the query is the only decision input. At "low" the query was
judged too simple to be worth looking anything up, so nothing
runs. Above that, both tools run concurrently and their output
is injected into the prompt of every agent in the loop, not
just the first one.

Failures are recorded, never raised. A dead Ollama degrades
the answer to model knowledge instead of failing the request.
"""

from __future__ import annotations

import asyncio
import logging
import time

from typing import Any

from config import (
    KB_TOP_K,
    KB_TIMEOUT,
    TOOLS,
    TOOLS_ENABLED,
    WEB_RESULTS,
    WEB_TIMEOUT,
)

from state import AgentState
from tools.base import (
    ToolRegistry,
    ToolResult,
)


logger = logging.getLogger(
    "ULTRON.tools.pre_retrieve"
)


# ============================================================
# GATE
# ============================================================

def plan_retrieval(
    query: str,
    effort: str,
) -> dict[str, bool]:
    """
    Decide which tools to run before the agent loop.

    The only input is the effort level that main.classify_effort
    already assigned to the query. The classifier decides
    whether the query is substantial enough to be worth
    looking anything up, and this function just honors it:

        effort "low"      -> no retrieval
        effort otherwise  -> knowledge base and web search

    Returns {"kb_search": bool, "web_search": bool}.

    Whatever is retrieved here is injected into the prompt of
    every agent in the loop, not just the first one.
    """

    plan = {
        "kb_search": False,
        "web_search": False,
    }

    if not TOOLS_ENABLED:

        return plan

    if not query or not query.strip():

        return plan

    if effort == "low":

        return plan

    plan["kb_search"] = TOOLS["kb_search"][
        "use_kb"
    ].get(effort, True)

    plan["web_search"] = TOOLS["web_search"][
        "use_web"
    ].get(effort, True)

    return plan


# ============================================================
# EXECUTION
# ============================================================

def _arguments_for(
    tool: str,
    query: str,
) -> dict[str, Any]:
    """
    Per-tool keyword arguments.
    """

    if tool == "web_search":

        return {
            "max_results": WEB_RESULTS,
            "fetch_top_n": 0,
        }

    if tool == "kb_search":

        return {
            "top_k": KB_TOP_K,
        }

    return {}


async def _run_tool(
    registry: ToolRegistry,
    tool: str,
    query: str,
    timeout: float,
) -> ToolResult:
    """
    Run one tool under a hard wall-clock budget.
    """

    try:

        return await asyncio.wait_for(
            registry.run(
                tool,
                query,
                agent="system",
                globally_enabled=TOOLS_ENABLED,
                **_arguments_for(
                    tool,
                    query,
                ),
            ),
            timeout=timeout,
        )

    except asyncio.TimeoutError:

        return ToolResult(
            tool=tool,
            query=query,
            ok=False,
            error=(
                f"{tool} exceeded its "
                f"{timeout:.0f}s budget and was "
                f"abandoned."
            ),
        )

    except Exception as exc:

        return ToolResult(
            tool=tool,
            query=query,
            ok=False,
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )


async def pre_retrieve(
    state: AgentState,
    registry: ToolRegistry,
) -> AgentState:
    """
    Run gated retrieval for a query and record the evidence on
    the state.

    Both tools run concurrently. Neither can raise.
    """

    query = state.query_memory.user_query

    if not query or not query.strip():

        return state

    plan = plan_retrieval(
        query,
        state.effort,
    )

    selected = [
        tool
        for tool, wanted in plan.items()
        if wanted and registry.has(tool)
    ]

    if not selected:

        logger.debug(
            "Pre-retrieval skipped for query in "
            "session %s",
            state.session_id,
        )

        return state

    started = time.monotonic()

    tasks = []

    for tool in selected:

        timeout = (
            WEB_TIMEOUT
            if tool == "web_search"
            else KB_TIMEOUT
        )

        tasks.append(
            _run_tool(
                registry,
                tool,
                query,
                timeout,
            )
        )

    results = await asyncio.gather(
        *tasks
    )

    for result in results:

        state.retrieval.add(
            result.tool,
            result,
        )

    elapsed = int(
        (time.monotonic() - started)
        * 1000
    )

    summary = ", ".join(
        r.summary() for r in results
    )

    logger.info(
        "Pre-retrieval finished in %dms | %s",
        elapsed,
        summary or "no results",
    )

    if state.retrieval.source_count:

        state.add_system_event(
            (
                f"Retrieved "
                f"{state.retrieval.source_count} "
                f"passage(s) from "
                f"{len(selected)} source(s) "
                f"in {elapsed}ms."
            ),
            {
                "tools": selected,
                "elapsed_ms": elapsed,
                "sources": (
                    state.retrieval.source_count
                ),
            },
        )

    return state


# ============================================================
# ON-DEMAND EXECUTION
# ============================================================

async def run_tool_requests(
    state: AgentState,
    registry: ToolRegistry,
    requests: list[dict[str, Any]],
    agent: str,
) -> list[ToolResult]:
    """
    Execute the tool requests an agent emitted.

    This is the escape hatch for agents that need more than
    the gated pre-retrieval found. Results are merged into the
    state so the agent sees them on its next turn.
    """

    if not requests:

        return []

    from config import (
        MAX_TOOL_REQUESTS_PER_TURN,
        MAX_TOOL_ROUNDS,
    )

    results: list[ToolResult] = []

    remaining_rounds = (
        MAX_TOOL_ROUNDS
        - state.retrieval.tool_rounds
    )

    if remaining_rounds <= 0:

        state.add_system_event(
            (
                f"Tool budget exhausted after "
                f"{MAX_TOOL_ROUNDS} rounds; "
                f"ignoring further requests."
            ),
            {"tool": "budget"},
        )

        return results

    for request in requests[:
        MAX_TOOL_REQUESTS_PER_TURN
    ]:

        tool = request.get(
            "tool",
            "",
        ).strip()

        query = str(
            request.get("query", "")
        ).strip()[:400]

        if not tool or not query:

            continue

        if not registry.has(tool):

            result = ToolResult(
                tool=tool,
                query=query,
                ok=False,
                error=(
                    f"Unknown tool '{tool}'."
                ),
            )

        else:

            kwargs = _arguments_for(
                tool,
                query,
            )

            # An agent may explicitly ask for a small number
            # of full pages.

            if tool == "web_search":

                try:

                    requested = int(
                        request.get(
                            "fetch_top_n",
                            0,
                        )
                    )

                except (TypeError, ValueError):

                    requested = 0

                kwargs["fetch_top_n"] = max(
                    0,
                    min(requested, 5),
                )

            timeout = (
                WEB_TIMEOUT
                if tool == "web_search"
                else KB_TIMEOUT
            )

            result = await _run_tool(
                registry,
                tool,
                query,
                timeout,
            )

        state.retrieval.add(
            result.tool,
            result,
        )

        results.append(result)

        logger.info(
            "Agent '%s' ran tool '%s' | ok=%s | %dms",
            agent,
            tool,
            result.ok,
            result.latency_ms,
        )

    if results:

        state.retrieval.tool_rounds += 1

        state.add_system_event(
            (
                f"Agent '{agent}' ran "
                f"{len(results)} tool request(s): "
                + ", ".join(
                    r.summary() for r in results
                )
            ),
            {
                "agent": agent,
                "round": (
                    state.retrieval.tool_rounds
                ),
            },
        )

    return results