"""
ULTRON MAIN
============================================================

FastAPI entry point and state-based execution engine.

Architecture:

    User
      ↓
    FastAPI
      ↓
    Session Memory
      ↓
    Agent State
      ↓
    Current Agent
      ↓
    Ollama
      ↓
    Agent Result
      ↓
    Validate Transition
      ↓
    Next Agent
      ↓
    ...
      ↓
    Final Agent
      ↓
    Response

ULTRON v1 intentionally has:

- No external tools
- No web search
- No code execution
- No vector database
- No knowledge graph

It does have:

- State-based agent routing
- Query memory
- Session memory
- Effort levels
- Loop limits
- Retry limits
- Ollama
- FastAPI
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import (
    AgentConnectionError,
    AgentEngine,
    AgentError,
    AgentResponseError,
    create_agent_engine,
)
from config import (
    AGENTS,
    APP_NAME,
    APP_VERSION,
    DEFAULT_EFFORT,
    EFFORT_LEVELS,
    HOST,
    PORT,
    SKILLS_ENABLED,
    SKILLS_REPO_DIR,
    TOOLS_ENABLED,
    WORK_AGENTS,
    validate_config,
)
from state import (
    AgentState,
    SessionMemory,
    create_agent_state,
    create_session_memory,
    save_completed_query,
)
from tools import get_tool_registry
from tools.kb_search import warm_index
from tools.pre_retrieve import (
    pre_retrieve,
    run_tool_requests,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger("ultron")


# ============================================================
# GLOBAL RUNTIME
# ============================================================

# In-memory session store.
#
# This is intentionally simple for ULTRON v1.
#
# Later this can become:
#
# Redis
# PostgreSQL
# SQLite
# MongoDB
# etc.

SESSIONS: dict[str, SessionMemory] = {}


# Shared agent engine.
AGENT_ENGINE: Optional[AgentEngine] = None

# Shared tool registry. Built lazily on first use because
# constructing it imports the web and knowledge base stacks.
TOOL_REGISTRY = None


# ============================================================
# SKILL ROUTER
# ============================================================
#
# The skill router lives in agent-skills-main/skill_router.py
# and is loaded lazily by file path so the skills repository
# never has to be on sys.path. Loading or routing failures
# never break a query; they only mean no skills are injected.

_skill_router_module = None
_skill_router_load_attempted = False


def get_skill_router():
    """
    Load agent-skills-main/skill_router.py once, or None.
    """

    global _skill_router_module
    global _skill_router_load_attempted

    if _skill_router_load_attempted:

        return _skill_router_module

    _skill_router_load_attempted = True

    if not SKILLS_ENABLED:

        logger.info(
            "Skill routing disabled "
            "(ULTRON_SKILLS_ENABLED=false)."
        )

        return None

    try:

        import importlib.util

        module_path = (
            Path(SKILLS_REPO_DIR) / "skill_router.py"
        )

        if not module_path.exists():

            logger.warning(
                "skill_router.py not found at %s; "
                "skill routing disabled.",
                module_path,
            )

            return None

        spec = importlib.util.spec_from_file_location(
            "skill_router",
            module_path,
        )

        module = importlib.util.module_from_spec(spec)

        spec.loader.exec_module(module)

        _skill_router_module = module

        index_path = (
            Path(SKILLS_REPO_DIR) / "skill_index.json"
        )

        logger.info(
            "Skill router loaded "
            "(index %s: %s).",
            index_path,
            "ready" if index_path.exists()
            else "MISSING - run build_skill_index.py",
        )

    except Exception:

        logger.exception(
            "Failed to load skill router; "
            "continuing without skills."
        )

        _skill_router_module = None

    return _skill_router_module


async def route_skills(state: AgentState) -> None:
    """
    Route the user's query against the skill catalog once,
    before the agent loop starts.

    Two synchronous Ollama calls run in a worker thread so
    the event loop is never blocked. Any failure leaves
    state.skills empty and the query proceeds unchanged.
    """

    router = get_skill_router()

    if router is None:

        return

    query = state.query_memory.user_query

    try:

        result = await asyncio.to_thread(
            router.route,
            query,
            False,
        )

    except Exception as exc:

        logger.exception(
            "Skill routing failed; continuing "
            "without skills."
        )

        state.add_system_event(
            f"Skill routing failed: {exc}",
            {"stage": "skill_route"},
        )

        return

    state.skills = result or {}

    selected = state.skills.get(
        "selected_skills",
        [],
    )

    if state.skills.get("skill_required") and selected:

        logger.info(
            "Skills selected: %s",
            ", ".join(
                s.get("id", "?")
                for s in selected
            ),
        )

        state.add_system_event(
            (
                "Skills selected: "
                + ", ".join(
                    s.get("name", s.get("id", ""))
                    for s in selected
                )
            ),
            {
                "stage": "skill_route",
                "skill_ids": [
                    s.get("id")
                    for s in selected
                ],
                "execution_guidance": (
                    state.skills.get(
                        "execution_guidance",
                        "",
                    )
                ),
            },
        )

    else:

        logger.info(
            "No skill required: %s",
            state.skills.get("reason", "")[:200],
        )


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    FastAPI startup/shutdown lifecycle.
    """

    global AGENT_ENGINE
    global TOOL_REGISTRY

    logger.info(
        "Starting %s v%s",
        APP_NAME,
        APP_VERSION,
    )

    validate_config()

    AGENT_ENGINE = create_agent_engine()

    logger.info(
        "Ollama endpoint initialized."
    )

    logger.info(
        "Loaded agents: %s",
        ", ".join(AGENTS.keys()),
    )

    logger.info(
        "Effort levels: %s",
        ", ".join(EFFORT_LEVELS.keys()),
    )

    if TOOLS_ENABLED:

        try:

            TOOL_REGISTRY = get_tool_registry()

            logger.info(
                "Tools registered: %s",
                ", ".join(
                    TOOL_REGISTRY.names()
                ),
            )

            # The knowledge base index parses a 1.37MB graph
            # and a 118MB embeddings file. Loading it now
            # keeps that cost off the first user request.
            #
            # Warm in a thread so startup is not blocked,
            # and tolerate failure: ULTRON still answers
            # without retrieval.

            await asyncio.to_thread(
                warm_index
            )

        except Exception:

            logger.exception(
                "Tool registry could not be "
                "initialized. ULTRON will run "
                "without retrieval."
            )

            TOOL_REGISTRY = None

    else:

        logger.info(
            "Retrieval tools are disabled."
        )

    yield

    logger.info(
        "Shutting down %s.",
        APP_NAME,
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "ULTRON state-based local AI agent system."
    ),
    lifespan=lifespan,
)


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):
    """
    Request body for /chat.
    """

    query: str = Field(
        ...,
        min_length=1,
        description="User query.",
    )

    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Existing session ID. "
            "If omitted, a new session is created."
        ),
    )

    effort: str = Field(
        default="auto",
        description=(
            "ULTRON effort level: "
            "low, medium, high, max, or auto."
        ),
    )

    images: Optional[list[str]] = Field(
        default=None,
        description=(
            "List of base64-encoded images "
            "for vision analysis."
        ),
    )

    chat_history: Optional[list[dict[str, str]]] = Field(
        default=None,
        description=(
            "Recent chat messages from the frontend. "
            "Used to give agents conversation context."
        ),
    )


# ============================================================
# RESPONSE MODELS
# ============================================================

class ChatResponse(BaseModel):
    """
    Response returned by /chat.
    """

    session_id: str

    query_id: str

    response: str

    effort: str

    loops: int

    retries: int

    agents_used: list[str]

    termination_reason: Optional[str]

    confidence: float

    elapsed_seconds: float

    # Retrieval telemetry, so a client can tell whether an
    # answer was grounded and which sources back it.

    sources_used: int = 0

    sources: list[dict] = Field(
        default_factory=list
    )

    retrieval_failures: list[dict] = Field(
        default_factory=list
    )

    tool_rounds: int = 0

    # Skill routing telemetry, so the client can show which
    # agent skills (if any) were applied to this query.

    skill_required: bool = False

    skills_used: list[dict] = Field(
        default_factory=list
    )

    skill_execution_guidance: str = ""


class SessionResponse(BaseModel):
    """
    Basic session information.
    """

    session_id: str

    created_at: str

    query_count: int


# ============================================================
# SESSION MANAGEMENT
# ============================================================

def get_or_create_session(
    session_id: Optional[str],
) -> SessionMemory:
    """
    Retrieve an existing session or create a new one.
    """

    if session_id:

        existing = SESSIONS.get(
            session_id
        )

        if existing is not None:

            return existing

    session = create_session_memory()

    SESSIONS[
        session.session_id
    ] = session

    return session


# ============================================================
# EFFORT VALIDATION
# ============================================================

def validate_effort(
    effort: str,
) -> str:
    """
    Validate and normalize the requested effort level.

    "auto" is valid here but must be resolved to a concrete
    level before creating state.
    """

    normalized = (
        effort
        .strip()
        .lower()
    )

    valid = {"low", "medium", "high", "max", "auto"}

    if normalized not in valid:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid effort '{effort}'. "
                f"Valid values: low, medium, high, max, auto"
            ),
        )

    return normalized


# ============================================================
# AUTO EFFORT CLASSIFIER
# ============================================================

# Keywords that indicate code-related tasks.
_CODE_KEYWORDS = {
    "code", "coding", "program", "python", "javascript",
    "typescript", "fastapi", "django", "flask", "api",
    "debug", "fix", "error", "bug", "function", "class",
    "endpoint", "route", "database", "sql", "html", "css",
    "react", "vue", "backend", "frontend", "server",
    "import", "def ", "async ", "await ", "return ",
    "write", "implement", "refactor", "deploy", "test",
    "unit test", "pytest", "npm", "pip", "docker",
}

# Keywords that indicate reasoning tasks.
_REASONING_KEYWORDS = {
    "why", "explain", "analyze", "compare", "difference",
    " pros ", " cons ", "tradeoff", "trade-off", "better",
    "best", "worst", "approach", "architecture", "design",
    "pattern", "should i", "which", "evaluate", "reason",
    "logic", "prove", "deduce", "strategy", "plan",
}

# Keywords that indicate research tasks.
_RESEARCH_KEYWORDS = {
    "research", "investigate", "find", "search", "lookup",
    "what is", "who is", "when did", "history", "overview",
    "summary", "summarize", "list", "examples", "alternatives",
    "options", "possibilities", "technologies", "frameworks",
}

# Keywords that indicate simple / low-effort tasks.
_SIMPLE_KEYWORDS = {
    "hi", "hello", "hey", "thanks", "thank you", "bye",
    "goodbye", "yes", "no", "ok", "okay", "sure",
    "how are you", "what's up", "good morning", "good night",
    "help", "who are you", "what can you do",
}


def classify_effort(query: str) -> str:
    """
    Automatically classify query complexity and return
    an effort level.

    Uses keyword matching and query length as signals.

    Returns: "low", "medium", "high", or "max"
    """

    q = query.strip().lower()
    words = q.split()
    word_count = len(words)

    # --------------------------------------------------
    # LOW: very short or simple greetings / small talk
    # --------------------------------------------------
    if word_count <= 6:

        for kw in _SIMPLE_KEYWORDS:

            if kw in q:

                return "low"

    if word_count <= 4:

        return "low"

    # --------------------------------------------------
    # Signal scoring
    # --------------------------------------------------
    score_code = sum(
        1 for kw in _CODE_KEYWORDS if kw in q
    )
    score_reason = sum(
        1 for kw in _REASONING_KEYWORDS if kw in q
    )
    score_research = sum(
        1 for kw in _RESEARCH_KEYWORDS if kw in q
    )

    combined = score_code + score_reason + score_research

    # --------------------------------------------------
    # MAX: very long + multiple signal categories
    # --------------------------------------------------
    if word_count > 80 and combined >= 4:

        return "max"

    # Multiple signals with long query
    categories = sum(
        1 for s in (score_code, score_reason, score_research)
        if s > 0
    )

    if word_count > 50 and categories >= 3:

        return "max"

    # --------------------------------------------------
    # HIGH: code or reasoning heavy, or long + complex
    # --------------------------------------------------
    if score_code >= 3:

        return "high"

    if score_reason >= 3:

        return "high"

    if word_count > 40 and combined >= 3:

        return "high"

    if categories >= 2 and word_count > 25:

        return "high"

    # --------------------------------------------------
    # MEDIUM: default for most real questions
    # --------------------------------------------------
    return "medium"


# ============================================================
# STATE LIMIT CHECKING
# ============================================================

def get_effort_limits(
    effort: str,
) -> tuple[int, int]:
    """
    Return:

        max_loops
        max_retries
    """

    config = EFFORT_LEVELS[
        effort
    ]

    return (
        config["max_loops"],
        config["max_retries"],
    )


def should_stop_before_agent(
    state: AgentState,
    max_loops: int,
) -> bool:
    """
    Determine whether ULTRON has reached its loop limit.

    Only work agents consume loop budget. The orchestrator
    and final agent are free.

    If the loop limit is reached, the final agent is still
    allowed to execute so it can produce the accumulated answer.
    """

    if state.loop_count < max_loops:

        return False

    # Allow final agent through even at the loop limit so it
    # can produce the accumulated answer.
    if state.current_agent == "final":

        return False

    return True


# ============================================================
# AGENT EXECUTION
# ============================================================

async def execute_state_machine(
    state: AgentState,
) -> AgentState:
    """
    Run the ULTRON state machine.

    The current agent decides the next agent.

    Example:

        orchestrator
             ↓
        research
             ↓
        reasoning
             ↓
        verification
             ↓
        final

    But this is NOT hardcoded.

    The actual route is determined dynamically by each agent.
    """

    if AGENT_ENGINE is None:

        raise RuntimeError(
            "ULTRON agent engine has not been initialized."
        )

    max_loops, max_retries = (
        get_effort_limits(
            state.effort
        )
    )

    logger.info(
        "Starting state machine | "
        "session=%s | "
        "query=%s | "
        "effort=%s | "
        "max_loops=%s | "
        "max_retries=%s",
        state.session_id,
        state.query_memory.query_id,
        state.effort,
        max_loops,
        max_retries,
    )

    # ----------------------------------------------------
    # SKILL ROUTING
    # ----------------------------------------------------
    #
    # Runs once, before the first agent, so every prompt in
    # the loop carries the same SELECTED SKILLS block.
    # Failures are already swallowed inside route_skills().

    await route_skills(state)

    # ----------------------------------------------------
    # PRE-RETRIEVAL
    # ----------------------------------------------------
    #
    # Runs once, before the first agent, so every prompt in
    # the loop carries the same evidence. Gated in
    # pre_retrieve.plan_retrieval and unable to raise.

    if TOOLS_ENABLED and TOOL_REGISTRY is not None:

        try:

            await pre_retrieve(
                state,
                TOOL_REGISTRY,
            )

        except Exception as exc:

            logger.exception(
                "Pre-retrieval failed; continuing "
                "without evidence.",
            )

            state.add_system_event(
                (
                    "Pre-retrieval could not "
                    f"complete: {exc}"
                ),
                {"stage": "pre_retrieve"},
            )

    while not state.finished:

        # ----------------------------------------------------
        # LOOP LIMIT
        # ----------------------------------------------------

        if should_stop_before_agent(
            state,
            max_loops,
        ):

            logger.warning(
                "Maximum loops reached: %s",
                max_loops,
            )

            # Use the last agent's response if available,
            # otherwise fall back to the generic message.
            last_response = (
                state.final_response
                or state.get_last_agent_response()
                or (
                    "I reached the maximum reasoning "
                    "effort for this request before "
                    "completing the task."
                )
            )

            state.mark_finished(
                response=last_response,
                reason="max_loops",
            )

            state.add_system_event(
                (
                    "Maximum loop limit reached."
                ),
                {
                    "max_loops": max_loops,
                },
            )

            break

        # ----------------------------------------------------
        # CURRENT AGENT
        # ----------------------------------------------------

        current_agent = (
            state.current_agent
        )

        state.mark_agent_started(
            current_agent
        )

        logger.info(
            "Executing agent '%s' | "
            "loop=%s | retry=%s",
            current_agent,
            state.loop_count,
            state.retry_count,
        )

        # ----------------------------------------------------
        # AGENT EXECUTION
        # ----------------------------------------------------

        try:

            result = await AGENT_ENGINE.run(
                state
            )

        except (
            AgentConnectionError,
            AgentResponseError,
            AgentError,
        ) as exc:

            logger.exception(
                "Agent '%s' failed.",
                current_agent,
            )

            # ----------------------------------------------
            # RETRY LIMIT
            # ----------------------------------------------

            if state.retry_count < max_retries:

                state.increment_retry()

                state.add_system_event(
                    (
                        f"Agent '{current_agent}' "
                        f"failed. Retrying."
                    ),
                    {
                        "error": str(exc),
                        "retry": state.retry_count,
                        "max_retries": max_retries,
                    },
                )

                logger.warning(
                    "Retrying agent '%s' "
                    "(%s/%s)",
                    current_agent,
                    state.retry_count,
                    max_retries,
                )

                continue

            # ----------------------------------------------
            # RETRIES EXHAUSTED
            # ----------------------------------------------

            state.add_system_event(
                (
                    f"Agent '{current_agent}' "
                    f"failed and retry limit "
                    f"was exhausted."
                ),
                {
                    "error": str(exc),
                    "max_retries": max_retries,
                },
            )

            state.mark_finished(
                response=(
                    "I couldn't complete the request "
                    "because an internal agent failed "
                    "after the allowed retries."
                ),
                reason="max_retries",
            )

            break

        except Exception as exc:

            logger.exception(
                "Unexpected error from agent '%s'.",
                current_agent,
            )

            state.add_system_event(
                (
                    f"Unexpected error in "
                    f"agent '{current_agent}'."
                ),
                {
                    "error": str(exc),
                },
            )

            state.mark_finished(
                response=(
                    "ULTRON encountered an unexpected "
                    "internal error while processing "
                    "the request."
                ),
                reason="agent_error",
            )

            break

        # ----------------------------------------------------
        # STORE AGENT RESPONSE
        # ----------------------------------------------------

        state.add_agent_memory(
            agent=result.agent,
            response=result.response,
            next_agent=result.next_agent,
            confidence=result.confidence,
        )

        state.set_confidence(
            result.confidence
        )

        # ----------------------------------------------------
        # TOOL DISPATCH
        # ----------------------------------------------------
        #
        # An agent can ask for more retrieval by emitting
        # "tool_requests". When it does, the tools run, the
        # evidence is merged into the state, and the same
        # agent runs again so it can read what came back.
        #
        # This deliberately happens before the loop increment
        # and before the self-routing guard, because the agent
        # has not transitioned anywhere and must not be charged
        # loop budget for asking a question.

        if (
            result.wants_tools
            and TOOLS_ENABLED
            and TOOL_REGISTRY is not None
        ):

            tool_results = (
                await run_tool_requests(
                    state=state,
                    registry=TOOL_REGISTRY,
                    requests=result.tool_requests,
                    agent=current_agent,
                )
            )

            if tool_results:

                if any(
                    r.ok for r in tool_results
                ):

                    logger.info(
                        "Re-running agent '%s' "
                        "with new evidence (%d round).",
                        current_agent,
                        state.retrieval.tool_rounds,
                    )

                    # Stay on this agent. It will see the
                    # updated RETRIEVED EVIDENCE block on its
                    # next turn and should then route or
                    # finish.

                    state.current_agent = (
                        current_agent
                    )

                    continue

                logger.warning(
                    "All tool requests from agent "
                    "'%s' failed; continuing without "
                    "new evidence.",
                    current_agent,
                )

        # ----------------------------------------------------
        # LOOP INCREMENT
        # ----------------------------------------------------

        # Work agents always consume loop budget on successful
        # execution, whether they finish or transition.
        if current_agent in WORK_AGENTS:

            state.increment_loop()

        # ----------------------------------------------------
        # AGENT FINISHED
        # ----------------------------------------------------

        if result.finished:

            state.mark_finished(
                response=result.response,
                reason="completed",
            )

            logger.info(
                "ULTRON finished through agent '%s'.",
                current_agent,
            )

            break

        # ----------------------------------------------------
        # TRANSITION
        # ----------------------------------------------------

        next_agent = (
            result.next_agent
        )

        state.record_transition(
            from_agent=current_agent,
            to_agent=next_agent,
            confidence=result.confidence,
        )

        # ----------------------------------------------------
        # SELF-ROUTING DETECTION
        # ----------------------------------------------------

        # If a work agent routes to itself, it is stuck.
        # Force it to finish with whatever it said.
        if (
            next_agent == current_agent
            and current_agent in WORK_AGENTS
        ):

            logger.info(
                "Agent '%s' routed to itself, "
                "forcing finish.",
                current_agent,
            )

            state.mark_finished(
                response=result.response,
                reason="self_routing",
            )

            break

        logger.info(
            "Transition: %s -> %s",
            current_agent,
            next_agent,
        )

        state.current_agent = next_agent

    # ========================================================
    # SESSION MEMORY
    # ========================================================

    save_completed_query(
        state
    )

    return state


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
) -> ChatResponse:
    """
    Main ULTRON chat endpoint.

    Example request:

    {
        "query": "Explain state based agents",
        "effort": "medium"
    }

    Or with an existing session:

    {
        "query": "Now modify that architecture",
        "session_id": "existing-session-id",
        "effort": "high"
    }
    """

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    query = request.query.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    effort = validate_effort(
        request.effort
    )

    # Resolve "auto" to a concrete effort level.
    if effort == "auto":

        effort = classify_effort(query)

        logger.info(
            "Auto effort classified as: %s",
            effort,
        )

    # --------------------------------------------------------
    # SESSION
    # --------------------------------------------------------

    session = get_or_create_session(
        request.session_id
    )

    # If the same query already exists in session memory
    # (e.g. from a regenerate/retry), remove the stale entry
    # so it gets re-processed fresh.
    session.queries = [
        q
        for q in session.queries
        if not (q.completed and q.user_query == query)
    ]

    # --------------------------------------------------------
    # CREATE STATE
    # --------------------------------------------------------

    state = create_agent_state(
        user_query=query,
        session_memory=session,
        effort=effort,
        images=request.images,
        chat_history=request.chat_history,
    )

    logger.info(
        "New query | "
        "session=%s | "
        "query=%s | "
        "effort=%s",
        state.session_id,
        state.query_memory.query_id,
        effort,
    )

    # --------------------------------------------------------
    # RUN STATE MACHINE
    # --------------------------------------------------------

    try:

        final_state = await execute_state_machine(
            state
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Fatal ULTRON execution error."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "ULTRON failed while processing "
                "the request."
            ),
        ) from exc

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start_time
    )

    response_text = (
        final_state.final_response
        or
        "ULTRON did not produce a final response."
    )

    # Skill routing summary for the client. The full
    # SKILL.md content stays server-side; the UI only
    # needs to know which skills were applied and why.

    skills_data = final_state.skills or {}

    skills_used = [
        {
            "id": s.get("id", ""),
            "name": s.get("name", ""),
            "path": s.get("path", ""),
            "reason": s.get("reason", ""),
            "relevance": s.get("relevance", 0.0),
            "priority": s.get("priority", 1),
        }
        for s in skills_data.get(
            "selected_skills",
            [],
        )
    ]

    return ChatResponse(
        session_id=(
            final_state.session_id
        ),

        query_id=(
            final_state.query_memory.query_id
        ),

        response=response_text,

        effort=final_state.effort,

        loops=final_state.loop_count,

        retries=final_state.retry_count,

        agents_used=(
            final_state.agents_used
        ),

        termination_reason=(
            final_state.termination_reason
        ),

        confidence=(
            final_state.confidence
        ),

        elapsed_seconds=round(
            elapsed,
            3,
        ),

        sources_used=(
            final_state.retrieval.source_count
        ),

        sources=(
            final_state.retrieval.kb_sources
            + final_state.retrieval.web_sources
        ),

        retrieval_failures=(
            final_state.retrieval.failures
        ),

        tool_rounds=(
            final_state.retrieval.tool_rounds
        ),

        skill_required=bool(
            skills_data.get(
                "skill_required",
                False,
            )
        ),

        skills_used=skills_used,

        skill_execution_guidance=(
            skills_data.get(
                "execution_guidance",
                "",
            )
            or ""
        ),
    )


# ============================================================
# NEW SESSION
# ============================================================

@app.post(
    "/session",
    response_model=SessionResponse,
)
async def create_session() -> SessionResponse:
    """
    Explicitly create a new ULTRON session.
    """

    session = create_session_memory()

    SESSIONS[
        session.session_id
    ] = session

    return SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        query_count=0,
    )


# ============================================================
# SESSION INFORMATION
# ============================================================

@app.get(
    "/session/{session_id}",
)
async def get_session(
    session_id: str,
) -> dict[str, Any]:
    """
    Return session memory.

    Useful for debugging ULTRON's memory system.
    """

    session = SESSIONS.get(
        session_id
    )

    if session is None:

        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    return session.to_dict()


# ============================================================
# SESSION DELETE
# ============================================================

@app.delete(
    "/session/{session_id}",
)
async def delete_session(
    session_id: str,
) -> dict[str, Any]:
    """
    Delete a session from memory.
    """

    if session_id not in SESSIONS:

        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    del SESSIONS[
        session_id
    ]

    return {
        "success": True,
        "session_id": session_id,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health",
)
async def health() -> dict[str, Any]:
    """
    Basic ULTRON health endpoint.
    """

    return {
        "status": "ok",
        "application": APP_NAME,
        "version": APP_VERSION,
        "agents": list(
            AGENTS.keys()
        ),
        "sessions": len(
            SESSIONS
        ),
    }


# ============================================================
# AGENT INFORMATION
# ============================================================

@app.get(
    "/agents",
)
async def agents() -> dict[str, Any]:
    """
    Return configured ULTRON agents.

    System prompts are intentionally not exposed.
    """

    result = {}

    for name, config in AGENTS.items():

        result[name] = {
            "model": config["model"],
            "purpose": config["purpose"],
        }

    return {
        "agents": result,
    }


# ============================================================
# EFFORT INFORMATION
# ============================================================

@app.get(
    "/effort",
)
async def effort() -> dict[str, Any]:
    """
    Return available effort levels.
    """

    return {
        "effort_levels": EFFORT_LEVELS,
        "default": DEFAULT_EFFORT,
    }


# ============================================================
# ROOT
# ============================================================

@app.get(
    "/",
)
async def root() -> dict[str, Any]:
    """
    Basic ULTRON information.
    """

    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "description": (
            "State-based local AI agent system."
        ),
        "architecture": (
            "state -> agent -> transition -> state"
        ),
        "agents": list(
            AGENTS.keys()
        ),
    }


# ============================================================
# FRONTEND
# ============================================================

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/ui", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """
    Serve the ULTRON chat frontend.
    """
    return FileResponse(
        FRONTEND_DIR / "index.html",
        media_type="text/html",
    )


if FRONTEND_DIR.is_dir():

    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static",
    )


# ============================================================
# LOCAL DEVELOPMENT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
    )