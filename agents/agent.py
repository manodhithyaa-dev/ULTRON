"""
ULTRON AGENT ENGINE
============================================================

Responsible for:

- Loading agent configuration
- Calling Ollama
- Building agent context
- Parsing structured agent output
- Validating next-agent transitions
- Handling malformed model responses
- Returning structured AgentResult objects

This file does NOT:

- Run FastAPI
- Manage sessions
- Control the main execution loop
- Store persistent memory

The execution loop lives in main.py.
State and memory live in state.py.
Agent configuration lives in config.py.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from config import (
    AGENTS,
    AGENT_NAMES,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_TEMPERATURE,
    MAX_EVIDENCE_CHARS,
    MAX_SKILL_CHARS,
    MAX_TOOL_REQUESTS_PER_TURN,
    MODELS,
    OLLAMA_BASE_URL,
    TOOLS,
    VALID_TRANSITIONS,
)
from state import AgentState


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("ultron.agent")


# ============================================================
# EXCEPTIONS
# ============================================================

class AgentError(Exception):
    """
    Base exception for ULTRON agent failures.
    """


class AgentConfigurationError(AgentError):
    """
    Raised when an agent is missing or incorrectly configured.
    """


class AgentConnectionError(AgentError):
    """
    Raised when ULTRON cannot communicate with Ollama.
    """


class AgentResponseError(AgentError):
    """
    Raised when the LLM response cannot be parsed or validated.
    """


class InvalidTransitionError(AgentError):
    """
    Raised when an agent attempts an invalid state transition.
    """


# ============================================================
# AGENT RESULT
# ============================================================

@dataclass
class AgentResult:
    """
    Structured result returned by an ULTRON agent.
    """

    agent: str

    response: str

    next_agent: str

    finished: bool

    confidence: float

    raw_response: str = ""

    # Sanitized retrieval requests the agent asked for.
    # Empty when the agent did not ask, or when the request
    # was rejected during validation.

    tool_requests: list[dict[str, Any]] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """
        Whether this result represents a successful agent run.
        """

        return self.error is None

    @property
    def wants_tools(self) -> bool:
        """
        Whether this result carries retrieval requests.
        """

        return bool(self.tool_requests)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert result into a JSON-serializable dictionary.
        """

        return {
            "agent": self.agent,
            "response": self.response,
            "next_agent": self.next_agent,
            "finished": self.finished,
            "confidence": self.confidence,
            "tool_requests": self.tool_requests,
            "raw_response": self.raw_response,
            "metadata": self.metadata,
            "error": self.error,
        }


# ============================================================
# OLLAMA CLIENT
# ============================================================

class OllamaClient:
    """
    Minimal asynchronous Ollama client.

    ULTRON v1 intentionally keeps the Ollama integration simple.

    No tools.
    No function calling.
    No external services.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = 300.0,
    ) -> None:

        self.base_url = base_url.rstrip("/")

        self.timeout = timeout

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """
        Send a chat request to Ollama.

        Uses Ollama's /api/chat endpoint.
        """

        url = (
            f"{self.base_url}/api/chat"
        )

        payload = {
            "model": model,

            "messages": messages,

            "stream": False,

            "options": {
                "temperature": temperature,

                "num_ctx": DEFAULT_CONTEXT_WINDOW,
            },
        }

        try:

            async with httpx.AsyncClient(
                timeout=self.timeout
            ) as client:

                response = await client.post(
                    url,
                    json=payload,
                )

        except httpx.RequestError as exc:

            raise AgentConnectionError(
                f"Could not connect to Ollama at "
                f"{self.base_url}: {exc}"
            ) from exc

        if response.status_code != 200:

            raise AgentConnectionError(
                f"Ollama returned HTTP "
                f"{response.status_code}: "
                f"{response.text}"
            )

        try:

            data = response.json()

        except json.JSONDecodeError as exc:

            raise AgentResponseError(
                "Ollama returned invalid JSON."
            ) from exc

        message = data.get(
            "message"
        )

        if not isinstance(
            message,
            dict,
        ):

            raise AgentResponseError(
                "Ollama response does not contain "
                "a valid message object."
            )

        content = message.get(
            "content"
        )

        if not isinstance(
            content,
            str,
        ):

            raise AgentResponseError(
                "Ollama response does not contain "
                "valid message content."
            )

        content = content.strip()

        if not content:

            raise AgentResponseError(
                "Ollama returned an empty response."
            )

        return content


# ============================================================
# JSON EXTRACTION
# ============================================================

def strip_markdown_fences(
    text: str,
) -> str:
    """
    Remove Markdown JSON fences if the model incorrectly adds them.

    Example:

        ```json
        {...}
        ```

    becomes:

        {...}
    """

    text = text.strip()

    if text.startswith(
        "```json"
    ):

        text = text[
            len("```json"):
        ]

        if text.endswith(
            "```"
        ):

            text = text[
                :-3
            ]

    elif text.startswith(
        "```"
    ):

        text = text[
            len("```"):
        ]

        if text.endswith(
            "```"
        ):

            text = text[
                :-3
            ]

    return text.strip()


def extract_json_object(
    text: str,
) -> dict[str, Any]:
    """
    Extract a JSON object from model output.

    The model should return pure JSON, but this function is
    deliberately tolerant of accidental surrounding text.
    """

    cleaned = strip_markdown_fences(
        text
    )

    # First attempt: entire response is JSON.
    try:

        parsed = json.loads(
            cleaned,
            strict=False,
        )

        if isinstance(
            parsed,
            dict,
        ):

            return parsed

    except json.JSONDecodeError:
        pass

    # Second attempt: locate the first JSON object.
    start = cleaned.find("{")

    if start == -1:

        raise AgentResponseError(
            "No JSON object found in agent response."
        )

    depth = 0

    in_string = False

    escape = False

    for index in range(
        start,
        len(cleaned),
    ):

        char = cleaned[index]

        if in_string:

            if escape:

                escape = False

            elif char == "\\":
                escape = True

            elif char == '"':
                in_string = False

            continue

        if char == '"':

            in_string = True

            continue

        if char == "{":

            depth += 1

        elif char == "}":

            depth -= 1

            if depth == 0:

                candidate = cleaned[
                    start:index + 1
                ]

                try:

                    parsed = json.loads(
                        candidate,
                        strict=False,
                    )

                except json.JSONDecodeError as exc:

                    raise AgentResponseError(
                        "Found a JSON-like object, "
                        "but it could not be parsed."
                    ) from exc

                if not isinstance(
                    parsed,
                    dict,
                ):

                    raise AgentResponseError(
                        "Agent JSON must be an object."
                    )

                return parsed

    raise AgentResponseError(
        "Could not extract a complete JSON object "
        "from agent response."
    )


# ============================================================
# VALUE NORMALIZATION
# ============================================================

def normalize_confidence(
    value: Any,
) -> float:
    """
    Normalize confidence to the range 0.0 - 1.0.
    """

    if value is None:
        return 0.0

    try:

        confidence = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return 0.0

    return max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )


def normalize_finished(
    value: Any,
) -> bool:
    """
    Normalize the finished field.
    """

    if isinstance(
        value,
        bool,
    ):

        return value

    if isinstance(
        value,
        str,
    ):

        return value.lower().strip() in {
            "true",
            "yes",
            "1",
            "finished",
            "done",
        }

    return bool(value)


def normalize_next_agent(
    value: Any,
) -> str:
    """
    Normalize the next_agent value.
    """

    if not isinstance(
        value,
        str,
    ):

        raise AgentResponseError(
            "next_agent must be a string."
        )

    return value.strip().lower()


def normalize_tool_requests(
    value: Any,
    agent_name: str,
) -> list[dict[str, Any]]:
    """
    Sanitize the optional "tool_requests" field.

    Agents are small local models, so this field is treated as
    untrusted: anything malformed is dropped rather than
    raised, because a bad tool request must never cost the
    agent the response it already produced.

    Checks applied:

    - must be a list
    - each entry must be an object with a known tool name
    - the tool must be permitted for this agent
    - the query must be a non-empty string, length capped
    - fetch_top_n is coerced into a small non-negative int
    - at most MAX_TOOL_REQUESTS_PER_TURN entries survive
    """

    if value is None:

        return []

    if isinstance(value, dict):
        # Tolerate a single object where a list was
        # expected.

        value = [value]

    if not isinstance(value, list):

        return []

    normalized: list[dict[str, Any]] = []

    for entry in value:

        if (
            len(normalized)
            >= MAX_TOOL_REQUESTS_PER_TURN
        ):

            break

        if not isinstance(entry, dict):

            continue

        tool = entry.get("tool")

        if not isinstance(tool, str):

            continue

        tool = tool.strip().lower()

        if tool not in TOOLS:

            continue

        permitted = TOOLS[
            tool
        ]["allowed_agents"]

        if permitted and (
            agent_name not in permitted
        ):

            continue

        query = entry.get("query")

        if not isinstance(query, str):

            continue

        query = query.strip()[:400]

        if not query:

            continue

        request: dict[str, Any] = {
            "tool": tool,
            "query": query,
        }

        if tool == "web_search":

            try:

                fetch_top_n = int(
                    entry.get(
                        "fetch_top_n",
                        0,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                fetch_top_n = 0

            request["fetch_top_n"] = max(
                0,
                min(fetch_top_n, 5),
            )

        normalized.append(request)

    return normalized


# ============================================================
# AGENT ENGINE
# ============================================================

class AgentEngine:
    """
    Main ULTRON agent execution engine.

    One AgentEngine instance can execute any configured ULTRON
    agent.
    """

    def __init__(
        self,
        ollama: Optional[OllamaClient] = None,
    ) -> None:

        self.ollama = (
            ollama
            if ollama is not None
            else OllamaClient()
        )

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    def get_agent_config(
        self,
        agent_name: str,
    ) -> dict[str, Any]:
        """
        Retrieve configuration for an agent.
        """

        if agent_name not in AGENTS:

            raise AgentConfigurationError(
                f"Unknown ULTRON agent: "
                f"{agent_name}"
            )

        return AGENTS[
            agent_name
        ]

    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    def build_system_prompt(
        self,
        agent_name: str,
    ) -> str:
        """
        Return the configured system prompt for an agent.
        """

        config = self.get_agent_config(
            agent_name
        )

        return config[
            "system_prompt"
        ].strip()

    # --------------------------------------------------------
    # USER CONTEXT
    # --------------------------------------------------------

    def build_context_prompt(
        self,
        state: AgentState,
    ) -> str:
        """
        Build the state context provided to the current agent.

        The context is intentionally structured so the model can
        inspect ULTRON's shared state.
        """

        context = state.to_context()

        return json.dumps(
            context,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # MESSAGES
    # --------------------------------------------------------

    def build_messages(
        self,
        agent_name: str,
        state: AgentState,
    ) -> list[dict[str, Any]]:
        """
        Build the complete Ollama chat message list.
        """

        system_prompt = (
            self.build_system_prompt(
                agent_name
            )
        )

        context = (
            self.build_context_prompt(
                state
            )
        )

        evidence = state.build_evidence_block(
            max_chars=MAX_EVIDENCE_CHARS
        )

        evidence_section = (
            f"\n\n{evidence}\n"
            if evidence
            else (
                "\n\nNo evidence was retrieved "
                "for this query. Rely on the "
                "conversation and your own "
                "knowledge, and be explicit about "
                "uncertainty.\n"
            )
        )

        skill_block = state.build_skill_block(
            max_chars=MAX_SKILL_CHARS
        )

        skill_section = (
            f"\n\n{skill_block}\n"
            if skill_block
            else (
                "\n\nNo skill applies to this "
                "query.\n"
            )
        )

        user_message = f"""
ULTRON CURRENT STATE

The following JSON contains the current state of ULTRON.

Read it carefully.

You are the current agent:
{agent_name}

You must perform your assigned role and decide the next state.
{evidence_section}
{skill_section}
CURRENT STATE:

{context}

Remember:

- Follow your system prompt.
- Use the retrieved evidence when it is relevant, and
  cite it. If it does not help, ignore it.
- Follow the SELECTED SKILLS instructions when they
  apply to this task.
- Never claim to have used a source that is not present.
- Return ONLY valid JSON.
"""

        user_msg: dict[str, Any] = {
            "role": "user",
            "content": user_message.strip(),
        }

        # Attach images if present on state.
        if state.images:

            user_msg["images"] = state.images

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            user_msg,
        ]

    # --------------------------------------------------------
    # TRANSITION VALIDATION
    # --------------------------------------------------------

    def validate_transition(
        self,
        current_agent: str,
        next_agent: str,
    ) -> None:
        """
        Validate that the requested transition is legal.

        This is critical because the LLM is allowed to choose
        the next agent, but it must not have unrestricted control
        over the state graph.
        """

        if current_agent not in AGENT_NAMES:

            raise InvalidTransitionError(
                f"Unknown current agent: "
                f"{current_agent}"
            )

        if next_agent not in AGENT_NAMES:

            raise InvalidTransitionError(
                f"Unknown next agent: "
                f"{next_agent}"
            )

        allowed = VALID_TRANSITIONS.get(
            current_agent,
            set(),
        )

        if next_agent not in allowed:

            raise InvalidTransitionError(
                f"Invalid transition: "
                f"{current_agent} -> {next_agent}"
            )

    # --------------------------------------------------------
    # RESULT VALIDATION
    # --------------------------------------------------------

    def validate_result(
        self,
        agent_name: str,
        data: dict[str, Any],
    ) -> AgentResult:
        """
        Validate and normalize the model-generated agent result.
        """

        if not isinstance(
            data,
            dict,
        ):

            raise AgentResponseError(
                "Agent output must be a JSON object."
            )

        response = data.get(
            "response"
        )

        if not isinstance(
            response,
            str,
        ):

            raise AgentResponseError(
                "Agent output requires a string "
                "'response' field."
            )

        response = response.strip()

        if not response:

            raise AgentResponseError(
                "Agent response cannot be empty."
            )

        next_agent = normalize_next_agent(
            data.get(
                "next_agent"
            )
        )

        finished = normalize_finished(
            data.get(
                "finished",
                False,
            )
        )

        confidence = normalize_confidence(
            data.get(
                "confidence",
                0.0,
            )
        )

        # A final agent must always terminate.
        if agent_name == "final":

            next_agent = "final"

            finished = True

        # Any agent may finish directly when it has completed
        # the task. The next_agent field is irrelevant when
        # finished so normalize it to the current agent.
        elif finished:

            next_agent = agent_name

        self.validate_transition(
            current_agent=agent_name,
            next_agent=next_agent,
        )

        tool_requests = (
            normalize_tool_requests(
                data.get("tool_requests"),
                agent_name,
            )
        )

        return AgentResult(
            agent=agent_name,
            response=response,
            next_agent=next_agent,
            finished=finished,
            confidence=confidence,
            tool_requests=tool_requests,
        )

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    async def run(
        self,
        state: AgentState,
    ) -> AgentResult:
        """
        Execute the current ULTRON agent.

        Flow:

            State
              ↓
            Agent config
              ↓
            Prompt
              ↓
            Ollama
              ↓
            JSON parsing
              ↓
            Transition validation
              ↓
            AgentResult
        """

        agent_name = (
            state.current_agent
        )

        config = self.get_agent_config(
            agent_name
        )

        model = config.get(
            "model"
        )

        if not model:

            raise AgentConfigurationError(
                f"Agent '{agent_name}' has no model."
            )

        # When images are attached and the current agent is not
        # already the vision agent, use the vision model instead.
        if (
            state.images
            and agent_name != "vision"
        ):

            model = MODELS.get(
                "vision",
                model,
            )

            logger.info(
                "Images detected, using vision model: %s",
                model,
            )

        logger.info(
            "Running agent=%s model=%s "
            "loop=%s retry=%s",
            agent_name,
            model,
            state.loop_count,
            state.retry_count,
        )

        messages = self.build_messages(
            agent_name,
            state,
        )

        raw_response = await self.ollama.chat(
            model=model,
            messages=messages,
            temperature=DEFAULT_TEMPERATURE,
        )

        logger.debug(
            "Raw response from %s: %s",
            agent_name,
            raw_response,
        )

        data = extract_json_object(
            raw_response
        )

        result = self.validate_result(
            agent_name,
            data,
        )

        result.raw_response = raw_response

        result.metadata = {
            "model": model,
            "purpose": config.get(
                "purpose",
                "",
            ),
        }

        logger.info(
            "Agent=%s selected next_agent=%s "
            "finished=%s confidence=%.2f",
            agent_name,
            result.next_agent,
            result.finished,
            result.confidence,
        )

        return result


# ============================================================
# RETRY HELPER
# ============================================================

async def run_agent_with_retry(
    engine: AgentEngine,
    state: AgentState,
) -> AgentResult:
    """
    Execute an agent once.

    Retry policy is intentionally handled by main.py because
    main.py owns the global execution loop and effort limits.

    This helper exists primarily as a clean API boundary.
    """

    return await engine.run(
        state
    )


# ============================================================
# ENGINE FACTORY
# ============================================================

def create_agent_engine() -> AgentEngine:
    """
    Create a default ULTRON agent engine.
    """

    return AgentEngine()