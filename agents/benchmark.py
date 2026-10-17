#!/usr/bin/env python3
"""
ULTRON benchmark runner
=======================

Place this file beside main.py in the ULTRON project root.

Sends HTTP POST requests to the running ULTRON main.py application's /chat
endpoint. It does not import internal agent modules or initialize a second
agent engine. Start the server separately before running this script.

Outputs:
  benchmark_results/<run_id>.jsonl  One complete record per test
  benchmark_results/<run_id>.json   Run metadata and aggregate metrics
  benchmark_results/<run_id>.csv    Flat result summary

Examples:
  python benchmark.py
  python benchmark.py --effort high
  python benchmark.py --repeat 3
  python benchmark.py --limit 5 --output ./my_benchmarks
  python benchmark.py --reuse-session
  python benchmark.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import statistics
import sys
import time
import traceback
import uuid
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep benchmark output readable. ULTRON's own logger still reports errors.
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Edit/add prompts here to match your own evaluation goals.
TEST_CASES: list[dict[str, Any]] = [
    # General reasoning and mathematics
    {
        "id": "R01",
        "category": "reasoning_math",
        "prompt": "A server processes 120 requests per minute. Traffic increases by 25% and then decreases by 20%. What is the final request rate? Show every calculation.",
        "effort": "medium",
    },
    {
        "id": "R02",
        "category": "formal_reasoning",
        "prompt": "All robotic arms are machines. Some machines are autonomous. Can we conclude that some robotic arms are autonomous? Explain using formal logic.",
        "effort": "medium",
    },
    {
        "id": "R03",
        "category": "math_hardware",
        "prompt": "A battery is rated at 7.4 V and 2000 mAh. Estimate its nominal energy in watt-hours. Explain why the actual usable energy may differ.",
        "effort": "medium",
    },
    {
        "id": "R04",
        "category": "ml_reasoning",
        "prompt": "A model has 96% accuracy on a dataset where 99% of samples are negative. Explain why this accuracy could be misleading and calculate the accuracy of a classifier that always predicts negative.",
        "effort": "medium",
    },
    {
        "id": "R05",
        "category": "reliability_math",
        "prompt": "A system has three independent components with reliabilities 0.99, 0.95, and 0.90 connected in series. Calculate total reliability. Then calculate it if the first two components are redundant in parallel and the third remains in series. Show the formulas.",
        "effort": "high",
    },
    {
        "id": "R06",
        "category": "resource_planning",
        "prompt": "You have 8 GB of RAM available. A model needs 6 GB, a browser needs 2 GB, and the operating system and other services need 1.5 GB. Explain whether all can run simultaneously without memory pressure, and recommend a practical solution.",
        "effort": "medium",
    },

    # Coding and debugging
    {
        "id": "C01",
        "category": "coding",
        "prompt": "Write a Python function deduplicate(items) that removes duplicates while preserving original order. It must support unhashable elements such as dictionaries. Provide complete code and tests for empty input, repeated integers, and repeated dictionaries.",
        "effort": "high",
    },
    {
        "id": "C02",
        "category": "debugging",
        "prompt": "Find and fix every issue in this Python code. Explain the problems and provide a corrected version, including tests for an empty list, a single number, and multiple numbers:\n\ndef average(values):\n    total = 0\n    for i in range(len(values) + 1):\n        total += values[i]\n    return total / len(values)",
        "effort": "high",
    },
    {
        "id": "C03",
        "category": "async_coding",
        "prompt": "Implement a Python asyncio function that processes 100 URLs concurrently with a maximum of 10 requests in flight. Include timeouts, exception handling, and per-URL results. Explain why creating 100 unrestricted tasks may be undesirable.",
        "effort": "high",
    },
    {
        "id": "C04",
        "category": "api_database_design",
        "prompt": "Design a FastAPI endpoint for creating a chat message using PostgreSQL. Include Pydantic validation, SQL transaction handling, authentication considerations, duplicate-request protection, and appropriate HTTP status codes. Provide runnable example code and identify assumptions.",
        "effort": "high",
    },

    # Agent routing and effort classification
    {
        "id": "A01",
        "category": "routing_simple",
        "prompt": "Hi! How are you?",
        "effort": "low",
    },
    {
        "id": "A02",
        "category": "routing_general",
        "prompt": "Explain the difference between TCP and UDP.",
        "effort": "medium",
    },
    {
        "id": "A03",
        "category": "routing_coding",
        "prompt": "Debug this Python function and write unit tests for its edge cases.",
        "effort": "high",
    },
    {
        "id": "A04",
        "category": "routing_complex",
        "prompt": "Analyze three architectural options, compare security and performance trade-offs, and recommend one for a production application.",
        "effort": "high",
    },
    {
        "id": "A05",
        "category": "multi_step",
        "prompt": "Design, implement, test, and review a FastAPI service with JWT authentication and PostgreSQL persistence. State assumptions and work through the requirements systematically.",
        "effort": "high",
    },
    {
        "id": "A06",
        "category": "reasoning_uncertainty",
        "prompt": "Explain why a database query is slow, identify likely causes, and propose a measurement-driven debugging plan without assuming access to the database.",
        "effort": "high",
    },

    # Instruction following
    {
        "id": "I01",
        "category": "instruction_following",
        "prompt": "Explain unit testing using exactly three numbered items. Each item must contain exactly one sentence. Do not include a heading or introduction.",
        "effort": "low",
    },
    {
        "id": "I02",
        "category": "structured_output",
        "prompt": 'Return valid JSON only, with exactly these keys: "language", "purpose", and "example". Use Python as the language and describe a function that adds two integers. Do not use Markdown code fences.',
        "effort": "low",
    },
    {
        "id": "I03",
        "category": "uncertainty_handling",
        "prompt": 'A user says: "My application is slow." Provide a diagnostic response that distinguishes known facts from assumptions. Ask for the three most useful missing details before recommending a specific fix.',
        "effort": "medium",
    },

    # Multi-turn memory probes. Run with --reuse-session and preserve this order.
    {
        "id": "M01",
        "category": "memory_setup",
        "prompt": "I am designing a chat service using FastAPI, PostgreSQL, and WebSockets. The service must support offline message delivery. We will use this architecture for the next questions.",
        "effort": "medium",
    },
    {
        "id": "M02",
        "category": "memory_recall",
        "prompt": "Which backend framework, database, communication protocol, and delivery requirement did I specify in my previous message?",
        "effort": "medium",
    },
    {
        "id": "M03",
        "category": "memory_application",
        "prompt": "Based on the architecture I specified earlier, propose a database schema and message-delivery flow for offline users. Do not replace my chosen technologies.",
        "effort": "high",
    },

    # Security and reliability
    {
        "id": "S01",
        "category": "security",
        "prompt": "Explain how to store passwords securely in a FastAPI application. Discuss password hashing, secure verification, and relevant protections.",
        "effort": "medium",
    },
    {
        "id": "S02",
        "category": "security_review",
        "prompt": "Review a WebSocket chat service for authentication, authorization, and message-validation risks. Identify concrete risks and mitigations.",
        "effort": "high",
    },
    {
        "id": "S03",
        "category": "failure_recovery",
        "prompt": "Explain what a robust agent service should do when its model server is unavailable and a generation request fails repeatedly. Discuss bounded retries, error reporting, and graceful failure.",
        "effort": "medium",
    },
    {
        "id": "S04",
        "category": "calibration",
        "prompt": "Explain why an AI-generated answer should not be treated as verified merely because its reported confidence is high.",
        "effort": "medium",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_to_dict(value: Any) -> dict[str, Any]:
    """Support both Pydantic v1 and v2."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    return {"value": str(value)}


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def safe_mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


async def post_chat(
    base_url: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """POST a query to the running main.py /chat endpoint."""
    url = base_url.rstrip("/") + "/chat"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _request() -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
                if not 200 <= response.status < 300:
                    raise RuntimeError(f"HTTP {response.status}: {raw[:2000]}")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {error_body[:2000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not connect to ULTRON at {url}: {exc.reason}. "
                "Start main.py first."
            ) from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"/chat returned invalid JSON: {raw[:2000]}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"/chat returned {type(parsed).__name__}, expected a JSON object")
        return parsed

    return await asyncio.to_thread(_request)


async def run_benchmark(args: argparse.Namespace) -> int:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"{run_id}.jsonl"
    json_path = output_dir / f"{run_id}.json"
    csv_path = output_dir / f"{run_id}.csv"

    tests = TEST_CASES[: args.limit] if args.limit else list(TEST_CASES)
    if args.repeat > 1:
        tests = [
            {**case, "repeat_index": repeat_index}
            for repeat_index in range(1, args.repeat + 1)
            for case in tests
        ]
    else:
        tests = [{**case, "repeat_index": 1} for case in tests]

    records: list[dict[str, Any]] = []
    shared_session_id: str | None = None

    print(f"ULTRON HTTP benchmark run: {run_id}")
    print(f"Endpoint: {args.base_url.rstrip('/')}/chat")
    print(f"Tests: {len(tests)} | Output: {output_dir}")
    print("Sending queries to the running main.py application; no internal modules imported.")

    for index, case in enumerate(tests, start=1):
        requested_effort = args.effort or case.get("effort", "auto")
        started_at = utc_now()
        started = time.perf_counter()

        print(
            f"[{index:02d}/{len(tests):02d}] "
            f"{case['id']} | {case['category']} | effort={requested_effort}",
            flush=True,
        )

        record: dict[str, Any] = {
            "run_id": run_id,
            "test_id": case["id"],
            "repeat_index": case["repeat_index"],
            "category": case["category"],
            "prompt": case["prompt"],
            "requested_effort": requested_effort,
            "started_at": started_at,
            "status": "error",
        }

        try:
            payload: dict[str, Any] = {
                "query": case["prompt"],
                "effort": requested_effort,
            }
            if args.reuse_session and shared_session_id:
                payload["session_id"] = shared_session_id

            response_data = await post_chat(args.base_url, payload, args.timeout)
            elapsed_wall = round(time.perf_counter() - started, 4)

            if args.reuse_session:
                shared_session_id = response_data.get("session_id", shared_session_id)

            response_text = response_data.get("response", "")
            if not isinstance(response_text, str):
                response_text = str(response_text)

            record.update(
                {
                    "status": "success",
                    "finished_at": utc_now(),
                    "wall_time_seconds": elapsed_wall,
                    "response": response_text,
                    "session_id": response_data.get("session_id"),
                    "query_id": response_data.get("query_id"),
                    "resolved_effort": response_data.get("effort"),
                    "loops": response_data.get("loops"),
                    "retries": response_data.get("retries"),
                    "agents_used": response_data.get("agents_used", []),
                    "termination_reason": response_data.get("termination_reason"),
                    "confidence": response_data.get("confidence"),
                    "elapsed_seconds_reported": response_data.get("elapsed_seconds"),
                    "sources_used": response_data.get("sources_used", 0),
                    "tool_rounds": response_data.get("tool_rounds", 0),
                    "skill_required": response_data.get("skill_required", False),
                    "skills_used": response_data.get("skills_used", []),
                    "retrieval_failures": response_data.get("retrieval_failures", []),
                    "response_characters": len(response_text),
                    "response_words": len(response_text.split()),
                    "http_response": response_data,
                }
            )
            print(
                f"    {elapsed_wall:.2f}s | loops={record.get('loops')} "
                f"| retries={record.get('retries')} | agents={record.get('agents_used')}",
                flush=True,
            )
        except Exception as exc:
            record.update(
                {
                    "finished_at": utc_now(),
                    "wall_time_seconds": round(time.perf_counter() - started, 4),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"    ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)

        records.append(record)
        # Write each completed test immediately, preserving results on interruption.
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    successful = [r for r in records if r.get("status") == "success"]
    failed = [r for r in records if r.get("status") != "success"]
    latencies = [
        float(r["wall_time_seconds"])
        for r in successful
        if r.get("wall_time_seconds") is not None
    ]
    loop_counts = [
        float(r["loops"]) for r in successful
        if isinstance(r.get("loops"), (int, float))
    ]
    retry_counts = [
        float(r["retries"]) for r in successful
        if isinstance(r.get("retries"), (int, float))
    ]
    confidence_values = [
        float(r["confidence"]) for r in successful
        if isinstance(r.get("confidence"), (int, float))
    ]

    summary = {
        "run_id": run_id,
        "created_at": utc_now(),
        "endpoint": args.base_url.rstrip("/") + "/chat",
        "requested_effort_override": args.effort,
        "repeat": args.repeat,
        "reuse_session": args.reuse_session,
        "test_count": len(tests),
        "success_count": len(successful),
        "failure_count": len(failed),
        "success_rate_percent": round(100 * len(successful) / len(records), 2) if records else 0.0,
        "latency_seconds": {
            "mean": safe_mean(latencies),
            "median": round(statistics.median(latencies), 4) if latencies else None,
            "min": round(min(latencies), 4) if latencies else None,
            "max": round(max(latencies), 4) if latencies else None,
        },
        "mean_loops": safe_mean(loop_counts),
        "mean_retries": safe_mean(retry_counts),
        "mean_reported_confidence": safe_mean(confidence_values),
        "failures": [
            {
                "test_id": r.get("test_id"),
                "repeat_index": r.get("repeat_index"),
                "error_type": r.get("error_type"),
                "error": r.get("error"),
            }
            for r in failed
        ],
        "artifacts": {
            "jsonl": str(jsonl_path),
            "json": str(json_path),
            "csv": str(csv_path),
        },
        "notes": [
            "Responses come from the running main.py /chat endpoint.",
            "Confidence is application-reported and is not independently calibrated.",
            "Token counts and tokens/second are not available in the current ChatResponse schema.",
            "This runner records answers and execution metrics; it does not automatically grade semantic correctness.",
        ],
    }
    write_json(json_path, {"summary": summary, "results": records})

    csv_columns = [
        "run_id", "test_id", "repeat_index", "category", "status",
        "requested_effort", "resolved_effort", "wall_time_seconds",
        "elapsed_seconds_reported", "loops", "retries", "confidence",
        "termination_reason", "agents_used", "sources_used", "tool_rounds",
        "skill_required", "response_words", "response_characters",
        "prompt", "response", "error_type", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["agents_used"] = " -> ".join(row.get("agents_used") or [])
            writer.writerow(row)

    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Completed: {len(successful)}/{len(tests)}")
    print(f"Success rate: {summary['success_rate_percent']}%")
    print(f"Mean latency: {summary['latency_seconds']['mean']} seconds")
    print(f"Median latency: {summary['latency_seconds']['median']} seconds")
    print(f"Mean loops: {summary['mean_loops']} | Mean retries: {summary['mean_retries']}")
    print(f"JSONL: {jsonl_path}")
    print(f"JSON:  {json_path}")
    print(f"CSV:   {csv_path}")

    return 0 if not failed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the real ULTRON agent pipeline.")
    parser.add_argument(
        "--effort",
        choices=["auto", "low", "medium", "high", "max"],
        default=None,
        help="Override effort for every test. Default: use each test's configured effort.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running ULTRON main.py server (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="HTTP timeout in seconds for each query (default: 600).",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the entire test set (default: 1).")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N test cases (0 = all).")
    parser.add_argument(
        "--output",
        default="./benchmark_results",
        help="Output directory (default: ./benchmark_results).",
    )
    parser.add_argument(
        "--reuse-session",
        action="store_true",
        help="Reuse one session for all prompts; otherwise each prompt gets a fresh session.",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_benchmark(parse_args())))
