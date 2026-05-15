"""LangSmith / LangChain tracing wiring + per-node latency instrumentation.

Two concerns live in this module:

1. ``init_tracing()`` reads ``LANGCHAIN_TRACING_V2`` + ``LANGSMITH_API_KEY``
   and, when both are set, forwards them into ``os.environ`` so the
   LangChain callback manager wires LangSmith automatically. Without a key,
   it is a complete no-op so local runs and CI never hit the network.

2. ``traceable_node()`` wraps an async LangGraph node so:
     * the node is annotated with ``langsmith.traceable`` (no-op when
       ``langsmith`` is not installed or tracing is off)
     * its wall-clock latency is appended to the existing ``trace`` list in
       the ``AgentState`` ("agent:topic produced 412 chars in 1820ms")

This gives the demo node-level observability for multi-step AI workflows
which the JD asks for ("distributed tracing or node-level observability to
diagnose complex bottlenecks") without forcing a cloud dependency.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

from backend.config import get_settings

logger = logging.getLogger(__name__)

_TRACING_ENABLED: bool | None = None

T = TypeVar("T")
NodeFn = Callable[..., Awaitable[dict[str, Any]]]


# --------------------------------------------------------------------------- #
# LangSmith env-var bootstrap
# --------------------------------------------------------------------------- #


def init_tracing() -> bool:
    """Initialise LangSmith tracing if env says so. Returns True if enabled.

    Idempotent — safe to call from FastAPI lifespan and from scripts.
    """
    global _TRACING_ENABLED
    if _TRACING_ENABLED is not None:
        return _TRACING_ENABLED

    settings = get_settings()
    want = bool(settings.langchain_tracing_v2)
    api_key = settings.langsmith_api_key or os.environ.get("LANGSMITH_API_KEY", "")

    if not want or not api_key:
        _TRACING_ENABLED = False
        logger.info(
            "LangSmith tracing: disabled "
            "(LANGCHAIN_TRACING_V2=%s, LANGSMITH_API_KEY=%s)",
            want,
            "set" if api_key else "unset",
        )
        return False

    # LangChain's callback manager reads these directly from os.environ.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

    _TRACING_ENABLED = True
    logger.info(
        "LangSmith tracing: enabled (project=%s)", settings.langsmith_project
    )
    return True


def is_tracing_enabled() -> bool:
    """Report current tracing status without re-initialising."""
    return bool(_TRACING_ENABLED)


# --------------------------------------------------------------------------- #
# @traceable shim
# --------------------------------------------------------------------------- #


def _traceable(name: str | None = None, run_type: str = "chain"):
    """Return langsmith.traceable when available, else an identity decorator.

    Keeping this indirection means importers don't have to guard every
    decoration site — when ``langsmith`` is not installed (eg minimal CI
    image) or when tracing is disabled at runtime, the wrapped function
    behaves exactly like the unwrapped one.
    """
    try:
        from langsmith import traceable as _ls_traceable  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        def _noop(fn: T) -> T:
            return fn

        return _noop

    def _decorate(fn: T) -> T:
        # langsmith.traceable returns a wrapped callable; cast back to T so
        # the decoration is transparent to static checkers.
        return cast(T, _ls_traceable(run_type=run_type, name=name or getattr(fn, "__name__", "node"))(fn))

    return _decorate


def traceable_node(name: str) -> Callable[[NodeFn], NodeFn]:
    """Wrap a LangGraph async node with LangSmith tracing + latency timing.

    The wrapper:

    * Applies ``@langsmith.traceable(run_type="chain", name=name)`` if the
      ``langsmith`` package is importable. Otherwise it's a no-op.
    * Measures wall-clock latency around the inner call and APPENDS a
      ``"trace"`` entry of the form ``"node:<name> took 1820ms"`` to the
      result dict so the frontend side-panel can render per-node timings.

    The wrapped function MUST return a ``dict[str, Any]`` (LangGraph node
    contract); any existing ``"trace"`` list inside the return value is
    preserved — the latency line is appended after the node's own lines.
    """

    def _decorator(fn: NodeFn) -> NodeFn:
        traced_fn = _traceable(name=name, run_type="chain")(fn)

        @functools.wraps(fn)
        async def _wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
            loop = asyncio.get_event_loop()
            t0 = loop.time()
            result = await traced_fn(*args, **kwargs)
            elapsed_ms = int((loop.time() - t0) * 1000)
            if not isinstance(result, dict):
                # Defensive: a node returning a non-dict would break LangGraph
                # anyway, but we don't want to crash here either.
                return result
            existing = list(result.get("trace") or [])
            existing.append(f"node:{name} took {elapsed_ms}ms")
            new_result = dict(result)
            new_result["trace"] = existing
            # node_latencies is a flat list of (name, ms) pairs — the
            # AgentState reducer ``operator.add`` merges entries from
            # fan-out branches automatically, and the side-panel UI
            # renders them as an aggregate breakdown.
            existing_latencies = list(new_result.get("node_latencies") or [])
            existing_latencies.append({"name": name, "ms": elapsed_ms})
            new_result["node_latencies"] = existing_latencies
            return new_result

        return _wrapped

    return _decorator
