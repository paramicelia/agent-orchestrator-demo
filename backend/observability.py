"""LangSmith / LangChain tracing wiring.

Enabling is fully opt-in. If ``LANGCHAIN_TRACING_V2`` is not truthy *or*
``LANGSMITH_API_KEY`` is missing, every helper here becomes a no-op so the
demo continues to work offline. When tracing IS enabled, we forward the
relevant variables into the process environment so the LangChain callback
manager picks them up automatically — there's nothing to wrap manually.
"""

from __future__ import annotations

import logging
import os

from backend.config import get_settings

logger = logging.getLogger(__name__)

_TRACING_ENABLED: bool | None = None


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
