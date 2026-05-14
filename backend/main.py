"""FastAPI app entry point."""

from __future__ import annotations

# Compat shim: pinned langchain-core 0.3.x reads langchain.debug/verbose/llm_cache,
# but newer langchain may not export them. Set safe defaults before any import chain
# that touches langchain-core (e.g. via langgraph).
import langchain  # noqa: E402

for _attr, _default in (("debug", False), ("verbose", False), ("llm_cache", None)):
    if not hasattr(langchain, _attr):
        setattr(langchain, _attr, _default)

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.agents.supervisor import build_graph
from backend.api.routes import router as api_router
from backend.config import get_settings
from backend.llm.groq_client import GroqClient
from backend.memory.mem0_client import Mem0Client

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise long-lived dependencies once per process."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.app_log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("Starting agent-orchestrator-demo")
    client = GroqClient()
    memory = Mem0Client()
    app.state.client = client
    app.state.memory = memory
    app.state.graph = build_graph(client, memory)
    logger.info("Graph compiled — ready")
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agent Orchestrator Demo",
        version="0.1.0",
        description=(
            "Multi-agent social concierge: LangGraph supervisor + mem0-style "
            "long-term memory + Groq model tiering."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    # Serve the demo frontend at /
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/")
        async def root() -> FileResponse:
            return FileResponse(str(frontend_dir / "index.html"))

    return app


app = create_app()
