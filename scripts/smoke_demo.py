"""End-to-end smoke demo.

Run with a real GROQ_API_KEY in the environment to see the full pipeline:

    GROQ_API_KEY=gsk_... python scripts/smoke_demo.py

Three user turns are sent with the same user_id so you can watch memory
accumulate across calls.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Make sure we can import backend.* even if run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.agents.supervisor import build_graph  # noqa: E402
from backend.llm.groq_client import GroqClient  # noqa: E402
from backend.memory.mem0_client import Mem0Client  # noqa: E402

USER_ID = "smoke_demo_user"

TURNS = [
    "Hey, I want to plan an interesting weekend. I'm into jazz and indie films.",
    "Cool — who from my circle could I invite to come along?",
    "Suggest one specific event I should actually book tonight.",
]


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


async def main() -> int:
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Aborting smoke demo.", file=sys.stderr)
        print("Sign up: https://console.groq.com/keys", file=sys.stderr)
        return 1

    client = GroqClient()
    memory = Mem0Client(db_path="./chroma_db_smoke")
    memory.reset(USER_ID)  # fresh slate for the demo
    graph = build_graph(client, memory)

    for i, msg in enumerate(TURNS, 1):
        banner(f"Turn {i}: {msg}")
        result = await graph.ainvoke(
            {
                "user_id": USER_ID,
                "message": msg,
                "agent_outputs": [],
                "trace": [],
            }
        )
        print(f"\n[intent] selected_agents = {result.get('selected_agents')}")
        print(f"[memory] hits used = {len(result.get('memory_context', []))}")
        for hit in result.get("memory_context", []):
            print(f"   - {hit.get('text')}")
        print(f"\n[reply]\n{result.get('final_response')}")
        print("\n[trace]")
        for line in result.get("trace", []):
            print(f"   · {line}")

    banner("Final stored memory for this user")
    for item in memory.get_all(USER_ID):
        print(f" - {item['text']}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
