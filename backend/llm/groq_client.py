"""Async Groq client wrapper with model tiering helpers.

Model tiering rationale
-----------------------
Lightweight tasks (intent classification, simple routing decisions, schema
extraction) should run on the cheap, fast 8B model. Reasoning-heavy tasks
(agent responses, summarisation, planning) get the 70B model. Routing this
explicitly cuts cost by ~10x without quality loss on simple tasks.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from groq import AsyncGroq

from backend.config import get_settings

logger = logging.getLogger(__name__)


class GroqClient:
    """Thin async wrapper around the Groq SDK with tiering."""

    SMART_MODEL_DEFAULT = "llama-3.3-70b-versatile"
    LITE_MODEL_DEFAULT = "llama-3.1-8b-instant"

    def __init__(
        self,
        api_key: str | None = None,
        smart_model: str | None = None,
        lite_model: str | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.groq_api_key
        self.smart_model = smart_model or settings.groq_smart_model or self.SMART_MODEL_DEFAULT
        self.lite_model = lite_model or settings.groq_lite_model or self.LITE_MODEL_DEFAULT
        # Lazy-init the SDK client so import-time succeeds without a key.
        self._client: AsyncGroq | None = None

    def _ensure_client(self) -> AsyncGroq:
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to .env or your environment. "
                "Sign up at https://console.groq.com/keys"
            )
        if self._client is None:
            self._client = AsyncGroq(api_key=self.api_key)
        return self._client

    async def call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.4,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Single LLM call returning the assistant message content."""
        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        logger.debug("groq.call model=%s messages=%d", model, len(messages))
        resp = await client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        return content

    async def call_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        """Single tool-aware LLM call.

        Returns a dict shaped like::

            {
                "content": str | None,
                "tool_calls": [
                    {"id": str, "name": str, "arguments": dict[str, Any]},
                    ...
                ],
                "finish_reason": str,
            }

        Groq follows the OpenAI tool-use protocol, so callers can implement
        the standard ``model -> tool_call -> tool result -> model`` loop on
        top of this single primitive. See ``backend/agents/event_agent.py``.
        """
        import json

        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        logger.debug(
            "groq.call_with_tools model=%s messages=%d tools=%d",
            model,
            len(messages),
            len(tools),
        )
        resp = await client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[dict[str, Any]] = []
        raw_calls = getattr(msg, "tool_calls", None) or []
        for tc in raw_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                logger.warning("call_with_tools: bad JSON arguments: %r", tc.function.arguments)
                args = {}
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                }
            )
        return {
            "content": msg.content,
            "tool_calls": tool_calls,
            "finish_reason": choice.finish_reason,
        }

    async def smart(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ) -> str:
        """Reasoning-heavy call (70B tier)."""
        messages = self._build_messages(prompt, system)
        return await self.call(
            self.smart_model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def lite(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> str:
        """Cheap, fast call (8B tier). Use for classifiers/routers."""
        messages = self._build_messages(prompt, system)
        response_format = {"type": "json_object"} if json_mode else None
        return await self.call(
            self.lite_model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    async def lite_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """Structured JSON output via the lite tier."""
        raw = await self.lite(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("lite_json: failed to parse JSON: %s | raw=%r", exc, raw)
            return {}

    @staticmethod
    def _build_messages(prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages
