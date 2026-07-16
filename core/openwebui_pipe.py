"""
title: LG Tutor (Socratic Guide)
author: J
version: 0.2.0
description: Bridges OpenWebUI to the LangGraph tutor FastAPI backend.
             Maps OpenWebUI's chat_id -> the graph's thread_id so each
             OpenWebUI conversation is its own persistent tutoring session.
             Streams the tutor's reply token-by-token and shows a status
             indicator while the evaluator/arc-planner phase runs.

INSTALL: OpenWebUI -> Admin Panel -> Functions -> (+) -> paste this file ->
         Save -> enable the toggle. Then set the Valves (gear icon) if your
         API isn't at the default URL. The tutor appears as a model named
         "LG Tutor" in the model picker.

NOTE: This file is the source of truth; the copy inside OpenWebUI's database
      must be re-pasted after edits here.
"""

import hashlib

import aiohttp
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        API_BASE_URL: str = Field(
            # Docker-network service name when both containers share a compose
            # network. For local dev outside Docker use http://localhost:8000;
            # for OpenWebUI-in-Docker -> host API use http://host.docker.internal:8000.
            default="http://lg-tutor-api:8000",
            description="Base URL of the tutor FastAPI server.",
        )
        API_KEY: str = Field(
            default="",
            description="Shared secret sent as X-API-Key. Must match the backend's TUTOR_API_KEY. Leave empty if the backend has auth disabled.",
        )
        REQUEST_TIMEOUT: int = Field(
            default=120,
            description="Seconds to wait for the tutor backend.",
        )
        STATUS_MESSAGE: str = Field(
            default="Reading your answer…",
            description="Status text shown while the tutor is thinking (before the reply starts streaming).",
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        # Registers the model shown in OpenWebUI's picker.
        return [{"id": "lg-tutor", "name": "LG Tutor"}]

    def _thread_id(self, body: dict, __metadata__: dict, __user__: dict) -> str:
        """chat_id is the stable per-conversation key; prefix with the user id
        so two users can never collide on a thread. Falls back to a hash of the
        first message if chat_id is missing (known OpenWebUI edge case)."""
        meta = __metadata__ or {}
        chat_id = meta.get("chat_id") or meta.get("session_id")
        if not chat_id:
            first = ""
            for m in body.get("messages", []):
                if m.get("role") == "user":
                    first = str(m.get("content", ""))
                    break
            chat_id = "fb-" + hashlib.sha256(first.encode()).hexdigest()[:16]
        user_id = (__user__ or {}).get("id", "anon")
        return f"{user_id}:{chat_id}"

    async def _emit_status(self, emitter, description: str, done: bool, hidden: bool = False):
        if emitter:
            await emitter(
                {
                    "type": "status",
                    "data": {"description": description, "done": done, "hidden": hidden},
                }
            )

    async def pipe(
        self,
        body: dict,
        __metadata__: dict = None,
        __user__: dict = None,
        __task__: str = None,
        __event_emitter__=None,
    ):
        # OpenWebUI routes background jobs (title generation, tags, etc.) to
        # the chat's model. Short-circuit them so they NEVER hit the graph and
        # advance the student's tutoring state.
        if __task__:
            yield "Socratic Tutoring Session"
            return

        # The graph is stateful (checkpointer); it only needs the newest
        # student message, not OpenWebUI's full replayed history.
        messages = body.get("messages", [])
        user_message = next(
            (
                str(m.get("content", ""))
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )

        payload = {
            "message": user_message,
            "thread_id": self._thread_id(body, __metadata__, __user__),
        }

        headers = {}
        if self.valves.API_KEY:
            headers["X-API-Key"] = self.valves.API_KEY

        # Fill the evaluator/arc-planner dead air with a visible status.
        await self._emit_status(__event_emitter__, self.valves.STATUS_MESSAGE, done=False)

        url = f"{self.valves.API_BASE_URL.rstrip('/')}/chat"
        timeout = aiohttp.ClientTimeout(total=self.valves.REQUEST_TIMEOUT)
        first_chunk = True

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        detail = (await response.text())[:300]
                        await self._emit_status(__event_emitter__, "", done=True, hidden=True)
                        yield f"⚠️ Tutor backend returned {response.status}: {detail}"
                        return

                    async for chunk in response.content.iter_any():
                        text = chunk.decode("utf-8", errors="ignore")
                        if not text:
                            continue
                        if first_chunk:
                            # First token has arrived — clear the status line.
                            await self._emit_status(__event_emitter__, "", done=True, hidden=True)
                            first_chunk = False
                        yield text
        except aiohttp.ClientError as e:
            await self._emit_status(__event_emitter__, "", done=True, hidden=True)
            yield f"⚠️ Could not reach the tutor backend: {e}"
            return
        except Exception as e:  # timeout & anything else — never leave status stuck
            await self._emit_status(__event_emitter__, "", done=True, hidden=True)
            yield f"⚠️ Tutor request failed: {e}"
            return

        # Safety: if the stream ended without a single chunk, clear the status.
        if first_chunk:
            await self._emit_status(__event_emitter__, "", done=True, hidden=True)
            yield "⚠️ The tutor sent an empty reply. Please try again."
