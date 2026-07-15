"""
title: LG Tutor (Socratic Guide)
author: J
version: 0.1.0
description: Bridges OpenWebUI to the LangGraph tutor FastAPI backend.
             Maps OpenWebUI's chat_id -> the graph's thread_id so each
             OpenWebUI conversation is its own persistent tutoring session.

INSTALL: OpenWebUI -> Admin Panel -> Functions -> (+) -> paste this file ->
         Save -> enable the toggle. Then set the Valves (gear icon) if your
         API isn't at the default URL. The tutor appears as a model named
         "LG Tutor" in the model picker.

NOTE: This file is the source of truth; the copy inside OpenWebUI's database
      must be re-pasted after edits here.
"""

import hashlib

import requests
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        API_BASE_URL: str = Field(
            # host.docker.internal reaches the host machine from inside
            # OpenWebUI's Docker container; use http://localhost:8000 if
            # OpenWebUI runs directly on the host.
            default="http://host.docker.internal:8000",
            description="Base URL of the tutor FastAPI server.",
        )
        REQUEST_TIMEOUT: int = Field(
            default=120,
            description="Seconds to wait for the tutor backend.",
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

    def pipe(
        self,
        body: dict,
        __metadata__: dict = None,
        __user__: dict = None,
        __task__: str = None,
    ):
        # OpenWebUI routes background jobs (title generation, tags, etc.) to
        # the chat's model. Short-circuit them so they NEVER hit the graph and
        # advance the student's tutoring state.
        if __task__:
            return "Socratic Tutoring Session"

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

        try:
            response = requests.post(
                f"{self.valves.API_BASE_URL.rstrip('/')}/chat",
                json=payload,
                stream=True,
                timeout=self.valves.REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            return f"⚠️ Could not reach the tutor backend: {e}"

        def stream():
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk

        return stream()
