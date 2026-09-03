"""
llm.py
======

Provider-agnostic LLM client for the contextual chatbot.

Uses OpenAI's API by default. If it isn't configured, falls back to an
offline demo mode -- a small rule-based / echo responder that requires no
API key and no network access, so the whole project runs end-to-end out
of the box.

Resolution order (first one that is configured wins):
    1. OpenAI     (OPENAI_API_KEY set, `openai` package installed)
    2. Offline demo mode

The public surface is a single class, `LLMClient`, with one method,
`generate(messages, system_prompt=None) -> str`, so `cli.py` doesn't need
to know or care which backend is actually in use.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Use the conversation history "
    "provided to stay consistent with what has already been discussed."
)


class LLMClient:
    """
    Wraps whichever LLM backend is available (OpenAI, or an offline demo
    responder) behind one simple `generate()` method.

    Attributes
    ----------
    provider: str
        One of "openai" or "demo" -- indicates which backend was
        actually selected, so the CLI can label output clearly
        (especially important for demo mode).
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        provider:
            Force a specific backend ("openai" or "demo"). If None,
            auto-detect based on environment variables and installed
            packages.
        model:
            Override the default model name for the chosen provider.
        """
        self.provider = provider or self._detect_provider()
        self.model = model or self._default_model_for(self.provider)
        self._client = None  # lazily created SDK client, if applicable

        if self.provider == "openai":
            self._client = self._build_openai_client()
        # "demo" provider needs no client.

    # ------------------------------------------------------------------ #
    # Provider detection / setup
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_provider() -> str:
        """Pick a provider based on which API key is present in the env."""
        if os.environ.get("OPENAI_API_KEY"):
            try:
                import openai  # noqa: F401
                return "openai"
            except ImportError:
                pass
        return "demo"

    @staticmethod
    def _default_model_for(provider: str) -> str:
        if provider == "openai":
            return os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        return "demo-echo-v1"

    @staticmethod
    def _build_openai_client():
        import openai  # imported lazily so the package is optional
        return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a reply given the conversation so far.

        Parameters
        ----------
        messages:
            List of {"role": "user"|"assistant"|"system", "content": str}
            dicts, oldest first. Any "system" entries (e.g. a folded-in
            memory summary) are treated as context, not as the persona
            prompt.
        system_prompt:
            The assistant's persona / instructions. Defaults to a generic
            helpful-assistant prompt.

        Returns
        -------
        str: the assistant's reply text.
        """
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        if self.provider == "openai":
            return self._generate_openai(messages, system_prompt)
        return self._generate_demo(messages, system_prompt)

    # ------------------------------------------------------------------ #
    # Backend-specific generation
    # ------------------------------------------------------------------ #

    def _generate_openai(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        chat_messages = [{"role": "system", "content": system_prompt}]
        for m in messages:
            role = "system" if m["role"] == "system" else m["role"]
            chat_messages.append({"role": role, "content": m["content"]})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------ #
    # Offline demo mode
    # ------------------------------------------------------------------ #

    def _generate_demo(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        """
        A small rule-based responder used when no API key is configured.

        It is intentionally simple: it recognizes a few conversational
        patterns (greetings, questions, thanks, farewells) and otherwise
        falls back to acknowledging the message while referencing how
        many turns of history it has access to -- demonstrating that
        conversation *memory* is working even without a real model.
        """
        user_turns = [m["content"] for m in messages if m["role"] == "user"]
        last_user_message = user_turns[-1] if user_turns else ""
        text = last_user_message.strip().lower()
        turn_count = len(user_turns)

        if not text:
            return "[demo mode] I didn't catch a message -- try typing something!"

        if "my name is" in text:
            name = last_user_message.split("my name is", 1)[-1].strip().split(".")[0]
            reply = f"Nice to meet you, {name}! I'll remember that for this conversation."
        elif any(g in text for g in ("hello", "hi", "hey", "greetings")):
            reply = "Hello! I'm running in offline demo mode (no API key configured)."
        elif any(text.startswith(w) for w in ("what", "why", "how", "who", "when", "where")) or text.endswith("?"):
            reply = (
                "That's a good question! In demo mode I can't reach a real "
                "language model, but I can see this is turn "
                f"{turn_count} of our conversation."
            )
        elif any(w in text for w in ("thanks", "thank you")):
            reply = "You're welcome!"
        elif any(w in text for w in ("bye", "goodbye", "see you")):
            reply = "Goodbye! This conversation has been saved to disk."
        else:
            reply = self._demo_echo(last_user_message, turn_count)

        return f"[demo mode] {reply}"

    @staticmethod
    def _demo_echo(message: str, turn_count: int) -> str:
        templates = [
            "I hear you saying: \"{msg}\" (that's message #{n} in this thread).",
            "Got it -- \"{msg}\". I'm keeping track of our conversation so far.",
            "Noted: \"{msg}\". Set OPENAI_API_KEY for real replies.",
        ]
        template = random.choice(templates)
        clipped = message if len(message) <= 200 else message[:200] + "…"
        return template.format(msg=clipped, n=turn_count)
