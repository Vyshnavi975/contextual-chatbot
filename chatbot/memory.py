"""
memory.py
=========

Persistent, multi-thread conversation memory for the contextual chatbot.

Design goals
------------
* Each named "thread" (conversation) is stored as its own JSON file on disk,
  so conversations survive across process restarts.
* History is capped so it doesn't grow unboundedly and blow past an LLM's
  context/token limits. When a thread gets long, the oldest messages are
  collapsed into a single rule-based "summary" entry instead of being
  dropped outright, so long-run context is preserved in a condensed form.
* No external dependencies -- this module only uses the Python standard
  library, so the storage/memory logic can be unit tested without any
  API keys or network access.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

# Roles used inside a conversation thread.
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SUMMARY = "summary"  # a condensed stand-in for older messages

# Default location where conversation threads are stored on disk.
DEFAULT_STORAGE_DIR = Path.home() / ".contextual_chatbot" / "conversations"

# A thread name may only contain characters that are safe for filenames.
_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _safe_thread_name(name: str) -> str:
    """Validate a thread name so it can be used directly as a filename."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Thread name must not be empty.")
    if not _VALID_NAME_RE.match(name):
        raise ValueError(
            "Thread name may only contain letters, numbers, '-' and '_' "
            f"(got: {name!r})."
        )
    return name


@dataclass
class Message:
    """A single turn in a conversation."""

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        return cls(
            role=data.get("role", ROLE_USER),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
        )


class ConversationMemory:
    """
    Manages a single named conversation thread: loading it from disk,
    appending new messages, saving it back, and keeping it within a
    manageable size via truncation + rule-based summarization.

    Parameters
    ----------
    thread_name:
        Logical name of the conversation (e.g. "default", "work-notes").
        Used as the JSON filename on disk.
    storage_dir:
        Directory that holds one JSON file per thread. Defaults to
        ``~/.contextual_chatbot/conversations``.
    max_messages:
        Maximum number of *full* messages to keep verbatim before the
        oldest ones are rolled up into a summary entry.
    max_chars_per_message:
        Safety cap on how many characters a single stored message may
        contain (very long pastes are truncated with a marker).
    """

    def __init__(
        self,
        thread_name: str = "default",
        storage_dir: Optional[Path] = None,
        max_messages: int = 20,
        max_chars_per_message: int = 4000,
    ) -> None:
        self.thread_name = _safe_thread_name(thread_name)
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
        self.max_messages = max_messages
        self.max_chars_per_message = max_chars_per_message
        self.messages: List[Message] = []

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #

    @property
    def path(self) -> Path:
        return self.storage_dir / f"{self.thread_name}.json"

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Load this thread's messages from disk, if a file exists."""
        if not self.path.exists():
            self.messages = []
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        except (json.JSONDecodeError, OSError):
            # Corrupt or unreadable file -- start fresh rather than crash.
            self.messages = []

    def save(self) -> None:
        """Persist the current message list to disk as JSON."""
        payload = {
            "thread_name": self.thread_name,
            "updated_at": time.time(),
            "messages": [m.to_dict() for m in self.messages],
        }
        # Write atomically: write to a temp file then replace, so a crash
        # mid-write never leaves a corrupt conversation file behind.
        tmp_path = self.path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.path)

    # ------------------------------------------------------------------ #
    # Mutating the conversation
    # ------------------------------------------------------------------ #

    def add_message(self, role: str, content: str) -> Message:
        """Append a message, enforce size limits, and save to disk."""
        if role not in (ROLE_USER, ROLE_ASSISTANT, ROLE_SUMMARY):
            raise ValueError(f"Unknown role: {role!r}")

        content = content if content is not None else ""
        if len(content) > self.max_chars_per_message:
            content = content[: self.max_chars_per_message] + " …[truncated]"

        message = Message(role=role, content=content)
        self.messages.append(message)
        self._truncate_if_needed()
        self.save()
        return message

    def clear(self) -> None:
        """Delete all messages in this thread (file stays, becomes empty)."""
        self.messages = []
        self.save()

    # ------------------------------------------------------------------ #
    # Truncation / summarization
    # ------------------------------------------------------------------ #

    def _truncate_if_needed(self) -> None:
        """
        Keep the stored history bounded.

        When the number of stored messages exceeds ``max_messages``, the
        oldest messages (excluding any existing summary at the front) are
        collapsed into a single rule-based summary message so the thread
        keeps a condensed memory of everything that happened, without the
        raw text growing forever or blowing past an LLM's context window.
        """
        if len(self.messages) <= self.max_messages:
            return

        # Keep an existing leading summary (if any) separate from the
        # messages we are about to fold in.
        has_leading_summary = bool(self.messages) and self.messages[0].role == ROLE_SUMMARY
        existing_summary = self.messages[0] if has_leading_summary else None
        rest = self.messages[1:] if has_leading_summary else self.messages[:]

        # How many of the most recent messages to keep verbatim.
        keep_tail = max(self.max_messages - 1, 1)
        to_summarize = rest[:-keep_tail] if len(rest) > keep_tail else []
        tail = rest[-keep_tail:] if len(rest) > keep_tail else rest

        if not to_summarize:
            # Nothing old enough to fold in yet.
            return

        new_summary_text = self._summarize_messages(to_summarize, existing_summary)
        summary_message = Message(role=ROLE_SUMMARY, content=new_summary_text)
        self.messages = [summary_message] + tail

    @staticmethod
    def _summarize_messages(messages: List[Message], existing_summary: Optional[Message]) -> str:
        """
        Produce a compact, rule-based (non-LLM) summary of older messages.

        This keeps memory management fully functional offline / without an
        API key. Each folded-in message contributes a short clipped snippet
        tagged with its role, so the gist of the conversation is retained
        even after the verbatim text is dropped.
        """
        prefix = "[Summary of earlier conversation] "
        parts: List[str] = []
        if existing_summary and existing_summary.content:
            prior = existing_summary.content
            if prior.startswith(prefix):
                prior = prior[len(prefix):]
            parts.append(prior)

        for m in messages:
            snippet = " ".join(m.content.split())  # collapse whitespace
            if len(snippet) > 120:
                snippet = snippet[:120] + "…"
            who = "User" if m.role == ROLE_USER else "Assistant"
            parts.append(f"{who} said: {snippet}")

        combined = " | ".join(parts)
        # Cap the overall summary length too, so it can never itself grow
        # without bound across many truncation passes.
        max_summary_chars = 2000
        if len(combined) > max_summary_chars:
            combined = "…" + combined[-max_summary_chars:]
        return f"{prefix}{combined}"

    # ------------------------------------------------------------------ #
    # Reading the conversation
    # ------------------------------------------------------------------ #

    def get_messages(self) -> List[Dict]:
        """Return all stored messages (including any summary) as dicts."""
        return [m.to_dict() for m in self.messages]

    def get_context_for_llm(self) -> List[Dict[str, str]]:
        """
        Build the message list to send to an LLM: role/content pairs only,
        with any summary message rendered as a leading system-style note.
        This is what keeps requests within a reasonable token budget.
        """
        context = []
        for m in self.messages:
            if m.role == ROLE_SUMMARY:
                context.append({"role": "system", "content": m.content})
            else:
                context.append({"role": m.role, "content": m.content})
        return context

    def __len__(self) -> int:
        return len(self.messages)

    # ------------------------------------------------------------------ #
    # Thread discovery (class-level helpers, not tied to one instance)
    # ------------------------------------------------------------------ #

    @staticmethod
    def list_threads(storage_dir: Optional[Path] = None) -> List[str]:
        """Return the names of all conversation threads found on disk."""
        directory = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
        if not directory.exists():
            return []
        names = sorted(p.stem for p in directory.glob("*.json"))
        return names

    @staticmethod
    def delete_thread(thread_name: str, storage_dir: Optional[Path] = None) -> bool:
        """Delete a thread's JSON file from disk. Returns True if removed."""
        directory = Path(storage_dir) if storage_dir else DEFAULT_STORAGE_DIR
        name = _safe_thread_name(thread_name)
        path = directory / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False
