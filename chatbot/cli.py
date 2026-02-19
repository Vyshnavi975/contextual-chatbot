"""
cli.py
======

Interactive command-line interface for the contextual chatbot.

Ties together `memory.ConversationMemory` (persistent multi-turn history)
and `llm.LLMClient` (provider-agnostic generation, with an offline demo
fallback) into a simple REPL.

Run it with:
    python3 main.py
    python3 main.py --thread work --storage-dir ./data

In-chat commands (typed at the prompt):
    /help              Show available commands
    /new <name>        Switch to (or create) a different conversation thread
    /threads           List all saved conversation threads
    /history           Print the full stored history for the current thread
    /clear             Erase the current thread's history
    /exit, /quit       Leave the chat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .llm import LLMClient
from .memory import ConversationMemory, ROLE_ASSISTANT, ROLE_USER

BANNER = """\
==================================================
  Contextual Chatbot
==================================================
Type a message and press Enter to chat.
Type /help to see available commands.
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextual-chatbot",
        description="A command-line chatbot with persistent multi-turn memory.",
    )
    parser.add_argument(
        "--thread",
        default="default",
        help="Name of the conversation thread to use (default: 'default').",
    )
    parser.add_argument(
        "--storage-dir",
        default=None,
        help="Directory to store conversation JSON files "
        "(default: ~/.contextual_chatbot/conversations).",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=20,
        help="Max verbatim messages kept per thread before older ones are "
        "summarized (default: 20).",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "demo"],
        default=None,
        help="Force a specific LLM backend instead of auto-detecting from "
        "environment variables.",
    )
    parser.add_argument(
        "--message",
        "-m",
        default=None,
        help="Send a single message non-interactively and print the reply "
        "(useful for scripting/testing). Skips the interactive REPL.",
    )
    return parser


class ChatSession:
    """Wires together memory + LLM client and runs the REPL loop."""

    def __init__(
        self,
        thread_name: str = "default",
        storage_dir: Optional[str] = None,
        max_messages: int = 20,
        provider: Optional[str] = None,
    ) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.max_messages = max_messages
        self.memory = ConversationMemory(
            thread_name=thread_name,
            storage_dir=self.storage_dir,
            max_messages=max_messages,
        )
        self.llm = LLMClient(provider=provider)

    def switch_thread(self, thread_name: str) -> None:
        self.memory = ConversationMemory(
            thread_name=thread_name,
            storage_dir=self.storage_dir,
            max_messages=self.max_messages,
        )

    def handle_message(self, user_text: str) -> str:
        """Store the user's message, get a reply, store it, and return it."""
        self.memory.add_message(ROLE_USER, user_text)
        context = self.memory.get_context_for_llm()
        reply = self.llm.generate(context)
        self.memory.add_message(ROLE_ASSISTANT, reply)
        return reply

    def print_history(self) -> None:
        messages = self.memory.get_messages()
        if not messages:
            print("(no messages yet in this thread)")
            return
        for m in messages:
            label = {"user": "You", "assistant": "Bot", "summary": "Summary"}.get(
                m["role"], m["role"]
            )
            print(f"[{label}] {m['content']}")

    def run_repl(self) -> None:
        print(BANNER)
        print(f"Provider: {self.llm.provider}" + (" (offline demo mode)" if self.llm.provider == "demo" else ""))
        print(f"Thread:   {self.memory.thread_name}  ({len(self.memory)} stored message(s))")
        print()

        while True:
            try:
                user_text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_text:
                continue

            if user_text.startswith("/"):
                if self._handle_command(user_text):
                    break
                continue

            reply = self.handle_message(user_text)
            print(f"Bot: {reply}")

    def _handle_command(self, command_line: str) -> bool:
        """Handle a slash command. Returns True if the REPL should exit."""
        parts = command_line.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            print("Goodbye!")
            return True
        elif cmd == "/help":
            print(
                "\nCommands:\n"
                "  /new <name>   Switch to (or create) a conversation thread\n"
                "  /threads      List all saved conversation threads\n"
                "  /history      Show the full history of the current thread\n"
                "  /clear        Erase the current thread's history\n"
                "  /exit, /quit  Leave the chat\n"
            )
        elif cmd == "/new":
            if not arg:
                print("Usage: /new <thread-name>")
            else:
                try:
                    self.switch_thread(arg)
                    print(f"Switched to thread '{arg}' "
                          f"({len(self.memory)} stored message(s)).")
                except ValueError as e:
                    print(f"Error: {e}")
        elif cmd == "/threads":
            threads = ConversationMemory.list_threads(self.storage_dir)
            if threads:
                print("Saved threads: " + ", ".join(threads))
            else:
                print("No saved threads yet.")
        elif cmd == "/history":
            self.print_history()
        elif cmd == "/clear":
            self.memory.clear()
            print(f"Cleared history for thread '{self.memory.thread_name}'.")
        else:
            print(f"Unknown command: {cmd}. Type /help for a list of commands.")
        return False


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    session = ChatSession(
        thread_name=args.thread,
        storage_dir=args.storage_dir,
        max_messages=args.max_messages,
        provider=args.provider,
    )

    if args.message is not None:
        # Non-interactive single-shot mode: send one message, print reply, exit.
        reply = session.handle_message(args.message)
        print(reply)
        return 0

    session.run_repl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
