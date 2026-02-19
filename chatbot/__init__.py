"""
Contextual Chatbot
===================

A small, dependency-light command-line chatbot that remembers multi-turn
conversation history across sessions.

Modules:
    memory  - Conversation storage, threading, and history truncation/summarization.
    llm     - Provider-agnostic LLM client (Anthropic / OpenAI / offline demo mode).
    cli     - Interactive command-line interface tying memory + llm together.
"""

__version__ = "1.0.0"
