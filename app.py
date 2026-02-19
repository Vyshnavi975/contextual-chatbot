#!/usr/bin/env python3
"""
app.py
======

Optional simple Flask web UI for the Contextual Chatbot.

This is a thin wrapper around the same `chatbot.memory` and `chatbot.llm`
modules used by the CLI -- it demonstrates that the chatbot's core logic
is independent of the interface. Flask is an optional dependency; this
file is only needed if you want the web UI (the CLI works without it).

Run with:
    pip install flask
    python3 app.py
Then open http://127.0.0.1:5000 in your browser.

Endpoints:
    GET  /                      Chat page (simple HTML form)
    POST /api/chat              {"thread": str, "message": str} -> {"reply": str}
    GET  /api/history/<thread>  -> {"messages": [...]}
    GET  /api/threads           -> {"threads": [...]}
"""

from __future__ import annotations

try:
    from flask import Flask, jsonify, request
except ImportError as exc:  # pragma: no cover - only hit if flask isn't installed
    raise SystemExit(
        "Flask is not installed. Run `pip install flask` to use the web UI, "
        "or use `python3 main.py` for the command-line chatbot instead."
    ) from exc

from chatbot.llm import LLMClient
from chatbot.memory import ROLE_ASSISTANT, ROLE_USER, ConversationMemory

app = Flask(__name__)

# One shared LLM client for the whole process (provider auto-detected once).
llm_client = LLMClient()

INDEX_HTML = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Contextual Chatbot</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
    #log { border: 1px solid #ccc; border-radius: 8px; padding: 12px; height: 420px; overflow-y: auto; margin-bottom: 12px; background: #fafafa; }
    .msg { margin: 8px 0; line-height: 1.4; }
    .user { color: #1a4; }
    .bot { color: #14a; }
    .role { font-weight: 600; margin-right: 6px; }
    form { display: flex; gap: 8px; }
    input[type=text] { flex: 1; padding: 8px; font-size: 1rem; }
    button { padding: 8px 16px; font-size: 1rem; cursor: pointer; }
    .banner { font-size: 0.85rem; color: #666; margin-bottom: 12px; }
  </style>
</head>
<body>
  <h1>Contextual Chatbot</h1>
  <div class="banner" id="provider-banner">Provider: loading...</div>
  <div id="log"></div>
  <form id="chat-form">
    <input type="text" id="message" placeholder="Type a message..." autocomplete="off" autofocus>
    <button type="submit">Send</button>
  </form>
  <script>
    const THREAD = "web-default";
    const log = document.getElementById('log');
    const form = document.getElementById('chat-form');
    const input = document.getElementById('message');
    const banner = document.getElementById('provider-banner');

    function addMessage(role, content) {
      const div = document.createElement('div');
      div.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
      const label = role === 'user' ? 'You' : (role === 'summary' ? 'Summary' : 'Bot');
      div.innerHTML = '<span class="role">' + label + ':</span>' + content;
      log.appendChild(div);
      log.scrollTop = log.scrollHeight;
    }

    async function loadHistory() {
      const res = await fetch('/api/history/' + THREAD);
      const data = await res.json();
      log.innerHTML = '';
      data.messages.forEach(m => addMessage(m.role, m.content));
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      addMessage('user', text);
      input.value = '';
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({thread: THREAD, message: text})
      });
      const data = await res.json();
      addMessage('assistant', data.reply || ('Error: ' + data.error));
      banner.textContent = 'Provider: ' + (data.provider || 'unknown');
    });

    loadHistory();
    banner.textContent = 'Provider: (send a message to detect)';
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    thread_name = data.get("thread", "web-default")
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "message must not be empty"}), 400

    try:
        memory = ConversationMemory(thread_name=thread_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    memory.add_message(ROLE_USER, message)
    reply = llm_client.generate(memory.get_context_for_llm())
    memory.add_message(ROLE_ASSISTANT, reply)

    return jsonify({"reply": reply, "provider": llm_client.provider})


@app.route("/api/history/<thread_name>")
def api_history(thread_name):
    try:
        memory = ConversationMemory(thread_name=thread_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"messages": memory.get_messages()})


@app.route("/api/threads")
def api_threads():
    return jsonify({"threads": ConversationMemory.list_threads()})


if __name__ == "__main__":
    app.run(debug=True)
