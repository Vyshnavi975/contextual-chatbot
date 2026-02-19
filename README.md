# Contextual Chatbot

A command-line (and optional Flask web) chatbot with **persistent, multi-turn
conversation memory**. Conversations are stored to disk as JSON, organized
into named threads, and automatically summarized/truncated once they get
long, so the bot stays within a reasonable context/token budget no matter
how long you talk to it.

It runs **end-to-end with zero configuration**: if no `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY` is set, it falls back to a clearly-labeled offline
"demo mode" that still exercises the full memory/threading pipeline without
any network access or API key.

---

## Features

- **Multi-turn memory** — every message (user + assistant) is stored and
  replayed as context on the next turn, so the bot remembers what you told it
  earlier in the conversation.
- **Persistent, on-disk storage** — each conversation thread is a plain JSON
  file (`~/.contextual_chatbot/conversations/<thread>.json` by default), so
  history survives across restarts.
- **Multiple named threads** — keep separate conversations (`work`,
  `personal`, `demo`, ...) side by side; switch between them with `/new`.
- **Automatic truncation + summarization** — once a thread grows past a
  configurable number of messages, the oldest ones are folded into a
  compact rule-based summary entry instead of being kept verbatim forever,
  keeping requests to the LLM bounded in size.
- **Provider-agnostic LLM backend** — picks Anthropic or OpenAI automatically
  based on which API key is set, or forces a specific provider with
  `--provider`.
- **Offline demo mode** — works out of the box with **no API key and no
  network access**, using a small rule-based responder. Every demo reply is
  clearly prefixed with `[demo mode]`.
- **Optional Flask web UI** (`app.py`) — the same memory/LLM logic exposed
  behind a tiny single-page chat interface and a JSON API.
- **Unit tested** — the storage/memory logic and the demo responder have
  unit tests that pass with no API key required.

---

## Project layout

```
contextual-chatbot/
├── main.py               # CLI entry point
├── app.py                # optional Flask web UI (uses the same chatbot package)
├── chatbot/
│   ├── __init__.py
│   ├── memory.py          # ConversationMemory: JSON storage, threads, truncation/summarization
│   ├── llm.py              # LLMClient: Anthropic / OpenAI / offline demo mode
│   └── cli.py               # interactive REPL + argument parsing
├── tests/
│   ├── test_memory.py     # unit tests for storage/threading/truncation logic
│   └── test_llm.py         # unit tests for the offline demo responder
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Setup / install

Requires **Python 3.8+**. No dependencies are required to run in demo mode.

```bash
git clone <this-repo-url>
cd contextual-chatbot

# Optional: create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Optional: install real-LLM and/or web-UI dependencies
pip install -r requirements.txt
# or install only what you need, e.g.:
#   pip install anthropic
#   pip install openai
#   pip install flask
```

### Plugging in a real API key

The app auto-detects the provider from environment variables — set **one**
of these before running:

```bash
# Use Anthropic (Claude)
export ANTHROPIC_API_KEY="sk-ant-..."

# Use OpenAI (GPT)
export OPENAI_API_KEY="sk-..."
```

Optional overrides:

```bash
export ANTHROPIC_MODEL="claude-3-5-haiku-20241022"   # default shown
export OPENAI_MODEL="gpt-4o-mini"                     # default shown
```

If neither key is set (or the matching SDK package isn't installed), the
app automatically falls back to **offline demo mode** — no crash, no silent
failure, just a clearly labeled `[demo mode]` responder.

---

## Usage

### Command-line, interactive

```bash
python3 main.py
```

```
==================================================
  Contextual Chatbot
==================================================
Type a message and press Enter to chat.
Type /help to see available commands.

Provider: demo (offline demo mode)
Thread:   default  (0 stored message(s))

You: Hello, my name is Vyshnavi.
Bot: [demo mode] Nice to meet you, Vyshnavi! I'll remember that for this conversation.
You: Thanks a lot!
Bot: [demo mode] You're welcome!
You: /history
[User] Hello, my name is Vyshnavi.
[Bot] [demo mode] Nice to meet you, Vyshnavi! I'll remember that for this conversation.
[User] Thanks a lot!
[Bot] [demo mode] You're welcome!
You: /exit
Goodbye!
```

In-chat commands:

| Command        | Description                                            |
|----------------|----------------------------------------------------------|
| `/help`        | Show available commands                                  |
| `/new <name>`  | Switch to (or create) a different conversation thread    |
| `/threads`     | List all saved conversation threads                      |
| `/history`     | Print the full stored history for the current thread     |
| `/clear`       | Erase the current thread's history                       |
| `/exit`, `/quit` | Leave the chat                                          |

### Command-line flags

```bash
python3 main.py --thread work-notes            # use/create a named thread
python3 main.py --storage-dir ./data            # custom storage location
python3 main.py --max-messages 30                # raise the truncation threshold
python3 main.py --provider demo                  # force offline demo mode
python3 main.py --message "Hello there!"          # single-shot, non-interactive
```

Single-shot example (useful for scripting/testing):

```bash
$ python3 main.py --message "What is Python?"
[demo mode] That's a good question! In demo mode I can't reach a real language model, but I can see this is turn 1 of our conversation.
```

### Multiple threads example

```bash
$ python3 main.py --thread work -m "Remind me to review the PR"
[demo mode] Got it -- "Remind me to review the PR". I'm keeping track of our conversation so far.

$ python3 main.py --thread personal -m "What's a good weekend recipe?"
[demo mode] That's a good question! In demo mode I can't reach a real language model, but I can see this is turn 1 of our conversation.

$ python3 -c "from chatbot.memory import ConversationMemory as C; print(C.list_threads())"
['personal', 'work']
```

### Optional Flask web UI

```bash
pip install flask
python3 app.py
# then open http://127.0.0.1:5000
```

`app.py` exposes:

- `GET /` — a minimal chat page (HTML + vanilla JS)
- `POST /api/chat` — `{"thread": "...", "message": "..."}` → `{"reply": "...", "provider": "..."}`
- `GET /api/history/<thread>` — full stored history for a thread
- `GET /api/threads` — list of saved threads

---

## Architecture overview

```
                ┌────────────┐
   user input → │   cli.py    │ → prints reply
                │  (REPL /    │
                │  argparse)  │
                └─────┬──────┘
                      │
        ┌─────────────┼──────────────┐
        ▼                            ▼
┌───────────────┐            ┌───────────────┐
│  memory.py     │            │   llm.py       │
│                │            │                │
│ ConversationMemory          │ LLMClient       │
│ - load/save JSON            │ - auto-detects  │
│ - add_message()             │   Anthropic /   │
│ - truncate + summarize      │   OpenAI / demo │
│ - list/delete threads       │ - generate()    │
└───────┬───────┘            └───────┬────────┘
        │                            │
        ▼                            ▼
  <thread>.json file          Anthropic / OpenAI
  on disk                     API, or offline
                               rule-based demo
```

1. **`chatbot/memory.py`** owns everything about *what has been said*:
   reading/writing a thread's JSON file, appending new messages, enforcing
   size limits, and — once a thread exceeds `max_messages` — folding the
   oldest messages into a single rule-based summary entry (role
   `"summary"`) so the stored history (and what gets sent to the LLM) stays
   bounded no matter how long the conversation runs.
2. **`chatbot/llm.py`** owns *how a reply is generated*: it auto-detects an
   available provider from environment variables (`ANTHROPIC_API_KEY` /
   `OPENAI_API_KEY`), lazily imports the matching optional SDK, and exposes
   one `generate(messages) -> str` method. If no provider is configured it
   uses a small rule-based responder (greeting/question/thanks/farewell
   pattern matching) so the whole app still works with zero setup.
3. **`chatbot/cli.py`** is the glue: an argparse-based entry point plus a
   REPL loop that reads user input, calls `ConversationMemory.add_message`,
   asks `LLMClient.generate` for a reply using the (possibly summarized)
   history, stores the reply, and prints it — along with slash commands for
   switching/listing/clearing threads.
4. **`app.py`** is an optional, independent Flask front-end that imports the
   same `chatbot.memory` / `chatbot.llm` modules, proving the core logic is
   fully decoupled from the interface.

### How truncation/summarization works

Each `ConversationMemory` is created with a `max_messages` cap (default 20).
Every time a message is added:

1. If the thread has more than `max_messages` stored, the oldest messages
   (past however many are needed to stay under the cap) are removed from
   the verbatim list.
2. Those removed messages are collapsed into a short, clipped summary line
   per message (e.g. `User said: ...`), joined together, and stored as a
   single leading message with role `"summary"`.
3. If a summary already exists at the front of the thread, its text is
   merged with the newly-removed messages rather than replaced, so context
   from earlier summarization passes is not lost.
4. When building the message list to send to an LLM
   (`get_context_for_llm()`), the summary entry is mapped to a `"system"`
   role so the model treats it as background context rather than a literal
   turn in the conversation.

This is a lightweight, dependency-free (no LLM call needed) approach —
good enough to bound token usage and works identically whether or not an
API key is configured.

---

## Running the tests

```bash
python3 -m pytest
# or, without pytest installed:
python3 -m unittest discover
```

Expected output:

```
17 passed in 0.05s
```

The tests cover:

- `tests/test_memory.py` — persisting/reloading messages, independent named
  threads, clearing and deleting threads, invalid thread names, corrupt
  JSON recovery, per-message character truncation, and the
  truncation/summarization behavior over a long conversation.
- `tests/test_llm.py` — offline demo-mode provider selection and its
  rule-based response patterns (greeting, question, farewell, empty input).

No API key or network access is required to run the test suite.

---

## License

MIT — see [LICENSE](LICENSE).
