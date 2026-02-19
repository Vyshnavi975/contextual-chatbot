"""
Unit tests for chatbot.memory.

These tests require no API key and no network access -- they only
exercise the on-disk JSON storage and truncation/summarization logic.
Run with either:
    python3 -m pytest
    python3 -m unittest discover
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from chatbot.memory import (
    ROLE_ASSISTANT,
    ROLE_SUMMARY,
    ROLE_USER,
    ConversationMemory,
)


class TestConversationMemoryBasics(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="chatbot_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_add_message_and_persist_to_disk(self):
        mem = ConversationMemory(thread_name="alpha", storage_dir=self.tmp_dir)
        mem.add_message(ROLE_USER, "Hello there")
        mem.add_message(ROLE_ASSISTANT, "Hi! How can I help?")

        self.assertEqual(len(mem), 2)
        self.assertTrue(mem.path.exists())

        # Reload from disk in a fresh instance and confirm it round-trips.
        mem2 = ConversationMemory(thread_name="alpha", storage_dir=self.tmp_dir)
        messages = mem2.get_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], ROLE_USER)
        self.assertEqual(messages[0]["content"], "Hello there")
        self.assertEqual(messages[1]["role"], ROLE_ASSISTANT)
        self.assertEqual(messages[1]["content"], "Hi! How can I help?")

    def test_multiple_named_threads_are_independent(self):
        mem_a = ConversationMemory(thread_name="thread-a", storage_dir=self.tmp_dir)
        mem_b = ConversationMemory(thread_name="thread-b", storage_dir=self.tmp_dir)

        mem_a.add_message(ROLE_USER, "message in A")
        mem_b.add_message(ROLE_USER, "message in B")

        self.assertEqual(len(mem_a), 1)
        self.assertEqual(len(mem_b), 1)
        self.assertNotEqual(mem_a.get_messages(), mem_b.get_messages())

        threads = ConversationMemory.list_threads(self.tmp_dir)
        self.assertIn("thread-a", threads)
        self.assertIn("thread-b", threads)

    def test_clear_empties_history_but_keeps_file(self):
        mem = ConversationMemory(thread_name="clearme", storage_dir=self.tmp_dir)
        mem.add_message(ROLE_USER, "hi")
        mem.clear()
        self.assertEqual(len(mem), 0)

        mem2 = ConversationMemory(thread_name="clearme", storage_dir=self.tmp_dir)
        self.assertEqual(len(mem2), 0)

    def test_delete_thread_removes_file(self):
        mem = ConversationMemory(thread_name="deleteme", storage_dir=self.tmp_dir)
        mem.add_message(ROLE_USER, "hi")
        self.assertTrue(mem.path.exists())

        removed = ConversationMemory.delete_thread("deleteme", storage_dir=self.tmp_dir)
        self.assertTrue(removed)
        self.assertFalse(mem.path.exists())

        # Deleting again returns False since the file is already gone.
        removed_again = ConversationMemory.delete_thread("deleteme", storage_dir=self.tmp_dir)
        self.assertFalse(removed_again)

    def test_invalid_thread_name_raises(self):
        with self.assertRaises(ValueError):
            ConversationMemory(thread_name="bad/name!", storage_dir=self.tmp_dir)
        with self.assertRaises(ValueError):
            ConversationMemory(thread_name="   ", storage_dir=self.tmp_dir)

    def test_corrupt_file_does_not_crash_load(self):
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        bad_path = self.tmp_dir / "corrupt.json"
        bad_path.write_text("{ this is not valid json", encoding="utf-8")

        mem = ConversationMemory(thread_name="corrupt", storage_dir=self.tmp_dir)
        self.assertEqual(len(mem), 0)  # falls back to an empty thread

    def test_long_message_is_truncated(self):
        mem = ConversationMemory(
            thread_name="longmsg", storage_dir=self.tmp_dir, max_chars_per_message=50
        )
        mem.add_message(ROLE_USER, "x" * 500)
        stored = mem.get_messages()[0]["content"]
        self.assertLessEqual(len(stored), 50 + len(" …[truncated]"))
        self.assertTrue(stored.endswith("…[truncated]"))


class TestConversationMemoryTruncation(unittest.TestCase):
    """Covers the summarization/truncation behavior for long conversations."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="chatbot_test_trunc_"))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_history_is_capped_at_max_messages_with_summary(self):
        max_messages = 6
        mem = ConversationMemory(
            thread_name="growing", storage_dir=self.tmp_dir, max_messages=max_messages
        )

        # Add far more messages than the cap allows.
        for i in range(20):
            role = ROLE_USER if i % 2 == 0 else ROLE_ASSISTANT
            mem.add_message(role, f"message number {i}")

        messages = mem.get_messages()
        # Never grows past the cap.
        self.assertLessEqual(len(messages), max_messages)
        # The oldest entry should now be a rolled-up summary.
        self.assertEqual(messages[0]["role"], ROLE_SUMMARY)
        # The most recent real message should still be present verbatim.
        self.assertEqual(messages[-1]["content"], "message number 19")

    def test_summary_grows_to_include_new_old_messages_over_time(self):
        max_messages = 4
        mem = ConversationMemory(
            thread_name="growing2", storage_dir=self.tmp_dir, max_messages=max_messages
        )
        for i in range(10):
            mem.add_message(ROLE_USER, f"turn {i}")

        first_summary = mem.get_messages()[0]["content"]

        for i in range(10, 15):
            mem.add_message(ROLE_USER, f"turn {i}")

        second_summary = mem.get_messages()[0]["content"]
        # The summary should have picked up more content as more messages
        # aged out of the verbatim window.
        self.assertNotEqual(first_summary, second_summary)
        self.assertTrue(second_summary.startswith("[Summary of earlier conversation]"))

    def test_context_for_llm_maps_summary_to_system_role(self):
        mem = ConversationMemory(thread_name="ctx", storage_dir=self.tmp_dir, max_messages=4)
        for i in range(10):
            mem.add_message(ROLE_USER, f"turn {i}")

        context = mem.get_context_for_llm()
        self.assertEqual(context[0]["role"], "system")
        # Remaining entries should only use user/assistant roles.
        for entry in context[1:]:
            self.assertIn(entry["role"], ("user", "assistant"))


if __name__ == "__main__":
    unittest.main()
