"""
Unit tests for chatbot.llm -- specifically the offline demo-mode responder,
which requires no API key and no network access.
"""

import os
import unittest
from unittest import mock

from chatbot.llm import LLMClient


class TestDemoModeSelection(unittest.TestCase):
    def test_defaults_to_demo_when_no_api_keys_set(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            # Also clear any keys that might be set in the real environment
            # for this process, so the test is deterministic.
            env = dict(os.environ)
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("OPENAI_API_KEY", None)
            with mock.patch.dict(os.environ, env, clear=True):
                client = LLMClient()
                self.assertEqual(client.provider, "demo")

    def test_explicit_provider_override(self):
        client = LLMClient(provider="demo")
        self.assertEqual(client.provider, "demo")


class TestDemoModeGeneration(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient(provider="demo")

    def test_reply_is_labeled_as_demo_mode(self):
        reply = self.client.generate([{"role": "user", "content": "Hello"}])
        self.assertTrue(reply.startswith("[demo mode]"))

    def test_greeting_gets_greeting_reply(self):
        reply = self.client.generate([{"role": "user", "content": "hello there"}])
        self.assertIn("Hello", reply)

    def test_question_gets_question_reply(self):
        reply = self.client.generate(
            [{"role": "user", "content": "What is the capital of France?"}]
        )
        self.assertIn("question", reply.lower())

    def test_empty_history_does_not_crash(self):
        reply = self.client.generate([])
        self.assertTrue(reply.startswith("[demo mode]"))

    def test_farewell_gets_farewell_reply(self):
        reply = self.client.generate([{"role": "user", "content": "goodbye"}])
        self.assertIn("Goodbye", reply)


if __name__ == "__main__":
    unittest.main()
