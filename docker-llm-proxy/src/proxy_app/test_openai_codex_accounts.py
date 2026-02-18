# SPDX-License-Identifier: MIT

import asyncio
import tempfile
import unittest
from pathlib import Path

from proxy_app.openai_codex_accounts import (
    OpenAICodexEventLog,
    account_id_for_key,
    is_valid_openai_api_key_shape,
    normalize_provider,
    parse_provider_api_keys,
    remove_provider_api_key,
    upsert_provider_api_key,
)


class OpenAICodexAccountHelpersTest(unittest.TestCase):
    def test_normalize_provider_aliases(self) -> None:
        self.assertEqual(normalize_provider("openai"), "openai")
        self.assertEqual(normalize_provider("codex"), "openai")
        self.assertEqual(normalize_provider("openai_codex"), "openai")
        self.assertEqual(normalize_provider("unknown"), "")

    def test_api_key_shape_validation(self) -> None:
        self.assertTrue(is_valid_openai_api_key_shape("sk-abcdefghijklmnopqrstuvwxyz"))
        self.assertFalse(is_valid_openai_api_key_shape("not-a-key"))
        self.assertFalse(is_valid_openai_api_key_shape("sk-short"))

    def test_upsert_parse_and_remove_provider_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"

            key_name_1, created_1 = upsert_provider_api_key(
                env_file, "openai", "sk-abcdefghijklmnopqrstuvwxyz"
            )
            self.assertTrue(created_1)
            self.assertEqual(key_name_1, "OPENAI_API_KEY_1")

            key_name_2, created_2 = upsert_provider_api_key(
                env_file, "openai", "sk-0123456789abcdefghijklmnopqrstuvwxyz"
            )
            self.assertTrue(created_2)
            self.assertEqual(key_name_2, "OPENAI_API_KEY_2")

            key_name_1_dup, created_dup = upsert_provider_api_key(
                env_file, "openai", "sk-abcdefghijklmnopqrstuvwxyz"
            )
            self.assertFalse(created_dup)
            self.assertEqual(key_name_1_dup, "OPENAI_API_KEY_1")

            parsed = parse_provider_api_keys(env_file, "openai")
            self.assertEqual(len(parsed), 2)
            self.assertEqual(parsed[0][0], "OPENAI_API_KEY_1")
            self.assertEqual(parsed[1][0], "OPENAI_API_KEY_2")

            removed = remove_provider_api_key(
                env_file, "openai", "sk-abcdefghijklmnopqrstuvwxyz"
            )
            self.assertEqual(removed, ["OPENAI_API_KEY_1"])

            parsed_after = parse_provider_api_keys(env_file, "openai")
            self.assertEqual(len(parsed_after), 1)
            self.assertEqual(parsed_after[0][0], "OPENAI_API_KEY_2")

    def test_account_id_is_stable(self) -> None:
        first = account_id_for_key("openai", "sk-abcdefghijklmnopqrstuvwxyz")
        second = account_id_for_key("openai", "sk-abcdefghijklmnopqrstuvwxyz")
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("openai:"))


class OpenAICodexEventLogTest(unittest.TestCase):
    def test_event_log_appends_and_lists_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            event_file = Path(temp_dir) / "events.jsonl"
            log = OpenAICodexEventLog(event_file, max_entries=3)

            async def run() -> None:
                await log.append_event("action.1", "ok", "first")
                await log.append_event("action.2", "ok", "second")
                await log.append_event("action.3", "ok", "third")
                await log.append_event("action.4", "ok", "fourth")
                latest = await log.list_events(limit=2)
                self.assertEqual(len(latest), 2)
                self.assertEqual(latest[0]["action"], "action.4")
                self.assertEqual(latest[1]["action"], "action.3")

            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
