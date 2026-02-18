# SPDX-License-Identifier: MIT

import asyncio
import time
import unittest
from unittest.mock import patch
import tempfile
from types import SimpleNamespace
from typing import Any, cast

from proxy_app.session_routing import extract_session_id, normalize_session_id
from rotator_library.session_affinity import SessionAffinityCache
from rotator_library.client import RotatingClient


class SessionRoutingExtractionTest(unittest.TestCase):
    def test_extract_session_id_prefers_metadata_over_body_and_headers(self):
        payload = {
            "session_id": "body-session",
            "metadata": {
                "session_id": "metadata-session",
            },
        }
        headers = {
            "x-session-id": "header-session",
        }
        self.assertEqual(
            extract_session_id(headers, payload),
            "metadata-session",
        )

    def test_extract_session_id_uses_header_when_payload_missing(self):
        payload = {"model": "openai/gpt-5.3-codex"}
        headers = {
            "x-session-id": "header-session",
        }
        self.assertEqual(extract_session_id(headers, payload), "header-session")

    def test_normalize_session_id_truncates_overlong_ids(self):
        source = "s" * 400
        normalized = normalize_session_id(source)
        self.assertEqual(len(normalized), 256)
        self.assertTrue(normalized.startswith("s" * 10))


class SessionAffinityCacheTest(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def test_session_affinity_keeps_concurrent_sessions_isolated(self):
        cache = SessionAffinityCache(max_entries=10, idle_ttl_seconds=3600)

        self.run_async(cache.set("openai", "session-a", "sk-a"))
        self.run_async(cache.set("openai", "session-b", "sk-b"))

        self.assertEqual(
            self.run_async(cache.get("openai", "session-a")),
            "sk-a",
        )
        self.assertEqual(
            self.run_async(cache.get("openai", "session-b")),
            "sk-b",
        )

    def test_session_affinity_clears_when_allowed_credentials_change(self):
        cache = SessionAffinityCache(max_entries=10, idle_ttl_seconds=3600)
        self.run_async(cache.set("openai", "session-a", "sk-a"))

        self.assertIsNone(
            self.run_async(
                cache.get(
                    "openai",
                    "session-a",
                    allowed_credentials=["sk-b"],
                )
            )
        )

        self.assertIsNone(self.run_async(cache.get("openai", "session-a")))

    def test_session_affinity_enforces_capacity_eviction(self):
        cache = SessionAffinityCache(max_entries=2, idle_ttl_seconds=3600)
        self.run_async(cache.set("openai", "session-a", "sk-a"))
        self.run_async(cache.set("openai", "session-b", "sk-b"))

        # Force session-a to be oldest so adding session-c evicts it.
        cache._entries[("openai", "session-a")].updated_at = time.time() - 100

        self.run_async(cache.set("openai", "session-c", "sk-c"))

        self.assertIsNone(self.run_async(cache.get("openai", "session-a")))
        self.assertEqual(self.run_async(cache.get("openai", "session-b")), "sk-b")
        self.assertEqual(self.run_async(cache.get("openai", "session-c")), "sk-c")

    def test_session_affinity_clear_credential_removes_all_bindings(self):
        cache = SessionAffinityCache(max_entries=10, idle_ttl_seconds=3600)
        self.run_async(cache.set("openai", "session-a", "sk-shared"))
        self.run_async(cache.set("openai", "session-b", "sk-shared"))
        self.run_async(cache.set("openai", "session-c", "sk-other"))

        removed = self.run_async(cache.clear_credential("openai", "sk-shared"))
        self.assertEqual(removed, 2)
        self.assertIsNone(self.run_async(cache.get("openai", "session-a")))
        self.assertIsNone(self.run_async(cache.get("openai", "session-b")))
        self.assertEqual(self.run_async(cache.get("openai", "session-c")), "sk-other")


class RotatingClientSessionAffinityIntegrationTest(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def test_client_keeps_session_on_same_credential_and_forwards_headers(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as data_dir:
                client = RotatingClient(
                    api_keys={
                        "openai": [
                            "sk-test-key-aaaaaaaaaaaaaaaaaaaa",
                            "sk-test-key-bbbbbbbbbbbbbbbbbbbb",
                        ]
                    },
                    configure_logging=False,
                    data_dir=data_dir,
                    global_timeout=10,
                    rotation_tolerance=0.0,
                )

                observed_calls = []

                async def mock_api_call(**kwargs):
                    observed_calls.append(
                        {
                            "api_key": kwargs.get("api_key"),
                            "extra_headers": dict(kwargs.get("extra_headers") or {}),
                        }
                    )
                    return {
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                        "choices": [{"message": {"content": "ok"}}],
                    }

                try:
                    with patch(
                        "rotator_library.client.random.shuffle", lambda values: None
                    ):
                        await client._execute_with_retry(
                            mock_api_call,
                            request=None,
                            model="openai/gpt-5.3-codex",
                            messages=[{"role": "user", "content": "first"}],
                            _proxy_session_id="session-a",
                        )
                        await client._execute_with_retry(
                            mock_api_call,
                            request=None,
                            model="openai/gpt-5.3-codex",
                            messages=[{"role": "user", "content": "second"}],
                            _proxy_session_id="session-a",
                        )
                        await client._execute_with_retry(
                            mock_api_call,
                            request=None,
                            model="openai/gpt-5.3-codex",
                            messages=[{"role": "user", "content": "third"}],
                            _proxy_session_id="session-b",
                        )

                    self.assertEqual(len(observed_calls), 3)
                    self.assertEqual(
                        observed_calls[0]["api_key"],
                        observed_calls[1]["api_key"],
                    )
                    self.assertNotEqual(
                        observed_calls[1]["api_key"],
                        observed_calls[2]["api_key"],
                    )

                    self.assertEqual(
                        observed_calls[0]["extra_headers"].get("session_id"),
                        "session-a",
                    )
                    self.assertEqual(
                        observed_calls[0]["extra_headers"].get("conversation_id"),
                        "session-a",
                    )
                    self.assertEqual(
                        observed_calls[2]["extra_headers"].get("session_id"),
                        "session-b",
                    )

                    snapshot = await client.get_openai_session_affinity_snapshot(
                        limit=10
                    )
                    self.assertEqual(snapshot.get("entry_count"), 2)
                finally:
                    await client.close()

        self.run_async(scenario())


class ChatCompletionsSessionPlumbingTest(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def test_chat_completions_passes_extracted_session_id_to_client(self):
        async def scenario():
            from proxy_app import main as proxy_main

            class FakeGateway:
                def resolve_for_request(self, request_data, available_providers=None):
                    return {"applied": False}

                def apply_route(self, request_data, route_decision):
                    return request_data

            class FakeClient:
                def __init__(self):
                    self.all_credentials = {"openai": ["sk-test"]}
                    self.calls = []

                async def acompletion(
                    self, request=None, _proxy_session_id=None, **kwargs
                ):
                    self.calls.append(
                        {
                            "session_id": _proxy_session_id,
                            "model": kwargs.get("model"),
                        }
                    )
                    return {
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "created": 0,
                        "model": kwargs.get("model"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "ok",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    }

            class FakeRequest:
                def __init__(self):
                    self.headers = {
                        "x-session-id": "session-from-header",
                    }
                    self._body = {
                        "model": "openai/gpt-5.3-codex",
                        "messages": [{"role": "user", "content": "hello"}],
                        "metadata": {"session_id": "session-from-metadata"},
                    }
                    self.app = SimpleNamespace(
                        state=SimpleNamespace(smart_gateway=FakeGateway())
                    )
                    self.url = "http://testserver/v1/chat/completions"
                    self.client = SimpleNamespace(host="127.0.0.1", port=50000)

                async def json(self):
                    return self._body

            fake_client = FakeClient()
            fake_request = FakeRequest()

            response = await proxy_main.chat_completions(
                request=cast(Any, fake_request),
                client=cast(Any, fake_client),
                _=None,
            )

            response = cast(dict[str, Any], response)
            self.assertEqual(response["object"], "chat.completion")
            self.assertEqual(len(fake_client.calls), 1)
            self.assertEqual(
                fake_client.calls[0]["session_id"],
                "session-from-metadata",
            )

        self.run_async(scenario())


if __name__ == "__main__":
    unittest.main()
