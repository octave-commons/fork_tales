# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

from __future__ import annotations

from typing import Any, Mapping

SESSION_ID_FIELD_CANDIDATES = (
    "session_id",
    "sessionId",
    "conversation_id",
    "conversationId",
    "thread_id",
    "threadId",
    "chat_id",
    "chatId",
)

SESSION_ID_HEADER_CANDIDATES = (
    "session_id",
    "x-session-id",
    "conversation_id",
    "x-conversation-id",
)


def normalize_session_id(value: Any) -> str:
    if value is None:
        return ""
    session_id = str(value).strip()
    if not session_id:
        return ""
    if len(session_id) > 256:
        return session_id[:256]
    return session_id


def extract_session_id_from_payload(payload: Mapping[str, Any] | None) -> str:
    if not isinstance(payload, Mapping):
        return ""

    metadata = payload.get("metadata")
    metadata_mapping = metadata if isinstance(metadata, Mapping) else {}

    for key in SESSION_ID_FIELD_CANDIDATES:
        session_id = normalize_session_id(metadata_mapping.get(key))
        if session_id:
            return session_id

    for key in SESSION_ID_FIELD_CANDIDATES:
        session_id = normalize_session_id(payload.get(key))
        if session_id:
            return session_id

    return ""


def extract_session_id(
    headers: Mapping[str, Any] | None,
    payload: Mapping[str, Any] | None,
) -> str:
    payload_session = extract_session_id_from_payload(payload)
    if payload_session:
        return payload_session

    if not isinstance(headers, Mapping):
        return ""

    lowered = {str(key).lower(): value for key, value in headers.items()}
    for key in SESSION_ID_HEADER_CANDIDATES:
        session_id = normalize_session_id(lowered.get(key))
        if session_id:
            return session_id

    return ""
