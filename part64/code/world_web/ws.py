from __future__ import annotations

import base64
import hashlib
import socket
import struct

from .constants import WS_MAGIC


WS_CLIENT_FRAME_MAX_BYTES = 1_048_576


def websocket_accept_value(client_key: str) -> str:
    accept_seed = client_key + WS_MAGIC
    digest = hashlib.sha1(accept_seed.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


def websocket_frame(opcode: int, payload: bytes = b"") -> bytes:
    data = bytes(payload)
    length = len(data)
    header = bytearray([0x80 | (opcode & 0x0F)])
    if length <= 125:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + data


def websocket_frame_text(message: str) -> bytes:
    return websocket_frame(0x1, message.encode("utf-8"))


def recv_ws_exact(connection: socket.socket, size: int) -> bytes | None:
    if size <= 0:
        return b""

    data = bytearray()
    while len(data) < size:
        try:
            chunk = connection.recv(size - len(data))
        except socket.timeout:
            if not data:
                raise
            continue
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def read_ws_client_frame(
    connection: socket.socket,
    *,
    max_payload: int = WS_CLIENT_FRAME_MAX_BYTES,
) -> tuple[int, bytes] | None:
    header = recv_ws_exact(connection, 2)
    if header is None:
        return None

    first, second = header
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    payload_len = second & 0x7F

    if payload_len == 126:
        extended = recv_ws_exact(connection, 2)
        if extended is None:
            return None
        payload_len = struct.unpack("!H", extended)[0]
    elif payload_len == 127:
        extended = recv_ws_exact(connection, 8)
        if extended is None:
            return None
        payload_len = struct.unpack("!Q", extended)[0]

    if not masked or payload_len > max_payload:
        return None

    mask_key = recv_ws_exact(connection, 4)
    if mask_key is None:
        return None

    payload = recv_ws_exact(connection, payload_len)
    if payload is None:
        return None

    if payload_len:
        payload = bytes(
            byte ^ mask_key[index % 4] for index, byte in enumerate(payload)
        )

    return opcode, payload


def consume_ws_client_frame(
    connection: socket.socket,
    *,
    max_payload: int = WS_CLIENT_FRAME_MAX_BYTES,
) -> bool:
    frame = read_ws_client_frame(connection, max_payload=max_payload)
    if frame is None:
        return False

    opcode, payload = frame
    if opcode == 0x8:
        try:
            connection.sendall(websocket_frame(0x8, payload[:125]))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        return False

    if opcode == 0x9:
        connection.sendall(websocket_frame(0xA, payload[:125]))
        return True

    if opcode in {0x0, 0x1, 0x2, 0xA}:
        return True

    return False
