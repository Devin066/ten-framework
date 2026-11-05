#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
"""WebSocket client for the SapinSapin TTS demo server.

Protocol (ws://<host>:8765/tts):
  * Send {"text": "...", "lang": "fil|ceb|hil", "voice": "<id>"} as JSON.
  * Receive binary PCM S16LE / 16 kHz / mono frames, then {"type": "end"},
    or {"type": "error", "message": "..."} after which the server closes
    the socket (code 1000).
  * There is no handshake, no auth and no cancel message. Text is limited
    to 500 characters per request.
  * One socket serves many sequential requests. The server closes a socket
    after about 30 s without a request, whether or not pings are sent.

Because of the idle close, the client keeps a ready socket by reconnecting
in the background whenever its socket closes. A flush closes the socket to
drop unread audio, and the background reconnect brings up a fresh one.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncIterator, Awaitable, Callable

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.protocol import State

from ten_ai_base.const import LOG_CATEGORY_VENDOR
from ten_runtime import AsyncTenEnv

from .config import MAX_TEXT_CHARS, SapinTTSConfig

# Event types communicated back to the extension.
EVENT_TTS_RESPONSE = 1
EVENT_TTS_END = 2
EVENT_TTS_ERROR = 3
EVENT_TTS_TTFB_METRIC = 5

WS_OPEN_TIMEOUT = 10.0
# Seconds to wait for the next message of a running request. The demo
# server has taken up to ~5 s for the first audio of a short sentence.
WS_RECV_TIMEOUT = 30.0
WS_CLOSE_TIMEOUT = 1.0
RECONNECT_BASE_DELAY = 0.5
RECONNECT_MAX_DELAY = 10.0

# Server messages that mean the request itself is invalid for this
# configuration, so retrying cannot help.
_FATAL_MESSAGES = ("unknown 'lang'",)
# HTTP statuses on the WebSocket upgrade that retrying cannot fix.
_FATAL_HTTP_STATUSES = {400, 401, 403, 404}

StatusCallback = Callable[[str, int, str], Awaitable[None]]

_SENTENCE_END = re.compile(r"(?<=[.!?。！？;；:,，])\s+")


class SapinTTSError(Exception):
    def __init__(self, message: str, fatal: bool = False, code: str = ""):
        super().__init__(message)
        self.message = message
        self.fatal = fatal
        self.code = code


def split_text(text: str, limit: int = MAX_TEXT_CHARS) -> list[str]:
    """Split text into pieces of at most `limit` characters.

    Prefers sentence boundaries, then whitespace, then a hard cut.
    Whitespace-only input yields no pieces.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        candidate = f"{current} {sentence}" if current else sentence
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            pieces.append(current)
        current = ""
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit + 1)
            if cut <= 0:
                cut = limit
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        current = sentence
    if current:
        pieces.append(current)
    return [piece for piece in pieces if piece]


def _usable(ws: ClientConnection | None) -> bool:
    return ws is not None and ws.state is State.OPEN


class SapinTTSClient:
    """Sapin TTS over one reusable WebSocket, kept open in the background."""

    def __init__(
        self,
        config: SapinTTSConfig,
        ten_env: AsyncTenEnv,
        on_status: StatusCallback | None = None,
    ):
        self.config = config
        self.ten_env = ten_env
        self._on_status = on_status

        self._ws: ClientConnection | None = None
        self._stopped = False
        # Bumped by cancel()/stop(). A get() started in an older generation
        # must not send or yield anything more.
        self._generation = 0
        self._connect_lock = asyncio.Lock()
        self._reconnect_task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()

        # TTFB tracking for the current request.
        self._sent_ts: float | None = None
        self._ttfb_sent = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the first socket so it is ready before the first text."""
        try:
            await self._connect()
        except SapinTTSError as e:
            if e.fatal:
                raise
            self.ten_env.log_warn(
                f"Sapin TTS initial connect failed, retrying: {e.message}",
                category=LOG_CATEGORY_VENDOR,
            )
            self._schedule_reconnect()

    async def stop(self) -> None:
        self._stopped = True
        self._generation += 1
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            await asyncio.gather(self._reconnect_task, return_exceptions=True)
            self._reconnect_task = None
        ws, self._ws = self._ws, None
        if ws is not None:
            await self._close_quietly(ws)
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def cancel(self) -> None:
        """Stop the running request and drop its unread audio.

        The server has no cancel message, so the socket is closed. The
        background reconnect then opens a fresh one for the next request.
        """
        self.ten_env.log_debug("Sapin TTS: cancelling current request")
        self._generation += 1
        self.reset_ttfb()
        ws, self._ws = self._ws, None
        if ws is not None:
            # Close without waiting for the peer: the generation bump
            # already stops the request from yielding more audio.
            self._spawn(self._close_quietly(ws))

    def reset_ttfb(self) -> None:
        self._sent_ts = None
        self._ttfb_sent = False

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def get(
        self, text: str
    ) -> AsyncIterator[tuple[bytes | int | SapinTTSError | None, int]]:
        """Synthesize one text and yield (payload, event) pairs."""
        pieces = split_text(text)
        if not pieces:
            yield None, EVENT_TTS_END
            return

        generation = self._generation
        for piece in pieces:
            async for payload, event in self._synthesize(piece, generation):
                yield payload, event
                if event == EVENT_TTS_ERROR:
                    return
            if generation != self._generation:
                return
        yield None, EVENT_TTS_END

    async def _synthesize(
        self, text: str, generation: int
    ) -> AsyncIterator[tuple[bytes | int | SapinTTSError, int]]:
        message = json.dumps(
            self.config.build_request(text), ensure_ascii=False
        )
        retried = False
        while True:
            try:
                ws = await self._ready_socket()
            except SapinTTSError as e:
                yield e, EVENT_TTS_ERROR
                return
            if generation != self._generation or self._stopped:
                self.ten_env.log_debug("Sapin TTS: cancelled before send")
                return

            got_audio = False
            carry = b""
            try:
                if self._sent_ts is None and not self._ttfb_sent:
                    self._sent_ts = time.monotonic()
                await ws.send(message)
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(), timeout=WS_RECV_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        await self._discard(ws)
                        yield SapinTTSError(
                            "Timed out waiting for Sapin TTS audio"
                        ), EVENT_TTS_ERROR
                        return
                    if generation != self._generation:
                        self.ten_env.log_debug("Sapin TTS: dropping late data")
                        return

                    if isinstance(raw, bytes):
                        # Keep PCM16 samples whole across frame boundaries.
                        data = carry + raw
                        cut = len(data) - (len(data) % 2)
                        carry = data[cut:]
                        if not cut:
                            continue
                        got_audio = True
                        if self._sent_ts is not None and not self._ttfb_sent:
                            self._ttfb_sent = True
                            ttfb_ms = int(
                                (time.monotonic() - self._sent_ts) * 1000
                            )
                            yield ttfb_ms, EVENT_TTS_TTFB_METRIC
                        yield data[:cut], EVENT_TTS_RESPONSE
                        continue

                    event = self._parse(raw)
                    kind = event.get("type")
                    if kind == "end":
                        return
                    if kind == "error":
                        detail = str(event.get("message") or "unknown error")
                        self.ten_env.log_error(
                            f"vendor_error: Sapin TTS: {detail}",
                            category=LOG_CATEGORY_VENDOR,
                        )
                        # The server closes the socket after an error.
                        await self._discard(ws)
                        yield SapinTTSError(
                            detail,
                            fatal=any(m in detail for m in _FATAL_MESSAGES),
                        ), EVENT_TTS_ERROR
                        return
                    self.ten_env.log_debug(f"Sapin TTS: ignoring {event}")

            except ConnectionClosed as e:
                await self._discard(ws)
                if generation != self._generation or self._stopped:
                    return
                if not got_audio and not retried:
                    # The idle close can race a request that was just sent.
                    # Nothing was emitted, so resending on a fresh socket
                    # cannot duplicate audio.
                    retried = True
                    self.ten_env.log_warn(
                        f"Sapin TTS socket closed before audio ({e}), retrying",
                        category=LOG_CATEGORY_VENDOR,
                    )
                    continue
                self.ten_env.log_error(
                    f"vendor_error: Sapin TTS socket closed: {e}",
                    category=LOG_CATEGORY_VENDOR,
                )
                yield SapinTTSError(
                    f"Sapin TTS connection closed: {e}"
                ), EVENT_TTS_ERROR
                return

    def _parse(self, raw: str) -> dict[str, Any]:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            return {"type": "unknown", "raw": raw[:200]}
        return event if isinstance(event, dict) else {"type": "unknown"}

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _ready_socket(self) -> ClientConnection:
        ws = self._ws
        if _usable(ws):
            return ws
        await self._connect()
        ws = self._ws
        if not _usable(ws):
            raise SapinTTSError("Sapin TTS connection is not available")
        return ws

    async def _connect(self) -> None:
        async with self._connect_lock:
            if _usable(self._ws) or self._stopped:
                return
            await self._notify("connecting")
            try:
                ws = await websockets.connect(
                    self.config.url,
                    open_timeout=WS_OPEN_TIMEOUT,
                    close_timeout=WS_CLOSE_TIMEOUT,
                    # The server ignores pings for its idle timeout.
                    ping_interval=None,
                    max_size=None,
                )
            except InvalidStatus as e:
                status = e.response.status_code
                await self._notify(
                    "disconnected", status, f"handshake rejected: {status}"
                )
                raise SapinTTSError(
                    f"Sapin TTS rejected the connection (HTTP {status})",
                    fatal=status in _FATAL_HTTP_STATUSES,
                    code=str(status),
                ) from e
            except Exception as e:  # pylint: disable=broad-exception-caught
                await self._notify("disconnected", 0, type(e).__name__)
                raise SapinTTSError(
                    f"Sapin TTS connect failed: {type(e).__name__}: {e}"
                ) from e

            if self._stopped:
                await self._close_quietly(ws)
                return
            self._ws = ws
            await self._notify("connected")
            self.ten_env.log_debug(
                "vendor_status: connected to sapin tts",
                category=LOG_CATEGORY_VENDOR,
            )
            self._spawn(self._watch(ws))

    async def _watch(self, ws: ClientConnection) -> None:
        """Report a closed socket and bring up a replacement."""
        try:
            await ws.wait_closed()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        if self._ws is ws:
            self._ws = None
        if self._stopped or self._ws is not None:
            # Stopped, or a newer socket is already installed.
            return
        self.ten_env.log_debug(
            f"vendor_status: sapin tts socket closed "
            f"({ws.close_code} {ws.close_reason})",
            category=LOG_CATEGORY_VENDOR,
        )
        await self._notify(
            "disconnected", ws.close_code or 0, ws.close_reason or "closed"
        )
        self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        if self._stopped:
            return
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        delay = RECONNECT_BASE_DELAY
        while not self._stopped and not _usable(self._ws):
            try:
                await self._connect()
                return
            except SapinTTSError as e:
                if e.fatal:
                    self.ten_env.log_error(
                        f"Sapin TTS reconnect stopped: {e.message}",
                        category=LOG_CATEGORY_VENDOR,
                    )
                    return
                self.ten_env.log_warn(
                    f"Sapin TTS reconnect failed, retrying in {delay}s: "
                    f"{e.message}",
                    category=LOG_CATEGORY_VENDOR,
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_DELAY)

    async def _discard(self, ws: ClientConnection | None) -> None:
        """Close `ws`; clear self._ws only if it is still that socket."""
        if ws is None:
            return
        if self._ws is ws:
            self._ws = None
        await self._close_quietly(ws)

    @staticmethod
    async def _close_quietly(ws: ClientConnection) -> None:
        try:
            await ws.close()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _notify(
        self, status: str, code: int = 0, message: str = ""
    ) -> None:
        if self._on_status is None:
            return
        try:
            await self._on_status(status, code, message)
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.ten_env.log_warn(f"Sapin TTS status callback failed: {e}")
