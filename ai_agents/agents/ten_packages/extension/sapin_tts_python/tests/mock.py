#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
"""An in-process Sapin TTS server and a recording extension tester."""

import asyncio
import json
import threading
from typing import Any, Callable

import websockets

from ten_ai_base.struct import TTSTextInput
from ten_runtime import Data, ExtensionTester, TenEnvTester

SERVED_LANGS = ("fil", "ceb", "hil")
# 3200 bytes = 100 ms of 16 kHz PCM16 mono, the server's frame size.
FRAME = b"\x11\x22" * 1600


class MockSapinServer:
    """Speaks the Sapin TTS protocol.

    `respond(request, ws, server)` handles each valid request; the default
    streams `server.audio` in `frame_size` frames and then {"type":"end"}.
    """

    def __init__(
        self,
        audio: bytes = FRAME * 3,
        frame_size: int = 3200,
        respond: Callable[..., Any] | None = None,
        frame_delay: float = 0.0,
    ):
        self.audio = audio
        self.frame_size = frame_size
        self.respond = respond
        self.frame_delay = frame_delay
        self.requests: list[dict] = []
        self.connections = 0
        self.sockets: list[Any] = []
        self.reject_status: int | None = None
        self._server = None
        self.port = 0

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/tts"

    async def __aenter__(self) -> "MockSapinServer":
        self._server = await websockets.serve(
            self._handler,
            "127.0.0.1",
            0,
            process_request=self._process_request,
        )
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *_exc) -> None:
        self._server.close()
        await self._server.wait_closed()

    def _process_request(self, connection, request):
        if self.reject_status is not None:
            return connection.respond(self.reject_status, "rejected\n")
        return None

    @staticmethod
    async def send(ws, payload: dict) -> None:
        await ws.send(json.dumps(payload))

    async def stream(self, ws, audio: bytes | None = None) -> None:
        audio = self.audio if audio is None else audio
        for i in range(0, len(audio), self.frame_size):
            await ws.send(audio[i : i + self.frame_size])
            if self.frame_delay:
                await asyncio.sleep(self.frame_delay)
        await self.send(ws, {"type": "end"})

    async def fail(self, ws, message: str) -> None:
        await self.send(ws, {"type": "error", "message": message})
        await ws.close(1000)

    async def _handler(self, ws) -> None:
        self.connections += 1
        self.sockets.append(ws)
        try:
            async for raw in ws:
                try:
                    request = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    await self.fail(ws, "invalid JSON")
                    return
                self.requests.append(request)
                if "text" not in request:
                    await self.fail(ws, "missing 'text'")
                    return
                if not str(request["text"]).strip():
                    await self.fail(ws, "'text' is empty")
                    return
                if len(request["text"]) > 500:
                    await self.fail(ws, "'text' exceeds 500 characters")
                    return
                if request.get("lang", "fil") not in SERVED_LANGS:
                    await self.fail(
                        ws, "unknown 'lang'; served: " + ", ".join(SERVED_LANGS)
                    )
                    return
                if self.respond is not None:
                    await self.respond(request, ws, self)
                else:
                    await self.stream(ws)
        except websockets.exceptions.ConnectionClosed:
            pass


class ThreadedMockSapinServer:
    """Runs MockSapinServer on its own loop, for extension-level tests
    where the TEN runtime owns the extension's event loop."""

    def __init__(self, **options):
        self.server = MockSapinServer(**options)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )

    def __enter__(self) -> MockSapinServer:
        self._thread.start()
        asyncio.run_coroutine_threadsafe(
            self.server.__aenter__(), self._loop
        ).result(5)
        return self.server

    def __exit__(self, *_exc) -> None:
        asyncio.run_coroutine_threadsafe(
            self.server.__aexit__(), self._loop
        ).result(5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)


def config_for(url: str, **params) -> dict:
    return {"params": {"url": url, "lang": "fil", **params}}


def send_text(
    ten_env_tester: TenEnvTester,
    request_id: str,
    text: str,
    text_input_end: bool,
    metadata: dict | None = None,
) -> None:
    tts_input = TTSTextInput(
        request_id=request_id,
        text=text,
        text_input_end=text_input_end,
        metadata=metadata or {"session_id": "s-1", "turn_id": 1},
    )
    data = Data.create("tts_text_input")
    data.set_property_from_json(None, tts_input.model_dump_json())
    ten_env_tester.send_data(data)


def send_flush(ten_env_tester: TenEnvTester, flush_id: str) -> None:
    flush = Data.create("tts_flush")
    flush.set_property_from_json(
        None, json.dumps({"flush_id": flush_id, "metadata": {}})
    )
    ten_env_tester.send_data(flush)


class RecordingTester(ExtensionTester):
    """Sends (request_id, text, text_input_end) inputs and records output.

    `timeline` holds every data name and ("audio", bytes) per frame, in
    arrival order. Stops after `expected_ends` tts_audio_end events, or on
    an error when `stop_on_error` is set.
    """

    def __init__(
        self,
        inputs: list[tuple[str, str, bool]],
        expected_ends: int = 1,
        stop_on_error: bool = False,
    ):
        super().__init__()
        self.inputs = inputs
        self.expected_ends = expected_ends
        self.stop_on_error = stop_on_error
        self.events: list[tuple[str, dict]] = []
        self.timeline: list[tuple[str, Any]] = []
        self.audio = b""
        self.sample_rates: set[int] = set()

    def on_start(self, ten_env_tester: TenEnvTester) -> None:
        for request_id, text, text_input_end in self.inputs:
            send_text(ten_env_tester, request_id, text, text_input_end)
        ten_env_tester.on_start_done()

    def on_data(self, ten_env: TenEnvTester, data) -> None:
        name = data.get_name()
        json_str, _ = data.get_property_to_json(None)
        payload = json.loads(json_str) if json_str else {}
        self.events.append((name, payload))
        # Error payloads carry the request ID as "id".
        self.timeline.append(
            (name, payload.get("request_id") or payload.get("id"))
        )
        if name == "error" and self.stop_on_error:
            ten_env.stop_test()
        if self.count("tts_audio_end") >= self.expected_ends:
            ten_env.stop_test()

    def on_audio_frame(self, ten_env: TenEnvTester, audio_frame) -> None:
        self.sample_rates.add(audio_frame.get_sample_rate())
        buf = audio_frame.lock_buf()
        try:
            chunk = bytes(buf)
        finally:
            audio_frame.unlock_buf(buf)
        self.audio += chunk
        self.timeline.append(("audio", len(chunk)))

    def count(self, name: str) -> int:
        return sum(1 for event, _ in self.events if event == name)

    def payloads(self, name: str) -> list[dict]:
        return [payload for event, payload in self.events if event == name]

    def end_of(self, request_id: str) -> dict:
        ends = [
            p
            for p in self.payloads("tts_audio_end")
            if p.get("request_id") == request_id
        ]
        assert len(ends) == 1, f"expected one end for {request_id}: {ends}"
        return ends[0]
