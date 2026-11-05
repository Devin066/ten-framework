#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
"""SapinTTSClient against an in-process Sapin WebSocket server."""

import asyncio
from unittest.mock import MagicMock

from sapin_tts_python.config import SapinTTSConfig
from sapin_tts_python.sapin_tts import (
    EVENT_TTS_END,
    EVENT_TTS_ERROR,
    EVENT_TTS_RESPONSE,
    EVENT_TTS_TTFB_METRIC,
    SapinTTSClient,
    SapinTTSError,
    split_text,
)
from sapin_tts_python.tests.mock import FRAME, MockSapinServer


def make_client(server: MockSapinServer, statuses=None, **params):
    config = SapinTTSConfig(params={"url": server.url, "lang": "fil", **params})
    config.update_params()

    async def on_status(status, code, message):
        if statuses is not None:
            statuses.append(status)

    return SapinTTSClient(
        config=config, ten_env=MagicMock(), on_status=on_status
    )


async def collect(client: SapinTTSClient, text: str) -> list[tuple]:
    return [item async for item in client.get(text)]


def audio_of(events: list[tuple]) -> bytes:
    return b"".join(d for d, e in events if e == EVENT_TTS_RESPONSE)


async def wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


def test_split_text_respects_limit_and_prefers_sentences():
    assert split_text("   \n\t") == []
    assert split_text("Kumusta ka?") == ["Kumusta ka?"]

    sentence = "Ito ay isang pangungusap na may sapat na haba. "
    pieces = split_text(sentence * 30, limit=100)
    assert all(len(piece) <= 100 for piece in pieces)
    assert all(piece.endswith(".") for piece in pieces)
    assert " ".join(pieces) == (sentence * 30).strip()

    # A single run with no spaces is hard-cut.
    assert split_text("x" * 250, limit=100) == ["x" * 100, "x" * 100, "x" * 50]


def test_streams_pcm_with_ttfb_then_end_and_status():
    async def _run():
        async with MockSapinServer() as server:
            statuses = []
            client = make_client(server, statuses, voice="FIL_0003")
            await client.start()
            events = await collect(client, "Kumusta ka?")
            await client.stop()

            kinds = [e for _, e in events]
            assert kinds[0] == EVENT_TTS_TTFB_METRIC
            assert kinds[-1] == EVENT_TTS_END
            assert audio_of(events) == server.audio
            assert server.requests == [
                {"text": "Kumusta ka?", "lang": "fil", "voice": "FIL_0003"}
            ]
            assert statuses[:2] == ["connecting", "connected"]

    asyncio.run(_run())


def test_one_socket_serves_sequential_requests():
    async def _run():
        async with MockSapinServer() as server:
            client = make_client(server)
            await client.start()
            for text in ["Una.", "Ikalawa.", "Ikatlo."]:
                client.reset_ttfb()
                events = await collect(client, text)
                assert events[-1][1] == EVENT_TTS_END
            await client.stop()
            assert server.connections == 1
            assert [r["text"] for r in server.requests] == [
                "Una.",
                "Ikalawa.",
                "Ikatlo.",
            ]

    asyncio.run(_run())


def test_odd_frames_keep_samples_whole():
    async def respond(_request, ws, server):
        await ws.send(b"")
        await ws.send(b"\x01")
        await ws.send(b"\x02\x03\x04\x05")
        await server.send(ws, {"type": "end"})

    async def _run():
        async with MockSapinServer(respond=respond) as server:
            client = make_client(server)
            events = await collect(client, "Oo.")
            await client.stop()
            chunks = [d for d, e in events if e == EVENT_TTS_RESPONSE]
            assert all(len(c) % 2 == 0 for c in chunks)
            assert b"".join(chunks) == b"\x01\x02\x03\x04"
            # TTFB is reported only once real audio arrived.
            assert [e for _, e in events].count(EVENT_TTS_TTFB_METRIC) == 1

    asyncio.run(_run())


def test_empty_text_ends_without_request():
    async def _run():
        async with MockSapinServer() as server:
            client = make_client(server)
            await client.start()
            assert await collect(client, " \n\t") == [(None, EVENT_TTS_END)]
            await client.stop()
            assert server.requests == []

    asyncio.run(_run())


def test_long_text_is_split_into_server_sized_requests():
    async def _run():
        async with MockSapinServer() as server:
            client = make_client(server)
            text = "Ito ay isang mahabang pangungusap para sa pagsubok. " * 25
            events = await collect(client, text)
            await client.stop()
            assert events[-1][1] == EVENT_TTS_END
            assert len(server.requests) > 1
            assert all(len(r["text"]) <= 500 for r in server.requests)
            assert audio_of(events) == server.audio * len(server.requests)
            assert [e for _, e in events].count(EVENT_TTS_TTFB_METRIC) == 1

    asyncio.run(_run())


def test_unknown_lang_is_fatal():
    async def _run():
        async with MockSapinServer() as server:
            client = make_client(server, lang="xx")
            events = await collect(client, "Hello.")
            await client.stop()
            (error, kind) = events[-1]
            assert kind == EVENT_TTS_ERROR
            assert isinstance(error, SapinTTSError)
            assert error.fatal
            assert "unknown 'lang'" in error.message

    asyncio.run(_run())


def test_vendor_error_is_non_fatal_and_next_request_reconnects():
    async def respond(request, ws, server):
        if request["text"] == "BAD":
            await server.fail(ws, "synthesis failed")
        else:
            await server.stream(ws)

    async def _run():
        async with MockSapinServer(respond=respond) as server:
            client = make_client(server)
            events = await collect(client, "BAD")
            error, kind = events[-1]
            assert kind == EVENT_TTS_ERROR and not error.fatal
            client.reset_ttfb()
            events = await collect(client, "Mabuti.")
            await client.stop()
            assert events[-1][1] == EVENT_TTS_END
            assert audio_of(events) == server.audio
            assert server.connections >= 2

    asyncio.run(_run())


def test_idle_close_is_reported_and_reconnected_in_background():
    async def _run():
        async with MockSapinServer() as server:
            statuses = []
            client = make_client(server, statuses)
            await client.start()
            # The server drops the idle socket, as the real one does at 30 s.
            await server.sockets[0].close(1000)
            await wait_until(lambda: server.connections == 2)
            await wait_until(lambda: statuses[-1] == "connected")
            assert statuses == [
                "connecting",
                "connected",
                "disconnected",
                "connecting",
                "connected",
            ]
            events = await collect(client, "Oo.")
            await client.stop()
            assert events[-1][1] == EVENT_TTS_END

    asyncio.run(_run())


def test_close_racing_a_new_request_is_retried_once():
    attempts = []

    async def respond(request, ws, server):
        attempts.append(request["text"])
        if len(attempts) == 1:
            # Idle timeout fires just as the request arrives.
            await ws.close(1000)
            return
        await server.stream(ws)

    async def _run():
        async with MockSapinServer(respond=respond) as server:
            client = make_client(server)
            events = await collect(client, "Oo.")
            await client.stop()
            assert events[-1][1] == EVENT_TTS_END
            assert audio_of(events) == server.audio
            assert attempts == ["Oo.", "Oo."]

    asyncio.run(_run())


def test_close_after_audio_is_an_error_not_a_replay():
    async def respond(_request, ws, _server):
        await ws.send(FRAME)
        await ws.close(1011)

    async def _run():
        async with MockSapinServer(respond=respond) as server:
            client = make_client(server)
            events = await collect(client, "Oo.")
            await client.stop()
            assert audio_of(events) == FRAME
            assert events[-1][1] == EVENT_TTS_ERROR
            assert len(server.requests) == 1

    asyncio.run(_run())


def test_cancel_mid_stream_drops_late_audio_and_next_request_works():
    async def _run():
        async with MockSapinServer(
            audio=FRAME * 50, frame_delay=0.02
        ) as server:
            client = make_client(server)
            await client.start()
            received = []

            async def consume():
                async for payload, event in client.get("Mahaba."):
                    received.append(event)
                    if event == EVENT_TTS_RESPONSE and len(received) == 3:
                        await client.cancel()

            await asyncio.wait_for(consume(), 5)
            assert EVENT_TTS_END not in received
            assert received.count(EVENT_TTS_RESPONSE) == 2

            client.reset_ttfb()
            server.audio = FRAME
            events = await collect(client, "Susunod.")
            await client.stop()
            assert events[-1][1] == EVENT_TTS_END
            assert audio_of(events) == FRAME

    asyncio.run(_run())


def test_cancel_while_connecting_sends_nothing():
    async def _run():
        async with MockSapinServer() as server:
            client = make_client(server)
            task = asyncio.create_task(collect(client, "Huwag ipadala."))
            await asyncio.sleep(0)
            await client.cancel()
            events = await asyncio.wait_for(task, 5)
            await client.stop()
            assert events == []
            assert server.requests == []

    asyncio.run(_run())


def test_rejected_upgrade_is_fatal_for_client_errors():
    async def _run():
        async with MockSapinServer() as server:
            server.reject_status = 404
            client = make_client(server)
            try:
                await client.start()
                raise AssertionError("expected a fatal error")
            except SapinTTSError as e:
                assert e.fatal and e.code == "404"
            await client.stop()

    asyncio.run(_run())


def test_unreachable_server_is_non_fatal_and_stop_ends_retries():
    async def _run():
        statuses = []
        config = SapinTTSConfig(params={"url": "ws://127.0.0.1:9/tts"})
        config.update_params()

        async def on_status(status, code, message):
            statuses.append(status)

        client = SapinTTSClient(config, MagicMock(), on_status=on_status)
        await client.start()  # Does not raise; retries in the background.
        events = await collect(client, "Oo.")
        assert events[-1][1] == EVENT_TTS_ERROR
        assert not events[-1][0].fatal
        await client.stop()
        count = len(statuses)
        await asyncio.sleep(0.8)
        assert len(statuses) == count
        assert "connected" not in statuses

    asyncio.run(_run())
