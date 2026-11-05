#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
"""SapinTTSExtension through the TEN runtime against a mock Sapin server."""

import json
import os
import shutil
import tempfile

from ten_runtime import TenEnvTester

from sapin_tts_python.tests.mock import (
    FRAME,
    RecordingTester,
    ThreadedMockSapinServer,
    config_for,
    send_flush,
    send_text,
)

REQUEST_END, INTERRUPTED, ERROR = 1, 2, 3
FATAL, NON_FATAL = -1000, 1000
BYTES_PER_MS = 16000 * 2 // 1000


def run(tester: RecordingTester, config: dict) -> RecordingTester:
    tester.set_test_mode_single("sapin_tts_python", json.dumps(config))
    error = tester.run()
    assert error is None, error.error_message() if error else ""
    return tester


def names(tester: RecordingTester) -> list[str]:
    return [name for name, _ in tester.timeline]


def test_basic_request_event_order_and_accounting():
    with ThreadedMockSapinServer() as server:
        tester = run(
            RecordingTester([("req-1", "Kumusta ka?", True)]),
            config_for(server.url),
        )

    order = [n for n in names(tester) if n.startswith("tts_") or n == "audio"]
    assert order[0] == "tts_audio_start"
    assert order[-1] == "tts_audio_end"
    assert set(order[1:-1]) == {"audio"}

    assert tester.audio == server.audio
    assert tester.sample_rates == {16000}
    end = tester.end_of("req-1")
    assert end["reason"] == REQUEST_END
    assert end["request_total_audio_duration_ms"] == (
        len(server.audio) // BYTES_PER_MS
    )
    assert end["request_event_interval_ms"] >= 0

    ttfb = [
        m for m in tester.payloads("metrics") if "ttfb" in m.get("metrics", {})
    ]
    assert len(ttfb) == 1
    assert ttfb[0]["metrics"]["ttfb"] >= 0


def test_appended_input_has_one_audio_lifecycle():
    with ThreadedMockSapinServer() as server:
        tester = run(
            RecordingTester(
                [
                    ("req-1", "Una.", False),
                    ("req-1", "Ikalawa.", False),
                    ("req-1", "Ikatlo.", True),
                ]
            ),
            config_for(server.url),
        )

    assert [r["text"] for r in server.requests] == [
        "Una.",
        "Ikalawa.",
        "Ikatlo.",
    ]
    assert tester.count("tts_audio_start") == 1
    assert tester.end_of("req-1")["reason"] == REQUEST_END
    assert tester.audio == server.audio * 3


def test_second_request_runs_after_first_completes():
    with ThreadedMockSapinServer() as server:
        tester = run(
            RecordingTester(
                [("req-1", "Una.", True), ("req-2", "Ikalawa.", True)],
                expected_ends=2,
            ),
            config_for(server.url),
        )

    timeline = tester.timeline
    assert timeline.index(("tts_audio_end", "req-1")) < timeline.index(
        ("tts_audio_start", "req-2")
    )
    assert tester.end_of("req-2")["reason"] == REQUEST_END
    assert server.connections == 1


def test_empty_text_ends_request_without_audio():
    with ThreadedMockSapinServer() as server:
        tester = run(
            RecordingTester([("req-1", "  \n", True)]),
            config_for(server.url),
        )

    assert server.requests == []
    assert tester.audio == b""
    assert tester.count("error") == 0
    end = tester.end_of("req-1")
    assert end["reason"] == REQUEST_END
    assert end["request_total_audio_duration_ms"] == 0


def test_error_before_final_text_keeps_request_open():
    async def respond(request, ws, server):
        if request["text"] == "BAD":
            await server.fail(ws, "synthesis failed")
        else:
            await server.stream(ws)

    with ThreadedMockSapinServer(respond=respond) as server:
        tester = run(
            RecordingTester(
                [("req-1", "BAD", False), ("req-1", "Mabuti.", True)]
            ),
            config_for(server.url),
        )

    errors = tester.payloads("error")
    assert len(errors) == 1
    assert errors[0]["code"] == NON_FATAL
    assert errors[0]["vendor_info"]["vendor"] == "sapin"
    assert tester.end_of("req-1")["reason"] == REQUEST_END
    assert tester.audio == server.audio


def test_failed_request_releases_buffered_request():
    async def respond(request, ws, server):
        if request["text"] == "BAD":
            await server.fail(ws, "synthesis failed")
        else:
            await server.stream(ws)

    with ThreadedMockSapinServer(respond=respond) as server:
        tester = run(
            RecordingTester(
                [("req-1", "BAD", True), ("req-2", "Mabuti.", True)],
                expected_ends=2,
            ),
            config_for(server.url),
        )

    timeline = tester.timeline
    assert timeline.index(("error", "req-1")) < timeline.index(
        ("tts_audio_end", "req-1")
    )
    assert tester.end_of("req-1")["reason"] == ERROR
    assert tester.end_of("req-2")["reason"] == REQUEST_END
    assert tester.audio == server.audio


def test_unknown_lang_is_fatal():
    with ThreadedMockSapinServer() as server:
        tester = run(
            RecordingTester([("req-1", "Hello.", True)]),
            config_for(server.url, lang="invalid"),
        )

    errors = tester.payloads("error")
    assert [e["code"] for e in errors] == [FATAL]
    assert "unknown 'lang'" in errors[0]["message"]
    assert tester.end_of("req-1")["reason"] == ERROR


def test_missing_url_is_fatal_and_requests_still_end():
    tester = run(
        RecordingTester([("req-1", "Hello.", True)]),
        {"params": {"url": ""}},
    )
    errors = tester.payloads("error")
    assert errors and all(e["code"] == FATAL for e in errors)
    assert tester.end_of("req-1")["reason"] == ERROR


def test_url_with_query_is_rejected_without_echoing_it():
    tester = run(
        RecordingTester([("req-1", "Hello.", True)]),
        {"params": {"url": "ws://example.test/tts?token=SECRET"}},
    )
    errors = tester.payloads("error")
    assert errors and errors[0]["code"] == FATAL
    assert all("SECRET" not in json.dumps(e) for e in errors)


class FlushTester(RecordingTester):
    """Flush req-1 after its first audio frame, then send req-2."""

    def __init__(self):
        super().__init__([("req-1", "Mahabang pangungusap.", True)])
        self.flushed = False
        self.req1_audio = 0

    def on_audio_frame(self, ten_env: TenEnvTester, audio_frame) -> None:
        super().on_audio_frame(ten_env, audio_frame)
        if ("tts_audio_end", "req-1") not in self.timeline:
            self.req1_audio += self.timeline[-1][1]
        if not self.flushed:
            self.flushed = True
            send_flush(ten_env, "flush-1")

    def on_data(self, ten_env: TenEnvTester, data) -> None:
        super().on_data(ten_env, data)
        name = data.get_name()
        if name == "tts_flush_end":
            send_text(ten_env, "req-2", "Susunod.", True)
        if ("tts_audio_end", "req-2") in self.timeline:
            ten_env.stop_test()


def test_flush_interrupts_drops_late_audio_and_next_request_works():
    with ThreadedMockSapinServer(audio=FRAME * 40, frame_delay=0.02) as server:
        tester = FlushTester()
        tester.expected_ends = 99
        run(tester, config_for(server.url))

    timeline = tester.timeline
    end1 = timeline.index(("tts_audio_end", "req-1"))
    flush_end = timeline.index(("tts_flush_end", None))
    start2 = timeline.index(("tts_audio_start", "req-2"))
    assert end1 < flush_end < start2
    # No req-1 audio after its interrupted end.
    assert all(name != "audio" for name, _ in timeline[end1:start2])

    interrupted = tester.end_of("req-1")
    assert interrupted["reason"] == INTERRUPTED
    assert interrupted["request_total_audio_duration_ms"] == (
        tester.req1_audio // BYTES_PER_MS
    )
    assert tester.end_of("req-2")["reason"] == REQUEST_END
    # The flushed stream's socket was dropped and a fresh one opened.
    assert server.connections >= 2


def test_dump_writes_exactly_the_emitted_audio():
    dump_dir = tempfile.mkdtemp(prefix="sapin_dump_")
    try:
        with ThreadedMockSapinServer() as server:
            tester = run(
                RecordingTester([("req-1", "Kumusta?", True)]),
                {**config_for(server.url), "dump": True, "dump_path": dump_dir},
            )
        path = os.path.join(dump_dir, "sapin_dump_req-1.pcm")
        with open(path, "rb") as f:
            assert f.read() == tester.audio
    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)


def test_no_dump_file_when_disabled():
    dump_dir = tempfile.mkdtemp(prefix="sapin_nodump_")
    try:
        with ThreadedMockSapinServer() as server:
            run(
                RecordingTester([("req-1", "Kumusta?", True)]),
                {**config_for(server.url), "dump_path": dump_dir},
            )
        assert os.listdir(dump_dir) == []
    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)


def test_connection_status_reports_connecting_then_connected():
    with ThreadedMockSapinServer() as server:
        tester = run(
            RecordingTester([("req-1", "Oo.", True)]),
            config_for(server.url),
        )

    statuses = [
        p["current"] for p in tester.payloads("connection_status_changed")
    ]
    assert statuses[:2] == ["connecting", "connected"]
    for payload in tester.payloads("connection_status_changed"):
        assert payload["vendor_info"]["vendor"] == "sapin"
