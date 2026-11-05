"""Sapin end/error frames must not count as audio."""

import json

from .ttfb_reference import TTFBReference
from .ttfb_hooks.sapin import SapinObserver


def request(text):
    return json.dumps({"text": text, "lang": "fil"})


def test_sapin_reference_counts_request_to_first_binary_audio(monkeypatch):
    clock = [0]
    monkeypatch.setattr(
        "tests.ttfb_hooks.base.time.monotonic_ns", lambda: clock[0]
    )
    reference = TTFBReference()
    observer = SapinObserver(reference)

    # Two sequential requests on the same socket.
    for request_id in ["first", "second"]:
        reference.begin_request(request_id)
        clock[0] += 100_000_000
        observer.observe_send(request("   "))
        clock[0] += 50_000_000
        observer.observe_send(request("Kumusta ka?"))
        observer.observe_receive(json.dumps({"type": "error", "message": "x"}))
        observer.observe_receive(b"")
        clock[0] += 30_000_000
        observer.observe_receive(b"\x01\x00" * 1600)
        clock[0] += 80_000_000
        observer.observe_receive(b"\x02\x00" * 1600)
        observer.observe_receive(json.dumps({"type": "end"}))
        assert reference.elapsed_ms(request_id) == 30


def test_sapin_reference_is_isolated_per_connection(monkeypatch):
    clock = [0]
    monkeypatch.setattr(
        "tests.ttfb_hooks.base.time.monotonic_ns", lambda: clock[0]
    )
    reference = TTFBReference()
    old_socket = SapinObserver(reference)
    new_socket = SapinObserver(reference)

    reference.begin_request("first")
    old_socket.observe_send(request("Una."))
    # The idle socket closed under the request; it is resent on a new one.
    clock[0] += 40_000_000
    new_socket.observe_send(request("Una."))
    clock[0] += 25_000_000
    new_socket.observe_receive(b"\x01\x00" * 1600)
    # Audio on a socket with no request is never attributed to a request.
    SapinObserver(reference).observe_receive(b"\x01\x00" * 1600)

    assert reference.elapsed_ms("first") == 65
