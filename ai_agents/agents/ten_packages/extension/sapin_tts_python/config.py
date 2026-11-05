#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
from __future__ import annotations

import copy
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ten_ai_base import utils

# The Sapin server always returns PCM S16LE, 16 kHz, mono.
SAMPLE_RATE = 16000
# The server rejects text longer than this.
MAX_TEXT_CHARS = 500
DEFAULT_LANG = "fil"

_SENSITIVE_SUBSTRINGS = ("key", "token", "secret", "signature", "password")

# Keys moved out of `params` into first-class fields. Everything else left
# in `params` is added to each synthesis request.
CONFIG_KEYS = ("lang", "voice")
ENDPOINT_KEYS = ("url",)


def _is_sensitive_key(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in _SENSITIVE_SUBSTRINGS)


def redact_url(url: str) -> str:
    """Keep only scheme, host, port and path of a URL for logging."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
    except ValueError:
        return "<invalid url>"
    redacted = f"{parsed.scheme}://{host}{port}{parsed.path}"
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        redacted += " (credentials, query or fragment redacted)"
    return redacted


class SapinTTSConfig(BaseModel):
    url: str = ""
    lang: str = DEFAULT_LANG
    voice: str = ""

    dump: bool = False
    dump_path: str = "/tmp"
    params: dict[str, Any] = Field(default_factory=dict)

    def update_params(self) -> None:
        params = self.params if isinstance(self.params, dict) else {}
        self.params = params

        # A non-empty top-level url wins over params.url.
        params_url = params.pop("url", None)
        if not self.url and params_url:
            self.url = str(params_url)

        for key in CONFIG_KEYS:
            if key in params:
                value = params.pop(key)
                # Empty env values (`${env:VAR|}`) keep the defaults.
                if value is not None and value != "":
                    setattr(self, key, str(value))

        self.url = str(self.url or "").strip()
        self.lang = str(self.lang or DEFAULT_LANG).strip()
        self.voice = str(self.voice or "").strip()

    def validate_config(self) -> None:
        if not self.url:
            raise ValueError("Sapin TTS url is required")
        try:
            parsed = urlparse(self.url)
            parsed.port  # pylint: disable=pointless-statement
        except ValueError as e:
            raise ValueError("Sapin TTS url is not a valid URL") from e
        # The URL is logged and reported as vendor metadata, so it must not
        # carry credentials. Do not echo it in the error.
        if parsed.username or parsed.password or parsed.query:
            raise ValueError(
                "Sapin TTS url must not contain credentials or a query"
            )
        if parsed.fragment:
            raise ValueError("Sapin TTS url must not contain a fragment")
        if parsed.scheme not in ("ws", "wss") or not parsed.hostname:
            raise ValueError("Sapin TTS url must be a ws:// or wss:// URL")
        if not self.lang:
            raise ValueError("Sapin TTS lang is required")

    def build_request(self, text: str) -> dict[str, Any]:
        request: dict[str, Any] = {"text": text, "lang": self.lang}
        if self.voice:
            request["voice"] = self.voice
        for key, value in self.params.items():
            if key in ENDPOINT_KEYS or key in request or value is None:
                continue
            request[key] = value
        return request

    def to_str(self, sensitive_handling: bool = True) -> str:
        if not sensitive_handling:
            return f"{self}"
        config = copy.deepcopy(self)
        config.url = redact_url(config.url)
        for key, value in list(config.params.items()):
            if value and _is_sensitive_key(key):
                config.params[key] = utils.encrypt(str(value))
        return f"{config}"
