"""Sapin TTS request frames and binary PCM audio frames."""

import json

from .base import ConnectionObserver


class SapinObserver(ConnectionObserver):
    def text_context(self, raw, request_id):
        if not isinstance(raw, str):
            return None
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(message, dict) and str(message.get("text") or "").strip():
            # Sapin serves one request at a time per connection.
            return request_id
        return None

    def audio_context(self, raw):
        # end/error events are JSON text; audio is raw binary PCM.
        if isinstance(raw, (bytes, bytearray)) and len(raw) > 0:
            return self.active_context
        return None
