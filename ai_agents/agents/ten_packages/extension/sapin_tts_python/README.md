# Sapin TTS Extension

Text-to-speech for TEN Framework using the SapinSapin TTS WebSocket server,
which speaks Filipino (`fil`), Cebuano (`ceb`) and Hiligaynon (`hil`).

## Configuration

All settings live under `params` in `property.json`:

```json
{
  "params": {
    "url": "${env:SAPIN_TTS_URL|ws://104.208.87.149:8765/tts}",
    "lang": "${env:SAPIN_TTS_LANG|fil}",
    "voice": "${env:SAPIN_TTS_VOICE|}"
  }
}
```

| Param | Default | Description |
| --- | --- | --- |
| `url` | — | WebSocket endpoint (`ws://` or `wss://`). Required. Must not contain credentials, a query or a fragment. |
| `lang` | `fil` | `fil`, `ceb` or `hil`. An unknown language is reported as a fatal error. |
| `voice` | server default | Voice ID, for example `FIL_0003`, `FIL_0000`, `CEB_0200`, `CEB_0288`, `HIL_0624`, `HIL_0619`. The server falls back to its default for an unknown voice. |

Any other keys in `params` are forwarded in each request. Top-level `dump`
and `dump_path` write one PCM file per request ID when `dump` is true.

The server has no authentication, so there is no API key.

## Audio

The server always returns PCM S16LE, 16 kHz, mono. The extension emits it as
is, so `sample_rate` is not configurable.

## Server behavior the extension handles

- **Length limit.** Text over 500 characters is split at sentence or word
  boundaries and sent as consecutive requests.
- **Idle close.** The server closes a socket after about 30 seconds without a
  request, pings or not. The extension reports `disconnected` and reconnects
  right away in the background, so a ready socket is waiting for the next
  text. A request that loses its socket before any audio arrives is resent
  once on a fresh socket.
- **Errors.** `{"type":"error"}` closes the socket. The request ends with a
  `NON_FATAL_ERROR`, or a `FATAL_ERROR` for an unknown `lang`, and the next
  request uses a new socket.
- **Flush.** There is no cancel message, so a flush closes the socket to drop
  unread audio and a fresh one is opened.

Subtitle timestamps are not supported: the server returns no word or segment
timing.

## Environment Variables

- `SAPIN_TTS_URL` — WebSocket endpoint.
- `SAPIN_TTS_LANG` — default language.
- `SAPIN_TTS_VOICE` — default voice ID.

## Testing

Extension-local tests (in-process mock server, no network):

```bash
task test-extension EXTENSION=agents/ten_packages/extension/sapin_tts_python
```

TTS guarder against the real server:

```bash
task tts-guarder-test EXTENSION=sapin_tts_python CONFIG_DIR=tests/configs
```

The guarder's sample-rate comparison is disabled for this extension because
the server has a single output rate.
