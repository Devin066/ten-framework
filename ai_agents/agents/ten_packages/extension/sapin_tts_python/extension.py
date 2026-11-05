#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import os
import time
import traceback

from ten_ai_base.const import LOG_CATEGORY_KEY_POINT, LOG_CATEGORY_TRANSCRIPTS
from ten_ai_base.helper import PCMWriter
from ten_ai_base.message import (
    ModuleError,
    ModuleErrorCode,
    ModuleErrorVendorInfo,
    ModuleType,
    TTSAudioEndReason,
)
from ten_ai_base.struct import TTSTextInput
from ten_ai_base.tts2 import AsyncTTS2BaseExtension
from ten_runtime import AsyncTenEnv

from .config import SAMPLE_RATE, SapinTTSConfig, redact_url
from .sapin_tts import (
    EVENT_TTS_END,
    EVENT_TTS_ERROR,
    EVENT_TTS_RESPONSE,
    EVENT_TTS_TTFB_METRIC,
    SapinTTSClient,
    SapinTTSError,
)


class SapinTTSExtension(AsyncTTS2BaseExtension):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.config: SapinTTSConfig | None = None
        self.client: SapinTTSClient | None = None
        self._config_error: str | None = None
        self._is_stopped = False

        self.current_request_id: str | None = None
        self.current_request_finished = False
        self.total_audio_bytes = 0
        self.recorder_map: dict[str, PCMWriter] = {}
        # Monotonic time tts_audio_start was sent for the current request.
        self._audio_start_ts: float | None = None
        self._request_finalized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_init(self, ten_env: AsyncTenEnv) -> None:
        await super().on_init(ten_env)
        try:
            config_json_str, _ = await self.ten_env.get_property_to_json("")
            if not config_json_str or config_json_str.strip() == "{}":
                raise ValueError(
                    "Configuration is empty. Required parameter 'url' is missing."
                )
            config = SapinTTSConfig.model_validate_json(config_json_str)
            config.update_params()
            # Validate before logging, so a URL that validation rejects
            # never reaches the log.
            config.validate_config()
            self.config = config
            ten_env.log_info(
                f"config: {config.to_str(sensitive_handling=True)}",
                category=LOG_CATEGORY_KEY_POINT,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._config_error = f"Initialization failed: {e}"
            ten_env.log_error(f"on_init failed: {traceback.format_exc()}")
            await self.send_tts_error(
                request_id="",
                error=self._fatal_error(self._config_error),
            )

    async def on_start(self, ten_env: AsyncTenEnv) -> None:
        await super().on_start(ten_env)
        if self.config is None or self._is_stopped:
            return
        self.client = SapinTTSClient(
            config=self.config, ten_env=ten_env, on_status=self._on_status
        )
        try:
            await self.client.start()
        except SapinTTSError as e:
            self._config_error = e.message
            await self.send_tts_error(
                request_id="", error=self._fatal_error(e.message, e.code)
            )

    async def on_stop(self, ten_env: AsyncTenEnv) -> None:
        self._is_stopped = True
        if self.client:
            await self.client.stop()
            self.client = None
        for request_id, recorder in list(self.recorder_map.items()):
            try:
                await recorder.flush()
            except Exception as e:  # pylint: disable=broad-exception-caught
                ten_env.log_error(
                    f"Error flushing PCMWriter for {request_id}: {e}"
                )
        self.recorder_map.clear()
        await super().on_stop(ten_env)

    async def on_deinit(self, ten_env: AsyncTenEnv) -> None:
        await super().on_deinit(ten_env)

    def vendor(self) -> str:
        return "sapin"

    def synthesize_audio_sample_rate(self) -> int:
        return SAMPLE_RATE

    def vendor_metadata(self) -> dict:
        if self.config is None:
            return {}
        metadata = {
            "url": redact_url(self.config.url),
            "lang": self.config.lang,
        }
        if self.config.voice:
            metadata["voice"] = self.config.voice
        return metadata

    async def _on_status(self, status: str, code: int, message: str) -> None:
        if status == "connecting":
            await self.on_connecting()
        elif status == "connected":
            await self.on_connected()
        else:
            await self.on_disconnected(code=code, message=message or "closed")

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    async def cancel_tts(self) -> None:
        """Flush: stop provider work and end the active request.

        The base class owns flush cleanup and sends tts_flush_end, so this
        does not call finish_request().
        """
        self.current_request_finished = True
        if self.client:
            await self.client.cancel()
        if self.current_request_id and not self._request_finalized:
            await self._end_request(TTSAudioEndReason.INTERRUPTED, finish=False)

    async def request_tts(self, t: TTSTextInput) -> None:
        try:
            self.ten_env.log_info(
                f"Requesting TTS for text: {t.text}, "
                f"text_input_end: {t.text_input_end}, "
                f"request_id: {t.request_id}",
                category=LOG_CATEGORY_TRANSCRIPTS,
            )

            if t.request_id != self.current_request_id:
                self._begin_request(t.request_id)
                await self._setup_recorder(t.request_id)
            elif self.current_request_finished:
                self.ten_env.log_warn(
                    f"Ignoring text for finished request {t.request_id}"
                )
                return

            if t.text_input_end:
                self.current_request_finished = True

            if self._config_error or self.client is None:
                await self._end_request(
                    TTSAudioEndReason.ERROR,
                    error=self._fatal_error(
                        self._config_error or "Sapin TTS is not configured"
                    ),
                )
                return

            if self._is_stopped:
                return

            if t.text.strip():
                await self._synthesize(t)

            if t.text_input_end and not self._request_finalized:
                await self._end_request(TTSAudioEndReason.REQUEST_END)

        except Exception as e:  # pylint: disable=broad-exception-caught
            if self._request_finalized:
                self.ten_env.log_debug(f"request_tts ended after finalize: {e}")
                return
            self.ten_env.log_error(
                f"Error in request_tts: {traceback.format_exc()}",
                category=LOG_CATEGORY_TRANSCRIPTS,
            )
            await self._end_request(
                TTSAudioEndReason.ERROR,
                error=self._non_fatal_error(str(e)),
            )

    def _begin_request(self, request_id: str) -> None:
        self.ten_env.log_debug(f"New TTS request with ID: {request_id}")
        if self.client:
            self.client.reset_ttfb()
        self.current_request_id = request_id
        self.current_request_finished = False
        self.total_audio_bytes = 0
        self._audio_start_ts = None
        self._request_finalized = False

    async def _synthesize(self, t: TTSTextInput) -> None:
        async for payload, event in self.client.get(t.text):
            if self._request_finalized:
                self.ten_env.log_debug("Request already finalized, dropping")
                break

            if event == EVENT_TTS_TTFB_METRIC and isinstance(payload, int):
                await self._start_audio()
                await self.send_tts_ttfb_metrics(
                    request_id=self.current_request_id,
                    ttfb_ms=payload,
                    extra_metadata={
                        "lang": self.config.lang,
                        "voice": self.config.voice,
                    },
                )

            elif event == EVENT_TTS_RESPONSE and isinstance(payload, bytes):
                if not payload:
                    continue
                await self._start_audio()
                self.total_audio_bytes += len(payload)
                await self._write_dump(payload)
                await self.send_tts_audio_data(payload)

            elif event == EVENT_TTS_END:
                break

            elif event == EVENT_TTS_ERROR:
                err = (
                    payload
                    if isinstance(payload, SapinTTSError)
                    else SapinTTSError(str(payload))
                )
                error = (
                    self._fatal_error(err.message, err.code)
                    if err.fatal
                    else self._non_fatal_error(err.message, err.code)
                )
                if t.text_input_end or err.fatal:
                    await self._end_request(
                        TTSAudioEndReason.ERROR, error=error
                    )
                else:
                    # Report it and keep the request open for later text.
                    await self.send_tts_error(
                        request_id=t.request_id, error=error
                    )
                break

    async def _start_audio(self) -> None:
        if self._audio_start_ts is not None:
            return
        self._audio_start_ts = time.monotonic()
        await self.send_tts_audio_start(request_id=self.current_request_id)

    async def _end_request(
        self,
        reason: TTSAudioEndReason,
        error: ModuleError | None = None,
        finish: bool = True,
    ) -> None:
        """Send the one tts_audio_end for this request, then finish it."""
        if self._request_finalized:
            return
        self._request_finalized = True
        # Later text for this request ID is ignored.
        self.current_request_finished = True
        request_id = self.current_request_id

        # Requests with no audio (empty text, error, flush) still get a
        # start/end pair; the interval then starts here.
        await self._start_audio()
        interval_ms = int((time.monotonic() - self._audio_start_ts) * 1000)
        duration_ms = int(self.total_audio_bytes * 1000 / (SAMPLE_RATE * 2))

        # The error goes out before the audio end, so it is reported here
        # rather than through finish_request().
        if error is not None:
            await self.send_tts_error(request_id=request_id, error=error)

        await self.send_tts_audio_end(
            request_id=request_id,
            request_event_interval_ms=interval_ms,
            request_total_audio_duration_ms=duration_ms,
            reason=reason,
        )

        recorder = self.recorder_map.pop(request_id, None)
        if recorder is not None:
            # A dump failure must never keep the request from finishing.
            try:
                await recorder.flush()
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.ten_env.log_error(f"Dump flush failed: {e}")

        if finish:
            await self.finish_request(request_id=request_id, reason=reason)

        self.ten_env.log_debug(
            f"Finalized request {request_id}, reason: {reason}, "
            f"interval: {interval_ms}ms, duration: {duration_ms}ms"
        )

    # ------------------------------------------------------------------
    # Errors and dump
    # ------------------------------------------------------------------

    def _fatal_error(self, message: str, code: str = "") -> ModuleError:
        return ModuleError(
            message=message,
            module=ModuleType.TTS,
            code=ModuleErrorCode.FATAL_ERROR,
            vendor_info=ModuleErrorVendorInfo(
                vendor=self.vendor(), code=code, message=message
            ),
        )

    def _non_fatal_error(self, message: str, code: str = "") -> ModuleError:
        return ModuleError(
            message=message,
            module=ModuleType.TTS,
            code=ModuleErrorCode.NON_FATAL_ERROR,
            vendor_info=ModuleErrorVendorInfo(
                vendor=self.vendor(), code=code, message=message
            ),
        )

    async def _setup_recorder(self, request_id: str) -> None:
        if not (self.config and self.config.dump):
            return
        for old_id in [rid for rid in self.recorder_map if rid != request_id]:
            try:
                await self.recorder_map.pop(old_id).flush()
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.ten_env.log_error(f"Error closing dump {old_id}: {e}")
        if request_id not in self.recorder_map:
            os.makedirs(self.config.dump_path, exist_ok=True)
            self.recorder_map[request_id] = PCMWriter(
                os.path.join(
                    self.config.dump_path, f"sapin_dump_{request_id}.pcm"
                )
            )

    async def _write_dump(self, data: bytes) -> None:
        recorder = self.recorder_map.get(self.current_request_id or "")
        if recorder is None:
            return
        try:
            await recorder.write(data)
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.ten_env.log_error(f"Dump write failed: {e}")
