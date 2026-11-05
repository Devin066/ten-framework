#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
import pytest

from sapin_tts_python.config import SapinTTSConfig


def make(**fields) -> SapinTTSConfig:
    config = SapinTTSConfig(**fields)
    config.update_params()
    return config


def test_params_become_fields_and_extras_pass_through():
    config = make(
        params={
            "url": "ws://host:8765/tts",
            "lang": "ceb",
            "voice": "CEB_0200",
            "speed": 1.1,
        }
    )
    config.validate_config()
    assert config.url == "ws://host:8765/tts"
    assert config.lang == "ceb"
    assert config.params == {"speed": 1.1}
    assert config.build_request("Maayong buntag.") == {
        "text": "Maayong buntag.",
        "lang": "ceb",
        "voice": "CEB_0200",
        "speed": 1.1,
    }


def test_top_level_url_wins_over_params_url():
    config = make(url="ws://top:1/tts", params={"url": "ws://nested:2/tts"})
    assert config.url == "ws://top:1/tts"
    assert "url" not in config.build_request("x")


def test_empty_env_values_keep_defaults():
    config = make(params={"url": "ws://h/tts", "lang": "", "voice": ""})
    assert config.lang == "fil"
    assert "voice" not in config.build_request("x")


def test_extra_params_cannot_override_text_or_lang():
    config = make(params={"url": "ws://h/tts", "text": "evil", "lang": "fil"})
    request = config.build_request("good")
    assert request["text"] == "good"


@pytest.mark.parametrize(
    "url",
    [
        "",
        "http://host/tts",
        "ws:///tts",
        "ws://user:pass@host/tts",
        "ws://host/tts?token=abc",
        "ws://host/tts#frag",
    ],
)
def test_invalid_urls_are_rejected_without_echo(url):
    config = make(params={"url": url})
    with pytest.raises(ValueError) as info:
        config.validate_config()
    assert "pass" not in str(info.value)
    assert "token=abc" not in str(info.value)


def test_to_str_redacts_sensitive_params():
    config = make(params={"url": "ws://h:8765/tts", "api_token": "s3cret"})
    text = config.to_str()
    assert "s3cret" not in text
    assert "ws://h:8765/tts" in text
