"""Tests for retry and JSON repair utilities."""

import pytest
from tools.retry import repair_json, parse_llm_json, with_retry


def test_repair_markdown_fence():
    raw = '```json\n{"compounds": ["SOFT", "HARD"]}\n```'
    result = parse_llm_json(raw)
    assert result["compounds"] == ["SOFT", "HARD"]
    print("\n✅ Markdown fence stripped and parsed")


def test_repair_preamble():
    raw = 'Sure! Here is the strategy:\n{"pit_laps": [35], "confidence": "high"}'
    result = parse_llm_json(raw)
    assert result["pit_laps"] == [35]
    print("\n✅ Preamble stripped and parsed")


def test_repair_trailing_text():
    raw = '{"compounds": ["SOFT", "HARD"]} Let me know if you need anything else!'
    result = parse_llm_json(raw)
    assert result["compounds"] == ["SOFT", "HARD"]
    print("\n✅ Trailing text stripped and parsed")


def test_clean_json_passes_through():
    raw = '{"compounds": ["SOFT", "HARD"], "pit_laps": [35]}'
    result = parse_llm_json(raw)
    assert result["pit_laps"] == [35]
    print("\n✅ Clean JSON passes through unchanged")


def test_with_retry_succeeds_on_second_attempt():
    attempts = {"count": 0}

    @with_retry(max_attempts=3, initial_delay=0.01)
    def flaky_function():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise ConnectionError("transient failure")
        return "success"

    result = flaky_function()
    assert result == "success"
    assert attempts["count"] == 2
    print(f"\n✅ Succeeded on attempt {attempts['count']}")


def test_with_retry_exhausted():
    @with_retry(max_attempts=3, initial_delay=0.01)
    def always_fails():
        raise ConnectionError("always fails")

    with pytest.raises(ConnectionError):
        always_fails()
    print("\n✅ Correctly raised after exhausting retries")