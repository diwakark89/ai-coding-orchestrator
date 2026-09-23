"""Tests for structured output parsing utilities (agents/parser.py)."""

import pytest
from pydantic import BaseModel, Field

from agentflow.agents.parser import (
    extract_json_block,
    extract_outer_json_string,
    parse_json_lines,
    parse_model_into_schema,
    parse_structured_json,
    strip_markdown_fences,
)
from agentflow.errors import StructuredParsingError


class SampleTaskProfile(BaseModel):
    name: str
    complexity: int
    risk_flags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. strip_markdown_fences
# ---------------------------------------------------------------------------


def test_strip_markdown_fences_with_json_specifier():
    text = """Here is the result:
```json
{
  "name": "auth-service",
  "complexity": 3
}
```
Done."""
    stripped = strip_markdown_fences(text, language="json")
    assert stripped == '{\n  "name": "auth-service",\n  "complexity": 3\n}'


def test_strip_markdown_fences_without_specifier():
    text = """```
{"status": "ok"}
```"""
    stripped = strip_markdown_fences(text)
    assert stripped == '{"status": "ok"}'


def test_strip_markdown_fences_prefers_matching_language():
    text = """```yaml
name: test
```
```json
{"name": "json_test"}
```"""
    stripped = strip_markdown_fences(text, language="json")
    assert stripped == '{"name": "json_test"}'


def test_strip_markdown_fences_no_fences():
    text = "Just plain text without fences"
    assert strip_markdown_fences(text) == text
    assert strip_markdown_fences("") == ""
    assert strip_markdown_fences("   ") == ""


# ---------------------------------------------------------------------------
# 2. extract_outer_json_string
# ---------------------------------------------------------------------------


def test_extract_outer_json_string_object():
    text = 'Leading explanation: {"a": 1, "b": [2, 3]} Trailing explanation.'
    extracted = extract_outer_json_string(text)
    assert extracted == '{"a": 1, "b": [2, 3]}'


def test_extract_outer_json_string_array():
    text = 'Prefix: ["one", "two", {"three": 3}] Suffix.'
    extracted = extract_outer_json_string(text)
    assert extracted == '["one", "two", {"three": 3}]'


def test_extract_outer_json_string_with_braces_in_strings():
    text = 'Prefix {"message": "hello {world}", "code": "}"} Postfix'
    extracted = extract_outer_json_string(text)
    assert extracted == '{"message": "hello {world}", "code": "}"}'


def test_extract_outer_json_string_with_escaped_quotes():
    text = r'Prefix {"msg": "escaped \" quote and {brace}"} Postfix'
    extracted = extract_outer_json_string(text)
    assert extracted == r'{"msg": "escaped \" quote and {brace}"}'


def test_extract_outer_json_string_unbalanced_returns_none():
    assert extract_outer_json_string('{"broken": 1') is None
    assert extract_outer_json_string("no braces here") is None
    assert extract_outer_json_string("") is None


# ---------------------------------------------------------------------------
# 3. extract_json_block
# ---------------------------------------------------------------------------


def test_extract_json_block_fenced():
    text = '```json\n{"foo": "bar"}\n```'
    assert extract_json_block(text) == '{"foo": "bar"}'


def test_extract_json_block_unfenced_embedded():
    text = 'Sure! Here is the JSON: {"foo": "bar"} Let me know if you need more.'
    assert extract_json_block(text) == '{"foo": "bar"}'


def test_extract_json_block_empty():
    assert extract_json_block("") == ""
    assert extract_json_block("   ") == ""


# ---------------------------------------------------------------------------
# 4. parse_structured_json
# ---------------------------------------------------------------------------


def test_parse_structured_json_direct():
    data = parse_structured_json('{"key": "value", "num": 42}')
    assert data == {"key": "value", "num": 42}


def test_parse_structured_json_fenced():
    text = """```json
{"items": [1, 2, 3]}
```"""
    data = parse_structured_json(text)
    assert data == {"items": [1, 2, 3]}


def test_parse_structured_json_embedded_in_prose():
    text = """I have analyzed your request. Here is the decision payload:
{"decision": "PROCEED", "model": "GPT-6 Sol"}
Please confirm if this works."""
    data = parse_structured_json(text)
    assert data == {"decision": "PROCEED", "model": "GPT-6 Sol"}


def test_parse_structured_json_empty_raises_error():
    with pytest.raises(StructuredParsingError, match="empty text"):
        parse_structured_json("")
    with pytest.raises(StructuredParsingError, match="empty text"):
        parse_structured_json("   \n\t  ")


def test_parse_structured_json_invalid_raises_error():
    with pytest.raises(StructuredParsingError, match="Failed to parse structured JSON"):
        parse_structured_json("This is purely prose without any valid JSON structure.")


# ---------------------------------------------------------------------------
# 5. parse_json_lines
# ---------------------------------------------------------------------------


def test_parse_json_lines_valid():
    jsonl = """
{"event": "start", "id": 1}
{"event": "chunk", "data": "hello"}

{"event": "end", "status": "ok"}
"""
    events = parse_json_lines(jsonl)
    assert len(events) == 3
    assert events[0]["event"] == "start"
    assert events[1]["data"] == "hello"
    assert events[2]["event"] == "end"


def test_parse_json_lines_ignore_invalid_true():
    jsonl = """
{"event": "start"}
Not JSON at all
{"event": "end"}
"""
    events = parse_json_lines(jsonl, ignore_invalid=True)
    assert len(events) == 2
    assert events[0]["event"] == "start"
    assert events[1]["event"] == "end"


def test_parse_json_lines_ignore_invalid_false_raises():
    jsonl = """
{"event": "start"}
Not JSON at all
"""
    with pytest.raises(StructuredParsingError, match="not valid JSON"):
        parse_json_lines(jsonl, ignore_invalid=False)


def test_parse_json_lines_non_object_with_ignore_invalid_false():
    jsonl = "[1, 2, 3]\n"
    with pytest.raises(StructuredParsingError, match="not a JSON object"):
        parse_json_lines(jsonl, ignore_invalid=False)


# ---------------------------------------------------------------------------
# 6. parse_model_into_schema
# ---------------------------------------------------------------------------


def test_parse_model_into_schema_success():
    text = """```json
{
  "name": "payment-api",
  "complexity": 4,
  "risk_flags": ["payments", "security"]
}
```"""
    model = parse_model_into_schema(text, SampleTaskProfile)
    assert isinstance(model, SampleTaskProfile)
    assert model.name == "payment-api"
    assert model.complexity == 4
    assert model.risk_flags == ["payments", "security"]


def test_parse_model_into_schema_validation_failure():
    # 'complexity' should be int, given unparseable value or missing required 'name'
    text = '{"complexity": "high"}'
    with pytest.raises(StructuredParsingError, match="Failed to validate data against schema"):
        parse_model_into_schema(text, SampleTaskProfile)


def test_parse_model_into_schema_non_dict_json():
    text = '["not", "a", "dict"]'
    with pytest.raises(StructuredParsingError, match="Expected JSON object"):
        parse_model_into_schema(text, SampleTaskProfile)
