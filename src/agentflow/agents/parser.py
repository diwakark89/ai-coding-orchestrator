"""Structured output parsing utilities for AI agent responses."""

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from agentflow.errors import StructuredParsingError

T = TypeVar("T", bound=BaseModel)

# Regex matching markdown code blocks with optional language specifier
MARKDOWN_FENCE_PATTERN = re.compile(
    r"```(?:([a-zA-Z0-9_-]+))?\s*\n?(.*?)\n?```",
    re.DOTALL,
)


def strip_markdown_fences(text: str, language: str | None = None) -> str:
    """Strip markdown code fence blocks from text.

    Args:
        text: Raw text potentially wrapped in markdown code fences.
        language: Optional language specifier (e.g. 'json', 'yaml') to match preferentially.

    Returns:
        Content extracted from the code fence, or the stripped text if no fence was matched.
    """
    clean = text.strip()
    if not clean:
        return ""

    matches = list(MARKDOWN_FENCE_PATTERN.finditer(clean))
    if not matches:
        return clean

    if language:
        lang_lower = language.lower()
        for match in matches:
            tag = (match.group(1) or "").lower()
            if tag == lang_lower:
                return match.group(2).strip()

    # Fall back to the first code fence found
    return matches[0].group(2).strip()


def extract_outer_json_string(text: str) -> str | None:
    """Scan text to extract the outermost balanced JSON object `{...}` or array `[...]`.

    Properly handles escaped quotes and braces nested within string literals.

    Args:
        text: Arbitrary text containing an embedded JSON block.

    Returns:
        The extracted JSON substring, or None if no balanced JSON structure is found.
    """
    start_brace = text.find("{")
    start_bracket = text.find("[")

    if start_brace == -1 and start_bracket == -1:
        return None

    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start_idx = start_brace
        open_char = "{"
        close_char = "}"
    else:
        start_idx = start_bracket
        open_char = "["
        close_char = "]"

    depth = 0
    in_string = False
    escape = False

    for i in range(start_idx, len(text)):
        char = text[i]
        if escape:
            escape = False
            continue
        if char == "\\":
            if in_string:
                escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx : i + 1]

    return None


def extract_json_block(text: str) -> str:
    """Extract candidate JSON substring from text.

    Attempts extraction via:
    1. Markdown code fences with optional 'json' tag.
    2. Outermost balanced JSON object or array scan.
    3. Falling back to stripped input text.

    Args:
        text: Raw agent output.

    Returns:
        Clean candidate JSON string.
    """
    clean = text.strip()
    if not clean:
        return ""

    # Check for markdown code fences first
    fenced = strip_markdown_fences(clean, language="json")
    if fenced != clean:
        return fenced

    # If no fences, try scanning for balanced outer JSON structure
    outer = extract_outer_json_string(clean)
    if outer is not None:
        return outer

    return clean


def parse_structured_json(text: str) -> Any:
    """Parse structured JSON from raw agent output with resilient fallback strategies.

    Args:
        text: Raw output containing JSON (directly, within code fences, or embedded in prose).

    Returns:
        Parsed JSON data (dict, list, or scalar).

    Raises:
        StructuredParsingError: If no valid JSON can be extracted or parsed.
    """
    clean = text.strip()
    if not clean:
        raise StructuredParsingError("Cannot parse structured JSON from empty text.")

    # Strategy 1: Direct JSON parsing
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Markdown fence stripping
    fenced = strip_markdown_fences(clean, language="json")
    if fenced and fenced != clean:
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Outermost balanced brace extraction
    outer = extract_outer_json_string(clean)
    if outer and outer != clean:
        try:
            return json.loads(outer)
        except json.JSONDecodeError:
            pass

    # Strategy 4: If fenced had outer text, try balanced brace extraction on the fenced content
    if fenced and fenced != clean:
        outer_fenced = extract_outer_json_string(fenced)
        if outer_fenced:
            try:
                return json.loads(outer_fenced)
            except json.JSONDecodeError:
                pass

    raise StructuredParsingError(
        f"Failed to parse structured JSON from text:\n{clean[:500]}"
        + ("..." if len(clean) > 500 else "")
    )


def parse_json_lines(text: str, ignore_invalid: bool = True) -> list[dict[str, Any]]:
    """Parse JSON Lines (JSONL) formatted output into a list of dictionaries.

    Args:
        text: Raw output containing one JSON object per line.
        ignore_invalid: If True, skip unparseable lines; if False, raise StructuredParsingError.

    Returns:
        List of parsed dictionary events.

    Raises:
        StructuredParsingError: If ignore_invalid is False and an invalid JSON line is encountered.
    """
    events: list[dict[str, Any]] = []
    for line_num, line in enumerate(text.splitlines(), start=1):
        line_str = line.strip()
        if not line_str:
            continue
        try:
            data = json.loads(line_str)
            if isinstance(data, dict):
                events.append(data)
            elif not ignore_invalid:
                raise StructuredParsingError(
                    f"Line {line_num} is valid JSON but not a JSON object: '{line_str}'"
                )
        except json.JSONDecodeError as e:
            if not ignore_invalid:
                raise StructuredParsingError(
                    f"Line {line_num} is not valid JSON: '{line_str}' ({e})"
                ) from e

    return events


def parse_model_into_schema(text: str, schema_class: type[T]) -> T:
    """Extract and validate a Pydantic schema model from text containing structured JSON.

    Args:
        text: Raw output containing structured JSON.
        schema_class: The Pydantic model class to validate into.

    Returns:
        Validated instance of schema_class.

    Raises:
        StructuredParsingError: If parsing fails or the data does not conform to schema_class.
    """
    parsed = parse_structured_json(text)
    if not isinstance(parsed, dict):
        raise StructuredParsingError(
            f"Expected JSON object for schema '{schema_class.__name__}', "
            f"got '{type(parsed).__name__}'."
        )

    try:
        return schema_class.model_validate(parsed)
    except ValidationError as e:
        raise StructuredParsingError(
            f"Failed to validate data against schema '{schema_class.__name__}': {e}"
        ) from e
