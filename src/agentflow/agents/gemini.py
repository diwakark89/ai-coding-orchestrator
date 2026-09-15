"""Google Gemini CLI agent adapter."""

import json
import re
from typing import Any

from agentflow.agents.base import (
    AdapterCapabilities,
    AgentRequest,
    AgentResult,
    BaseAgentAdapter,
    Provider,
)
from agentflow.process.executor import ProcessExecutor, ProcessResult

GEMINI_MODEL_ALIASES: dict[str, str] = {
    "gemini 3.8 flash": "gemini-3.8-flash",
    "gemini flash": "gemini-3.8-flash",
    "flash": "gemini-3.8-flash",
}

SESSION_ID_PATTERN = re.compile(
    r"(?:session[_\s-]?id|session)\s*[:=]\s*([a-zA-Z0-9_-]+)", re.IGNORECASE
)


class GeminiAdapter(BaseAgentAdapter):
    """Adapter for locally installed Gemini CLI."""

    def __init__(
        self,
        command: str = "gemini",
        executor: ProcessExecutor | None = None,
    ) -> None:
        super().__init__(command=command, executor=executor)

    @property
    def provider(self) -> Provider:
        return Provider.GOOGLE

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            supports_resume=True,
            supports_read_only_mode=True,
            supports_structured_output=True,
            supports_model_selection=True,
        )

    def _resolve_model(self, model: str) -> str:
        """Resolve friendly model names to CLI-accepted model identifiers."""
        cleaned = model.strip().lower()
        return GEMINI_MODEL_ALIASES.get(cleaned, model)

    def build_args(self, request: AgentRequest, session_id: str | None = None) -> list[str]:
        """Construct argument array for Gemini CLI."""
        args: list[str] = [self.command, "--output-format", "json"]

        model = self._resolve_model(request.model)
        args.extend(["--model", model])

        if session_id:
            args.extend(["--resume", session_id])

        if request.read_only:
            args.append("--read-only")

        if request.extra_args:
            args.extend(request.extra_args)

        # Pass prompt safely as a single argument
        args.append(request.prompt)
        return args

    def parse_result(
        self,
        proc_res: ProcessResult,
        request: AgentRequest,
        session_id: str | None = None,
    ) -> AgentResult:
        """Parse Gemini CLI output and extract text, session ID, and raw events."""
        extracted_text = proc_res.stdout
        extracted_session_id = session_id
        raw_events: list[dict[str, Any]] = []

        raw_stdout = proc_res.stdout.strip()
        if raw_stdout:
            try:
                data = json.loads(raw_stdout)
                if isinstance(data, dict):
                    raw_events.append(data)
                    # Support multiple standard response envelopes
                    if "response" in data:
                        extracted_text = str(data["response"])
                    elif "text" in data:
                        extracted_text = str(data["text"])
                    elif "content" in data:
                        extracted_text = str(data["content"])
                    elif "candidates" in data and isinstance(data["candidates"], list):
                        parts: list[str] = []
                        for cand in data["candidates"]:
                            if isinstance(cand, dict):
                                content = cand.get("content", {})
                                if isinstance(content, dict):
                                    for part in content.get("parts", []):
                                        if isinstance(part, dict) and "text" in part:
                                            parts.append(str(part["text"]))
                                        elif isinstance(part, str):
                                            parts.append(part)
                                elif isinstance(content, str):
                                    parts.append(content)
                        if parts:
                            extracted_text = "".join(parts)
                    elif "output" in data:
                        extracted_text = str(data["output"])

                    extracted_session_id = (
                        data.get("session_id")
                        or data.get("sessionId")
                        or (data.get("session") or {}).get("id")
                        or extracted_session_id
                    )
                elif isinstance(data, list):
                    raw_events.extend(item for item in data if isinstance(item, dict))
                    extracted_text = raw_stdout
            except json.JSONDecodeError:
                # Handle possible JSONL stream output
                parsed_lines: list[dict[str, Any]] = []
                text_parts: list[str] = []
                for line in raw_stdout.splitlines():
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        line_data = json.loads(line_str)
                        if isinstance(line_data, dict):
                            parsed_lines.append(line_data)
                            if "text" in line_data:
                                text_parts.append(str(line_data["text"]))
                            elif "response" in line_data:
                                text_parts.append(str(line_data["response"]))
                            if "session_id" in line_data:
                                extracted_session_id = str(line_data["session_id"])
                    except json.JSONDecodeError:
                        continue

                if parsed_lines:
                    raw_events = parsed_lines
                    if text_parts:
                        extracted_text = "\n".join(text_parts)

        # Fallback regex search for session ID
        if not extracted_session_id:
            match = SESSION_ID_PATTERN.search(proc_res.stdout) or SESSION_ID_PATTERN.search(
                proc_res.stderr
            )
            if match:
                extracted_session_id = match.group(1)

        return AgentResult(
            provider=self.provider,
            model=request.model,
            session_id=extracted_session_id,
            exit_code=proc_res.exit_code,
            text=extracted_text,
            raw_events=raw_events,
            started_at=proc_res.started_at,
            completed_at=proc_res.completed_at,
            timed_out=proc_res.timed_out,
            stderr=proc_res.stderr,
        )
