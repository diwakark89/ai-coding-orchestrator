"""Claude Code CLI agent adapter."""

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
from agentflow.agents.parser import strip_markdown_fences
from agentflow.process.executor import ProcessExecutor, ProcessResult

CLAUDE_MODEL_ALIASES: dict[str, str] = {
    "claude sonnet 5": "sonnet",
    "claude sonnet": "sonnet",
    "claude opus 5.5": "claude-opus-5-5",
    "claude opus": "claude-opus-5-5",
    "claude-sonnet-5": "sonnet",
    "claude-opus-5-5": "claude-opus-5-5",
    # Legacy Opus 5 names are upgraded to Opus 5.5 so older routing.yaml files keep working.
    "claude opus 5": "claude-opus-5-5",
    "claude-opus-5": "claude-opus-5-5",
}

UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
SESSION_ID_PATTERN = re.compile(
    r"(?:session[_\s-]?id|session)\s*[:=]\s*([a-zA-Z0-9_-]+)", re.IGNORECASE
)


class ClaudeAdapter(BaseAgentAdapter):
    """Adapter for locally installed Claude Code CLI."""

    def __init__(
        self,
        command: str = "claude",
        executor: ProcessExecutor | None = None,
    ) -> None:
        super().__init__(command=command, executor=executor)

    @property
    def provider(self) -> Provider:
        return Provider.ANTHROPIC

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
        return CLAUDE_MODEL_ALIASES.get(cleaned, model)

    def build_args(self, request: AgentRequest, session_id: str | None = None) -> list[str]:
        """Construct non-interactive argument array for Claude Code CLI."""
        args: list[str] = [self.command, "--print", "--output-format", "json"]

        model = self._resolve_model(request.model)
        args.extend(["--model", model])

        if request.read_only:
            args.extend(["--permission-mode", "plan"])

        if session_id:
            args.extend(["--resume", session_id])

        if request.extra_args:
            args.extend(request.extra_args)

        # Pass prompt safely as a single argument (ProcessExecutor avoids shell concatenation)
        args.append(request.prompt)
        return args

    def parse_result(
        self,
        proc_res: ProcessResult,
        request: AgentRequest,
        session_id: str | None = None,
    ) -> AgentResult:
        """Parse Claude Code output and extract text, session ID, and events."""
        extracted_text = proc_res.stdout
        extracted_session_id = session_id
        raw_events: list[dict[str, Any]] = []

        raw_stdout = proc_res.stdout.strip()
        if raw_stdout:
            try:
                data = json.loads(raw_stdout)
                if isinstance(data, dict):
                    raw_events.append(data)
                    extracted_text = (
                        data.get("result")
                        or data.get("text")
                        or data.get("content")
                        or data.get("message")
                        or raw_stdout
                    )
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
                # Try stripping markdown code fences
                fenced = strip_markdown_fences(raw_stdout, language="json")
                if fenced != raw_stdout:
                    try:
                        fenced_data = json.loads(fenced)
                        if isinstance(fenced_data, dict):
                            raw_events.append(fenced_data)
                            extracted_text = (
                                fenced_data.get("result")
                                or fenced_data.get("text")
                                or fenced_data.get("content")
                                or fenced_data.get("message")
                                or fenced
                            )
                            extracted_session_id = (
                                fenced_data.get("session_id")
                                or fenced_data.get("sessionId")
                                or (fenced_data.get("session") or {}).get("id")
                                or extracted_session_id
                            )
                        elif isinstance(fenced_data, list):
                            raw_events.extend(
                                item for item in fenced_data if isinstance(item, dict)
                            )
                            extracted_text = fenced
                    except json.JSONDecodeError:
                        pass

                if not raw_events:
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
                                if "result" in line_data:
                                    text_parts.append(str(line_data["result"]))
                                elif "text" in line_data:
                                    text_parts.append(str(line_data["text"]))
                                if "session_id" in line_data:
                                    extracted_session_id = str(line_data["session_id"])
                                elif "sessionId" in line_data:
                                    extracted_session_id = str(line_data["sessionId"])
                        except json.JSONDecodeError:
                            continue

                    if parsed_lines:
                        raw_events = parsed_lines
                        if text_parts:
                            extracted_text = "\n".join(text_parts)

        # Fallback session ID detection via regex if not found in structured JSON
        if not extracted_session_id:
            match = SESSION_ID_PATTERN.search(proc_res.stdout) or SESSION_ID_PATTERN.search(
                proc_res.stderr
            )
            if match:
                extracted_session_id = match.group(1)
            else:
                uuid_match = UUID_PATTERN.search(proc_res.stdout)
                if uuid_match:
                    extracted_session_id = uuid_match.group(0)

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
