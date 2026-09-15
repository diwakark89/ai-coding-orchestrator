"""OpenAI Codex CLI agent adapter."""

import json
from typing import Any

from agentflow.agents.base import (
    AdapterCapabilities,
    AgentRequest,
    AgentResult,
    BaseAgentAdapter,
    Provider,
)
from agentflow.errors import UnsupportedCapabilityError
from agentflow.process.executor import ProcessExecutor, ProcessResult

CODEX_MODEL_ALIASES: dict[str, str] = {
    "gpt-5.6 luna": "gpt-5.6-luna",
    "gpt-5.6 terra": "gpt-5.6-terra",
    "luna": "gpt-5.6-luna",
    "terra": "gpt-5.6-terra",
}


class CodexAdapter(BaseAgentAdapter):
    """Adapter for locally installed OpenAI Codex CLI."""

    def __init__(
        self,
        command: str = "codex",
        executor: ProcessExecutor | None = None,
    ) -> None:
        super().__init__(command=command, executor=executor)

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            supports_resume=False,
            supports_read_only_mode=False,
            supports_structured_output=True,
            supports_model_selection=True,
        )

    def _resolve_model(self, model: str) -> str:
        """Resolve friendly model names to CLI-accepted model identifiers."""
        cleaned = model.strip().lower()
        return CODEX_MODEL_ALIASES.get(cleaned, model)

    def build_args(self, request: AgentRequest, session_id: str | None = None) -> list[str]:
        """Construct non-interactive argument array for Codex CLI."""
        args: list[str] = [self.command, "exec", "--json"]

        model = self._resolve_model(request.model)
        args.extend(["--model", model])

        if request.extra_args:
            args.extend(request.extra_args)

        # Pass prompt safely as a single argument
        args.append(request.prompt)
        return args

    async def resume(self, session_id: str, request: AgentRequest) -> AgentResult:
        """Explicitly reject session resume as unsupported by Codex CLI."""
        raise UnsupportedCapabilityError("Codex CLI adapter does not support session resume.")

    def parse_result(
        self,
        proc_res: ProcessResult,
        request: AgentRequest,
        session_id: str | None = None,
    ) -> AgentResult:
        """Parse Codex CLI output, capturing errors distinctly from model output."""
        extracted_text = proc_res.stdout
        raw_events: list[dict[str, Any]] = []

        raw_stdout = proc_res.stdout.strip()
        if raw_stdout:
            try:
                data = json.loads(raw_stdout)
                if isinstance(data, dict):
                    raw_events.append(data)
                    extracted_text = (
                        data.get("output")
                        or data.get("text")
                        or data.get("content")
                        or data.get("message")
                        or raw_stdout
                    )
                elif isinstance(data, list):
                    raw_events.extend(item for item in data if isinstance(item, dict))
                    extracted_text = raw_stdout
            except json.JSONDecodeError:
                # Handle streaming JSONL lines
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
                            if "message" in line_data:
                                text_parts.append(str(line_data["message"]))
                            elif "output" in line_data:
                                text_parts.append(str(line_data["output"]))
                            elif "content" in line_data:
                                text_parts.append(str(line_data["content"]))
                    except json.JSONDecodeError:
                        continue

                if parsed_lines:
                    raw_events = parsed_lines
                    if text_parts:
                        extracted_text = "\n".join(text_parts)

        return AgentResult(
            provider=self.provider,
            model=request.model,
            session_id=None,
            exit_code=proc_res.exit_code,
            text=extracted_text,
            raw_events=raw_events,
            started_at=proc_res.started_at,
            completed_at=proc_res.completed_at,
            timed_out=proc_res.timed_out,
            stderr=proc_res.stderr,
        )
