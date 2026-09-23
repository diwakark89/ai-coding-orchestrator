"""OpenAI Codex CLI agent adapter."""

import json
import os
import re
import tomllib
from pathlib import Path
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
    "gpt-6 luna": "gpt-6-luna",
    "gpt-6 sol": "gpt-6-sol",
    "luna": "gpt-6-luna",
    "sol": "gpt-6-sol",
    # Legacy GPT-5.6 names are upgraded to GPT-6 so older routing.yaml files keep working.
    "gpt-5.6 luna": "gpt-6-luna",
    "gpt-5.6 terra": "gpt-6-sol",
    "gpt-5.6-luna": "gpt-6-luna",
    "gpt-5.6-terra": "gpt-6-sol",
    "terra": "gpt-6-sol",
}


class CodexAdapter(BaseAgentAdapter):
    """Adapter for locally installed OpenAI Codex CLI."""

    def __init__(
        self,
        command: str = "codex",
        executor: ProcessExecutor | None = None,
        config_home: Path | None = None,
    ) -> None:
        super().__init__(command=command, executor=executor)
        self.config_home = config_home or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))

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

        # Codex merges CLI config overrides with user/project config. An empty table does
        # not remove inherited MCP servers, so disable each configured server by name.
        if request.provider_options.get("enable_mcp") is not True:
            for name in self._configured_mcp_servers(request.effective_working_directory):
                if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                    raise ValueError(f"Cannot safely disable MCP server with name {name!r}")
                args.extend(["--config", f"mcp_servers.{name}.enabled=false"])

        model = self._resolve_model(request.model)
        args.extend(["--model", model])

        if request.extra_args:
            args.extend(request.extra_args)

        # Pass prompt safely as a single argument
        args.append(request.prompt)
        return args

    def _configured_mcp_servers(self, working_directory: Path) -> list[str]:
        """Find user and project MCP entries without reading or emitting their credentials."""
        config_paths = [self.config_home / "config.toml"]
        config_paths.extend(
            parent / ".codex" / "config.toml"
            for parent in reversed((working_directory, *working_directory.parents))
            if parent != Path.home()
        )
        names: set[str] = set()
        for path in config_paths:
            if not path.is_file():
                continue
            with path.open("rb") as config_file:
                config = tomllib.load(config_file)
            names.update(config.get("mcp_servers", {}))
        return sorted(names)

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
