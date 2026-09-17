"""Google Antigravity CLI (`agy`) agent adapter.

`agy` is a distinct binary and flag dialect from the `gemini` CLI -- not a renamed drop-in
(confirmed against antigravity.google/docs/cli/headless: prompt is passed via `-p`, resume is
`--continue`/`--conversation` rather than `--resume`, and unattended runs require
`--dangerously-skip-permissions` since there is no equivalent of `gemini`'s `--read-only`).
Its JSON response envelope isn't documented in detail, so this reuses `GeminiAdapter`'s
schema-agnostic `parse_result()` (multiple candidate field names, regex fallback) unchanged.
"""

from agentflow.agents.base import AdapterCapabilities, AgentRequest
from agentflow.agents.gemini import GeminiAdapter
from agentflow.process.executor import ProcessExecutor


class AntigravityAdapter(GeminiAdapter):
    """Adapter for locally installed Google Antigravity CLI (`agy`)."""

    def __init__(
        self,
        command: str = "agy",
        executor: ProcessExecutor | None = None,
    ) -> None:
        super().__init__(command=command, executor=executor)

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            supports_resume=True,
            supports_read_only_mode=False,  # no --read-only equivalent; see build_args
            supports_structured_output=True,
            supports_model_selection=True,
        )

    def build_args(self, request: AgentRequest, session_id: str | None = None) -> list[str]:
        """Construct argument array for the Antigravity CLI (`agy`)."""
        model = self._resolve_model(request.model)
        args: list[str] = [
            self.command,
            "-p",
            request.prompt,
            "--output-format",
            "json",
            "--model",
            model,
        ]

        if session_id:
            args.extend(["--conversation", session_id])

        # agy has no --read-only flag; --sandbox restricts tool access for read-only requests,
        # otherwise unattended runs need --dangerously-skip-permissions or they block waiting
        # for interactive tool approval.
        args.append("--sandbox" if request.read_only else "--dangerously-skip-permissions")

        if request.extra_args:
            args.extend(request.extra_args)

        return args
