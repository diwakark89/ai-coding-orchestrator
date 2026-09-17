"""Starter `.ai-orchestrator/routing.yaml` generation for `agentflow init`.

Detection is a heuristic starting point only -- there is no way to reliably infer a project's
full test/build setup from marker files alone, so the generated config always needs a human
review pass before real tasks are routed against it.
"""

from pathlib import Path

from agentflow.config.models import (
    DocumentationConfig,
    LimitsConfig,
    ProjectConfig,
    ProjectMetaConfig,
    VerificationGroup,
)
from agentflow.routing.complexity import DEFAULT_COMPLEXITY_CONFIG
from agentflow.routing.rules import DEFAULT_MODELS_CONFIG, DEFAULT_ROUTING_RULES

_SKIP_DIR_NAMES = frozenset(
    {"node_modules", ".venv", "venv", "__pycache__", "dist", "build", "target"}
)

# Marker filename -> (group kind, guessed test command). Order matters: first match per
# directory wins so a directory with both package.json and pyproject.toml isn't double-counted
# under two different guessed commands for the same group.
_MARKERS: tuple[tuple[str, str, str], ...] = (
    ("package.json", "node", "npm test"),
    ("pyproject.toml", "python", "pytest"),
    ("requirements.txt", "python", "pytest"),
    ("pom.xml", "java", "mvn test"),
)

_MAX_DEPTH = 3


def _marker_for(directory: Path) -> tuple[str, str, str] | None:
    """Return (marker_name, group_kind, command) for the first matching marker in `directory`."""
    for marker_name, group_kind, command in _MARKERS:
        if (directory / marker_name).exists():
            return marker_name, group_kind, command
    return None


def detect_verification_groups(root_path: Path) -> dict[str, VerificationGroup]:
    """Heuristically detect per-directory test groups by walking the repo for marker files.

    Walks up to `_MAX_DEPTH` directories deep, skipping dependency/build directories and any
    hidden directory. Each match becomes one `verification:` group, keyed by directory name
    (or the detected group kind at the repo root).
    """
    groups: dict[str, VerificationGroup] = {}

    def _register(name: str, marker_relative: str, command: str) -> None:
        unique_name = name
        suffix = 2
        while unique_name in groups:
            unique_name = f"{name}-{suffix}"
            suffix += 1
        groups[unique_name] = VerificationGroup(detect=[marker_relative], commands=[command])

    root_marker = _marker_for(root_path)
    if root_marker is not None:
        marker_name, group_kind, command = root_marker
        _register(group_kind, marker_name, command)

    def _walk(directory: Path, depth: int) -> None:
        if depth > _MAX_DEPTH:
            return
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir() or entry.name.startswith(".") or entry.name in _SKIP_DIR_NAMES:
                continue
            marker = _marker_for(entry)
            if marker is not None:
                marker_name, _group_kind, command = marker
                relative_marker = (entry / marker_name).relative_to(root_path).as_posix()
                _register(entry.name, relative_marker, command)
            _walk(entry, depth + 1)

    _walk(root_path, 1)
    return groups


def build_starter_config(root_path: Path) -> ProjectConfig:
    """Assemble a starter ProjectConfig for `root_path`, seeded from AgentFlow's V1 defaults.

    `documentation` is left disabled: there is no reliable way to guess which docs a project
    wants kept in sync, so that section is a deliberate no-op until the user configures it.
    """
    verification = detect_verification_groups(root_path)
    return ProjectConfig(
        version=1,
        project=ProjectMetaConfig(name=root_path.name),
        models=DEFAULT_MODELS_CONFIG,
        complexity=DEFAULT_COMPLEXITY_CONFIG,
        routing=DEFAULT_ROUTING_RULES,
        verification=verification or None,
        limits=LimitsConfig(),
        documentation=DocumentationConfig(enabled=False, candidate_files=[]),
    )
