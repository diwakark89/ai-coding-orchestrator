"""Conservative worktree change detection and verification group selection."""

import fnmatch
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from agentflow.config.models import VerificationGroup
from agentflow.errors import ProcessExecutionError
from agentflow.process.executor import ProcessExecutor


@dataclass(frozen=True)
class ChangeSnapshot:
    """Paths and content fingerprints for staged, unstaged, and untracked changes."""

    fingerprints: dict[str, str]
    reliable: bool
    reason: str = ""

    @property
    def paths(self) -> set[str]:
        return set(self.fingerprints)

    def changed_since(self, previous: "ChangeSnapshot") -> set[str]:
        if not self.reliable or not previous.reliable:
            return self.paths | previous.paths
        return {
            path
            for path in self.paths | previous.paths
            if self.fingerprints.get(path) != previous.fingerprints.get(path)
        }


def _fingerprint(worktree: Path, relative: str) -> str | None:
    path = worktree / relative
    if not path.resolve().is_relative_to(worktree.resolve()):
        return None
    if path.is_symlink():
        return hashlib.sha256(os.readlink(path).encode("utf-8")).hexdigest()
    if not path.exists():
        return "deleted"
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


async def snapshot_changes(executor: ProcessExecutor, worktree: Path) -> ChangeSnapshot:
    """Read Git porcelain v1 with NUL paths; never mutate or stage the worktree."""
    try:
        result = await executor.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=worktree,
        )
        if result.exit_code != 0:
            return ChangeSnapshot({}, False, "Git status failed")
        records = result.stdout.split("\0")
        fingerprints: dict[str, str] = {}
        index = 0
        while index < len(records) and records[index]:
            record = records[index]
            if len(record) < 4 or record[2] != " ":
                return ChangeSnapshot({}, False, "Git status was incomplete")
            status = record[:2]
            paths = [record[3:]]
            if "R" in status or "C" in status:
                index += 1
                if index >= len(records) or not records[index]:
                    return ChangeSnapshot({}, False, "Git rename status was incomplete")
                paths.append(records[index])
            for raw_path in paths:
                relative = raw_path.replace("\\", "/")
                fingerprint = _fingerprint(worktree, relative)
                if fingerprint is None:
                    return ChangeSnapshot({}, False, "Changed path could not be inspected")
                fingerprints[relative] = fingerprint
            index += 1
        return ChangeSnapshot(fingerprints, True)
    except (OSError, ValueError, ProcessExecutionError):
        return ChangeSnapshot({}, False, "Git changes could not be inspected")


# Prose documentation cannot change build or test outcomes, so an unmapped doc edit must
# not widen verification to every group. A group can still claim docs via scope_paths.
_DOCUMENTATION_SUFFIXES = (".md", ".markdown", ".rst", ".adoc")


def _is_documentation(path: str) -> bool:
    return path.lower().endswith(_DOCUMENTATION_SUFFIXES)


@dataclass(frozen=True)
class GroupSelection:
    """Selected groups in config order, with concise user-facing reasons."""

    groups: list[str]
    reasons: dict[str, list[str]]
    suppressed: dict[str, str]
    broad: bool = False
    documentation_only: list[str] = field(default_factory=list)


def select_groups(
    config: dict[str, VerificationGroup],
    applicable: list[str],
    paths: set[str],
    *,
    reliable: bool,
) -> GroupSelection:
    """Select the deepest component owner of each path, plus explicit mappings.

    Documentation files select nothing unless a group's scope_paths claims them.
    """
    reasons: dict[str, list[str]] = {name: [] for name in applicable}
    broad_reason: str | None = None
    documentation: list[str] = []
    if not reliable or not paths:
        broad_reason = "change scope unavailable"
    else:
        for path in sorted(paths):
            explicit = [
                name
                for name in applicable
                if any(
                    fnmatch.fnmatchcase(path, pattern.replace("\\", "/"))
                    for pattern in config[name].scope_paths
                )
            ]
            if not explicit and _is_documentation(path):
                documentation.append(path)
                continue
            if "/" not in path and not explicit:
                broad_reason = f"shared root file: {path}"
                break
            owners: list[str] = []
            deepest = 0
            for name in applicable:
                directory = config[name].working_directory.replace("\\", "/")
                directory = directory.removeprefix("./").rstrip("/")
                if directory != "." and path.startswith(directory + "/"):
                    depth = len(directory.split("/"))
                    if depth > deepest:
                        owners = [name]
                        deepest = depth
                    elif depth == deepest:
                        owners.append(name)
            # Explicit mappings can name other affected components, but an
            # ancestor directory alone does not make its reactor an owner.
            matches = list(dict.fromkeys([*explicit, *owners]))
            if not matches:
                broad_reason = f"unmapped shared file: {path}"
                break
            for name in matches:
                reasons[name].append(f"changed: {path}")

    if broad_reason is not None:
        for name in applicable:
            reasons[name] = [broad_reason]
        selected = list(applicable)
    else:
        selected = [name for name in applicable if reasons[name]]

    suppressed: dict[str, str] = {}
    active = set(selected)
    incoming = {name: 0 for name in selected}
    for name in selected:
        for duplicate in config[name].supersedes:
            if duplicate in incoming:
                incoming[duplicate] += 1
    ready = [name for name in selected if incoming[name] == 0]
    superseder_order: list[str] = []
    while ready:
        name = ready.pop(0)
        superseder_order.append(name)
        for duplicate in config[name].supersedes:
            if duplicate in incoming:
                incoming[duplicate] -= 1
                if incoming[duplicate] == 0:
                    ready.append(duplicate)
    for name in superseder_order:
        if name not in active:
            continue
        descendants: list[str] = []
        pending = list(config[name].supersedes)
        seen: set[str] = set()
        while pending:
            duplicate = pending.pop()
            if duplicate in seen:
                continue
            seen.add(duplicate)
            descendants.append(duplicate)
            if duplicate in config:
                pending.extend(config[duplicate].supersedes)
        for duplicate in descendants:
            if duplicate in active:
                active.remove(duplicate)
                suppressed[duplicate] = name
    return GroupSelection(
        groups=[name for name in selected if name in active],
        reasons={name: reasons[name] for name in selected if name in active},
        suppressed=suppressed,
        broad=broad_reason is not None,
        documentation_only=documentation,
    )
