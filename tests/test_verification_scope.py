"""Change-aware verification selection and repair cache integration tests."""

import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentflow.config.models import ProjectConfig, VerificationGroup
from agentflow.persistence.database import DatabaseManager
from agentflow.process.executor import ProcessExecutor, ProcessResult
from agentflow.workflow.verification import VerificationRunner, VerificationStatus
from agentflow.workflow.verification_scope import select_groups, snapshot_changes


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _worktree(git_repo: Path, tmp_path: Path) -> Path:
    for directory, marker in (
        ("front-end", "package.json"),
        ("web-service", "pom.xml"),
        ("ai-engine", "pyproject.toml"),
        ("data-processor", "pyproject.toml"),
    ):
        component = git_repo / directory
        component.mkdir()
        (component / marker).write_text("marker\n", encoding="utf-8")
        (component / "source.txt").write_text("base\n", encoding="utf-8")
    (git_repo / "package.json").write_text("root\n", encoding="utf-8")
    _git(git_repo, "add", "-A")
    _git(git_repo, "commit", "-m", "Add component markers")
    worktree = tmp_path / "work tree with spaces"
    _git(git_repo, "worktree", "add", "-b", "agentflow/scope-test", str(worktree))
    return worktree


def _groups() -> dict[str, VerificationGroup]:
    return {
        "root-node": VerificationGroup(detect=["package.json"], commands=["check-root"]),
        "ai-engine": VerificationGroup(
            detect=["ai-engine/pyproject.toml"],
            working_directory="ai-engine",
            commands=["check-ai"],
        ),
        "data-processor": VerificationGroup(
            detect=["data-processor/pyproject.toml"],
            working_directory="data-processor",
            commands=["check-data"],
        ),
        "web-service": VerificationGroup(
            detect=["web-service/pom.xml"],
            working_directory="web-service",
            commands=["check-web"],
        ),
        "front-end": VerificationGroup(
            detect=["front-end/package.json"],
            working_directory="front-end",
            commands=["check-front"],
            supersedes=["root-node"],
        ),
    }


class CountingExecutor(ProcessExecutor):
    def __init__(
        self,
        outcomes: dict[str, list[int]] | None = None,
        failure_output: dict[str, str] | None = None,
    ) -> None:
        self.git = ProcessExecutor()
        self.outcomes = outcomes or {}
        self.failure_output = failure_output or {}
        self.calls: list[tuple[str, Path]] = []

    async def run(
        self,
        cmd_args: Sequence[str | Path],
        cwd: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        input_data: str | bytes | None = None,
    ) -> ProcessResult:
        if str(cmd_args[0]) == "git":
            return await self.git.run(
                cmd_args, cwd=cwd, env=env, timeout=timeout, input_data=input_data
            )
        assert cwd is not None
        command = str(cmd_args[0])
        self.calls.append((command, Path(cwd)))
        codes = self.outcomes.get(command, [])
        exit_code = codes.pop(0) if codes else 0
        now = datetime.now(timezone.utc)
        return ProcessResult(
            command=[str(arg) for arg in cmd_args],
            exit_code=exit_code,
            stdout=(
                self.failure_output.get(command, "FAILED test_case - AssertionError")
                if exit_code
                else ""
            ),
            stderr="",
            started_at=now,
            completed_at=now,
        )


def _runner(tmp_path: Path, git_repo: Path, executor: CountingExecutor) -> VerificationRunner:
    db = DatabaseManager(tmp_path / "scope.db")
    db.initialize()
    db.upsert_project("project", "Project", str(git_repo))
    db.create_run("run-scope", "project", "Scope test")
    return VerificationRunner(db, executor=executor)


@pytest.mark.asyncio
async def test_snapshot_and_selection_cover_staged_unstaged_untracked_windows_worktree(
    git_repo: Path, tmp_path: Path
):
    worktree = _worktree(git_repo, tmp_path)
    (worktree / "front-end" / "source.txt").write_text("staged\n", encoding="utf-8")
    _git(worktree, "add", "front-end/source.txt")
    (worktree / "web-service" / "source.txt").write_text("unstaged\n", encoding="utf-8")
    (worktree / "ai-engine" / "new file.txt").write_text("untracked\n", encoding="utf-8")

    snapshot = await snapshot_changes(ProcessExecutor(), worktree)
    selected = select_groups(_groups(), list(_groups()), snapshot.paths, reliable=snapshot.reliable)

    assert snapshot.reliable
    assert snapshot.paths == {
        "front-end/source.txt",
        "web-service/source.txt",
        "ai-engine/new file.txt",
    }
    assert selected.groups == ["ai-engine", "web-service", "front-end"]
    assert "changed: front-end/source.txt" in selected.reasons["front-end"]


@pytest.mark.asyncio
async def test_cross_component_rename_selects_source_and_destination(
    git_repo: Path, tmp_path: Path
):
    worktree = _worktree(git_repo, tmp_path)
    _git(worktree, "mv", "front-end/source.txt", "web-service/moved.txt")

    snapshot = await snapshot_changes(ProcessExecutor(), worktree)
    selected = select_groups(_groups(), list(_groups()), snapshot.paths, reliable=snapshot.reliable)

    assert snapshot.reliable
    assert snapshot.paths == {"front-end/source.txt", "web-service/moved.txt"}
    assert selected.groups == ["web-service", "front-end"]


@pytest.mark.asyncio
async def test_shared_file_falls_back_to_all_and_explicit_overlap_removes_duplicate(
    git_repo: Path, tmp_path: Path
):
    worktree = _worktree(git_repo, tmp_path)
    (worktree / ".editorconfig").write_text("changed root\n", encoding="utf-8")
    snapshot = await snapshot_changes(ProcessExecutor(), worktree)
    groups = _groups()

    selected = select_groups(groups, list(groups), snapshot.paths, reliable=snapshot.reliable)

    assert selected.broad
    assert selected.groups == ["ai-engine", "data-processor", "web-service", "front-end"]
    assert selected.suppressed == {"root-node": "front-end"}
    assert all("shared root file" in reason[0] for reason in selected.reasons.values())

    executor = CountingExecutor()
    runner = _runner(tmp_path, git_repo, executor)
    result = await runner.run("run-scope", worktree, groups)
    assert result.status == VerificationStatus.PASSED
    assert "check-root" not in [command for command, _ in executor.calls]
    assert len(executor.calls) == 4


def test_maven_module_change_runs_only_the_module_group():
    groups = {
        "reactor": VerificationGroup(working_directory="web-service", supersedes=["module"]),
        "module": VerificationGroup(working_directory="web-service/module"),
    }
    selected = select_groups(
        groups, list(groups), {"web-service/module/src/Main.java"}, reliable=True
    )
    assert selected.groups == ["module"]
    assert selected.suppressed == {}

    parent_change = select_groups(groups, list(groups), {"web-service/pom.xml"}, reliable=True)
    assert parent_change.groups == ["reactor"]

    cross_module = select_groups(
        {
            **groups,
            "other": VerificationGroup(working_directory="web-service/other"),
        },
        [*groups, "other"],
        {"web-service/module/src/Main.java", "web-service/other/src/Other.java"},
        reliable=True,
    )
    assert cross_module.groups == ["module", "other"]


def test_explicit_scope_mapping_cannot_drop_changed_component_owner():
    groups = {
        "root": VerificationGroup(scope_paths=["front-end/**"]),
        "front-end": VerificationGroup(working_directory="front-end"),
    }
    selected = select_groups(groups, list(groups), {"front-end/src/page.tsx"}, reliable=True)
    assert selected.groups == ["root", "front-end"]


def test_unmapped_nested_shared_file_falls_back_to_all_groups():
    groups = {
        "front-end": VerificationGroup(working_directory="front-end"),
        "web-service": VerificationGroup(working_directory="web-service"),
    }
    selected = select_groups(groups, list(groups), {"common/schema.json"}, reliable=True)
    assert selected.broad
    assert selected.groups == ["front-end", "web-service"]
    assert "unmapped shared file" in selected.reasons["front-end"][0]


def test_invalid_overlap_cycle_is_rejected():
    with pytest.raises(ValueError, match="cycle"):
        ProjectConfig.model_validate(
            {
                "project": {"name": "sample"},
                "verification": {
                    "a": {"supersedes": ["b"]},
                    "b": {"supersedes": ["a"]},
                },
            }
        )


def test_overlap_chain_never_uses_a_suppressed_group_as_its_cover():
    groups = {
        "module": VerificationGroup(working_directory="web-service/module"),
        "parent": VerificationGroup(working_directory="web-service", supersedes=["module"]),
        "reactor": VerificationGroup(working_directory="web-service", supersedes=["parent"]),
    }
    selected = select_groups(groups, list(groups), {"settings.gradle"}, reliable=True)
    assert selected.groups == ["reactor"]
    assert selected.suppressed["parent"] == "reactor"


@pytest.mark.asyncio
async def test_runner_executes_only_changed_nested_module_even_with_reactor_overlap(
    git_repo: Path, tmp_path: Path
):
    worktree = _worktree(git_repo, tmp_path)
    module = worktree / "web-service" / "application"
    module.mkdir()
    (module / "pom.xml").write_text("module marker\n", encoding="utf-8")
    (module / "source.java").write_text("changed\n", encoding="utf-8")
    executor = CountingExecutor()
    runner = _runner(tmp_path, git_repo, executor)
    groups = {
        "reactor": VerificationGroup(
            detect=["web-service/pom.xml"],
            working_directory="web-service",
            commands=["check-reactor"],
            supersedes=["application"],
        ),
        "application": VerificationGroup(
            detect=["web-service/application/pom.xml"],
            working_directory="web-service/application",
            commands=["check-application"],
        ),
    }

    result = await runner.run("run-scope", worktree, groups)

    assert result.status == VerificationStatus.PASSED
    assert result.groups_run == ["application"]
    assert executor.calls == [("check-application", module)]


@pytest.mark.asyncio
async def test_repair_reruns_failed_group_first_and_retains_unaffected_passes(
    git_repo: Path, tmp_path: Path
):
    worktree = _worktree(git_repo, tmp_path)
    (worktree / "web-service" / "source.txt").write_text("web change\n", encoding="utf-8")
    (worktree / "front-end" / "source.txt").write_text("front change\n", encoding="utf-8")
    executor = CountingExecutor({"check-front": [1, 0]})
    runner = _runner(tmp_path, git_repo, executor)
    groups = _groups()

    initial = await runner.run("run-scope", worktree, groups)
    assert initial.status == VerificationStatus.FAILED
    assert initial.groups_run == ["web-service", "front-end"]
    assert initial.failed_group == "front-end"

    (worktree / "front-end" / "source.txt").write_text("front repair\n", encoding="utf-8")
    (worktree / "ai-engine" / "new.txt").write_text("new component\n", encoding="utf-8")
    after = await runner.run_incremental("run-scope", worktree, groups)

    assert after.status == VerificationStatus.PASSED
    assert after.phase == "incremental"
    assert after.groups_run == ["ai-engine", "web-service", "front-end"]
    assert after.rerun_groups == ["front-end", "ai-engine"]
    assert after.cached_groups == ["web-service"]
    # No redundant full pass: web-service's files are unchanged since it passed.
    assert [command for command, _ in executor.calls] == [
        "check-web",
        "check-front",
        "check-front",
        "check-ai",
    ]
    assert all(cwd.is_relative_to(worktree) for _, cwd in executor.calls)


@pytest.mark.asyncio
async def test_repair_edit_in_a_passed_group_invalidates_its_retained_pass(
    git_repo: Path, tmp_path: Path
):
    worktree = _worktree(git_repo, tmp_path)
    (worktree / "web-service" / "source.txt").write_text("web change\n", encoding="utf-8")
    (worktree / "front-end" / "source.txt").write_text("front change\n", encoding="utf-8")
    executor = CountingExecutor({"check-front": [1, 0]})
    runner = _runner(tmp_path, git_repo, executor)
    groups = _groups()

    await runner.run("run-scope", worktree, groups)
    (worktree / "front-end" / "source.txt").write_text("front repair\n", encoding="utf-8")
    (worktree / "web-service" / "source.txt").write_text("web touched\n", encoding="utf-8")
    after = await runner.run_incremental("run-scope", worktree, groups)

    assert after.status == VerificationStatus.PASSED
    assert after.cached_groups == []
    assert [command for command, _ in executor.calls] == [
        "check-web",
        "check-front",
        "check-front",
        "check-web",
    ]


@pytest.mark.asyncio
async def test_failing_incremental_pass_is_not_accepted_as_repaired(git_repo: Path, tmp_path: Path):
    worktree = _worktree(git_repo, tmp_path)
    (worktree / "front-end" / "source.txt").write_text("changed\n", encoding="utf-8")
    executor = CountingExecutor({"check-front": [1, 1]})
    runner = _runner(tmp_path, git_repo, executor)

    initial = await runner.run("run-scope", worktree, _groups())
    assert initial.status == VerificationStatus.FAILED
    (worktree / "front-end" / "source.txt").write_text("repaired\n", encoding="utf-8")
    after = await runner.run_incremental("run-scope", worktree, _groups())

    assert after.status == VerificationStatus.FAILED
    assert after.failed_group == "front-end"
    assert len(executor.calls) == 2


@pytest.mark.asyncio
async def test_documentation_only_edit_after_pass_retains_every_pass(
    git_repo: Path, tmp_path: Path
):
    worktree = _worktree(git_repo, tmp_path)
    (worktree / "front-end" / "source.txt").write_text("changed\n", encoding="utf-8")
    executor = CountingExecutor()
    runner = _runner(tmp_path, git_repo, executor)

    assert (await runner.run("run-scope", worktree, _groups())).success
    (worktree / "front-end" / "NOTES.md").write_text("doc only\n", encoding="utf-8")
    after = await runner.run_incremental("run-scope", worktree, _groups())

    assert after.success
    assert after.cached_groups == ["front-end"]
    assert len(executor.calls) == 1


@pytest.mark.asyncio
async def test_bounded_reproduction_retains_original_failure_evidence(
    git_repo: Path, tmp_path: Path
):
    worktree = _worktree(git_repo, tmp_path)
    (worktree / "front-end" / "source.txt").write_text("changed\n", encoding="utf-8")
    executor = CountingExecutor(
        {"check-front": [1, 0]},
        failure_output={
            "check-front": "FAIL front-end/test_widget.ts: Exceeded timeout of 5000 ms"
        },
    )
    runner = _runner(tmp_path, git_repo, executor)
    log_dir = tmp_path / "logs"
    result = await runner.run("run-scope", worktree, _groups(), log_dir=log_dir)

    result = await runner.reproduce_failure(
        "run-scope", worktree, _groups(), result, log_dir=log_dir
    )

    assert result.status == VerificationStatus.INTERMITTENT
    assert result.first_failure is not None and not result.first_failure.success
    assert result.reproduction_result is not None and result.reproduction_result.success
    assert len(executor.calls) == 2
    assert len(runner.db_manager.list_verification_runs("run-scope")) == 2


@pytest.mark.asyncio
async def test_representative_skillify_style_two_repair_cycles_execute_four_not_fifteen(
    git_repo: Path, tmp_path: Path
):
    worktree = _worktree(git_repo, tmp_path)
    (worktree / "web-service" / "source.txt").write_text("web change\n", encoding="utf-8")
    (worktree / "front-end" / "source.txt").write_text("front change\n", encoding="utf-8")
    (worktree / "common").mkdir()
    (worktree / "common" / "feature.md").write_text("docs\n", encoding="utf-8")
    groups = _groups()
    executor = CountingExecutor({"check-front": [1, 1, 0]})
    runner = _runner(tmp_path, git_repo, executor)

    initial = await runner.run("run-scope", worktree, groups)
    assert initial.status == VerificationStatus.FAILED
    assert initial.groups_run == ["web-service", "front-end"]
    (worktree / "front-end" / "source.txt").write_text("repair one\n", encoding="utf-8")
    assert (
        await runner.run_incremental("run-scope", worktree, groups)
    ).status == VerificationStatus.FAILED
    (worktree / "front-end" / "source.txt").write_text("repair two\n", encoding="utf-8")
    final = await runner.run_incremental("run-scope", worktree, groups)

    assert final.status == VerificationStatus.PASSED
    assert [name for name, _ in executor.calls] == [
        "check-web",
        "check-front",
        "check-front",
        "check-front",
    ]
    # The prior marker-only loop ran five groups on each of these three passes.
    legacy_executor = CountingExecutor()
    for _ in range(3):
        for name, group in groups.items():
            cwd = worktree / group.working_directory
            await legacy_executor.run([group.commands[0]], cwd=cwd)
    assert len(legacy_executor.calls) == 15
    assert len(executor.calls) == 4


def test_documentation_changes_do_not_widen_selection():
    groups = {
        "front-end": VerificationGroup(working_directory="front-end"),
        "web-service": VerificationGroup(working_directory="web-service"),
        "content": VerificationGroup(working_directory="web-service/content"),
    }
    selected = select_groups(
        groups,
        list(groups),
        {
            "common/docs/api/frontend-to-web-service.md",
            "README.md",
            "front-end/src/page.tsx",
            "web-service/content/src/Service.java",
        },
        reliable=True,
    )
    assert not selected.broad
    assert selected.groups == ["front-end", "content"]
    assert selected.documentation_only == [
        "README.md",
        "common/docs/api/frontend-to-web-service.md",
    ]

    docs_only = select_groups(groups, list(groups), {"docs/guide.rst"}, reliable=True)
    assert docs_only.groups == []
    assert not docs_only.broad


def test_scope_paths_can_claim_documentation_for_a_group():
    groups = {
        "docs-site": VerificationGroup(working_directory="site", scope_paths=["docs/**"]),
        "front-end": VerificationGroup(working_directory="front-end"),
    }
    selected = select_groups(groups, list(groups), {"docs/intro.md"}, reliable=True)
    assert selected.groups == ["docs-site"]
    assert selected.documentation_only == []
