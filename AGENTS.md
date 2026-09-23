# AgentFlow — AI Agent Operating Manual & Architecture Guide (`AGENTS.md`)

Welcome, coding agent. This repository houses **AgentFlow** (also known as `ai-coding-orchestrator`), a reusable, local, multi-agent coding orchestrator.

This document is your primary operating manual when reading, designing, testing, and modifying this codebase. Always adhere strictly to the rules, invariants, toolchains, and phased delivery protocols established here.

---

## 1. Overview & Core Philosophy

AgentFlow coordinates locally installed coding-agent CLIs (Claude Code, OpenAI Codex CLI, Gemini CLI) across end-to-end software development workflows:
- **Interactive Planning:** Multi-turn requirement gathering and architecture planning via Claude.
- **Deterministic Routing:** Configuration-driven worker model selection without autonomous agent discretion over models.
- **Isolated Implementation:** Autonomous code changes executed strictly inside isolated Git worktrees via Codex CLI (GPT-6 Luna / Sol).
- **Deterministic Verification:** Project-defined builds, tests, and linters run by the orchestrator (actual process exit codes dictate success, never model claims).
- **Independent Review:** Read-only inspection of final diffs via Gemini 3.8 Flash (escalating to Claude Sonnet/Opus for security/architecture).
- **Documentation & Completion:** Post-verification documentation sync and human approval.

### Primary References
- **Implementation Blueprint:** [phased-implementation-plan.md](phased-implementation-plan.md) — Step-by-step phased instructions, milestone definitions, and test criteria.
- **Technical Architecture:** [technical-design-document.md](technical-design-document.md) — Comprehensive technical design, schemas, state machine, and data flow.

---

## 2. Non-Negotiable Architectural Invariants

Every agent working on this codebase must adhere to the following rules without exception:

1. **No Direct AI Provider APIs:**
   - Never import or call Anthropic, OpenAI, or Google generative AI client libraries/APIs.
   - Never manage API keys, tokens, or sessions directly.
   - AI integrations interact solely with locally installed, subscription-authenticated CLIs (`claude`, `codex`, `gemini`).

2. **Forbidden Frameworks & Dependencies:**
   - Do NOT introduce **LangChain**, **LangGraph**, **CrewAI**, or **AutoGen**.
   - Do NOT introduce external databases (e.g. Postgres, Redis, MongoDB), message queues (e.g. RabbitMQ, Kafka), or background worker daemons (e.g. Celery).
   - Use only the approved technology stack: **Python 3.12+**, **Typer**, **Rich**, **Pydantic v2**, **PyYAML**, **SQLite (standard library)**, **pytest**, **asyncio**, and **Git CLI**.

3. **Deterministic Model Routing:**
   - AI models must never decide which model executes the next stage.
   - Routing is calculated deterministically from the Pydantic `TaskProfile` and `.ai-orchestrator/routing.yaml` using exact precedence rules (see [technical-design-document.md](technical-design-document.md) §18).

4. **Strict V1 Model Pool:**
   - Permitted models: `Claude Sonnet 5`, `Claude Opus 5.5`, `GPT-6 Luna`, `GPT-6 Sol`, `Gemini 3.8 Flash`.
   - **Explicitly Excluded:** `GPT-5.4 Mini` and `GPT-5.6 Sol` must never appear in routing tables, defaults, fallbacks, or escalation chains.

5. **Single-Writer Constraint:**
   - Only one AI agent may modify a worktree at any given moment.
   - Reviewers and planners must remain strictly read-only. Concurrent writers are forbidden.

6. **Deterministic Verification over Model Output:**
   - Never mark verification successful because an LLM claimed tests passed.
   - Real test/build commands must execute in subprocesses, and only process exit codes (`exit_code == 0`) determine pass/fail.

7. **Zero Shell Concatenation:**
   - Never execute commands via `shell=True` or concatenate user/task strings into shell commands.
   - All subprocess calls must use `asyncio.create_subprocess_exec()` with structured argument lists.

8. **Relative Paths in Documentation:**
   - Any doc in this repo (`AGENTS.md`, `README.md`, or anything an agent writes/edits) must reference other repo files with **relative paths only** — never absolute filesystem paths (`C:\...`, `/home/...`) or `file:///` URIs.
   - Absolute paths break for every other clone, machine, and OS.

---

## 3. Development Environment & `uv` Workflow

The repository standardizes on **`uv`** as the package and project manager.

### Prerequisites
- Python >= 3.12
- `uv` installed (`pip install uv` or native binary)
- Git CLI (available in `PATH`)

### Essential Commands

```bash
# Set up virtual environment and install dependencies in editable mode
uv venv
uv pip install -e ".[dev]"

# Run test suite
uv run pytest

# Run tests with verbose output and coverage
uv run pytest -v --cov=agentflow

# Code formatting and lint checking
uv run ruff check .
uv run ruff format --check .

# Apply lint fixes and formatting
uv run ruff check --fix .
uv run ruff format .

# Type checking
uv run mypy src tests

# Test global CLI entry point
uv run agentflow --version
uv run agentflow doctor
```

### Global CLI Installation (for integration testing)
```bash
uv tool install --editable .
agentflow doctor
```

---

## 4. Repository Structure & Module Architecture

Code lives under `src/agentflow/`, organized by responsibility (`config/`, `project/`, `task/`, `routing/`, `agents/`, `workflow/`, `process/`, `git/`, `persistence/`, `observability/`, `concurrency/`, `ui/`). This mirrors the logical package architecture defined in [technical-design-document.md](technical-design-document.md) — read that doc's module architecture section rather than relying on a tree here, since the actual layout is always the source of truth.

> [!NOTE]
> Do not create empty placeholder files or modules for future phases until the active milestone requires them.

---

## 5. Phased Delivery Protocol & Verification Gates

Implementation is organized into **8 Milestones** spanning **11 Phases** (full detail in [phased-implementation-plan.md](phased-implementation-plan.md)).

### Milestone Roadmap

| Milestone | Phases | Scope | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Milestone 1** | **Phase 1** | Project Foundation & CLI | Packaging, Typer CLI, global config, project discovery, SQLite init, process executor, `agentflow doctor`. |
| **Milestone 2** | **Phase 2** | Provider CLI Adapter Layer | `AgentAdapter` interface, Claude/Codex/Gemini adapters, structured output, session resume detection. |
| **Milestone 3** | **Phase 3** | Interactive Planning Workflow | Multi-turn planning, user question loop, Opus escalation, `approved-plan.md`, Pydantic `TaskProfile`. |
| **Milestone 4** | **Phase 4** | Deterministic Routing Engine | Complexity scoring, hard-risk overrides, first-match rules, precedence engine, `agentflow route`. |
| **Milestone 5** | **Phases 5–6** | Worktree Implementation & Verification | Git worktrees, Codex execution, write lock, real test/build runner, bounded repair loop. |
| **Milestone 6** | **Phases 7–9** | Review, Docs & Final Approval | Gemini review, structured findings fix cycle, documentation sync, final summary & human sign-off. |
| **Milestone 7** | **Phase 10** | Resume & Crash Recovery | `agentflow resume`, run locks, multi-project concurrency, stale worktree cleanup, `agentflow runs`. |
| **Milestone 8** | **Phase 11** | Observability & Analytics | Structured event logging, local metric aggregation, `agentflow stats` (informational only). |

### Phase Execution Rules for Coding Agents
1. **Determine Active Milestone:** Always inspect repository state to confirm which phases are completed before writing code.
2. **Strict Phase Boundaries:** Never implement forward-looking speculative features or placeholder stubs for later phases. Focus solely on the active phase.
3. **Phase 4 Stability Boundary:** Phase 4 (Deterministic Routing) is a **hard gate**. You must never begin Phase 5 (autonomous worktree modifications) until Phase 4 passes 100% of its table-driven routing reproducibility tests.
4. **Verification Gates:** Before declaring a phase complete:
   - All modules must import cleanly without syntax or type errors.
   - All automated unit/integration tests for current and previous phases must pass (`uv run pytest`).
   - CLI commands introduced by the phase must be manually or programmatically verified.
   - Zero regression to existing functionality.

---

## 6. Orchestrated Agents & Deterministic Routing Reference

AgentFlow orchestrates three CLI toolsets. Full specs for every role and schema below: [technical-design-document.md](technical-design-document.md) §13–§27.

### 1. Agent Roles & V1 Model Allocations
| Role | Assigned Model | Provider CLI | Purpose & Trigger |
| :--- | :--- | :--- | :--- |
| **Default Planner** | `Claude Sonnet 5` | `claude` | Interactive repository analysis, question loop, plan creation |
| **Architecture Planner** | `Claude Opus 5.5` | `claude` | Escalated planning for architecture changes, new services, datastores, payments |
| **Lightweight Coder** | `GPT-6 Luna` | `codex` | Low complexity (score 0–2), no hard risks |
| **Standard / High Coder**| `GPT-6 Sol` | `codex` | Medium/high complexity or any hard risk flag |
| **Implementation Escalation** | `Claude Sonnet 5` | `claude` | Repeated implementation or complex repair failures |
| **Default Reviewer** | `Gemini 3.8 Flash`| `gemini` | Fast, independent read-only diff inspection post-verification |
| **Deep Reviewer** | `Claude Sonnet 5` | `claude` | Deep inspection when security, auth, concurrency, or payments are modified |
| **Architecture Reviewer**| `Claude Opus 5.5` | `claude` | Structural review when `architecture_change = true` |
| **Documenter** | `Gemini 3.8 Flash`| `gemini` | Sync project docs (`architecture.md`, etc.) based on final diff |

### 2. Core Schemas & Contracts
Canonical schemas (`TaskProfile`, Routing Decision Contract, `AgentRequest`/`AgentResult`, Review Findings) are defined in the TDD — read them there rather than duplicating field lists here.

### 3. Deterministic Precedence Hierarchy
When evaluating routing for any workflow stage:
1. **Explicit User Override** (CLI flag or manual input)
2. **Hard Risk Rules** (e.g. `force_standard_if_any` triggers Sol)
3. **Workflow Stage Rules** (Planning, Review, Documentation specific rules)
4. **Project-Specific Rules** (Configured top-to-bottom in `routing.yaml`, first match wins)
5. **Complexity Rules** (Low -> Luna, Medium/High -> Sol)
6. **Default Fallback**

---

## 7. Safety, Security & Subprocess Standards

1. **Subprocess Isolation (`process/executor.py`):**
   ```python
   # Always use argument arrays; never pass a shell string
   proc = await asyncio.create_subprocess_exec(
       *cmd_args,
       cwd=working_dir,
       stdout=asyncio.subprocess.PIPE,
       stderr=asyncio.subprocess.PIPE,
       env=clean_env,
   )
   ```
2. **Working Directory & Worktree Safety (`git/worktree.py`):**
   - The user's active Git working directory must never be modified by code generation agents.
   - Worktrees are created in `~/.agentflow/worktrees/<project-name>/<run-id>` with branch `agentflow/<run-id>`.
   - Never run `git reset --hard`, `git clean -fd`, `git push --force`, or auto-merge without explicit human confirmation.
3. **Path Handling & Portability:**
   - Always use `pathlib.Path` for filesystem interactions.
   - Canonicalize repository paths via `.resolve()` before computing SHA-256 project identifiers.
   - Support both POSIX and Windows path representations.
4. **Crash Recovery & Concurrency Safety:**
   - Persist workflow state transitions in SQLite before triggering external processes.
   - Implement both run-level locks (`.run.lock`) and worktree writer locks (`.writer.lock`) to prevent race conditions across concurrent AgentFlow runs.
