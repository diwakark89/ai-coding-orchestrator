# AgentFlow — Phased Implementation Plan

**Status:** FINALIZED V1 Implementation Plan  
**Source:** AgentFlow Technical Design Document  
**Goal:** Build AgentFlow incrementally so every phase is runnable, testable, and does not depend on unfinished later phases.

---

# Global Instructions for the Coding Agent

Before making changes:

1. Inspect the existing repository.
2. Do not assume files/classes/modules exist unless confirmed.
3. If the repository is empty, initialize the project using the structure described in this plan.
4. Do not implement functionality from later phases early unless it is required to keep the current phase clean and functional.
5. Every phase must:
   - compile/import successfully;
   - pass its own tests;
   - preserve all previous tests;
   - leave the CLI runnable.
6. Do not introduce:
   - LangChain;
   - LangGraph;
   - CrewAI;
   - AutoGen;
   - external databases;
   - message queues;
   - direct AI APIs.
7. AI integrations must use local installed CLIs only.
8. Use Python 3.12+.
9. Use:
   - Typer;
   - Rich;
   - Pydantic v2;
   - PyYAML;
   - SQLite;
   - pytest;
   - asyncio;
   - Git CLI.
10. Use `asyncio.create_subprocess_exec()` for process execution.
11. Never build shell commands by concatenating user-provided input.
12. Prefer small, testable modules over large orchestration classes.

---

# Phase 1 — Project Foundation and CLI [DONE]

## Objective

Create the standalone AgentFlow Python application with:

- packaging;
- global CLI;
- configuration loading;
- project discovery;
- SQLite initialization;
- safe subprocess execution;
- `agentflow doctor`.

No AI agent invocation yet.

---

## Files / Areas to Inspect

If repository already exists, inspect:

- `pyproject.toml`
- existing Python package layout;
- existing CLI framework;
- existing tests;
- configuration handling;
- logging;
- persistence.

If starting from scratch, create the minimal package structure required for this phase.

Proposed logical modules:

```text
src/agentflow/
├── cli.py
├── application.py
├── config/
├── project/
├── process/
├── persistence/
└── ui/
```

Do not create empty placeholder modules for future phases unless needed.

---

## Step-by-Step Instructions

### 1. Initialize Python project

Configure:

- Python >= 3.12;
- Typer;
- Rich;
- Pydantic v2;
- PyYAML;
- pytest.

Configure executable:

```text
agentflow
```

mapped to the CLI entry point.

---

### 2. Implement global configuration

Support:

```text
~/.agentflow/config.yaml
```

Schema:

```yaml
version: 1

cli:
  claude:
    command: claude

  codex:
    command: codex

  gemini:
    command: gemini

storage:
  database: ~/.agentflow/agentflow.db

worktrees:
  root: ~/.agentflow/worktrees

logging:
  root: ~/.agentflow/logs
```

Requirements:

- expand `~`;
- validate with Pydantic;
- produce readable validation errors;
- create sensible defaults when file does not exist;
- do not overwrite an existing file automatically.

---

### 3. Implement project discovery

From current working directory:

1. Resolve canonical directory.
2. Walk upward.
3. Detect Git repository root.
4. Locate:

```text
.ai-orchestrator/routing.yaml
```

Return a structured `ProjectContext`.

Support explicit project path:

```bash
agentflow --project /path/to/repo ...
```

or equivalent `-C`.

Explicit path must override auto-discovery.

---

### 4. Implement project identity

Generate project ID from canonical repository path.

Example concept:

```text
SHA256(canonical_repository_path)
```

Do not use directory name as unique ID.

---

### 5. Implement project routing configuration loader

For Phase 1, validate at least:

```yaml
version: 1

project:
  name: example
```

Later phases will extend the schema.

Reject unsupported configuration versions.

---

### 6. Implement safe process executor

Create reusable abstraction for subprocess execution.

Requirements:

- use `asyncio.create_subprocess_exec`;
- accept argument arrays;
- support working directory;
- capture stdout;
- capture stderr;
- return exit code;
- support timeout;
- never use `shell=True`.

Result contract should contain:

```text
command
exit_code
stdout
stderr
started_at
completed_at
timed_out
```

---

### 7. Implement SQLite initialization

Default:

```text
~/.agentflow/agentflow.db
```

Create initial tables:

### projects

```text
id
name
repository_path
config_hash
created_at
last_used_at
```

### runs

```text
id
project_id
task
status
created_at
updated_at
```

Do not add future tables until they are required.

Use schema migrations or an explicit schema-version mechanism from the start.

---

### 8. Implement `agentflow doctor`

Check:

- Python compatibility;
- Git availability;
- Claude CLI availability;
- Codex CLI availability;
- Gemini CLI availability;
- global data directory writable;
- SQLite writable;
- current Git repository;
- routing configuration presence.

Do not require all provider CLIs to exist for the command itself to run.

Display:

```text
✓ Git
✓ Claude CLI
✗ Gemini CLI
...
```

Return non-zero exit code when critical environment requirements are missing.

---

### 9. Implement minimal CLI commands

Required:

```bash
agentflow --version

agentflow doctor

agentflow status
```

For Phase 1, `status` may report that no active run exists.

---

## Verification / Test Criteria

Automated tests must cover:

- global config parsing;
- invalid config rejection;
- `~` expansion;
- repository root discovery;
- nested-directory project discovery;
- explicit project override;
- canonical project ID stability;
- process stdout capture;
- process stderr capture;
- process exit code;
- timeout handling;
- SQLite initialization;
- doctor result generation.

Manual verification:

```bash
agentflow --version
agentflow doctor
```

Both must execute successfully.

---

## Phase 1 Completion Criteria

Phase is complete only when:

- package installs locally;
- `agentflow` runs globally;
- project discovery works;
- SQLite initializes safely;
- safe subprocess execution exists;
- all tests pass.

---

# Phase 2 — Provider CLI Adapter Layer [DONE]

## Objective

Introduce a provider-independent AI agent interface and working adapters for:

- Claude CLI;
- Codex CLI;
- Gemini CLI.

Do not implement planning/routing yet.

---

## Files / Areas to Inspect

Inspect:

- process executor from Phase 1;
- configuration objects;
- CLI executable configuration;
- existing tests.

Create logical package:

```text
agentflow/agents/
```

---

## Step-by-Step Instructions

### 1. Define provider enum

Support:

```text
ANTHROPIC
OPENAI
GOOGLE
```

Do not use arbitrary provider strings throughout application code.

---

### 2. Define common request contract

Create provider-independent `AgentRequest`.

Required fields:

```text
role
prompt
repository_path
working_directory
model
read_only
timeout_seconds
```

Optional provider-specific settings should live in a separate extensible structure rather than polluting the common contract.

---

### 3. Define common result contract

Create `AgentResult`.

Required:

```text
provider
model
session_id
exit_code
text
raw_events
started_at
completed_at
timed_out
```

---

### 4. Define AgentAdapter interface

Required operations:

```text
start(request)
resume(session_id, request)
```

If a provider does not support resume in a usable way, return a well-defined unsupported capability rather than simulating it silently.

---

### 5. Implement ClaudeAdapter

Responsibilities:

- construct CLI argument array;
- select configured model;
- support structured/non-interactive invocation;
- parse response;
- capture session ID when available;
- support resume when available;
- honor read-only/planning permission mode where supported.

Do not embed authentication logic.

---

### 6. Implement CodexAdapter

Responsibilities:

- use Codex CLI;
- support model selection;
- use non-interactive execution;
- support structured output when available;
- capture CLI errors distinctly from model output;
- operate against specified working directory.

Do not give Codex unrestricted shell access beyond what its CLI configuration permits.

---

### 7. Implement GeminiAdapter

Responsibilities:

- use Gemini CLI;
- support explicit model selection;
- parse structured output;
- support session resume when available;
- operate against specified repository/worktree.

---

### 8. Implement adapter registry

Example concept:

```text
Provider → AgentAdapter
```

Workflow code later must not instantiate provider-specific adapters directly.

---

### 9. Add adapter capability metadata

Each adapter should report capabilities such as:

```text
supports_resume
supports_read_only_mode
supports_structured_output
supports_model_selection
```

Do not hardcode these assumptions in workflow logic.

---

### 10. Extend `agentflow doctor`

Validate each configured CLI using a lightweight command.

Do not send model requests during doctor checks.

---

## Verification / Test Criteria

Use mocked/fake CLI executables for automated tests.

Cover:

- correct arguments generated;
- prompt passed as one argument/input safely;
- model passed correctly;
- working directory honored;
- stdout parsed;
- stderr captured;
- non-zero exit code preserved;
- timeout preserved;
- session ID parsed;
- unsupported resume handled explicitly.

Manual verification:

Run one harmless prompt against each locally installed CLI from a test repository.

---

## Phase 2 Completion Criteria

The application can invoke all three provider CLIs through the same internal interface without workflow code knowing CLI-specific details.

---

# Phase 3 — Interactive Planning Workflow

## Objective

Implement the planning workflow using:

- Claude Sonnet 5 by default;
- Claude Opus 5 for architecture escalation;
- interactive user decisions;
- approved plan artifact;
- machine-readable TaskProfile.

No implementation agent is invoked yet.

---

## Files / Areas to Inspect

Inspect:

- Claude adapter;
- project context;
- run persistence;
- terminal UI.

Create logical modules:

```text
workflow/planning
task/profile
ui/questions
ui/approval
```

---

## Step-by-Step Instructions

### 1. Add workflow states

At minimum:

```text
NEW
PROJECT_READY
PLANNING
WAITING_FOR_USER
PLAN_READY
PLAN_APPROVED
TASK_CLASSIFIED
BLOCKED
FAILED
CANCELLED
```

Persist every state transition.

---

### 2. Implement run creation

Command:

```bash
agentflow run "task description"
```

For Phase 3:

- create run;
- discover project;
- start planning only;
- stop after plan approval and TaskProfile generation.

Do not modify source code.

---

### 3. Define planning prompt contract

Sonnet must be instructed to:

- inspect repository;
- inspect relevant architecture/documentation;
- understand the requested change;
- identify ambiguity;
- ask only material questions;
- present alternatives when needed;
- recommend one;
- avoid implementation;
- produce a structured final plan.

---

### 4. Implement interactive planning loop

The orchestrator controls the terminal.

Example:

```text
Sonnet:
There are two valid approaches...

1. ...
2. ...

Recommended: 1

Choose:
```

User response is sent back through resumed planner session.

Persist:

- question;
- user answer;
- planner session ID.

---

### 5. Define Opus escalation criteria

Planning must escalate to Opus when the task contains or discovers:

```text
architecture_change
new_service
new_datastore
security_boundary_change
payment
cross_service_ownership_change
```

For Phase 3 these flags may initially be returned by Sonnet.

The orchestrator must record why escalation occurred.

---

### 6. Produce `approved-plan.md`

Required sections:

```text
Objective
Existing Behavior
Required Behavior
Architecture Impact
Data Changes
API Changes
Implementation Steps
Validation
Authorization
Concurrency
Idempotency
Tests
Documentation
Risks
Acceptance Criteria
```

Sections may say "Not applicable" where appropriate.

---

### 7. Define TaskProfile Pydantic schema

Include at least:

```text
stage
technologies
affected_layers
estimated_files

schema_change
destructive_schema_change
api_contract_change

authentication
authorization
data_ownership

transaction_logic
concurrency
idempotency

payment
security_boundary_change

external_integration
new_dependency
new_service
new_datastore

architecture_change
ai_or_rag
performance_sensitive
```

---

### 8. Generate machine-readable TaskProfile

Planner output must conform exactly to schema.

Reject malformed output.

Do not silently infer missing booleans from prose.

If parsing fails:

- allow bounded retry;
- otherwise mark planning `BLOCKED`.

---

### 9. Implement plan approval

User options:

```text
Approve
Continue planning
Cancel
```

Only `Approve` transitions to `PLAN_APPROVED`.

---

### 10. Persist artifacts

Store:

```text
task.md
approved-plan.md
task-profile.json
```

under run artifacts.

---

## Verification / Test Criteria

Tests must cover:

- state transitions;
- waiting for user;
- session resume;
- user decision persistence;
- TaskProfile validation;
- malformed planner result retry;
- plan approval;
- cancellation;
- Opus escalation trigger.

Manual workflow:

```bash
agentflow run "Add a simple endpoint"
```

must end with:

```text
PLAN_APPROVED
TASK_CLASSIFIED
```

without editing repository code.

---

# Phase 4 — Deterministic Routing Engine

## Objective

Implement the finalized deterministic model-selection policy.

This phase must contain **no model judgment about exact worker selection**.

---

## Files / Areas to Inspect

Inspect:

- TaskProfile;
- routing configuration;
- project config loader.

Create:

```text
routing/
├── engine
├── matcher
├── rules
├── decision
└── complexity
```

---

## Step-by-Step Instructions

### 1. Extend `routing.yaml` schema

Support:

```text
models
complexity
routing
escalation
verification
```

Validate all model references.

Reject references to undefined model aliases.

---

### 2. Implement complexity scorer

Complexity must derive from:

```text
estimated_files
+
configured flags
```

Example:

```text
1-3 files = 0
4-7 = 1
8+ = 2
```

and configured flag values.

Return:

```text
score
LOW | MEDIUM | HIGH
contributing factors
```

---

### 3. Implement hard-risk evaluation

Default VitePrep configuration includes:

```text
authentication
authorization
data_ownership
concurrency
idempotency
transaction_logic
destructive_schema_change
payment
security_boundary_change
ai_or_rag
```

These force at least standard implementation tier.

---

### 4. Implement routing precedence

Exactly:

```text
1. User override
2. Hard risk
3. Workflow stage
4. Project-specific rule
5. Complexity
6. Default
```

Do not change this ordering.

---

### 5. Implement first-match semantics

Rules at the same priority level execute top-to-bottom.

First match wins.

---

### 6. Define RoutingDecision

Include:

```text
stage
provider
model
role
matched_rule
reason
complexity_score
complexity
risk_flags
```

---

### 7. Add V1 model aliases

Configuration must support:

Planning:

```text
Claude Sonnet 5
Claude Opus 5
```

Implementation:

```text
GPT-5.6 Luna
GPT-5.6 Terra
Claude Sonnet 5
```

Review/documentation:

```text
Gemini 3.8 Flash
Claude Sonnet 5
Claude Opus 5
```

Do not include:

```text
GPT-5.4 Mini
GPT-5.6 Sol
```

---

### 8. Implement `agentflow route`

Command:

```bash
agentflow route "task"
```

It should:

1. create planning/classification context;
2. generate TaskProfile;
3. calculate complexity;
4. route;
5. display explanation;
6. not create worktree;
7. not modify application files.

---

### 9. Display routing explanation

Example:

```text
Stage: IMPLEMENTATION
Score: 7
Complexity: HIGH

Matched:
implementation.force-standard

Risk flags:
- concurrency
- idempotency

Selected:
OpenAI / GPT-5.6 Terra
```

---

### 10. Persist RoutingDecision

Create:

```text
routing-decision.json
```

and database record.

---

## Verification / Test Criteria

This phase needs extensive table-driven tests.

Mandatory scenarios:

### Case A

```text
2 files
no risk
```

Expected:

```text
Luna
```

### Case B

```text
5 files
normal business logic
```

Expected:

```text
Terra
```

### Case C

```text
2 files
authorization=true
```

Expected:

```text
Terra
```

### Case D

```text
concurrency=true
idempotency=true
```

Expected:

```text
Terra
```

### Case E

```text
architecture_change=true
planning
```

Expected:

```text
Opus
```

### Case F

User override provided.

Expected:

User override wins unless configuration explicitly prohibits the model.

### Case G

Same TaskProfile + same config repeated 100 times.

Expected:

Identical RoutingDecision every time.

---

# Phase 5 — Git Worktree and Implementation Workflow

## Objective

Implement source-code execution safely inside isolated Git worktrees.

---

## Files / Areas to Inspect

Inspect:

- project identity;
- process executor;
- routing decision;
- Codex adapter;
- state machine.

Create:

```text
git/
workflow/implementation
```

---

## Step-by-Step Instructions

### 1. Implement worktree manager

Default root:

```text
~/.agentflow/worktrees/
```

Path format:

```text
<project-name>/<run-id>/
```

---

### 2. Create dedicated branch

Example:

```text
agentflow/<run-id>
```

Validate branch does not already exist.

---

### 3. Worktree safety rules

Do not:

- modify primary working tree;
- reset main branch;
- delete user branches;
- force clean primary repository.

---

### 4. Add implementation states

```text
WORKTREE_READY
IMPLEMENTING
VERIFYING
REPAIRING
```

---

### 5. Route implementation model

Use RoutingDecision.

V1:

```text
LOW → Luna
MEDIUM → Terra
HIGH → Terra
```

Hard-risk always at least Terra.

---

### 6. Build implementation prompt

Provide implementation agent:

- approved plan;
- TaskProfile;
- repository context;
- acceptance criteria;
- scope constraints;
- worktree path.

Explicit instruction:

```text
Do not redesign approved architecture.
If plan is invalid or impossible, stop and report blocker.
```

---

### 7. Invoke Codex adapter

Run against worktree only.

Capture:

- session;
- result;
- changed files;
- Git diff.

---

### 8. Detect no-op implementation

If implementation claims success but Git diff is empty while changes were expected, mark stage failed or blocked.

---

### 9. Add write lock

Only one writer may modify a worktree at a time.

---

## Verification / Test Criteria

Use a small fixture Git repository.

Test:

- branch creation;
- worktree creation;
- primary working tree unchanged;
- implementation executed in worktree;
- diff captured;
- worktree locking;
- existing branch conflict handling;
- cleanup after failed creation.

Manual test:

Run a trivial change using Luna and verify source changes appear only in the AgentFlow worktree.

---

# Phase 6 — Deterministic Verification and Repair

## Objective

Run real build/test/lint commands and automatically repair bounded failures.

---

## Files / Areas to Inspect

Inspect:

- project config;
- process executor;
- implementation workflow;
- routing engine.

Create:

```text
workflow/verification
workflow/repair
```

---

## Step-by-Step Instructions

### 1. Add verification configuration

Example:

```yaml
verification:
  backend:
    detect:
      - pom.xml
    commands:
      - ./mvnw test
      - ./mvnw verify

  frontend:
    detect:
      - package.json
    commands:
      - npm run lint
      - npm run typecheck
      - npm test
      - npm run build

  python:
    detect:
      - pyproject.toml
    commands:
      - pytest
```

---

### 2. Detect applicable verification groups

Run only groups whose configured detection conditions match.

---

### 3. Execute commands sequentially

For each:

- command;
- exit code;
- stdout;
- stderr;
- duration.

Stop current verification pass after configured fatal failure if appropriate.

---

### 4. Define verification result

Overall status:

```text
PASSED
FAILED
ERROR
TIMED_OUT
```

---

### 5. Implement simple failure classifier

First use deterministic patterns where possible:

```text
lint
formatting
missing import
type mismatch
compilation
unit test
integration test
transaction
database
unknown
```

Do not initially build sophisticated AI classification.

---

### 6. Implement repair routing

Simple/local failures:

```text
Luna
```

Examples:

- missing import;
- formatting;
- obvious type issue;
- straightforward fixture issue.

Complex failures:

```text
Terra
```

Examples:

- business logic;
- SQL;
- transaction;
- concurrency;
- idempotency;
- multi-file regression.

---

### 7. Implement escalation

```text
Luna repair fails twice
→ Terra
```

```text
Terra repeatedly fails
→ Sonnet
```

If Sonnet identifies architecture blocker:

```text
→ planning/Opus
```

---

### 8. Re-run full verification after repair

Do not only rerun the previously failing command before declaring success.

Final success requires configured verification sequence to pass.

---

### 9. Persist results

Store:

```text
verification.json
```

and DB records.

---

## Verification / Test Criteria

Automated fixture scenarios:

- all commands pass;
- compile failure;
- lint failure;
- timeout;
- repair succeeds first time;
- Luna fails twice → Terra;
- Terra fails configured limit → escalation;
- final verification re-runs complete sequence.

No AI-generated statement may mark verification successful.

Only process exit codes may do so.

---

# Phase 7 — Independent Review and Fix Cycle

## Objective

Implement independent AI review after deterministic verification passes.

---

## Files / Areas to Inspect

Inspect:

- final Git diff;
- TaskProfile;
- routing;
- verification results;
- Gemini adapter.

Create:

```text
workflow/review
```

---

## Step-by-Step Instructions

### 1. Add review states

```text
REVIEWING
REVIEW_FIXING
REVIEW_APPROVED
```

---

### 2. Select reviewer deterministically

Default:

```text
Gemini 3.8 Flash
```

Deep review:

```text
Claude Sonnet 5
```

when risk includes configured sensitive areas.

Architecture review:

```text
Claude Opus 5
```

when architecture changed.

---

### 3. Provide review context

Reviewer gets:

- approved plan;
- TaskProfile;
- final diff;
- verification results.

Reviewer must not modify files directly.

---

### 4. Define structured findings

Each finding:

```text
severity
category
file
line
problem
recommendation
```

Severity:

```text
CRITICAL
HIGH
MEDIUM
LOW
```

---

### 5. Define default enforcement

```text
CRITICAL → mandatory fix
HIGH → mandatory fix
MEDIUM → configurable
LOW → informational
```

---

### 6. Send mandatory findings to implementation worker

Use original implementation worker unless routing requires escalation.

Reviewer remains read-only.

---

### 7. Re-run full verification

Every code fix requires complete verification.

---

### 8. Re-review

Reviewer verifies mandatory issues resolved.

Limit fix cycles.

---

### 9. Persist review

Store:

```text
review.md
```

plus machine-readable findings.

---

## Verification / Test Criteria

Test:

- default Gemini review;
- deep Sonnet review based on flags;
- architecture Opus review;
- CRITICAL forces fix;
- HIGH forces fix;
- LOW does not block;
- review fix triggers verification;
- maximum review cycles enforced.

---

# Phase 8 — Documentation Workflow

## Objective

Update documentation only after implementation and review are stable.

---

## Files / Areas to Inspect

Inspect project-configured documentation rules.

Do not assume documentation filenames.

Create:

```text
workflow/documentation
```

---

## Step-by-Step Instructions

### 1. Add documentation configuration

Allow project to define:

```yaml
documentation:
  enabled: true

  candidate_files:
    - architecture.md
    - implemented-features.md
```

Files are project-specific.

---

### 2. Use Gemini 3.8 Flash by default

Provide:

- approved plan;
- TaskProfile;
- final diff;
- verification results;
- final review findings.

---

### 3. Documentation must describe final implementation

Do not blindly copy original design if implementation changed.

---

### 4. Documentation write safety

Documentation is another writer stage.

It must obey the same one-writer lock.

---

### 5. Re-run lightweight verification if documentation can affect build

Example:

- generated docs;
- markdown lint;
- docs included in static build.

Project config determines commands.

---

### 6. Produce documentation summary

Store:

```text
documentation-summary.md
```

---

## Verification / Test Criteria

Test:

- documentation disabled;
- documentation enabled;
- missing candidate file handled;
- Gemini receives final implementation context;
- only intended documentation files changed where scoped.

---

# Phase 9 — Final Approval and Run Completion

## Objective

Provide a safe human-controlled completion workflow.

---

## Step-by-Step Instructions

### 1. Produce final summary

Include:

```text
Task
Plan
Selected models
Routing reasons
Files changed
Verification status
Review status
Documentation changes
Outstanding warnings
Worktree
Branch
```

---

### 2. Add final states

```text
READY_FOR_APPROVAL
COMPLETED
CANCELLED
BLOCKED
FAILED
```

---

### 3. User approval options

At minimum:

```text
Approve completion
Keep worktree for manual inspection
Cancel
```

Do not automatically merge to the user's branch.

---

### 4. Do not auto-push

V1 must not:

- push automatically;
- merge automatically;
- create production deployments.

Those can be future explicit features.

---

## Verification / Test Criteria

A complete run must reach:

```text
READY_FOR_APPROVAL
```

only if:

- plan approved;
- implementation exists;
- deterministic verification passed;
- mandatory review findings resolved;
- required documentation completed.

---

# Phase 10 — Resume, Crash Recovery, and Multi-Project Reliability

## Objective

Make AgentFlow production-safe for regular local use.

---

## Files / Areas to Inspect

Inspect:

- all persisted workflow state;
- sessions;
- worktrees;
- locks;
- subprocess lifecycle.

---

## Step-by-Step Instructions

### 1. Implement `agentflow resume`

Example:

```bash
agentflow resume RUN-001
```

Read persisted state and determine next legal action.

---

### 2. Resume planning sessions

When provider supports session continuation, reuse session ID.

If session is unavailable:

- reconstruct context from persisted artifacts;
- start a new session;
- record that recovery occurred.

---

### 3. Handle interrupted implementation

Before rerunning:

1. inspect worktree;
2. inspect Git diff;
3. inspect recorded implementation state;
4. avoid blindly repeating completed edits.

---

### 4. Add run-level locks

Prevent two processes from controlling the same run.

---

### 5. Add worktree writer locks

Prevent concurrent writers.

---

### 6. Support simultaneous different projects

Example:

```text
Run A → VitePrep worktree
Run B → German project worktree
```

Both may execute concurrently.

---

### 7. Handle stale locks

Locks must contain enough metadata to determine whether owning process still exists.

Do not silently delete an apparently active lock.

---

### 8. Add cleanup command

Example:

```bash
agentflow cleanup
```

May remove:

- completed stale worktrees;
- stale logs;
- stale lock files;

only after verifying they are safe to remove.

---

### 9. Implement `agentflow runs`

Show:

```text
Run ID
Project
Task
Status
Current Stage
Created
Updated
```

---

### 10. Implement improved `agentflow status`

Show active/current run details.

---

## Verification / Test Criteria

Test:

- crash after plan;
- crash during implementation;
- crash after implementation before verification;
- resume with existing diff;
- stale lock;
- two projects executing simultaneously;
- same run cannot be controlled twice;
- cleanup does not remove active worktree.

---

# Phase 11 — Observability and Routing Analytics

## Objective

Collect enough local data to improve routing without making routing adaptive.

---

## Step-by-Step Instructions

### 1. Add structured event logging

Example:

```json
{
  "runId": "RUN-001",
  "stage": "IMPLEMENTATION",
  "event": "AGENT_STARTED",
  "provider": "openai",
  "model": "gpt-5.6-terra"
}
```

---

### 2. Capture metrics

At minimum:

```text
model selected
matched routing rule
implementation attempts
verification attempts
first-pass success
review findings
review severity
escalation count
stage duration
```

---

### 3. Add local reporting command

Example:

```bash
agentflow stats
```

Possible output:

```text
Luna
Tasks: 24
First-pass verification: 87%
Escalations: 3

Terra
Tasks: 18
First-pass verification: 94%
```

---

### 4. Do not auto-change routing

Analytics are informational only.

Routing changes remain manual changes to:

```text
routing.yaml
```

---

## Verification / Test Criteria

Statistics must be reproducible from persisted run data.

No external telemetry service required.

---

# Final End-to-End Workflow

When all phases are complete:

```text
agentflow run "Implement referral expiration"
```

must perform:

```text
Project discovery
      ↓
Sonnet planning
      ↓
Interactive decisions
      ↓
Optional Opus architecture escalation
      ↓
Plan approval
      ↓
TaskProfile
      ↓
Deterministic routing
      ↓
Luna or Terra
      ↓
Git worktree implementation
      ↓
Real build/test/lint
      ↓
Repair/escalation
      ↓
Gemini independent review
      ↓
Sonnet/Opus deeper review if required
      ↓
Fix + re-verification
      ↓
Gemini documentation
      ↓
Final summary
      ↓
User approval
```

---

# Required V1 Model Rules

The following must remain unchanged unless project configuration explicitly changes them later:

```text
Planning default
→ Claude Sonnet 5

Architecture planning
→ Claude Opus 5

Low-complexity implementation
→ GPT-5.6 Luna

Medium implementation
→ GPT-5.6 Terra

High implementation
→ GPT-5.6 Terra

Implementation escalation
→ Claude Sonnet 5

Default review
→ Gemini 3.8 Flash

Deep review
→ Claude Sonnet 5

Architecture review
→ Claude Opus 5

Documentation
→ Gemini 3.8 Flash
```

Explicitly excluded from V1:

```text
GPT-5.4 Mini
GPT-5.6 Sol
```

---

# Recommended Build Order

Do not attempt the full orchestrator in one coding-agent task.

Use these implementation milestones:

```text
Milestone 1
Phase 1 [DONE]

Milestone 2
Phase 2 [DONE]


Milestone 3
Phase 3

Milestone 4
Phase 4

Milestone 5
Phases 5–6

Milestone 6
Phases 7–9

Milestone 7
Phase 10

Milestone 8
Phase 11
```

**Phase 4 — deterministic routing — should be treated as a critical stability boundary.** Do not proceed to autonomous code implementation until its rule precedence, hard-risk overrides, complexity scoring, and reproducibility tests are all passing.

# V1 Definition of Done

AgentFlow V1 is complete when:

1. It is installed once and usable globally.
2. It works against multiple independent Git repositories.
3. Each repository controls behavior through `.ai-orchestrator/routing.yaml`.
4. Claude Sonnet can conduct interactive planning.
5. Opus can handle architecture escalation.
6. TaskProfile generation is schema-validated.
7. Model routing is deterministic and explainable.
8. Luna and Terra can implement through Codex CLI.
9. Implementation occurs in isolated Git worktrees.
10. Real project verification determines success.
11. Repair and escalation have bounded retry loops.
12. Gemini performs independent review.
13. Mandatory findings are resolved and re-verified.
14. Gemini can update configured documentation.
15. Runs survive process interruption and can resume.
16. Multiple projects can be handled without changing AgentFlow source code.
17. No direct AI provider API is used.
18. GPT-5.4 Mini and GPT-5.6 Sol are not used.
