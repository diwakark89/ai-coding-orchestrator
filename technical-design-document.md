# Technical Design Document

## Local Multi-Agent Coding Orchestrator

**Status:** FINALIZED for V1  
**Document Type:** Technical Design Document  
**Primary Goal:** Build a reusable local orchestrator that coordinates Claude Code, Codex CLI, and Gemini CLI using existing subscription-authenticated CLIs, without directly integrating provider APIs.

---

# 1. Objective

Build a standalone local CLI application that coordinates multiple AI coding agents across software-development workflows.

The orchestrator must:

- use installed AI coding CLIs rather than direct AI APIs;
- support multiple projects from one global installation;
- perform interactive planning with Claude;
- deterministically select the implementation/review model;
- execute implementation through the appropriate coding CLI;
- run builds, tests, linting, and other verification itself;
- perform independent code review;
- coordinate repair/review loops;
- generate/update documentation;
- preserve workflow state so interrupted tasks can resume;
- isolate agent-generated changes using Git worktrees;
- remain project-independent;
- allow project-specific behavior through one configuration file.

The same orchestrator executable must work across different repositories without application-code changes to the orchestrator.

---

# 2. Core Design Principles

## 2.1 CLI-Based AI Integration

The orchestrator MUST invoke official locally installed coding-agent CLIs.

Supported providers for V1:

- Claude Code
- OpenAI Codex CLI
- Gemini CLI

The orchestrator MUST NOT:

- call provider LLM APIs directly;
- extract OAuth/session tokens from provider CLIs;
- invoke undocumented provider endpoints;
- manage provider credentials itself.

Authentication remains the responsibility of each installed CLI.

---

## 2.2 Deterministic Model Routing

AI models MUST NOT have final authority over worker-model selection.

Models may classify the task and recommend characteristics, but the orchestrator determines the final provider/model through configuration.

Routing must be:

- explicit;
- deterministic;
- explainable;
- configuration-driven;
- reproducible.

Given the same `TaskProfile` and routing configuration, the same routing result must be produced.

---

## 2.3 Project Independence

The orchestrator source code MUST NOT contain assumptions such as:

- Java → Sol
- React → Luna
- Spring Boot → specific model
- PostgreSQL → specific model

Technology may be used as a routing signal, but routing policy belongs to project configuration.

---

## 2.4 One Writer at a Time

Only one AI agent may modify a working tree at a time.

Example:

```text
Codex WRITE
    ↓
Verification
    ↓
Gemini READ/REVIEW
    ↓
Codex WRITE FIXES
```

Multiple coding agents MUST NOT concurrently modify the same working tree.

---

## 2.5 Deterministic Verification

Models do not determine whether implementation works.

The orchestrator runs actual:

- compiler;
- build;
- unit tests;
- integration tests;
- lint;
- type checking;
- static analysis;

and determines success using actual command exit codes.

---

# 3. V1 Model Pool

The following model pool is FINALIZED for V1.

| Responsibility                    | Model            |
| --------------------------------- | ---------------- |
| Default interactive planning      | Claude Sonnet 5  |
| Architecture / ambiguous planning | Claude Opus 5.5    |
| Low-complexity implementation     | GPT-6 Luna     |
| Standard implementation           | GPT-6 Sol    |
| High-complexity implementation    | GPT-6 Sol    |
| Implementation escalation         | Claude Sonnet 5  |
| Independent default review        | Gemini 3.8 Flash |
| Deep code review                  | Claude Sonnet 5  |
| Architecture-critical review      | Claude Opus 5.5    |
| Documentation                     | Gemini 3.8 Flash |

Explicitly excluded:

- GPT-5.4 Mini
- GPT-5.6 Sol

They MUST NOT appear in:

- routing;
- fallback;
- escalation;
- benchmarks;
- manual selection;

unless the configuration is deliberately changed in the future.

---

# 4. High-Level Architecture

```text
                            USER
                              │
                              ▼
                    ┌───────────────────┐
                    │   AgentFlow CLI   │
                    └────────┬──────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
           Project       Workflow      Persistence
           Context        Engine         SQLite
                │            │
                │            ▼
                │      Task Analyzer
                │            │
                │            ▼
                │       TaskProfile
                │            │
                │            ▼
                │       Model Router
                │            │
                │       routing.yaml
                │            │
                ▼            ▼
            Git Worktree   Agent Adapter
                             │
              ┌──────────────┼───────────────┐
              │              │               │
              ▼              ▼               ▼
          Claude CLI      Codex CLI       Gemini CLI
```

---

# 5. Technology Stack

## 5.1 Language

**Python 3.12+**

Reasons:

- strong subprocess support;
- straightforward asynchronous process handling;
- excellent CLI ecosystem;
- simple filesystem/Git integration;
- good JSON/YAML/schema validation support;
- minimal runtime overhead;
- platform-portable.

---

## 5.2 Libraries

| Concern                 | Technology                            |
| ----------------------- | ------------------------------------- |
| CLI                     | Typer                                 |
| Terminal UI             | Rich                                  |
| Configuration           | YAML                                  |
| Schema validation       | Pydantic v2                           |
| Async process execution | `asyncio`                             |
| Persistent state        | SQLite                                |
| Testing                 | pytest                                |
| Packaging               | uv                                    |
| Git integration         | Git CLI                               |
| Logging                 | Python logging + structured JSON logs |

V1 SHOULD NOT use:

- LangChain;
- LangGraph;
- CrewAI;
- AutoGen;
- Kubernetes;
- message queues;
- external databases;
- cloud orchestration services.

---

# 6. Installation Model

AgentFlow is installed once globally on the developer machine.

Example:

```bash
uv tool install agentflow
```

or during development:

```bash
uv tool install --editable .
```

The following command must then work from any directory:

```bash
agentflow --version
```

---

# 7. Multi-Project Model

Each project contains its own project-specific configuration.

Example:

```text
viteprep/
├── .ai-orchestrator/
│   └── routing.yaml
└── ...

german-learning/
├── .ai-orchestrator/
│   └── routing.yaml
└── ...

stock-analysis/
├── .ai-orchestrator/
│   └── routing.yaml
└── ...
```

AgentFlow itself is shared globally.

---

# 8. Configuration Layers

## 8.1 Global Machine Configuration

Location:

```text
~/.agentflow/config.yaml
```

Purpose:

- CLI executable locations;
- global data directory;
- SQLite database location;
- worktree root;
- logging defaults.

Example:

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

Global configuration MUST NOT contain project routing policy.

---

## 8.2 Project Configuration

Location:

```text
<repository>/.ai-orchestrator/routing.yaml
```

This file is the **single source of truth for model-routing behavior**.

It contains:

- project metadata;
- stack information;
- model catalog aliases;
- complexity scoring;
- risk flags;
- routing rules;
- escalation rules;
- verification commands;
- review rules.

---

# 9. Project Discovery

When:

```bash
cd ~/projects/viteprep
agentflow plan "Implement referral expiry"
```

AgentFlow must:

1. determine current directory;
2. walk upward until finding Git repository root;
3. locate `.ai-orchestrator/routing.yaml`;
4. validate configuration;
5. initialize the run.

Equivalent explicit invocation:

```bash
agentflow plan \
  --project ~/projects/viteprep \
  "Implement referral expiry"
```

Supported alias:

```bash
agentflow -C ~/projects/viteprep run "..."
```

Explicit `--project` takes precedence over current-directory discovery.

---

# 10. Project Identity

Directory names MUST NOT be treated as globally unique project identifiers.

Internally:

```text
project_id = SHA256(canonical_repository_path)
```

Project configuration may additionally provide:

```yaml
project:
  name: viteprep
```

This name is for presentation, not uniqueness.

---

# 11. Task Workflow

Primary workflow:

```text
NEW TASK
   │
   ▼
PROJECT DISCOVERY
   │
   ▼
PLANNING
   │
   ▼
USER QUESTIONS / DECISIONS
   │
   ▼
APPROVED PLAN
   │
   ▼
TASK PROFILE
   │
   ▼
DETERMINISTIC ROUTING
   │
   ▼
CREATE WORKTREE
   │
   ▼
IMPLEMENTATION
   │
   ▼
BUILD / TEST / LINT
   │
   ├──── FAILURE ───► REPAIR / ESCALATION
   │
   ▼
INDEPENDENT REVIEW
   │
   ├──── FINDINGS ──► REPAIR
   │
   ▼
RE-VERIFY
   │
   ▼
DOCUMENTATION
   │
   ▼
OPTIONAL ARCHITECTURE REVIEW
   │
   ▼
FINAL USER APPROVAL
   │
   ▼
DONE
```

---

# 12. Workflow State Machine

Suggested states:

```text
NEW
PROJECT_READY

PLANNING
WAITING_FOR_USER
PLAN_READY
PLAN_APPROVED

TASK_CLASSIFIED
ROUTED

WORKTREE_READY

IMPLEMENTING
VERIFYING
REPAIRING

REVIEWING
REVIEW_FIXING
REVIEW_APPROVED

DOCUMENTING

ARCHITECTURE_REVIEW

READY_FOR_APPROVAL
COMPLETED

BLOCKED
FAILED
CANCELLED
```

State transitions must be persisted.

A process crash must not lose workflow progress.

---

# 13. Planning

## 13.1 Default Planner

Default:

```text
Claude Sonnet 5
```

Sonnet must:

- inspect the repository;
- understand existing architecture;
- understand the requested change;
- identify ambiguity;
- ask the user questions when decisions materially affect implementation;
- identify alternative approaches when appropriate;
- recommend one;
- produce the implementation plan.

---

## 13.2 Opus Escalation

Claude Opus 5.5 is used when planning identifies:

- architecture changes;
- new services;
- new databases/datastores;
- security-boundary changes;
- major authentication/authorization redesign;
- payment architecture;
- difficult distributed consistency;
- major cross-service ownership changes;
- major RAG architecture;
- unresolved architectural ambiguity.

---

# 14. Planner Output Contract

The planner must produce two distinct artifacts.

## 14.1 Human-Readable Plan

Example sections:

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

Stored as:

```text
approved-plan.md
```

---

## 14.2 Machine-Readable Task Profile

The planner must not select the exact implementation model.

Instead it produces facts describing the work.

---

# 15. TaskProfile

Conceptual Pydantic contract:

```python
class TaskProfile(BaseModel):
    stage: Stage

    technologies: set[str]
    affected_layers: set[str]

    estimated_files: int

    schema_change: bool = False
    destructive_schema_change: bool = False

    api_contract_change: bool = False

    authentication: bool = False
    authorization: bool = False
    data_ownership: bool = False

    transaction_logic: bool = False
    concurrency: bool = False
    idempotency: bool = False

    payment: bool = False
    security_boundary_change: bool = False

    external_integration: bool = False
    new_dependency: bool = False

    new_service: bool = False
    new_datastore: bool = False

    architecture_change: bool = False

    ai_or_rag: bool = False
    performance_sensitive: bool = False
```

The schema may be expanded later without changing the core routing architecture.

---

# 16. Complexity Scoring

Complexity MUST be computed by the orchestrator.

The planner MUST NOT simply return:

```text
complexity = HIGH
```

without objective signals.

Example configuration:

```yaml
complexity:
  file_count:
    "1-3": 0
    "4-7": 1
    "8+": 2

  flags:
    schema_change: 1
    api_contract_change: 1

    transaction_logic: 2
    concurrency: 2
    idempotency: 2

    authentication: 2
    authorization: 2
    data_ownership: 2

    payment: 3
    security_boundary_change: 3

    external_integration: 1
    new_dependency: 1

    architecture_change: 3
    ai_or_rag: 2
    performance_sensitive: 1

  thresholds:
    low:
      min: 0
      max: 2

    medium:
      min: 3
      max: 5

    high:
      min: 6
```

---

# 17. Hard Risk Overrides

Some tasks must not use the lightweight worker even when the calculated complexity is low.

Default hard-risk flags:

```yaml
force_standard_if_any:
  - authentication
  - authorization
  - data_ownership
  - concurrency
  - idempotency
  - transaction_logic
  - destructive_schema_change
  - payment
  - security_boundary_change
  - ai_or_rag
```

Example:

```text
Files changed: 2
Complexity score: 2
Authorization: true
```

Result:

```text
GPT-6 Sol
```

not Luna.

---

# 18. Routing Precedence

Routing MUST use this exact precedence:

```text
1. Explicit user override
2. Hard safety/risk rule
3. Workflow-stage rule
4. Project-specific rule
5. Complexity rule
6. Default
```

Rules must use **first-match wins** semantics within their priority level.

---

# 19. Implementation Routing

Default V1 implementation routing:

```text
LOW
+
no hard-risk flags
      ↓
GPT-6 Luna
```

```text
MEDIUM
      ↓
GPT-6 Sol
```

```text
HIGH
      ↓
GPT-6 Sol
```

Sol may use stronger CLI reasoning configuration where supported for HIGH tasks.

---

# 20. Implementation Escalation

Default escalation chain:

```text
Luna
  ↓
Sol
  ↓
Sonnet 5
```

If the problem is identified as architectural:

```text
Implementation blocked
      ↓
Opus 5.5 planning
      ↓
Updated plan
      ↓
Implementation routed again
```

Opus SHOULD NOT normally be used merely as a stronger coding retry.

---

# 21. Review Routing

Default:

```text
Gemini 3.8 Flash
```

Use Sonnet 5 for deeper review when code affects:

- authentication;
- authorization;
- payments;
- concurrency;
- transactions;
- data ownership;
- security-sensitive behavior.

Use Opus 5.5 for architecture review where:

```text
architecture_change = true
```

---

# 22. Documentation Routing

Default documentation model:

```text
Gemini 3.8 Flash
```

Documentation changes must use actual implementation state rather than the original proposal alone.

Documentation agent should receive:

- approved plan;
- final Git diff;
- verification results;
- review findings;
- final corrected implementation.

---

# 23. Example `routing.yaml`

```yaml
version: 1

project:
  name: viteprep

technologies:
  backend:
    - java-25
    - spring-boot-4
    - python
    - fastapi

  frontend:
    - react
    - nextjs
    - typescript

  data:
    - postgresql
    - pgvector
    - redis

  ai:
    - embeddings
    - vector-search
    - rag

models:
  planner:
    default:
      provider: anthropic
      model: sonnet-5

    architecture:
      provider: anthropic
      model: opus-5-5

  implementation:
    lightweight:
      provider: openai
      model: gpt-6-luna

    standard:
      provider: openai
      model: gpt-6-sol

    escalation:
      provider: anthropic
      model: sonnet-5

  review:
    default:
      provider: google
      model: gemini-3.8-flash

    deep:
      provider: anthropic
      model: sonnet-5

    architecture:
      provider: anthropic
      model: opus-5-5

  documentation:
    default:
      provider: google
      model: gemini-3.8-flash

complexity:
  file_count:
    "1-3": 0
    "4-7": 1
    "8+": 2

  flags:
    schema_change: 1
    api_contract_change: 1

    transaction_logic: 2
    concurrency: 2
    idempotency: 2

    authentication: 2
    authorization: 2
    data_ownership: 2

    payment: 3
    security_boundary_change: 3

    external_integration: 1
    new_dependency: 1

    architecture_change: 3
    ai_or_rag: 2
    performance_sensitive: 1

  thresholds:
    low:
      max: 2

    medium:
      min: 3
      max: 5

    high:
      min: 6

routing:
  planning:
    architecture_if_any:
      - architecture_change
      - new_service
      - new_datastore
      - security_boundary_change
      - payment
      - cross_service_ownership_change

    default: planner.default
    architecture: planner.architecture

  implementation:
    force_standard_if_any:
      - authentication
      - authorization
      - data_ownership
      - concurrency
      - idempotency
      - transaction_logic
      - destructive_schema_change
      - payment
      - security_boundary_change
      - ai_or_rag

    rules:
      - id: low-complexity
        when:
          complexity: low
        use: implementation.lightweight

      - id: medium-complexity
        when:
          complexity: medium
        use: implementation.standard

      - id: high-complexity
        when:
          complexity: high
        use: implementation.standard

  review:
    deep_if_any:
      - authentication
      - authorization
      - payment
      - concurrency
      - transaction_logic
      - data_ownership
      - destructive_schema_change

    architecture_if:
      architecture_change: true

    default: review.default
    deep: review.deep
    architecture: review.architecture

  documentation:
    default: documentation.default

escalation:
  implementation:
    lightweight:
      verification_failure_limit: 2
      next: implementation.standard

    standard:
      failure_limit: 2
      next: implementation.escalation

    escalation:
      architecture_blocker:
        next: planner.architecture
```

---

# 24. Routing Decision Contract

Every selection must produce an explainable result.

Conceptual model:

```python
class RoutingDecision(BaseModel):
    stage: Stage

    provider: Provider
    model: str
    role: str

    matched_rule: str
    reason: str

    complexity_score: int
    complexity: ComplexityLevel

    risk_flags: list[str]
```

Example terminal output:

```text
Routing Decision
─────────────────────────────────

Stage:             IMPLEMENTATION
Complexity score:  8
Complexity:        HIGH

Risk:
✓ transaction_logic
✓ concurrency
✓ idempotency

Matched rule:
implementation.force-standard

Selected:
Provider: OpenAI
Model:    GPT-6 Sol

Reason:
Concurrency and idempotency require at least
the standard implementation tier.
```

Routing decisions must be persisted.

---

# 25. Agent Adapter Abstraction

All AI CLIs must implement a common interface.

Conceptually:

```python
class AgentAdapter(Protocol):

    async def start(
        self,
        request: AgentRequest
    ) -> AgentResult:
        ...

    async def resume(
        self,
        session_id: str,
        request: AgentRequest
    ) -> AgentResult:
        ...
```

Implementations:

```text
ClaudeAdapter
CodexAdapter
GeminiAdapter
```

The Workflow Engine MUST NOT contain CLI-specific invocation logic.

---

# 26. Agent Request

Conceptual contract:

```python
class AgentRequest(BaseModel):
    role: AgentRole
    prompt: str

    repository_path: Path
    worktree_path: Path | None

    model: str

    read_only: bool

    timeout_seconds: int | None
```

---

# 27. Agent Result

```python
class AgentResult(BaseModel):
    provider: Provider
    model: str

    session_id: str | None

    exit_code: int

    text: str
    raw_events: list[dict]

    started_at: datetime
    completed_at: datetime
```

---

# 28. Process Execution

Use:

```python
asyncio.create_subprocess_exec()
```

Do NOT use:

```python
os.system(...)
```

or unescaped shell interpolation.

Arguments must be passed individually to prevent shell injection and quoting errors.

Conceptual example:

```python
process = await asyncio.create_subprocess_exec(
    "codex",
    "exec",
    "--json",
    prompt,
    cwd=worktree_path,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
)
```

---

# 29. Interactive Planning Sessions

Planning must support multi-turn interaction.

Example:

```text
Sonnet:
Should existing active access extend from:
A. Current expiry
B. Purchase timestamp

User:
A
```

AgentFlow must resume the existing planner session where supported.

Session IDs must be persisted.

The orchestrator owns:

- user input;
- UI;
- workflow status;
- session mapping.

The provider TUI SHOULD NOT become the primary orchestrator interface.

---

# 30. Git Isolation

Every implementation run must use a Git worktree.

Global location:

```text
~/.agentflow/worktrees/
```

Example:

```text
~/.agentflow/worktrees/
├── viteprep/
│   ├── RUN-001/
│   └── RUN-002/
└── german-learning/
    └── RUN-003/
```

Conceptual creation:

```bash
git worktree add \
  ~/.agentflow/worktrees/viteprep/RUN-001 \
  -b agentflow/RUN-001
```

Main developer working tree remains untouched.

---

# 31. Verification

Verification commands belong in project configuration.

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
      - python -m pytest
```

Each group may specify a repository-relative `working_directory` (default `.`).
Detection paths remain relative to the worktree root; commands execute from the
group's resolved directory inside that worktree. Generated groups for nested projects
set this field to the directory containing their detected project marker.
Python pytest commands use the matching component virtualenv from the original
project (`.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on Unix)
and execute with the worktree component as the working directory. Legacy bare
`pytest` commands receive the same resolution. Missing or unusable interpreters
produce an explicit verification error instead of falling back to `PATH`.

Group selection starts with Git's staged, unstaged, and untracked worktree paths.
`detect` controls whether a group exists for this checkout; it does not by itself
make the group run. A changed path selects the most specific configured
`working_directory` that contains it. A module change selects its module group,
not an ancestor reactor group. Groups tied at the same directory depth all run
unless an explicit overlap declaration covers one. Changes across components
select the union. A root-level file or a
path outside every configured group is treated as shared and selects every
applicable group. Empty or unreadable Git change scope also selects every
applicable group. `scope_paths` adds explicit repository-relative glob mappings
for shared files when a narrower mapping is known. Explicit mappings also add
affected groups for component paths. Configured patterns are an assertion of
ownership; include every affected group when mapping shared code.

Use `supersedes` only when one group performs the *same required checks* as the
named group. For a Skillify-style root `npm test` that invokes the same Jest
suite as `front-end`, the `front-end` group may declare
`supersedes: [root-node]`. If the root group contains additional checks, split
the Jest command into its own group before declaring overlap. A Maven reactor
group may declare `supersedes` for module groups only when the reactor command
actually executes their required tests. A module-only change still runs its
module group; the reactor can cover it when both groups are selected for a
shared change or explicit mapping. AgentFlow never infers overlap from
`npm`, Maven, or command names. Existing configurations remain valid; add
`supersedes` explicitly to eliminate known duplicate suites.
Generated Node groups require an actual `scripts.test` entry in their
`package.json`. For an existing profile whose root package has no test script,
remove the root `npm test` group; `supersedes` describes real coverage and is
not a way to hide an invalid command.

Each verification result records the selected groups, path-based reasons,
suppressed overlaps, rerun groups, cached passes, failed group, and command
results. After a repair, review fix, or documentation edit, AgentFlow reruns
the previously failed group first, then every group selected by paths changed
since the last pass (and any required group not yet run); a pass is retained
only for a group none of whose inputs changed, so no separate full "final gate"
pass is needed. Documentation files (`.md`, `.markdown`, `.rst`, `.adoc`) select
no group and never trigger the shared-file fallback unless a group's
`scope_paths` claims them explicitly. Agents that edit the worktree are told
AgentFlow runs verification itself, to run only the narrowest targeted checks,
and to report environment problems instead of editing test tooling. Missing interpreters, commands, or dependencies block without using a
coding-agent repair attempt. Timeout-like test failures receive one bounded
reproduction check; a passing retry is reported as intermittent and blocks
instead of counting as a successful verification.
The reproduction command is capped by `limits.reproduction_timeout_seconds`
(default 300 seconds) and is attempted once for each failing command and
unchanged worktree state.

AgentFlow must capture:

- command;
- exit code;
- stdout;
- stderr;
- start/end timestamps.

---

# 32. Repair Loop

Verification failure:

```text
Implementation
      ↓
Verification failed
      ↓
Classify failure
      ↓
Route repair
      ↓
Repair
      ↓
Verification
```

Examples appropriate for Luna:

- missing import;
- straightforward compile error;
- lint;
- formatting;
- simple type mismatch;
- localized test fixture issue.

Examples requiring Sol:

- incorrect business logic;
- transaction failure;
- concurrency bug;
- idempotency bug;
- SQL behavior;
- complex regression.

The failure classifier may use fixed rules first and AI classification only if necessary.

---

# 33. Retry Limits

The orchestrator must prevent infinite agent loops.

Example:

```yaml
limits:
  planning_turns: 20

  implementation_attempts: 3

  lightweight_verification_failures: 2

  standard_failures: 2

  review_fix_cycles: 2
```

If exceeded:

```text
BLOCKED
```

The user receives:

- current state;
- failures;
- attempts performed;
- recommended next action.

---

# 34. Code Review Contract

Review output should be structured.

Example:

```json
{
  "status": "CHANGES_REQUIRED",
  "findings": [
    {
      "severity": "HIGH",
      "category": "CONCURRENCY",
      "file": "ReferralService.java",
      "line": 87,
      "problem": "Concurrent redemption can consume the same invite twice.",
      "recommendation": "Enforce atomic redemption in the database transaction."
    }
  ]
}
```

Allowed severity:

```text
CRITICAL
HIGH
MEDIUM
LOW
```

Default handling:

```text
CRITICAL → must fix
HIGH     → must fix
MEDIUM   → configurable
LOW      → optional
```

---

# 35. Persistent Storage

Use global SQLite:

```text
~/.agentflow/agentflow.db
```

Suggested entities:

## projects

```text
id
name
repository_path
config_hash
created_at
last_used_at
```

## runs

```text
id
project_id
task
status
created_at
updated_at
```

## stages

```text
id
run_id
stage
status
attempt_count
started_at
completed_at
```

## agent_sessions

```text
id
run_id
stage
provider
model
cli_session_id
created_at
```

## decisions

```text
id
run_id
question
answer
created_at
```

## routing_decisions

```text
id
run_id
stage
task_profile_json
complexity_score
complexity_level
matched_rule
provider
model
reason
created_at
```

## verification_runs

```text
id
run_id
command
exit_code
stdout_path
stderr_path
started_at
completed_at
```

---

# 36. Run Artifacts

Human-readable run artifacts are stored per project:

```text
.ai-orchestrator/
├── routing.yaml
└── runs/
    └── RUN-001/
        ├── task.md
        ├── approved-plan.md
        ├── task-profile.json
        ├── routing-decision.json
        ├── implementation-summary.md
        ├── verification.json
        ├── review.md
        ├── documentation-summary.md
        └── final-summary.md
```

Recommended:

```gitignore
.ai-orchestrator/runs/
```

Commit:

```text
.ai-orchestrator/routing.yaml
```

Do not commit local execution logs by default.

---

# 37. CLI Commands

## Initialize project

```bash
agentflow init
```

Responsibilities:

- detect repository;
- detect common technologies;
- generate starter configuration;
- require user review before use.

---

## Start task

```bash
agentflow plan "Implement invitation expiry"
```

---

## Explicit project

```bash
agentflow plan \
  --project ~/projects/viteprep \
  "Implement invitation expiry"
```

---

## Resume

```bash
agentflow resume RUN-001
```

---

## Status

```bash
agentflow status
```

---

## Recent runs

```bash
agentflow runs
```

---

## Routing-only mode

```bash
agentflow route "Add ownership validation to exam endpoint"
```

This mode:

- analyzes;
- generates TaskProfile;
- computes routing;
- displays result;

but MUST NOT modify source code.

---

## Environment check

```bash
agentflow doctor
```

Expected checks:

```text
✓ Git
✓ Python runtime
✓ Claude CLI
✓ Codex CLI
✓ Gemini CLI
✓ CLI authentication
✓ project configuration
✓ Git repository
✓ writable AgentFlow storage
```

---

# 38. Suggested Source Layout

```text
ai-coding-orchestrator/
│
├── pyproject.toml
├── README.md
│
└── src/
    └── agentflow/
        │
        ├── cli.py
        ├── application.py
        │
        ├── config/
        │   ├── loader.py
        │   ├── global_config.py
        │   └── project_config.py
        │
        ├── project/
        │   ├── discovery.py
        │   └── context.py
        │
        ├── task/
        │   ├── analyzer.py
        │   ├── profile.py
        │   └── complexity.py
        │
        ├── routing/
        │   ├── engine.py
        │   ├── matcher.py
        │   ├── rules.py
        │   └── decision.py
        │
        ├── agents/
        │   ├── base.py
        │   ├── claude.py
        │   ├── codex.py
        │   └── gemini.py
        │
        ├── workflow/
        │   ├── engine.py
        │   ├── states.py
        │   ├── planning.py
        │   ├── implementation.py
        │   ├── verification.py
        │   ├── repair.py
        │   ├── review.py
        │   └── documentation.py
        │
        ├── process/
        │   ├── executor.py
        │   └── streaming.py
        │
        ├── git/
        │   ├── repository.py
        │   ├── worktree.py
        │   └── diff.py
        │
        ├── persistence/
        │   ├── database.py
        │   ├── models.py
        │   └── repositories.py
        │
        └── ui/
            ├── console.py
            ├── approval.py
            └── questions.py
```

Do not create unnecessary abstractions before their corresponding feature phase is implemented.

---

# 39. Security Requirements

## Process execution

Never interpolate task content directly into shell commands.

Use argument arrays.

---

## Credentials

AgentFlow MUST NOT read or persist:

- provider API keys unless explicitly configured for some future feature;
- OAuth tokens;
- CLI authentication cookies;
- browser sessions.

Provider CLIs remain credential owners.

---

## Repository protection

Implementation occurs in isolated worktrees.

AgentFlow must not:

- force-push, or push at all;
- merge without the user choosing `merge` at the final approval gate (or running
  `agentflow merge <run-id>`);
- delete branches without confirmation;
- modify the main working tree during an agent run.

An approved merge records the branch the primary checkout was on when the run's
worktree was created, commits the worktree's changes once (hooks run normally), and
cherry-picks that single commit onto the primary checkout. It proceeds only when the
checkout is on that branch with no tracked changes and no merge/rebase/cherry-pick in
progress; a conflicting pick is aborted so the checkout is left exactly as it was, and
the worktree is kept for `agentflow merge`. After a successful merge the worktree and
`agentflow/<run-id>` branch are removed. `agentflow cleanup` never removes a completed
run's worktree that still holds unmerged changes.

---

## Dangerous commands

V1 should block or require explicit approval for agent-requested commands such as:

```text
rm -rf
DROP DATABASE
DROP TABLE
git reset --hard
git clean -fd
git push --force
```

The exact mechanism depends on what command-control capabilities each provider CLI exposes.

---

# 40. Concurrency

V1 supports multiple simultaneous runs provided they operate in different worktrees.

Each run must have:

```text
unique run ID
unique worktree
unique branch
isolated stage state
```

The same branch/worktree must not have two active writer stages simultaneously.

Recommended locking:

```text
run-level lock
+
worktree-level lock
```

SQLite transactions can protect workflow-state updates.

---

# 41. Crash Recovery

Every stage transition must be persisted before the next stage begins.

Example:

```text
IMPLEMENTING
    ↓
process crash
```

After restart:

```bash
agentflow resume RUN-001
```

AgentFlow determines:

- current state;
- existing worktree;
- agent session ID where available;
- previous verification results;
- next legal transition.

External agent execution must be treated as potentially non-idempotent.

Before rerunning an implementation stage, inspect the current Git diff.

---

# 42. Observability

V1 does not require external observability infrastructure.

Store structured local events:

```json
{
  "timestamp": "...",
  "runId": "RUN-001",
  "stage": "IMPLEMENTATION",
  "event": "AGENT_STARTED",
  "provider": "openai",
  "model": "gpt-6-sol"
}
```

Useful metrics:

- tasks by model;
- first-pass verification rate;
- escalation count;
- implementation retry count;
- review findings by model;
- average stage duration;
- routing-rule frequency.

These metrics can later help tune `routing.yaml`.

---

# 43. Routing Benchmarking — FUTURE

Routing rules should initially be manually configured.

Later, collect outcome data such as:

```text
first-pass build success
first-pass test success
review severity
unnecessary file modifications
repair attempts
execution duration
```

This can inform project-specific routing changes.

The router itself should remain deterministic.

Do NOT introduce self-modifying routing in V1.

---

# 44. Versioning

Project configuration must contain:

```yaml
version: 1
```

AgentFlow must validate supported configuration versions.

If the orchestrator introduces incompatible routing changes later:

```text
version: 2
```

Migration should be explicit.

---

# 45. V1 Non-Goals

The following are explicitly out of scope:

- direct AI APIs;
- remote AgentFlow service;
- web dashboard;
- mobile UI;
- cloud-hosted orchestration;
- multi-user collaboration;
- Kubernetes;
- autonomous merging;
- autonomous production deployment;
- unrestricted shell execution;
- parallel writes by multiple agents;
- ML-based routing;
- self-modifying routing policy;
- provider credential management.

---

# 46. Implementation Phases

## Phase 1 — Foundation

Implement:

- Python package;
- Typer CLI;
- global installation;
- global config;
- project discovery;
- routing config loading;
- Pydantic validation;
- `agentflow doctor`;
- SQLite initialization;
- process execution abstraction.

No AI orchestration required yet.

---

## Phase 2 — AI CLI Adapters

Implement:

- ClaudeAdapter;
- CodexAdapter;
- GeminiAdapter;
- structured output parsing;
- session IDs;
- resume where available;
- timeout;
- process exit handling;
- logging.

---

## Phase 3 — Interactive Planning

Implement:

- Sonnet planner;
- repository inspection;
- user question loop;
- plan generation;
- approval flow;
- Opus escalation;
- TaskProfile generation.

---

## Phase 4 — Deterministic Routing

Implement:

- complexity calculation;
- hard-risk rules;
- first-match rules;
- routing precedence;
- routing result explanation;
- `agentflow route`;
- comprehensive unit tests.

This phase is considered critical infrastructure.

---

## Phase 5 — Worktree + Implementation

Implement:

- Git worktree creation;
- unique task branch;
- Luna invocation;
- Sol invocation;
- write locking;
- diff capture.

---

## Phase 6 — Verification + Repair

Implement:

- project verification commands;
- command-result persistence;
- simple failure classification;
- repair loop;
- Luna → Sol escalation;
- retry limits.

---

## Phase 7 — Review + Documentation

Implement:

- Gemini review;
- structured findings;
- repair loop;
- deep Sonnet review where required;
- optional Opus architecture review;
- Gemini documentation workflow.

---

## Phase 8 — Reliability Hardening

Implement:

- crash recovery;
- resume;
- run locks;
- concurrent-project support;
- cleanup;
- timeout handling;
- stale-worktree handling;
- config hashing;
- audit logs.

---

# 47. Acceptance Criteria

V1 is complete when all of the following are true.

## Multi-project

The globally installed AgentFlow can execute against at least two independent repositories without changing AgentFlow source code.

---

## Configuration

Each repository can control routing using only:

```text
.ai-orchestrator/routing.yaml
```

---

## Planning

Sonnet can interactively gather missing requirements and produce an approved plan.

Opus can be invoked for configured architecture escalations.

---

## Routing

Given a valid TaskProfile:

- model selection is deterministic;
- selection is reproducible;
- the matched rule is shown;
- the selection reason is shown.

---

## Implementation

AgentFlow can invoke:

- Luna;
- Sol;

through Codex CLI based on routing.

---

## Verification

Actual project verification commands determine success/failure.

LLM claims do not override failed commands.

---

## Review

Gemini can independently review the implementation and produce structured findings.

Critical/high findings can be routed through the repair cycle.

---

## Git safety

All code modifications occur in isolated Git worktrees.

The developer's original working tree remains unchanged.

---

## Recovery

An interrupted run can be resumed without recreating the task from scratch.

---

# 48. Final Architecture Decision

The system is intentionally separated into four concerns:

```text
┌──────────────────────────────┐
│ 1. Task Understanding        │
│ Claude Sonnet / Opus         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 2. TaskProfile               │
│ Provider-independent facts   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 3. Deterministic Router      │
│ routing.yaml                 │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 4. Execution                 │
│ Claude / Codex / Gemini CLI  │
└──────────────────────────────┘
```

The orchestrator implementation remains generic.

The project configuration determines how that generic engine behaves.

This separation is the central architectural requirement of AgentFlow V1.
