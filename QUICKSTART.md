# AgentFlow Quick Start

AgentFlow (CLI command `agentflow`) is a **local, reusable multi-agent coding orchestrator**. It
drives coding-agent CLIs you already have installed — `claude`, `codex`, `gemini` — through a
deterministic pipeline:

```
plan → route → implement (isolated git worktree) → verify/repair → review → document → human approval
```

AgentFlow never calls an AI provider's API directly and never handles your credentials. It only
shells out to the CLI binaries you've separately installed and logged into
(`asyncio.create_subprocess_exec` with an argument array — never a shell string), and it never
auto-pushes, auto-merges, or auto-deploys. Every run stops for your approval before anything leaves
the disposable worktree it worked in.

This guide uses a real project as the worked example throughout: a Next.js + Spring Boot +
FastAPI monorepo, referred to below by the relative path `..\skillify` — a sibling directory next
to `ai-coding-orchestrator` (adjust the relative path if your own project lives elsewhere).

> **Heads up:** this is a young tool. A few things below still work more narrowly than their name
> suggests — read [§9 Known Limitations](#9-known-limitations) before you rely on them in anger.

---

## 1. Prerequisites & install

You need, on the machine running AgentFlow:

- **Python ≥ 3.12** and **Git** (hard requirements — AgentFlow refuses to run without them).
- **[`uv`](https://docs.astral.sh/uv/)** to install AgentFlow itself.
- At least one coding-agent CLI, installed **and separately authenticated by you**:
  - [`claude`](https://docs.claude.com/en/docs/claude-code) (Anthropic)
  - `codex` (OpenAI)
  - `gemini` (Google) — or Google's newer [`agy`](https://antigravity.google/docs/cli/) CLI
    (Antigravity); see the `dialect` note in §2, since `agy`'s flag syntax is genuinely different
    from `gemini`'s and needs a config setting, not just a renamed command.

  AgentFlow doesn't do auth for you — if a CLI needs `claude login` / `codex login` / `gemini
  auth`/`agy auth`, run that yourself first.

### Install AgentFlow

There are two ways to install it. **If you just want to run AgentFlow against your own
projects** (the normal case, and what every other example in this guide assumes), use Option A.
Use Option B only if you're changing AgentFlow's own source code.

#### Option A: install as a global command (recommended)

Run this **once**, from *inside* the `ai-coding-orchestrator` repo you cloned/downloaded — `.`
below means "this directory":

```bash
cd ai-coding-orchestrator
uv tool install --editable .
```

`--editable` links the global command back to this repo's source, so a later `git pull` here
takes effect immediately, with no reinstall needed.

That's it — `agentflow` is now a command on your PATH, usable from **any directory, against any
project**. You do **not** need to `cd` back into `ai-coding-orchestrator` to use it. For example,
from inside your own project:

```bash
cd ..\skillify
agentflow --version
agentflow doctor
```

This is why every other `agentflow ...` example in this guide (including the ones using
`-C "..\skillify"` from elsewhere) never shows a `cd` into `ai-coding-orchestrator` first — they
all assume you installed it this way.

**If `agentflow: command not found` after installing:** `uv tool install` puts the command in
`~/.local/bin` (`%USERPROFILE%\.local\bin` on Windows), which isn't always on PATH yet. Run `uv
tool update-shell` and open a **new** terminal window, then try again.

#### Option B: local dev environment (only if you're modifying AgentFlow itself)

Run every command below from *inside* the `ai-coding-orchestrator` repo — this does **not**
create a global `agentflow` command, so every future invocation needs both `uv run` and this
directory as your working directory:

```bash
cd ai-coding-orchestrator
uv venv
uv pip install -e ".[dev]"
uv run agentflow --version
```

This also installs the test/lint tooling (`pytest`, `ruff`, `mypy`) that Option A skips. To point
this dev copy at another project without leaving `ai-coding-orchestrator`, use `-C`:

```bash
uv run agentflow -C "..\skillify" doctor
```

> **A note on the paths in the rest of this guide:** every example below uses
> `-C "..\skillify"`, which assumes you're sitting in `ai-coding-orchestrator` (or another sibling
> directory) when you run it. If you're already `cd`'d into your own project instead — e.g. right
> after Option A's last step above — just drop the `-C "..\skillify"` argument entirely;
> `agentflow` auto-discovers the project from your current directory, so plain `agentflow doctor`
> (no flag) does the same thing.

### Check your environment: `agentflow doctor`

```bash
agentflow doctor -C "..\skillify"
```

`doctor` checks, live, every time you run it:

| Check | Critical? | What it actually does |
|---|---|---|
| Python runtime | Yes | `sys.version_info >= (3, 12)` |
| Git | Yes | `shutil.which("git")` + `git --version` |
| Claude / Codex / Gemini CLI | **No — warning only** | `shutil.which(<cmd>)` + `<cmd> --version` (5s timeout) |
| AgentFlow storage writable | Yes | `~/.agentflow/{database,worktrees,logs}` dirs |
| SQLite | Yes | opens `~/.agentflow/agentflow.db` |
| Project | Only if `--project` given | valid git repo; `.ai-orchestrator/routing.yaml` parses if present |

`doctor` exits non-zero only on a **critical** failure. A missing coding-agent CLI is reported as a
warning and does not fail the command. Each CLI check is also compared against the last time you
ran `doctor` (persisted in `~/.agentflow/agentflow.db`): if a CLI was available before and is now
missing, that row gets a distinct **regressed** marker (a red `!` instead of the usual yellow one)
and its details are prefixed `Previously available, now missing:` — so a CLI that quietly stopped
working mid-project stands out from an ordinary "never installed this one" warning the next time
you happen to run `doctor`.

---

## 2. Create a project-specific profile (`.ai-orchestrator/routing.yaml`)

### Quick way: `agentflow init`

```bash
agentflow init -C "..\skillify"
```

This walks the project (up to 3 directories deep, skipping `node_modules`/`.venv`/`dist`/`build`/
etc.) looking for `package.json`, `pyproject.toml`/`requirements.txt`, and `pom.xml`, and writes a
starter `.ai-orchestrator/routing.yaml` with a `verification:` group per match (guessed commands:
`npm test` / `pytest` / `mvn test`), AgentFlow's built-in default `models:`/`routing:`/`complexity:`
/`limits:` sections, and `documentation:` left **disabled** (there's no reliable way to guess which
docs a project wants kept in sync, so that section is a deliberate no-op until you configure it).
It refuses to overwrite an existing profile unless you pass `--force`.

**Detection is a heuristic starting point, not a guarantee** — it's marker-file matching, not real
project understanding, and the guessed `verification.commands` are frequently wrong (e.g. a root
`package.json` with no `test` script). Always read the generated file and fix the commands before
running real tasks against it. For example, running this against Skillify detects one group per
service directory (`front-end`, `web-service`, `data-processor`, `ai-engine`, …) plus, correctly,
one group per Maven submodule inside `web-service`'s multi-module layout — a good starting map of
the repo, but the exact `commands:` (e.g. `npm test` vs. Skillify's real `pnpm -C front-end test`)
still need a human pass.

If `.ai-orchestrator/routing.yaml` is missing entirely (you skip `init` and never hand-write one),
AgentFlow does **not** error or prompt — it silently falls back to its built-in default model
bindings and rules. So an absent profile isn't visible as an error; it just means the project
quietly gets the same generic defaults as every other unconfigured project.

### Or hand-write one

You can also write `.ai-orchestrator/routing.yaml` yourself — useful for understanding the schema,
or for correcting what `init` generated. Here's a complete, valid example for Skillify (adapted
from the design doc's own worked example, which already uses `viteprep` — Skillify's public brand
name — with `verification.commands` filled in using Skillify's real per-service test commands:
`pnpm test` for the Next.js front-end, `mvn test` for the two Spring Boot services, `pytest` for
the two FastAPI services):

1. In your target project's repo root, create the folder and file:
   ```bash
   mkdir "..\skillify\.ai-orchestrator"
   ```
2. Save the following as
   `..\skillify\.ai-orchestrator\routing.yaml`:

   ```yaml
   version: 1

   project:
     name: viteprep

   models:
     planner:
       default:
         provider: anthropic
         model: Claude Sonnet 5
       architecture:
         provider: anthropic
         model: Claude Opus 5

     implementation:
       lightweight:
         provider: openai
         model: GPT-5.6 Luna
       standard:
         provider: openai
         model: GPT-5.6 Terra
       escalation:
         provider: anthropic
         model: Claude Sonnet 5

     review:
       default:
         provider: google
         model: Gemini 3.8 Flash
       deep:
         provider: anthropic
         model: Claude Sonnet 5
       architecture:
         provider: anthropic
         model: Claude Opus 5

     documentation:
       default:
         provider: google
         model: Gemini 3.8 Flash

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
       default: planner.default
       architecture: planner.architecture

     implementation:
       force_standard_if_any:
         - authentication
         - authorization
         - data_ownership
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
         - data_ownership
       architecture_if:
         architecture_change: true
       default: review.default
       deep: review.deep
       architecture: review.architecture

     documentation:
       default: documentation.default

   verification:
     front-end:
       detect:
         - front-end/package.json
       commands:
         - "pnpm -C front-end test"
     web-service:
       detect:
         - web-service/pom.xml
       commands:
         - "mvn -f web-service test"
     cleanup-service:
       detect:
         - cleanup-service/pom.xml
       commands:
         - "mvn -f cleanup-service test"
     data-processor:
       detect:
         - data-processor/pyproject.toml
       commands:
         - "pytest data-processor/tests"
     ai-engine:
       detect:
         - ai-engine/pyproject.toml
       commands:
         - "pytest ai-engine/tests"

   limits:
     planning_turns: 20
     implementation_attempts: 3
     lightweight_verification_failures: 2
     standard_failures: 2
     review_fix_cycles: 2

   documentation:
     enabled: true
     candidate_files:
       - README.md
       - PROJECT.md
   ```
3. Validate it loaded: `agentflow doctor -C "..\skillify"` — the "Project" check parses this file
   and reports whether it's valid.

### What each section means

| Section | Purpose |
|---|---|
| `models` | Concrete provider+model for each named role (`planner.default`, `implementation.lightweight`, etc.) — everything else references these by alias. |
| `complexity` | Turns a `TaskProfile` (file count + risk flags the planner reports) into a low/medium/high complexity score. |
| `routing` | Per-stage precedence rules: which risk flags force escalation, which complexity maps to which implementation tier. First match wins. |
| `verification` | Named test groups: `detect` files that gate whether a group applies, `commands` to run — this is what "deterministic verify" actually executes after implementation. |
| `limits` | Bounded retry/escalation counts, so a stuck stage can't loop forever. |
| `documentation` | Which docs the documentation stage is allowed to touch. |

There is also an optional **global** machine config at `~/.agentflow/config.yaml` for CLI command
names (`cli.claude.command`, etc. — override if your binary isn't literally named
`claude`/`codex`/`gemini`) and the storage/worktrees/logs roots. AgentFlow never creates this file
for you — if it's absent, it just uses built-in defaults in memory; create it by hand only if you
need to override something (e.g. `$env:AGENTFLOW_CONFIG_PATH` also lets you point at a different
path).

**Using Google's Antigravity CLI (`agy`) instead of `gemini`:** don't just set
`cli.gemini.command: agy` — `agy`'s non-interactive flag syntax is genuinely different from
`gemini`'s (prompt via `-p`, resume via `--continue`/`--conversation`, no `--read-only`
equivalent), so it also needs `dialect: antigravity` to route through the matching adapter:
```yaml
# ~/.agentflow/config.yaml
version: 1
cli:
  gemini:
    command: agy
    dialect: antigravity
```
Once set, `doctor`'s Google-provider row is labeled `"Antigravity/Gemini CLI"` and checks for
`agy` instead of `gemini`. If `agy` isn't on PATH even though it's installed, check where its
installer actually put it (on Windows this may be `%LOCALAPPDATA%\antigravity\bin`, which the
installer doesn't always add to PATH) rather than assuming AgentFlow can't find a correctly-PATH'd
binary.

---

## 3. The command surface

| Command | Does | Key flags |
|---|---|---|
| `agentflow doctor` | Checks Python/Git/CLI presence + storage | `-C/--project` |
| `agentflow init` | Writes a starter `.ai-orchestrator/routing.yaml` | `-C/--project`, `--force` |
| `agentflow status` | Shows active runs for the current project | `-C/--project` |
| `agentflow run "<task>"` | **Planning only** — no code touched | `-C/--project` |
| `agentflow route "<task>"` | Plans, classifies, shows the routing decision — no code touched | `-C/--project`, `--override-provider`, `--override-model`, `--override-stage` |
| `agentflow implement "<task>"` | Plan → route → implement (isolated worktree) → verify/repair | `-C/--project`, `--override-provider`, `--override-model`, `--override-stage` |
| `agentflow complete "<task>"` | Full pipeline through review, docs, and **human approval** | `-C/--project`, `--override-provider`, `--override-model`, `--override-stage` |
| `agentflow resume <RUN-ID>` | Resume a run left mid-flight by a crash **or a BLOCKED stage** | `-C/--project`, `--override-provider`, `--override-model`, `--override-stage` |
| `agentflow runs` | Table of recent runs (status/stage/timestamps) | `--all/-a`, `--limit N` |
| `agentflow cleanup` | Remove worktrees/logs for finished runs, clear stale locks | `-C/--project` (default: all tracked projects) |
| `agentflow stats` | Local routing/verification metrics | `--all/-a` |

Note `--override-provider`/`--override-model`/`--override-stage` exist on
`route`/`implement`/`complete`/`resume` — **not** on `run` (which is planning-only and doesn't
route). `-C/--project` can go before or after the subcommand.

`complete` never auto-pushes, auto-merges, or deploys — it stops at a final human-approval prompt.

---

## 4. Plan a feature — sample prompt

```bash
agentflow -C "..\skillify" complete "Add a GET /api/v1/health endpoint to web-service that reports database connectivity status"
```

What happens:

1. **Planning** — Claude (Sonnet, escalating to Opus if the task trips an architecture-risk flag
   like `new_service`/`payment`) runs an interactive, multi-turn Q&A with you, bounded by
   `limits.planning_turns` (default 20). It produces an `approved-plan.md` and a machine-readable
   `task-profile.json` (risk flags like `authentication`, `schema_change`, etc.).
2. **Routing** — a pure, deterministic function of that `task-profile.json` and your `routing.yaml`
   picks the provider/model for implementation. No AI model chooses this.
3. **Implementation** — the routed CLI (e.g. Codex/Luna for a low-complexity change) implements the
   plan inside a disposable git worktree — your primary working tree is never touched.
4. **Verify/repair** — runs your `verification.commands`; failures trigger a bounded
   lightweight→standard→escalation repair loop.
5. **Review** — an independent, read-only pass (Gemini by default, escalating to Claude for
   security/auth/payment-flagged changes).
6. **Documentation** — syncs the files listed in `documentation.candidate_files` (skipped entirely,
   with no state transition, if `documentation.enabled` is false or the list is empty).
7. **Approval** — you're asked to approve before anything is considered done.

Try smaller, cheaper commands first while learning the tool:

```bash
# See the routing decision only — no code, no CLI cost
agentflow -C "..\skillify" route "Add ownership validation to the exam endpoint"

# Plan interactively without implementing yet
agentflow -C "..\skillify" run "Add a GET /api/v1/health endpoint to web-service"
```

---

## 5. See the output of every stage

There's no `agentflow logs`/`show` command. Two ways to inspect a run:

**Quick summaries:**
```bash
agentflow -C "..\skillify" status          # active runs for this project
agentflow -C "..\skillify" runs --limit 20 # recent runs, any status
agentflow -C "..\skillify" stats           # routing/verification metrics
```

**Full detail — read the artifact files directly**, under
`<repo>\.ai-orchestrator\runs\<RUN-ID>\`:

| File | Stage | Notes |
|---|---|---|
| `task.md` | Planning | Raw task description as given |
| `approved-plan.md` | Planning | The plan you approved |
| `task-profile.json` | Planning | Machine-readable risk/complexity facts |
| `routing-decision.json` | Routing | Which provider/model was picked, and why |
| `implementation-summary.md` | Implementation | Human-readable only |
| `verification.json` | Verify/repair | Human-readable / audit only |
| `review.md`, `review-findings.json` | Review | Human-readable / audit only |
| `documentation-summary.md` | Documentation | Human-readable only |
| `final-summary.md` | Approval | Final CLI output, human-readable |

**Important, verified in the source:** within a single live run, one stage does **not** hand off to
the next by re-reading these files — everything passes as in-memory Python objects. The files are
written for you, and for `resume`'s two purposes: reconstructing a crashed run's context, and (for
`verification.json`/`review-findings.json` specifically) printing an informational note when
resuming a `BLOCKED` run so a leftover artifact from the failed attempt doesn't sit there silently
unread. Only `approved-plan.md` + `task-profile.json` + `routing-decision.json` actually gate what
`resume` does next (whether to re-plan, re-implement, or just re-verify); `verification.json` and
`review-findings.json` are read back for that informational note only, never to skip a step — the
run always genuinely re-verifies and re-reviews on resume, regardless of what those files say.
`implementation-summary.md` and `documentation-summary.md` remain purely for you to read.
Documentation now has a real `DOCUMENTING` pipeline state like every other stage, so it does show
up uniformly in `status`/`runs` output when it actually runs.

---

## 6. Continue after a stuck stage

```bash
agentflow resume <RUN-ID>
```

`resume` covers two different kinds of interruption:

- **A process crash.** The `agentflow` process itself died mid-stage (killed, machine restarted,
  Ctrl+C) while the run's database row was left in a live, non-terminal state. `resume` reuses any
  worktree changes already made rather than redoing them, and reconstructs the plan/CLI session
  from persisted state where possible.
- **A `BLOCKED` run** — any coding-agent CLI failure (non-zero exit, the CLI binary vanished, or
  the CLI returned a usage-limit/quota error) marks the run `BLOCKED`. `resume` looks up the state
  the run was in immediately before it blocked (from the persisted state-transition history) and
  retries **that stage** — e.g. a run blocked mid-`IMPLEMENTING` gets a fresh implementation
  attempt; a run blocked mid-`REVIEWING` re-enters at verification and proceeds through review
  again. It prints a note if a leftover `verification.json`/`review-findings.json` from the failed
  attempt is sitting on disk, purely informational.

It still **refuses**:
- A run that's still `NEW` ("nothing happened yet — start a new run instead").
- A run in a genuinely terminal state: `COMPLETED`, `FAILED`, or `CANCELLED`.
- A `BLOCKED` run whose only recorded prior state is `NEW` (blocked before anything real
  happened) — nothing to retry, start a new run instead.

```bash
agentflow -C "..\skillify" runs   # find the RUN-ID, its Stage, and the blocker_reason
agentflow resume RUN-AB12CD34      # retries from that stage
```

If the same platform is still the problem (e.g. Claude's usage limit hasn't reset yet), pair this
with `--override-provider`/`--override-model`/`--override-stage` — see the next section.

---

## 7. Fall back to another coding platform when one hits its usage limit

The mechanism is `--override-provider`/`--override-model`/`--override-stage`, on
`route`/`implement`/`complete`/`resume`. `--override-provider` and `--override-model` are required
together; provider is one of `anthropic`/`openai`/`google` (case-insensitive). `--override-stage`
picks which single stage the override applies to: `implementation` (the default when you omit the
flag), `review`, or `documentation` — planning isn't overridable this way (its model escalation is
driven by the planner's own output, not the routing engine). **Every other stage keeps routing
normally** — an override never leaks into a stage you didn't name.

Continuing the §6 scenario — Claude's usage limit got hit mid-implementation, the run is `BLOCKED`
— resume it straight onto a fallback provider for that stage:

```bash
agentflow -C "..\skillify" resume RUN-AB12CD34 --override-provider openai --override-model "GPT-5.6 Terra" --override-stage implementation
```

If the limit was hit during review instead (e.g. a security-flagged change escalated review to
Claude), point the same flags at that stage:

```bash
agentflow resume RUN-AB12CD34 --override-provider google --override-model "Gemini 3.8 Flash" --override-stage review
```

Or, if you'd rather not resume this particular run at all, the flags work identically on a brand
new `implement`/`complete` call for the same task.

**There is still no automatic detection of "usage limit exhausted" and no automatic switch to a
fallback provider.** The only automatic escalation AgentFlow does on its own is the repair
workflow's lightweight → standard → escalation chain (`limits.*_failures`), and that triggers on
**verification failures**, never on a provider outage or quota error. You still have to notice the
failure yourself (`agentflow runs` / the `blocker_reason`) and choose the fallback — the override
flags are a manual switch, not a safety net that fires itself.

---

## 8. Cleaning up

```bash
agentflow -C "..\skillify" cleanup
```

Removes worktrees/logs for runs that ended `COMPLETED`/`BLOCKED`/`CANCELLED`/`FAILED`, and clears
stale writer locks left by a dead process. It never touches a worktree still guarded by a *live*
process's lock.

---

## 9. Known limitations

Read this before you plan a workflow around AgentFlow — these are gaps found by reading the actual
source, not guesses. A previous pass of this guide listed several more items here (no `init`,
`BLOCKED` runs being unresumable, the override silently applying only to implementation despite
being undocumented as such, doctor never caching CLI availability, and the documentation stage
lacking a real pipeline state) — all of those are now fixed, as described in §2/§6/§7 above. What's
still genuinely missing:

- **No automatic usage-limit/quota detection or fallback.** AgentFlow never inspects a CLI's exit
  code or stderr to guess "this looks like a rate limit" and switch providers on its own. You still
  have to notice the failure yourself (`agentflow runs` / the `blocker_reason`) and choose a
  fallback with `--override-provider`/`--override-model`/`--override-stage`. The only *automatic*
  escalation AgentFlow does is the repair workflow's lightweight → standard → escalation chain, and
  that only triggers on verification failures, never on a provider outage or quota error.
- **The per-stage override doesn't reach planning.** `--override-stage` accepts
  `implementation`/`review`/`documentation` only — planning's Sonnet→Opus escalation is driven by
  the planner's own structured output, not the routing engine, so there's no equivalent lever for
  it yet. `resume` still retries a planning-stage `BLOCKED` run (it reuses the same crash-recovery
  path as a mid-planning process crash), but that retry always goes back to Claude — if Claude's
  limit is what blocked it and hasn't reset yet, the retry will likely block again the same way.
- **Resuming a `BLOCKED` review/documentation failure re-runs verification and review too**, even
  if only the later stage actually failed — `resume` always re-verifies before re-entering review,
  by deliberate design (the same safety choice that applies to crash recovery). Usually cheap, but
  not a targeted "retry just this one stage in isolation."
- **`verification.json`/`review-findings.json` are read back only for an informational note on
  resume**, not to skip or validate a re-run against what they claim happened. `resume` always
  redoes verification/review for real, regardless of what a stale artifact says.

None of the above stops the tool from being useful today. Keep task descriptions handy so you can
paste them into a fresh run in the one case (a planning-stage block) where resume genuinely can't
help, and expect a resumed review/documentation retry to also re-verify along the way.
