# 2026-06-24 · Claude Code automation setup

## Problem / Motivation

Cassandra enforces several conventions by discipline alone: a documented prose
style (no em or en dashes, use the middle dot), a per-stage evidence standard, a
falsifiability requirement for every hypothesis and ADR, determinism (seeds
`7 11 19`), and a durable hand-written `RESULTS.md` while `runs/` is gitignored.
The repository had no `.claude/` configuration, so none of these rules were backed
by tooling; each depended on the model or author remembering. This change adds the
first project-level Claude Code automations, targeting the rules that are easy to
violate silently and expensive to get wrong (a lost stage record, a style
regression, a non-falsifiable draft).

## What Changed

| File | Description |
| --- | --- |
| `.claude/settings.json` | Registers the two hooks (PostToolUse style lint, Stop provenance reminder) using the exec command form. |
| `.claude/hooks/check_md_dashes.py` | PostToolUse hook: flags em or en dashes in an edited Markdown file, nudges a fix, fails open. |
| `.claude/hooks/results_provenance.py` | Stop hook: non-blocking reminder when `runs/` artifacts are newer than `RESULTS.md`. |
| `.claude/agents/hypothesis-auditor.md` | Read-only subagent auditing a hypothesis or ADR draft against the evidence standard and falsifiability. |
| `.claude/agents/stage-evidence-verifier.md` | Read-only subagent verifying a `RESULTS.md` stage or `runs/*.md` summary against the per-stage standard. |
| `.claude/skills/new-hypothesis/SKILL.md` | User-invocable skill scaffolding `docs/hypotheses/NNN-slug.md` with the full template. |
| `.claude/skills/new-adr/SKILL.md` | User-invocable skill scaffolding `docs/decisions/NNNN-slug.md` with the reopen condition. |

## Implementation Approach

**Hooks.** Configured in `settings.json` under `hooks.PostToolUse` (matcher
`Write|Edit|MultiEdit`) and `hooks.Stop` (no matcher). Both use the exec form,
`command: "python"` with `args: ["${CLAUDE_PROJECT_DIR}/.claude/hooks/<script>.py"]`,
so the script path is passed as an argv element and never traverses shell quoting.
Each script reads the event JSON from stdin, runs its logic in Python, and prints a
JSON object on stdout with exit code 0.

- `check_md_dashes.py` reads `tool_input.file_path`. If it is a `.md` file inside the
  project tree (with `AGENT.md` exempt) that contains U+2014 or U+2013, it returns
  `hookSpecificOutput.additionalContext` so the model self-corrects, plus a
  `systemMessage` for user visibility, naming the offending lines. The whole body is
  wrapped so any error exits 0.
- `results_provenance.py` reads `cwd`. If any `runs/*.md` or `runs/*.jsonl` is newer
  than `RESULTS.md`, it returns a `systemMessage` listing the artifacts. It checks
  `stop_hook_active` to avoid re-firing, and it never blocks the stop.

**Subagents.** Markdown with YAML frontmatter (`name`, `description`, `tools: Read,
Grep, Glob`), read-only by tool restriction. Each encodes a numbered checklist drawn
from `CLAUDE.md` and the goal-loop guardrails, and a fixed output format (verdict,
a PASS/FAIL table, a line-referenced fix or gap list).

**Skills.** `SKILL.md` with frontmatter (`name`, `description`,
`disable-model-invocation: true`), making them user-only because they write numbered
files. Each encodes the numbering rule, an orient step that reads current ground
truth, the canonical section template, and the prose-style plus README and memory
reminders.

## Mathematical / Statistical Details

None. This change is structural configuration and tooling; it introduces no formula,
statistical test, or numeric algorithm. The provenance hook uses a plain file
modification-time comparison (newest `runs/` artifact versus `RESULTS.md`), described
above.

## Design Decisions

- **Exec form over shell form.** The project path contains a non-ASCII segment
  (`Masaüstü`) and sits under OneDrive. The exec form passes the script path as an
  argv element, avoiding shell quoting and encoding pitfalls, and is the documented
  preferred form for path placeholders.
- **Python scripts over inline shell.** Portable across the PowerShell-primary,
  Git-Bash-available Windows setup; Python is confirmed on PATH (torch
  `2.12.1+cu126`). The logic is testable in isolation, which it was.
- **Fail open, non-blocking.** Both hooks exit 0 on any error, and neither blocks an
  edit or traps a stop. The dash hook uses `additionalContext` (advisory) rather than
  exit 2 (blocking error); the Stop hook uses only `systemMessage`. A style or
  provenance check must never obstruct work.
- **`AGENT.md` exempt from the dash lint.** It carries 119 legacy em or en dashes and
  is a prose artifact; linting it would emit noise on every edit. The actively edited
  docs already follow the rule, so the hook protects against regressions rather than
  policing legacy text. A one-time `AGENT.md` cleanup is a separate, optional task.
- **Project-scoped enforcement.** The dash lint applies only to files under the
  project root, derived from the script's own location at `<project>/.claude/hooks/`.
  Files outside the repository, in particular the private memory notes under
  `~/.claude/`, are not project docs and are skipped. This guard was added after the
  hook correctly flagged pre-existing dashes in a memory file on first real use: the
  right fix was to match the rule's documented scope, project docs, rather than to
  reformat out-of-scope notes.
- **Provenance heuristic uses mtime, not git.** `runs/` is gitignored, so `git
  status` cannot see it. Comparing the newest `runs/` artifact mtime to `RESULTS.md`
  is stateless and needs no session diff. False positives are harmless because the
  message is non-blocking.
- **Subagents are read-only.** Auditing must not mutate the artifact under review, so
  tools are restricted to Read, Grep, Glob.
- **Skills are user-only.** They create numbered files, a side effect; the author
  decides when to invoke `/new-hypothesis` or `/new-adr`, rather than the model
  spawning them autonomously.
- **Alternatives considered and deferred.** A PreToolUse determinism guard (warn when
  `cassandra_compare.py` runs without `--seeds 7 11 19` or on a non-CUDA device) was
  considered but deferred as potentially noisy against legitimate smoke tests; it can
  be added later. MCP servers (context7) and plugins were out of scope for this
  request, which was hooks, subagents, and skills.

## Verification

- `settings.json` parses as valid JSON.
- `check_md_dashes.py`, tested by subprocess against the real non-ASCII path with five
  inputs: a dirty `.md` with an em and an en dash returns `additionalContext` plus
  `systemMessage` naming both lines (exit 0); a clean `.md` produces no output; a
  non-`.md` path produces no output; `AGENT.md` produces no output (exempt); a missing
  `tool_input` produces no output (exit 0). The later project-scope guard was verified
  too: an in-project `.md` with a dash flags, while an out-of-project `.md` and the
  real `~/.claude` memory file are skipped.
- `results_provenance.py`, tested with `stop_hook_active: true` (silent), missing
  `cwd` (silent), the real project `cwd` (silent, because `RESULTS.md` is current),
  and a synthetic tree where `runs/` is newer than `RESULTS.md` (emits the reminder).
  All exit 0.
- Grep of `.claude/` for em or en dashes returns only the intentional literals in
  `check_md_dashes.py`; no Markdown file added here contains them.
- Live behavior is confirmed on first use: a fresh Claude Code session loads
  `.claude/settings.json` and fires the hooks on the next Edit, Write, and Stop. The
  subagents are available as `hypothesis-auditor` and `stage-evidence-verifier`; the
  skills as `/new-hypothesis` and `/new-adr`. Claude Code may ask the user to review
  or trust project hooks before they run.

## Related Docs

- Conventions enforced: `CLAUDE.md` (prose style, evidence standard, determinism,
  `RESULTS.md` as the durable record); the goal-loop guardrails in
  `prompts/claude-goal-loop.md`.
- Artifacts the skills scaffold: `docs/hypotheses/`, `docs/decisions/`.
- Hooks reference used: code.claude.com/docs/en/hooks.
