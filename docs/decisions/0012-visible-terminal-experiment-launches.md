# ADR 0012 · Visible Terminal Experiment Launches

## Status

Accepted · permanent workflow rule

## Context

Mert wants Cassandra research runs to be visible while they execute. Hidden or
tool-only terminals make the work harder to supervise, especially for long
experiment matrices, CUDA smokes, and unattended goal-loop runs.

## Decision

Codex must launch every real experiment run, smoke test, comparison matrix, and
evidence-producing training command from a visible `powershell.exe` terminal.
The Codex tool terminal may still be used for file inspection, AST parsing,
artifact summarization, and other short non-experiment commands.

## Consequences

Future stage runs should be started with a visible PowerShell window, with logs
and JSONL artifacts still written under `experiments/tiny_language_lab/runs`.
This is a workflow requirement, not an experiment variable, and it should not be
silently relaxed in later phases.
