---
name: memory-update
description: Record session outcomes in the persistent memory directory so the next session starts oriented. Use at the end of any session that resolved a verdict, changed a plan, or hit an operational incident.
---

# memory-update · keep future sessions oriented, never misled

When to use: the session produced something a future session must know
that the repo itself does not record (verdict consequences, operational
gotchas, user directives, pending gates). The memory lives at
`~/.claude/projects/<project>/memory/`: one fact per file with
frontmatter, plus a one-line pointer per file in `MEMORY.md`.

## Exact steps, in order

1. Decide what is memory-worthy: user directives and fork answers,
   verdicts with their consequences, operational incidents and their
   fixes, pending gates with owners and deadlines. NOT memory-worthy:
   anything the repo already records (code, RESULTS.md content, git
   history) or session-local details.
2. Check for an existing memory file that covers the topic; UPDATE it
   rather than creating a near-duplicate. Delete or correct memories that
   turned out wrong.
3. Write or edit the memory file: frontmatter (`name`, one-line
   `description`, `metadata.type` of user, feedback, project, or
   reference), then the fact. For feedback and project types include a
   `**Why:**` and `**How to apply:**`. Convert every relative date to an
   absolute one. Link related memories with `[[name]]`.
4. Add or refresh the one-line pointer in `MEMORY.md`
   (`- [Title](file.md) · hook`). Never put the memory's content in the
   index.
5. Sweep for stale statements: search the memory directory for claims the
   session just invalidated (a "pending" that resolved, a path that
   moved) and fix every occurrence, not just one.

## Full example of a good final output (real file: `memory/musahit-nightly-gpu-collision.md`)

```markdown
---
name: musahit-nightly-gpu-collision
description: MUSAHIT_Nightly Windows task (daily 02:00) runs an Ollama pipeline on the GPU; how to skip one night without admin rights
metadata:
  type: project
---

`MUSAHIT_Nightly` is a Windows scheduled task (root path `\`, daily 02:00,
runs as an ACL-protected task that plain-user `Disable-ScheduledTask` cannot
touch). It executes
`C:\Users\senso\OneDrive\Masaüstü\MÜŞAHİT\scripts\scheduling\run_nightly.ps1`,
which starts Ollama if needed and runs `python -m musahit.pipeline run`.
Ollama loads a local LLM onto the RTX 4070 (8 GB), so it collides with any
overnight CASSANDRA training run.

**Why:** overnight Stage-55-class runs and the 02:00 MUSAHIT firing contend
for VRAM; Mert asked for one-night suppression on 2026-07-06.

**How to apply:** `run_nightly.ps1` has a one-shot skip guard (added
2026-07-06): create
`C:\Users\senso\OneDrive\Masaüstü\MÜŞAHİT\scripts\scheduling\SKIP_NEXT_RUN.flag`
(any content) and the next firing exits cleanly, deleting the flag; every
later night runs normally. To cancel a pending skip, delete the flag. For a
permanent disable, the task needs an elevated
`schtasks /Change /TN MUSAHIT_Nightly /DISABLE`. When scheduling overnight
CASSANDRA experiments, either plan them to finish before 02:00 or drop the
flag. Verify the guard still exists in the script before relying on it. See
[[phase4-stage55-flagship-midrun]].
```

## Mistakes to avoid (each one actually happened here)

- **Stale memory misleads.** "Stage 51 sample review STILL pending"
  survived in THREE places (a memory file, the index line, an ADR
  status) after the review passed; all three needed the same-day fix.
  Sweep every occurrence.
- **Do not bloat the index.** The `MEMORY.md` index once accumulated a
  multi-thousand-word single line; the index is one line per memory,
  content lives in the files.
- **Absolute dates only.** "Before the internship" rots; "before
  2026-07-20" does not.
- **Record the fix, not just the incident.** The OneDrive checkpoint
  crash memory is useful because it carries the recovery recipe and the
  new canonical paths, not merely "a crash happened".
- **Verify recalled facts before acting on them.** Memories are
  point-in-time; a remembered flag or path may have been renamed (the
  ladder-rung offset drifted from +1 to +8 across phases and the memory
  had to be corrected).
