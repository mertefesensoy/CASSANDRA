#!/usr/bin/env python3
"""Stop hook: remind to record lab runs into the durable RESULTS.md log.

In this project ``experiments/tiny_language_lab/runs/`` is gitignored and is a
local artifact; the durable, hand-written record of every stage is RESULTS.md
(CLAUDE.md). This hook fires when the agent stops and, if any ``runs/*.md`` or
``runs/*.jsonl`` is newer than RESULTS.md, surfaces a non-blocking
``systemMessage`` so an unrecorded stage is not forgotten.

It never blocks stopping (no exit 2, no ``decision: block``), guards against
re-trigger loops via ``stop_hook_active``, and fails open on any error.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_REL = Path("experiments/tiny_language_lab")
RUNS_REL = LAB_REL / "runs"
RESULTS_REL = LAB_REL / "RESULTS.md"
MAX_REPORTED = 5


def main() -> int:
    data = json.loads(sys.stdin.read())
    # Do not re-fire while already continuing from a stop hook.
    if data.get("stop_hook_active"):
        return 0
    cwd = data.get("cwd")
    if not cwd:
        return 0

    root = Path(cwd)
    runs_dir = root / RUNS_REL
    results = root / RESULTS_REL
    if not runs_dir.is_dir() or not results.is_file():
        return 0

    results_mtime = results.stat().st_mtime
    newer = []
    for pattern in ("*.md", "*.jsonl"):
        for f in runs_dir.glob(pattern):
            try:
                if f.stat().st_mtime > results_mtime:
                    newer.append(f)
            except OSError:
                continue
    if not newer:
        return 0

    newer.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    shown = [p.name for p in newer[:MAX_REPORTED]]
    extra = len(newer) - len(shown)
    listing = ", ".join(shown)
    if extra > 0:
        listing += f", and {extra} more"

    msg = (
        f"Provenance reminder: {len(newer)} run artifact(s) in "
        f"{RUNS_REL.as_posix()} are newer than RESULTS.md ({listing}). "
        f"runs/ is gitignored, so record the stage in RESULTS.md to keep the "
        f"durable evidence log current (CLAUDE.md)."
    )
    print(json.dumps({"systemMessage": msg}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Fail open: never trap a stop because of the reminder.
        sys.exit(0)
