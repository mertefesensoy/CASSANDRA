#!/usr/bin/env python3
"""PostToolUse hook: flag em dash (U+2014) and en dash (U+2013) in edited Markdown.

Cassandra prose style (CLAUDE.md) forbids em and en dashes; the convention is to
use the middle dot (U+00B7) or restructure the sentence. This hook reads the
PostToolUse JSON on stdin, and when an edited ``*.md`` file contains a forbidden
dash it returns ``additionalContext`` so the model fixes it, plus a
``systemMessage`` so the user sees the warning.

It fails open: any error, missing field, or unreadable file results in exit 0, so
the style check can never block or break an edit. AGENT.md is exempt as a known
legacy outlier (see docs/implementations/2026-06-24-claude-code-automation-setup.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EM_DASH = "—"  # em dash
EN_DASH = "–"  # en dash
MIDDLE_DOT = "·"  # middle dot, the project-approved substitute
EXEMPT_BASENAMES = {"AGENT.md"}
MAX_REPORTED = 12


def main() -> int:
    data = json.loads(sys.stdin.read())
    tool_input = data.get("tool_input") or {}
    file_path = tool_input.get("file_path")
    if not file_path:
        return 0

    path = Path(file_path)
    if path.suffix.lower() != ".md":
        return 0
    if path.name in EXEMPT_BASENAMES:
        return 0
    # Only enforce the project prose style inside the project tree. Files outside
    # it (for example private memory notes under ~/.claude) are not project docs.
    # The project root is this script's grandparent: <project>/.claude/hooks/.
    project_dir = Path(__file__).resolve().parents[2]
    try:
        path.resolve().relative_to(project_dir)
    except ValueError:
        return 0
    if not path.is_file():
        return 0

    text = path.read_text(encoding="utf-8", errors="replace")
    hits = [
        lineno
        for lineno, line in enumerate(text.splitlines(), start=1)
        if EM_DASH in line or EN_DASH in line
    ]
    if not hits:
        return 0

    shown = hits[:MAX_REPORTED]
    extra = len(hits) - len(shown)
    lines_desc = ", ".join(f"line {n}" for n in shown)
    if extra > 0:
        lines_desc += f", and {extra} more"

    context = (
        f"Prose-style check (CLAUDE.md): the Markdown file you just edited, "
        f"{path.name}, contains forbidden em ('{EM_DASH}') or en ('{EN_DASH}') "
        f"dash characters at {lines_desc}. Replace each one with the middle dot "
        f"'{MIDDLE_DOT}' or restructure the sentence, then continue. This is the "
        f"project's documented prose style; do not introduce these dashes."
    )
    out = {
        "systemMessage": (
            f"Style: {len(hits)} em/en dash(es) in {path.name} "
            f"({lines_desc}). Use '{MIDDLE_DOT}' or restructure per CLAUDE.md."
        ),
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": context,
        },
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Fail open: never block or break an edit because of the style check.
        sys.exit(0)
