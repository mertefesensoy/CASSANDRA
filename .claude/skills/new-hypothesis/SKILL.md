---
name: new-hypothesis
description: Scaffold a new Cassandra hypothesis at docs/hypotheses/NNN-slug.md with the full evidence standard and falsifiable pass-or-fail line. Use when starting a new falsifiable research question for the tiny language lab. Auto-numbers from the existing hypotheses and applies the project prose style.
disable-model-invocation: true
---

Scaffold a new hypothesis document for the Cassandra tiny language lab. This skill
writes one file and does not run experiments (that is Codex's lane). The argument,
if given, is the topic or slug.

## Steps

1. Resolve the next number. Glob `docs/hypotheses/*.md`, take the highest leading
   three-digit number `NNN`, and use the next integer, zero-padded to three digits.
   Letter-suffixed variants (for example `009b`) count as their base number. If the
   new work is a variant of an existing hypothesis, ask the author whether to use a
   letter suffix instead of a new number.

2. Fix the topic and slug. Use the argument if present. Otherwise ask the author one
   concise question for the topic. The slug is short, lowercase, hyphenated.

3. Orient on ground truth before writing. Read the latest "## Stage" section of
   `experiments/tiny_language_lab/RESULTS.md`, the newest
   `experiments/tiny_language_lab/runs/*.md` summaries, the `README.md` "Next ladder"
   to get the current rung number, any cited `research/` Gemini notes, and the one or
   two most recent files in `docs/hypotheses/` as structural exemplars.

4. Write `docs/hypotheses/NNN-slug.md` using the canonical section template below.
   Fill every section with real content grounded in step 3. Do not invent metrics:
   any number is either measured with a citation to a run file, or written as "to be
   measured by Codex".

5. Apply the prose style. No em dashes and no en dashes anywhere; use the middle dot
   or restructure the sentence. (A PostToolUse hook also checks this.)

6. After writing, remind the author to: add a "Next ladder" rung pointer in
   `README.md` marked specced or awaiting Codex with no fabricated numbers; confirm
   the prior-art flag names what Gemini should compare; and update memory
   (`MEMORY.md` index line and the goal-loop memory file). Print the created path.

## Canonical section template

```
# Hypothesis NNN · <one-line falsifiable claim>

- Status: OPEN. Specced for Codex as Stage <N> (README ladder rung <R>).
- Date: <YYYY-MM-DD>
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: <R> (Codex stage number <N>)
- Builds on: <prior stages, ADRs, and Gemini notes, with paths>

## Why this, and why now
<the frontier this addresses; why it is the most valuable next question>

## The mechanism (or competing explanations)
<the falsifiable mechanism, or the two explanations the stage distinguishes>

## Hypothesis
<the precise claim, plus the result that would kill it, in both directions>

## The reference points
<the exact arms or configs, including the baseline it must beat>

## Primary decision metric and pass or fail line
<val NLL and/or copy-probe accuracy; the explicit CONFIRM and KILL thresholds>

## Risks and confounds
<seed variance, small probe sets, device caveat, fair-shot controls>

## What result would change the plan
<CONFIRM path, each KILL path, and the artifact each would produce>

## Handoff to Codex (implemented as Codex stage <N>, README ladder rung <R>)
<files to modify; the exact cassandra_compare.py command with seeds 7 11 19;
 the metric that decides; the restated pass-or-fail line>

## Prior-art flag for Gemini
<what known work this resembles and what Gemini should compare before any claim>

## Links
<RESULTS.md stage, runs/*.md, prior ADRs and hypotheses, Gemini notes, README>
```

Keep the artifact small enough for Codex to finish on CPU, and keep it falsifiable:
the pass-or-fail line is mandatory.
