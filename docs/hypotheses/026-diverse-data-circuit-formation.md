# Hypothesis 026 · The in-context copy circuit is formed by diverse broad data, not by steps alone: TinyStories-only checkpoints stay circuit-absent while broad-bearing lineages cross the presence line, and the surviving Stage 58/59 checkpoint ladders decide it without training a single new model

- Status: OPEN. Specced for Codex as Stage 60 (README ladder rung 71).
  Audited 2026-07-22 (hypothesis-auditor pass; all twelve required fixes
  applied: exhaustive precedence-ordered partition with E-steps first,
  quantified E-diverse conjunct, corrected inventory incl. seed 19's
  every-1,000 grids, corrected Stage 56 vocab mechanism, named wrapper
  with hash rules, nearest-checkpoint broad-step matching, determinism
  re-probe). Eval-only by user directive (2026-07-22): this stage trains
  nothing; any follow-up intervention needs its own pre-registered line
  and a read-back before launch.
- Date: 2026-07-22
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 71 (Codex stage number 60)
- Builds on: the fired ADR 0016 clause-3 reopen
  (`runs/stage59_cold_letters_probe.json`: Stage 58 COLD 85M broad-only
  scored `0.194336` constrained-choice accuracy, chance `0.0625`, line
  `0.1625`, choice MRR `0.401049`, raw accuracy `0.126953`), the Phase 5
  flagship probe row (`runs/phase5_behavior_letters_probe.json`: 201.6M
  TinyStories-only scored `0.060547`, MRR `0.217769`, on the IDENTICAL
  1,024-case probe file), ADR 0008 (rank-2 closure this does not touch),
  Gemini notes
  `research/theme_2_in_context_learning_and_rag/11_induction_heads_and_zero_shot_failure.md`
  and `12_forgetting_of_structural_circuits.md` (note 12's mechanism
  survey is used; its Cassandra evidence paragraph is corrected in the
  note's addendum and in ADR 0018: no lineage ever went simple-then-broad
  into those two probe rows), and the surviving checkpoint inventory
  under `C:\cassandra_runs` (68 Stage 58 checkpoints verified present
  2026-07-22, including the full seed-7 three-arm every-5,000 ladders,
  the CURRICULUM phase-1 TinyStories-only ladders on all three seeds,
  seed 19's every-1,000 recovery-cadence grids, the Stage 59 `w = 0.10`
  ladders, and four Stage 56 Recipe v1 finals).

## Why this, and why now

Part 0 produced the phase's most surprising verified fact: a 2.4x smaller
model trained only on broad text copies in-context at three times chance,
while the narrow-trained flagship sits at chance on the identical probe.
Two stories fit that pair. E-diverse: diverse data is what forms the
circuit (Olsson et al.'s induction-head account; TinyStories' repetitive
register never pressures precise in-context routing). E-steps: any data
forms it eventually, and the flagship row is explained by something else
(scale, schedule, corpus register), so the pair is a coincidence of
lineages. The two probe rows alone cannot separate these, because the two
models differ in size, data, AND steps simultaneously.

The lab already owns the experiment that separates them. The Stage 58/59
checkpoint ladders hold TinyStories-only checkpoints (CURRICULUM phase 1,
three seeds), broad-only ladders (COLD, three seeds), an interleaved
ladder (MIXTURE), low-dose ladders (Stage 59 `w = 0.10`, two seeds), and
a hard domain-shift lineage (CURRICULUM phase 2 continues phase 1) at one
fixed 85.11M architecture. Probing every surviving checkpoint with the
same frozen probe maps circuit presence against data source, dose, and
steps, seed-paired, for about fifteen GPU-seconds per checkpoint.

## Design (eval-only)

One command shape, iterated over the checkpoint inventory:

```powershell
python .\experiments\tiny_language_lab\eval_letters_copy_probe.py `
  --checkpoint <path.pt> --device cuda `
  --out .\experiments\tiny_language_lab\runs\stage60_probe_<name>.json `
  --summary .\experiments\tiny_language_lab\runs\stage60_probe_<name>.md
```

All probe flags stay at the Part 0 defaults (`--lines 1024 --seed
20260709 --max-cases 1024`, probe file
`corpus/letters_copy_probe_seed.txt`). The probe tool computes no hashes
and no consolidated output, so one new committed wrapper,
`experiments/tiny_language_lab/make_stage60_circuit_matrix.py`, drives
the sweep: it computes each checkpoint's SHA-256 and the probe file's
SHA-256 per row, invokes the probe, and appends one row to
`runs/stage60_circuit_matrix.jsonl` plus a `.md` table (checkpoint,
lineage, seed, global step, broad steps seen, TinyStories steps seen,
choice accuracy, choice MRR, raw accuracy, NLL, hash fields). Hash rule:
where a Stage 58/59 record already names the checkpoint's SHA-256, the
wrapper verifies it and excludes-and-flags a mismatch; where no recorded
hash exists (most intermediate ladder checkpoints), the wrapper records
the fresh hash and marks the row `hash_unverified` rather than excluding
it.

Probe set, in priority order:

1. CURRICULUM phase-1 ladders, all seeds (TinyStories-only: seed 7 at
   5,000 / 10,000 / 12,500; seed 11 at 5,000 and the 5,952 final; seed 19
   at EVERY 1,000 steps from 1,000 through 5,000 plus the 5,952 final,
   thanks to its recovery-run checkpoint cadence). The E-steps killer
   rows, and seed 19 gives the finest narrow-only grid.
2. COLD ladders, all seeds (broad-only formation curve; seed 7 already
   has its final probed at `0.194336`).
3. CURRICULUM phase-2 ladders, all seeds (the domain-shift lineage:
   whatever phase 1 formed, does broad training build, keep, or destroy
   it? Seed 19 again carries an every-1,000 grid from 6,000 to 20,000).
4. MIXTURE seed-7 ladder (complete every-5,000 grid, 8 intermediates
   plus final) and the Stage 59 `w = 0.10` seed-11/19 ladders (both
   complete on disk): formation under interleaving and low dose.
5. Stage 56 Recipe v1 finals (four fp32 checkpoints verified under
   `C:\cassandra_runs\stage56_broadchar_checkpoints`): the
   recipe-dependence read. Known constraint: their 27-character
   text8-native vocabulary contains every probe letter but lacks the
   probe text's newline, comma, and period, so `validate_codec` in the
   probe tool is expected to raise. Attempt the row, and if it raises,
   record the incompatibility as the row's outcome; adapting the probe
   text is out of scope for this stage.

## Pre-registered classification and decision lines

Per row: PRESENT = choice accuracy at or above `0.1625` (the registered
reopen line, chance plus `0.10`); ABSENT = at or below `0.1125` (chance
plus `0.05`); GRAY = between. Binomial sd at chance with 1,024 cases is
about `0.0076`, so the ABSENT ceiling sits about 6.6 sd above chance and
the two lines are about 6.6 sd apart: the gray band is WIDE relative to
noise, which is exactly why a single-row fluke cannot jump both lines.

Primary verdict, evaluated in this precedence order, read across ALL
probed seeds (seed 7 carries the complete three-arm ladders; seeds 11 and
19 contribute every row their surviving checkpoints allow):

1. **E-steps (KILL, checked first)** = any TinyStories-only row on ANY
   seed reads PRESENT. Narrow data alone can form the circuit and the
   Part 0 contrast was a coincidence of lineages; the diverse-data claim
   dies. This outranks every other rule, including the replica rule.
2. **E-diverse (CONFIRM)** = every TinyStories-only row on every probed
   seed is ABSENT, AND the seed-7 COLD ladder reaches PRESENT at some
   step, AND the seed-7 phase-2 final sits at least `0.05` choice
   accuracy above the seed-7 phase-1 final row (or reads PRESENT
   outright). Diverse broad data is the formation ingredient; the narrow
   register forms nothing.
3. **E-gray (INCONCLUSIVE, the explicit catch-all)** = everything else:
   a GRAY TinyStories-only row, a COLD ladder that never reads PRESENT
   despite the final's known `0.194336` (probe instability, not
   physics), a phase-2 lineage that stays flat within `0.05` of its
   phase-1 final, or any other combination the first two branches do not
   claim. Report, no strong claim, and the follow-up design (if any)
   goes back through intake.

Replica rule (subordinate to the partition above): an ABSENT seed-11 or
seed-19 COLD final while seed 7 reads E-diverse downgrades CONFIRM to
INCONCLUSIVE; matching signs are recorded as robustness. (A PRESENT
TinyStories-only replica row is not a downgrade; it is E-steps by
branch 1.)

Harness determinism check, pre-registered: the already-probed COLD
seed-7 final (`0.194336`, `runs/stage59_cold_letters_probe.json`) is
re-probed once through the Stage 60 wrapper; exact reproduction is
required before any other row is trusted.

Screening reads, descriptive, pre-registered, firing no verdict:

- **Formation onset**: first global step at which each lineage crosses
  `0.1625`, per seed; MIXTURE and `w = 0.10` onsets compared to COLD at
  matched TOTAL steps, and CURRICULUM phase 2 compared to COLD at matched
  BROAD steps (curriculum broad steps = global step minus its phase-1
  step count). The broad-step grids do not align exactly (seed-7 phase-2
  broad steps land at 2,500 / 7,500 / ... against COLD's 5,000 grid), so
  each comparison uses the nearest surviving COLD checkpoint and
  discloses the step difference in the matrix table.
- **Destruction screen** (the corrected version of Gemini note 12's
  story): any within-lineage drop of at least `0.05` choice accuracy
  between consecutively probed checkpoints is flagged as a candidate
  destruction event. A flagged event justifies proposing a follow-up
  intervention; it does not authorize one (eval-only directive).

## Expected signal, baseline, and honesty constraints

- Anchors already measured: COLD 85M final `0.194336`; flagship 201.6M
  TinyStories-only `0.060547`; both on the frozen probe this stage
  reuses.
- Expected under E-diverse: phase-1 rows near `0.06`, COLD ladder rising
  through GRAY to PRESENT somewhere in the tens of thousands of steps,
  phase-2 climbing from its phase-1 value after the shift.
- Expected under E-steps: some phase-1 row at or above `0.1625` despite
  the flagship's chance-level row (size or schedule would then be the
  live variable, not data).
- The probe is one task family (identity copy over a 16-letter payload).
  A verdict here is about THIS circuit's formation, not reasoning in
  general; the model card language stays scoped.
- Every row records the checkpoint SHA-256 and the probe SHA-256; a row
  whose checkpoint hash does not match its Stage 58/59 record is
  excluded and flagged rather than silently probed.
- Failed loads, missing checkpoints, and vocabulary incompatibilities are
  recorded as rows with a reason, never dropped.

## Risks

- **Grid resolution.** Every Stage 58/59 arm's every-5,000 grid survives
  complete (and seed 19 adds every-1,000 grids), but onset estimates are
  still quantized to the surviving grid; a first-crossing step means
  "first surviving checkpoint at or past the crossing" and says so.
- **One probe family.** All conclusions are per-probe-family; a circuit
  invisible to the letters probe is out of scope by construction.
- **Gray-band pileup.** If most rows land GRAY, the stage resolves
  E-gray and the honest cost is one GPU-hour, not a wrong claim.
- **Cross-size transfer.** The flagship row is 201.6M and every ladder
  row is 85.11M; size is disclosed as uncontrolled in the E-steps read,
  which is why the KILL line keys on the 85M TinyStories-only rows, not
  on the flagship row.

## What result would change the plan

- **E-diverse** grounds Stage 61's instrumentation (the pure-broad 200M
  flagship's per-checkpoint probes become a formation curve at scale,
  ADR 0018) and makes "diverse data forms structure cheaply" a
  publishable mechanism claim for the release narrative.
- **E-steps** kills the diverse-data story before it reaches any model
  card, and the behavior axis pivots to what DOES separate the two
  anchor rows (size or schedule), through intake.
- A flagged destruction event revives the corrected note-12 question as
  a candidate follow-up hypothesis (its own sign-off, per the eval-only
  directive).
- E-gray sends the axis back to intake with the matrix as evidence.

## Links

- `docs/decisions/0018-phase-6-redesign-circuit-mapping-and-instrumented-flagship.md` (the authorizing redesign ADR)
- `docs/decisions/0016-phase-5-closeout-developmental-null-recipe-frontier.md` (clause 3, fired)
- `runs/stage59_cold_letters_probe.json` · `runs/phase5_behavior_letters_probe.json` (the two anchor rows)
- `runs/stage59_verdict.json` (the E-partial context this redesign follows)
- `research/theme_2_in_context_learning_and_rag/11_induction_heads_and_zero_shot_failure.md` · `12_forgetting_of_structural_circuits.md` (with correction addendum)
- `experiments/tiny_language_lab/eval_letters_copy_probe.py` · `corpus/letters_copy_probe_seed.meta.json`
