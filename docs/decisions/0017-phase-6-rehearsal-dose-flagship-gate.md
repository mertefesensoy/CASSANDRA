# ADR 0017 · Phase 6 runs the rehearsal dose-response as its centerpiece, and a pre-registered gate decides whether Phase 6.1 scales it into a published dual-register flagship

- Status: Accepted (user-directed, 2026-07-21)
- Date: 2026-07-21
- Author: Claude (hypothesis, ADR, and roadmap role)
- Resolves: the Phase 6 direction fork (user answered in session 2026-07-21:
  run the probe plus H025; if it works, scale to a bigger flagship published
  on Hugging Face; if not, set back and iterate).
- Builds on: ADR 0016 (Phase 5 closeout and its reopen clauses), H025
  (`docs/hypotheses/025-rehearsal-dose-response-and-mixing-law.md`, audited,
  rung 69 = Codex Stage 59), ADR 0014 (evaluation conventions), ADR 0015 D2
  and `docs/open-source-release-and-disk-plan.md` (release machinery, updated
  2026-07-21: Apache-2.0 chosen, fresh-init public repo), ADR 0012 (visible
  launches), ADR 0011 (human review rule), and the Stage 55 flagship
  precedent (`RESULTS.md`, 201,609,249 parameters).

## Context

Phase 5 closed with two facts and one loose end. Fact one: acquisition order
is a practical null at fixed compute (H024 E-null, seed-robust). Fact two:
recipe engineering moved matched-budget broad performance by roughly 24 times
the largest ordering effect (Recipe v2, ADR 0016). The loose end: Stage 58's
MIXTURE arm measured exactly one point on the rehearsal dose-response curve,
and that point shows near-perfect narrow-register retention (`0.826285`
versus `3.556502` TinyStories bits/char) for a `+0.027977` broad-text toll.

The user's Phase 6 aim, stated 2026-07-21: if H025 validates a solid, cost
efficient way to build a model that keeps its grammar fluency (the
TinyStories register) while learning broad text, then Phase 6.1 scales that
recipe into a bigger flagship published on Hugging Face. If H025 does not
validate it, Phase 6 sets back and iterates instead of scaling. This ADR
turns that aim into pre-registered gates so the scale-up decision is read
from decision lines, not enthusiasm.

## Decision

1. **D1, the centerpiece.** Stage 59 executes H025 exactly as audited:
   Part 0 (letters probe on the Stage 58 COLD final, reopen line `0.1625`),
   Part 1 (3.2M proxy mixture sweep at `w` in `{0, 0.05, 0.10, 0.20, 0.30,
   0.50}` times seeds `7 11 19`, mixing-law fit, 85M prediction published
   BEFORE Part 2), Part 2 (two 85M MIXTURE `w = 0.10` arms at 20,000 steps,
   seeds 11 and 19, against the existing paired COLD baselines). The
   hypothesis doc is the authority on every decision line; if this ADR and
   H025 disagree, H025 wins and the disagreement is flagged.
2. **D2, the Phase 6.1 gate, pre-registered.** Stage 60 (rung 70, the
   flagship scale-up) launches IF AND ONLY IF H025 reads
   **E-cheap-rehearsal** under H025's own partition, including the
   majority-of-three read after the registered seed-7 escalation (a split
   resolves by majority of the three seeds, not by both original seeds
   flipping). "Unresolved INCONCLUSIVE" means the escalation was declined
   by the user, or the three-seed read still fails to produce a CONFIRM
   majority. Every other H025 outcome (E-costly-rehearsal, E-partial,
   unresolved INCONCLUSIVE) returns Phase 6 to intake carrying the measured
   dose-response curve; no flagship launches on a null, a graded, or an
   unresolved verdict. Part 0's outcome does not gate Stage 60 either way;
   it feeds a separate intake candidate.
3. **D3, the Stage 60 flagship spec (conditional).** A 200M-class
   dual-register model in the Stage 55 configuration lineage (201,609,249
   parameters at the Stage 55 vocabulary; the union 33-character vocabulary
   moves the exact count slightly, as it did at 85M where 85,097,499 became
   85,106,721, so the D3a-analog sizing gate re-prints and records the new
   exact count, and a changed count is NOT a sizing-gate failure; block
   256, RoPE, Muon), union 33-character vocabulary, Recipe v2 (fp32, cosine
   to `--lr-final-frac 0.1`, `--checkpoint-every 5000`,
   `--checkpoint-keep 2`), trained on ONE continuous mixture corpus at dose
   `w_flag`. Registered dose rule: `w_flag = w*` ONLY IF H025's secondary
   transfer read registered USEFUL AND the fit's predicted toll `d(w*)`
   sits inside the CONFIRM band (at or below `+0.010`) AND
   `|w* - 0.10| <= 0.10`; otherwise `w_flag = 0.10`, the only dose directly
   confirmed at 85M. Any use of `w*` is a disclosed extrapolation beyond
   the confirmed point and the model card says so. The step budget
   comes from a confirm-first sustained 5,000-step throughput measure at
   the full configuration, targeting a 30 to 40 GPU-hour envelope with
   50,000 steps as the ceiling; the measured seconds-per-step number, not an
   assumption, sets the count. A kill-and-resume drill precedes the long
   run (ADR 0013 precedent).
4. **D4, Stage 60 success bars, pre-registered, evaluated in this order.**
   The four bars: (a) deterministic chunked text8 TEST bits/char strictly
   below the 85M COLD anchor `1.357318`; (b) final TinyStories retention at
   least `1.0` bits/char better than the 85M no-rehearsal anchor `3.556502`
   (cross-scale transfer of the anchor is disclosed in the model card);
   (c) the letters probe scored and reported either way; (d) the user's
   sample review passes (ADR 0011 rule). Precedence: FIRST, if (a) fails,
   the verdict is FAIL regardless of (b): no publication, Phase 6 returns
   to intake. SECOND, if (a) holds and (b) fails, the verdict is GRADED:
   the publication decision escalates to the user with the miss stated
   first. THIRD, if (a) and (b) hold but (c) is missing or (d) does not
   pass, the model stays unpublished and the decision escalates to the
   user ((c) is Codex-fixable by running the probe; (d) is the user's
   judgment alone). PUBLISH-WORTHY = all four. Honesty note baked into the
   bar: measured capacity gains at these budgets have been small (the
   Phase 4 flagship beat the 25.25M reference by about `0.012` in-domain),
   so bar (a) demands strict improvement and reports the margin rather than
   gating on a margin; the model card must print that margin next to the
   known between-seed spread `0.003035` so a within-noise win cannot be
   over-read. The GPT-2 117M zero-shot anchor `1.17` (Radford et al. 2019)
   is aspirational context in reports, never a gate.
5. **D5, the publication package (conditional on D4 plus the user's go).**
   fp16 safetensors model-only export to a public Hugging Face repo, fp32
   resume-capable checkpoint to a private Hub repo (the archive-then-delete
   rule from the release plan holds: uploads verified by checksum BEFORE
   any local deletion), model card per ADR 0015 D6 with every number
   carrying its regenerating command, TinyStories CDLA-Sharing-1.0 and
   text8 Wikipedia lineage disclosures, no corpus uploads, Apache-2.0 code.
   The user presses every public button; Codex prepares everything up to
   that line.
6. **D6, operational constraints for the window.** The external internship
   (2026-07-20 to 2026-08-21) makes weekday daytimes autonomous: long runs
   launch in the evening or on weekends through the ADR 0012 visible
   launcher with the keep-awake helper, checkpoints go only to
   `C:\cassandra_runs` (never OneDrive, never `%TEMP%`), the 15 GiB disk
   gate binds, the 02:00 MUSAHIT GPU collision is managed one night at a
   time with `SKIP_NEXT_RUN.flag`, and any pause or crash follows the
   run-pitstop protocol with SHA-256 lineage recorded.
7. **D7, the Gemini lane.** Before Part 1 runs: functional forms for data
   mixing laws at character-level small scale (which parametric family to
   fit, what the literature reports for extrapolation error). Standing:
   replay-dose numbers from the continual-learning literature to contextualize
   the measured curve. Contingent: induction-head elicitation methods if
   Part 0 fires its `0.1625` line. The Phase 7 RL-reasoning note stays
   parked until Phase 6 resolves.

## Scope and what this decision does not claim

- Stage 60 exists only behind D2's gate. Nothing in this ADR predicts
  E-cheap-rehearsal; ADR 0016 explicitly warns the toll may be
  composition-independent.
- H024's E-null on ordering stands; Stage 60's mixture is dose-driven
  composition, not an ordering claim.
- The equal-compute 85M-versus-200M control stays registered and
  unscheduled; Stage 60 plus the Stage 58 COLD arm yield only a
  non-compute-matched capacity read, and any capacity claim in the model
  card says so.
- "Published on Hugging Face" is a user action. This ADR authorizes
  preparation, never the push.
- The BPE substrate contingency stays unwritten unless a Phase 6
  retrospective intake revives it.

## What would reopen or reverse this decision

1. If Stage 60 launches and lands FAIL under D4, the D2 gate is spent:
   Phase 6 returns to intake with candidates ranked by ADR 0016 Decision 3
   (recipe and capacity frontier first), and no second flagship launches
   inside Phase 6.
2. If H025's secondary transfer read registers NOT USEFUL while the primary
   confirms, `w_flag` falls back to `0.10` and proxy-predicted data
   decisions stay out of the toolkit (this is a fallback inside D3, not a
   reversal).
3. If the sustained throughput measure prices even 30,000 steps above the
   40 GPU-hour envelope, Stage 60 pauses for a user decision on budget
   versus size (a smaller-than-200M flagship is a user fork, not a Codex
   improvisation).
4. If Part 0 fires (`>= 0.1625`), the behavior axis reopens as a SEPARATE
   intake candidate; it does not modify the D2 gate.

## Implementation status (updated 2026-07-22)

Implementation: Stage 59 COMPLETE on 2026-07-21, verdict **E-partial**
under H025's partition (seed 11 CONFIRM-side at `d = +0.009654` /
`r = +2.703539`; seed 19 neither side at `d = +0.012353` / `r = +2.669048`;
no KILL-side seed, so no escalation; both instability guards passed; every
deciding number verified against `runs/stage59_verdict.json` and its named
source files at read-back). **D2 therefore does NOT fire: Stage 60 is not
authorized**, and Phase 6 returns to intake carrying the measured
dose-response curve (proxy dose means in `runs/stage59_mixing_law_fit.md`;
the 85M toll at `w = 0.10` is about `0.011` bits/char while retention gain
saturates near `2.7` by that dose). The D3 `w_flag` machinery is moot. The
secondary transfer read registered NOT_DECISION_GRADE (predicted `0.023045`
versus measured `0.011004`, ratio `2.094`, wrong side of the `0.020` line),
so reopen item 2's fallback stands in its stronger form: proxy mixing laws
are out of the toolkit at this scale. Part 0 fired the ADR 0016 clause-3
behavior-axis reopen (`0.194336` versus `0.1625`); per D2 that outcome is
an intake candidate (H026 class) and never touched the gate. The
falsifiability ordering held: the frozen fit file predates the first
Part 2 launch by 39 minutes in the launcher logs. Intake candidates now on
the table, per this ADR and ADR 0016 Decision 3: the recipe and capacity
frontier, the H026 diverse-data circuit-formation axis, and any freshly
grounded dose follow-up (which would need a new ADR; this gate is spent).

## Links

- `docs/hypotheses/025-rehearsal-dose-response-and-mixing-law.md` (the authority for Stage 59)
- `docs/decisions/0016-phase-5-closeout-developmental-null-recipe-frontier.md`
- `docs/phase6-codex-goal-prompt.md` (the execution handoff this ADR authorizes)
- `docs/open-source-release-and-disk-plan.md` · `docs/phase5-licensing-notes.md` · `docs/phase5-model-card-draft.md`
- `docs/decisions/0011-phase-3-coherence-checkpoint-and-crossover-scaling-law.md` · `0012-visible-terminal-experiment-launches.md` · `0013-phase-4-free-accelerator-floor-scaling-flagship.md` · `0014-phase-4-closeout-flagship-verified-evaluation-standard.md` · `0015-phase-5-developmental-training-and-open-source-posture.md`
- `experiments/tiny_language_lab/RESULTS.md` (Stages 55 through 58)
