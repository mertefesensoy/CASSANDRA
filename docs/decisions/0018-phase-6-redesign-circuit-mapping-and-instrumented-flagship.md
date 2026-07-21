# ADR 0018 · Phase 6 redesign: circuit mapping becomes the centerpiece, and the flagship returns as a pure-broad 200M instrumented build whose checkpoint ladder feeds the science

- Status: Accepted (user-directed, 2026-07-22)
- Date: 2026-07-22
- Author: Claude (hypothesis, ADR, and roadmap role)
- Resolves: the Phase 6 retrospective intake that ADR 0017's spent gate
  required (user answered in session 2026-07-22: H026 circuit mapping as
  centerpiece, pure-broad 200M instrumented flagship, H026 eval-only).
- Builds on: ADR 0017 and its implementation status (Stage 59 E-partial,
  D2 gate not fired, transfer read NOT_DECISION_GRADE), the fired
  ADR 0016 clause-3 behavior-axis reopen
  (`runs/stage59_cold_letters_probe.json`, `0.194336` versus line
  `0.1625`), H026
  (`docs/hypotheses/026-diverse-data-circuit-formation.md`), the Stage 59
  dose curve (`runs/stage59_mixing_law_fit.md`,
  `runs/stage59_verdict.json`), and Gemini notes 12 and 19 (2026-07-21).

## Context, including one evidence correction that must not steer the plan

Stage 59 resolved E-partial: the low-dose rehearsal toll (`+0.009654` and
`+0.012353` bits/char, seeds 11 and 19) straddles the CONFIRM line, the
retention gain is large and consistent (`+2.70` and `+2.67`), and the
proxy mixing law over-predicted the measured cost by `2.094x` on the
wrong side of the KILL line. The ADR 0017 D2 flagship gate is spent.
Meanwhile Part 0 fired the behavior-axis reopen with the phase's
strongest surprise: the broad-only 85M copies in-context at `0.194336`
while the narrow-only 201.6M flagship sits at `0.060547` on the identical
frozen probe.

Gemini's note 12 correctly surveys the function-vector-shift and
induction-head literature, and its recommendation (probe the surviving
checkpoints) is adopted. Its Cassandra evidence paragraph, however,
inverts the lab's own record: it describes the `0.194` model as
simple-data-trained and the `0.061` model as that same model "trained
further on text8", a lineage that does not exist. The `0.194` row is the
Stage 58 COLD 85M trained ONLY on text8; the `0.061` row is the Phase 4
flagship 201.6M trained ONLY on TinyStories. No circuit-destruction event
is in evidence anywhere yet; what is in evidence is circuit FORMATION
under diverse data and non-formation under narrow data, with size
uncontrolled between the two rows. A correction addendum is appended to
the note file, and H026 is framed on the corrected reading, with the
destruction question demoted to a pre-registered screening read over the
CURRICULUM domain-shift lineage, where it can actually be measured.

Gemini's note 19 (proxy scale breakdown) matches the measured transfer
failure and supports keeping proxy-predicted mixtures out of the toolkit,
as ADR 0017's implementation status already records.

## Decision

1. **D1, the centerpiece.** Stage 60 executes H026 exactly as written:
   the eval-only probe matrix over every surviving Stage 58/59
   checkpoint, with the pre-registered PRESENT/ABSENT/GRAY lines
   (`0.1625` / `0.1125`), the E-diverse / E-steps / E-gray verdict
   partition read on seed 7 with seed 11/19 sign replicas, and the
   formation-onset and destruction screening reads. H026 is the authority
   if this ADR and it disagree. Cost: about one GPU-hour of probes.
2. **D2, the eval-only boundary.** Stage 60 trains nothing. Any
   intervention it motivates (a rescue run, an ablation, a continuation)
   is a separate hypothesis with its own pre-registered line, written
   only after the Stage 60 read-back, per the user's budget directive.
3. **D3, the flagship returns as Stage 61 (rung 72), pure-broad and
   instrumented.** A 200M-class Recipe v2 model in the Stage 55
   configuration lineage (Stage 55's vocabulary was ALREADY the
   33-character set, so the sizing gate expects EXACTLY 201,609,249
   parameters; a differing count flags a configuration error, which is
   the gate's purpose), block 256,
   RoPE, Muon, fp32, cosine to `--lr-final-frac 0.1`, trained on the
   PURE text8 broad corpus (at the 50,000-step ceiling that is
   204,800,000 characters, about 2.28 epochs of the 90M-character train
   split). No mixture, no dose claim: the dose axis stays closed and this
   build makes a capacity claim only. Budget discipline carries over from
   ADR 0017 D3 unchanged: sizing gate, sustained 5,000-step throughput
   measure setting the step count inside a 30 to 40 GPU-hour envelope
   with 50,000 steps as the ceiling, kill-and-resume drill before launch,
   visible launcher, keep-awake, `C:\cassandra_runs` only, 15 GiB disk
   gate, MUSAHIT nights planned.
4. **D4, the instrumentation contract.** `--checkpoint-every 5000` with
   `--checkpoint-keep` set to at least 12 (keep-all for the run), so that
   EVERY 5,000-step checkpoint survives until probed. The measured
   Stage 55 artifact is `1.51 GiB` per 200M checkpoint including Muon
   optimizer state (`docs/phase4-flagship-midrun-report.md`), so ten
   intermediates plus the final is roughly 16 to 18 GB; C: holds well
   over 100 GiB free, and intermediates may be deleted only AFTER their
   probe and retention rows are recorded, finals never. Each checkpoint
   gets, within a day of being written: the letters probe row (the 200M
   formation curve, H026's scale extension) and the TinyStories retention
   row (the no-rehearsal forgetting anchor at 200M). These are logged to
   `runs/stage61_instrumentation.jsonl` as they land, so a crash never
   costs the curve.
5. **D5, publish-worthiness bars for Stage 61, pre-registered, evaluated
   in this order.** (a) Deterministic chunked text8 TEST bits/char
   strictly below the 85M COLD anchor `1.357318`
   (`runs/stage58_dev_cold_85m_b42000_seed7_text8_test.json`); FAIL if
   not, no publication, back to intake. (b) The instrumentation curves complete
   (probe and retention rows for every surviving checkpoint) and the
   final's letters probe row recorded; this is Codex-fixable and blocks
   publication until fixed, never fires FAIL. (c) The user's sample
   review passes (ADR 0011); the user's judgment alone. PUBLISH-WORTHY =
   all three. The model card prints the (a) margin next to the known
   between-seed spread `0.003035` (the Stage 56 20k replica spread,
   `RESULTS.md` Stage 56), discloses that TinyStories retention is an
   expected non-goal for this pure-broad build (anchor: the 85M
   no-rehearsal retention was `3.556502` bits/char,
   `runs/stage58_dev_cold_retention.json`), and keeps the GPT-2 117M
   zero-shot `1.17` anchor as context, never a gate. Launch ordering,
   stated here so the goal prompt is not the sole authority: Stage 61
   MAY launch before Stage 60's verdict is read (both are measurement),
   but no packaging step proceeds before the Stage 60 read-back AND
   these bars. Hugging Face packaging follows ADR 0017 D5 unchanged:
   Codex prepares, the user presses every public button.
6. **D6, the Gemini lane.** Requested next: induction-head measurement
   methods that go beyond behavioral probes (attention-pattern
   prefix-matching scores, ablation protocols) sized to a 12-layer 85M
   char model, for the possible post-Stage-60 mechanistic follow-up; and
   the note-12 revision acknowledging the corrected evidence reading.

## Scope and what this decision does not claim

- Nothing here reopens the dose axis (E-partial stands), the ordering
  axis (ADR 0016), or proxy mixtures (NOT_DECISION_GRADE stands).
- Stage 61 makes no dual-register claim and no mechanism claim; its
  probe curve is measurement, interpreted only through H026's frame.
- The Stage 61 capacity comparison against the 85M anchor is
  budget-matched in steps, not in compute; the model card says so.
- Publication remains a user action.

## What would reopen or reverse this decision

1. If Stage 60 reads E-steps, Stage 61's instrumentation stays (it is
   cheap and descriptive) but the release narrative drops the
   diverse-data mechanism story, and the behavior axis goes back to
   intake on the size/schedule fork.
2. If the Stage 61 sizing or throughput gates price 30,000 steps above
   the 40 GPU-hour envelope, launch pauses for the user's size-versus-
   budget fork (ADR 0017 reopen item 3, carried over).
3. If Stage 61 lands FAIL under D5(a), no second flagship launches in
   Phase 6, and the capacity story is recorded as measured, not spun.
4. If the user's A/B round-2 votes (still open from Phase 5) contradict
   the release narrative, the narrative is rewritten before any
   packaging proceeds (ADR 0015 reversal clause, still live).

## Links

- `docs/hypotheses/026-diverse-data-circuit-formation.md` (Stage 60 authority)
- `docs/decisions/0017-phase-6-rehearsal-dose-flagship-gate.md` (the spent gate and carried-over discipline)
- `docs/decisions/0016-phase-5-closeout-developmental-null-recipe-frontier.md`
- `runs/stage59_verdict.json` · `runs/stage59_mixing_law_fit.md` · `runs/stage59_cold_letters_probe.json`
- `research/theme_2_in_context_learning_and_rag/12_forgetting_of_structural_circuits.md` (correction addendum) · `research/theme_3_training_dynamics_and_curriculums/19_scale_breakdown_in_proxy_mixing.md`
- `docs/phase6-codex-goal-prompt.md` (superseded Stage 60 section; redesign addendum)
- `docs/open-source-release-and-disk-plan.md` · `docs/phase5-model-card-draft.md`
