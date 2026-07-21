# ADR 0016 · Phase 5 closes: acquisition order is a practical null at the registered budget, and the recipe frontier, not the curriculum, is what actually moved

- Status: Accepted
- Date: 2026-07-21
- Author: Claude (hypothesis, ADR, and roadmap role)
- Resolves: H024 (`docs/hypotheses/024-developmental-acquisition-order.md`,
  now E-null) and ADR 0015 D1's developmental centerpiece; records the
  completion of ADR 0015's empirical program. Release posture (ADR 0015
  D2/D5 user-gated items) is explicitly NOT resolved here.
- Builds on: Stages 56, 57, 58 (`experiments/tiny_language_lab/RESULTS.md`),
  the Stage 58 evidence set (`runs/stage58_dev_*_text8_test.json`,
  `runs/stage58_dev_*_retention.json`), the closeout reports
  (`docs/phase5-final-report.md`, `docs/phase5-completion-audit.md`,
  `docs/phase5-developmental-midrun-report.md`), the figures
  (`docs/figures/phase5/`), and Gemini's post-Phase-5 notes
  (`research/theme_3_training_dynamics_and_curriculums/17_data_mixing_laws_and_doremi.md`,
  `18_catastrophic_forgetting_and_replay_buffers.md`,
  `research/theme_2_in_context_learning_and_rag/11_induction_heads_and_zero_shot_failure.md`).

## Context

ADR 0015 made the developmental experiment the centerpiece of Phase 5:
at one fixed 85.11M-parameter, 42,000-step compute budget, does a learned
TinyStories childhood (CURRICULUM), or an interleaved rehearsal mixture
(MIXTURE), beat cold broad training (COLD) on deterministic text8 TEST
bits/char? H024 pre-registered the trichotomy: E-curriculum at or below
`-0.05` bits/char, E-interfere at or above `+0.05`, E-null between them,
with reduced-budget seed replicas testing the sign and an escalation rule
for marginal or sign-flipping outcomes. H024 froze 50,000 steps subject to
a hard confirm-first clause delegating the exact count to a sustained
throughput re-measure; that re-measure (5,000 steps in 4,992.0457 seconds,
`runs/stage58_dev_throughput_85m_b5000_seed7.jsonl`, recorded in
`docs/phase5-developmental-midrun-report.md`) set the budget at 42,000
steps, about 11.65 GPU-hours per arm. That pre-authorized adjustment moved
the childhood fraction from the registered 25 percent to 12,500/42,000,
about 29.8 percent, and the MIXTURE corpus was rebuilt at the 25:59 ratio
to stay dose-matched (`docs/phase5-completion-audit.md`).

Codex completed all three seed-7 arms, the full-source retention curves,
and both reduced-budget replicas between 2026-07-09 and 2026-07-21,
surviving one CUDA illegal-instruction crash, one OneDrive-adjacent
final-write failure isolated to a restricted process context, and several
power-safety pauses, all documented with checkpoint SHA-256 lineage.
This ADR is the read-back verdict on that evidence. Every deciding number
below was verified against the raw run artifact named beside it, not
against prose.

## Evidence

1. **Primary H024 contrast, verified against the raw TEST reports**
   (`runs/stage58_dev_cold_85m_b42000_seed7_text8_test.json`,
   `runs/stage58_dev_curriculum_phase2_85m_b42000_seed7_text8_test.json`,
   and the four `b20000_seed11`/`b20000_seed19` cold and curriculum
   reports; all rows deterministic chunked text8 TEST over 4,999,936
   characters):

   | Seed | Budget | COLD | CURRICULUM | CURRICULUM minus COLD |
   | ---: | ---: | ---: | ---: | ---: |
   | 7 | 42,000 | 1.357318 | 1.362414 | +0.005096 |
   | 11 | 20,000 | 1.410154 | 1.419999 | +0.009845 |
   | 19 | 20,000 | 1.410779 | 1.418571 | +0.007791 |

   All three signs favor COLD; every magnitude sits far inside the
   `+/-0.05` E-null band. The escalation rule does not fire: the seed-7
   margin is not marginal (it is `0.005`, a tenth of the line) and both
   replicas reproduce its sign.
2. **Secondary arm.** MIXTURE scored `1.385295`
   (`runs/stage58_dev_mixture_85m_b42000_seed7_text8_test.json`),
   `+0.027977` versus COLD and `+0.022881` versus CURRICULUM, inside the
   registered practical band.
3. **Retention is real but secondary.** On the fixed 1,499,904-character
   TinyStories validation sample (`runs/stage58_dev_*_retention.json`),
   MIXTURE finishes at `0.826285` bits/char versus CURRICULUM `3.529069`
   and COLD `3.556502`. CURRICULUM's childhood is erased within about
   2,500 broad steps (0.878 at step 12,500, 3.189 at step 15,000): the
   catastrophic-forgetting cliff in fig2. Rehearsal preserves the narrow
   register at a `+0.028` broad-text cost; front-loading preserves
   nothing and still costs `+0.005`.
4. **Stability guard.** Every seed-7 arm's final sampled broad validation
   NLL improved on its own step-25,000 value: COLD `0.884350` versus
   `0.986930` (`-0.102580`), CURRICULUM `0.920510` versus `0.985283`
   (`-0.064773`, safely below the registered `1.035283` instability
   line), MIXTURE `0.905836` versus `1.007650` (`-0.101814`), verified
   against the run jsonl finals. No arm is inconclusive under H024's
   instability rule.
5. **The recipe frontier moved more than the curriculum ever could.**
   At matched budget and matched seed (20,000 steps), Recipe v2's COLD
   beats Stage 56's Recipe v1 by `-0.122473` (seed 11) and `-0.118812`
   (seed 19) bits/char
   (`runs/stage56_broadchar_85m_b20000_seed1{1,9}_text8_test.json`
   versus `runs/stage58_dev_cold_85m_b20000_seed1{1,9}_text8_test.json`).
   That is roughly 24 times the seed-7 primary order effect, and about
   4.4 times the largest order effect Stage 58 measured at all
   (MIXTURE's `+0.028`).
   Caveat, stated plainly: this is a bundle comparison, not a single
   factor. Recipe v2 differs from v1 in its cosine schedule (the one
   Stage 57 gate that measured a gain, `-0.068177` sampled NLL), the
   33-character union vocabulary (85,106,721 versus 85,097,499
   parameters, a 0.011 percent capacity difference), and checkpoint
   retention plumbing. The new fig4 plots both this paired contrast and
   the phase arc (flagship zero-shot `2.8817`, Recipe v1 at 50k
   `1.4857`, Recipe v2 COLD at 42k `1.3573`).
6. **Behavior axis unchanged.** The letters-only constrained-choice copy
   probe on the flagship scored `0.060547` versus `0.062500` chance
   (`runs/phase5_behavior_letters_probe.json`); nothing in Stage 58
   reopens ADR 0008's closure.

## Decision

1. **Phase 5's empirical program is CLOSED as delivered, and H024
   resolves E-null, seed-robust in sign.** At this scale, substrate
   pair, and budget, acquisition order is not a lever: a learned
   TinyStories childhood does not earn back the broad-text exposure it
   displaces, and interleaved rehearsal costs slightly more broad-text
   performance than it is worth on the primary metric.
2. **The order-of-acquisition axis closes at this budget and dose.**
   Like every closed branch before it (retrieval, data-side selection,
   analytic-base formation, rank-2 general copy), it stays closed unless
   a reopen clause below fires. The E-null is a practical decision under
   the registered `0.05` line, not a proof of equivalence, and it is a
   fixed-compute claim only.
3. **The measured lesson of Phase 5 is that recipe beats ordering.**
   Stage 57's gate-driven Recipe v2 delivered a matched-budget
   improvement more than an order of magnitude above H024's deltas
   (about 24 times the primary delta, 4.4 times the largest). Roadmap
   consequence: Phase 6 hypotheses should spend compute on levers with
   Stage-57-style measurable gates (data mixture composition, schedule,
   vocabulary, capacity) rather than on further ordering permutations.
4. **MIXTURE's retention result is the anchor for the Phase 6 intake.**
   Stage 58 measured one point on the rehearsal dose-response curve:
   at the 12,500/42,000 TinyStories dose (29.8 percent, corpus ratio
   25:59), retention improves by `2.7` bits/char for a `+0.028`
   broad-text cost. Gemini's data-mixing
   and replay notes propose the natural falsifiable follow-up: sweep the
   dose (including the 5 to 10 percent replay region) with tiny proxy
   models, fit a mixing law, and test whether the proxy-predicted
   optimum transfers to 85M. This becomes the leading H025 candidate,
   pending intake.
5. **Release actions stay user-gated.** Nothing here authorizes history
   surgery, license selection, Hub upload, or a public push. Those
   remain the user's explicit calls under ADR 0015 D2/D5 and
   `docs/phase5-success-criteria.md`.

## Scope and what this decision does not claim

- The E-null binds at 85.11M parameters, the TinyStories/text8 substrate
  pair, the re-measured 42,000-step budget, and the registered 12,500
  childhood steps (51.2M characters; the resulting 29.8 percent fraction
  follows from the budget re-measure, it was not itself registered).
  Added-compute childhoods, other dose points, other domain pairs, and
  other scales are all outside it.
- The recipe-versus-ordering lesson (Decision 3) is an observed
  magnitude comparison inside this lab, not a general law.
- The full-budget contrast has one seed; the replicas certify direction,
  not effect-size equality at 42,000 steps.
- MIXTURE versus CURRICULUM remains an order-plus-schedule-regime
  comparison, as H024 registered.

## What would reopen or reverse this decision

1. If a Phase 6 mixture-dose or mixing-law experiment produces a
   composition that beats COLD by at least `0.05` text8 TEST bits/char
   at matched budget, the ordering/composition axis reopens under the
   new hypothesis (composition, not order, would then be the live
   variable).
2. If an added-compute childhood (childhood steps NOT deducted from the
   broad budget) beats the matched-broad-budget baseline by at least
   `0.05`, the fixed-compute framing gets an explicit annotation and a
   new hypothesis.
3. If the letters copy probe run against the broad-trained Stage 58 COLD
   checkpoint scores at or above `0.1625` constrained-choice accuracy
   (chance `0.0625` plus `0.10`, the registered reopen threshold in
   `runs/phase5_behavior_letters_probe.json`), the behavior axis reopens
   under a new hypothesis about diverse-data circuit formation (this is
   a cheap, already-tooled probe and a natural Phase 6 intake item).
   **FIRED 2026-07-21** (Stage 59 Part 0,
   `runs/stage59_cold_letters_probe.json`): constrained-choice accuracy
   `0.194336` on the same 1,024-case probe file the flagship scored
   `0.060547` on, with choice MRR `0.401049` versus the flagship's
   `0.217769` and raw full-vocabulary accuracy `0.126953` (twice chance).
   The behavior axis REOPENS as a Phase 6 intake candidate (H026 class,
   diverse-data circuit formation); per ADR 0017 D2 this does not gate or
   modify the Stage 60 decision. Single checkpoint, single seed: the
   intake spec should replicate the probe on the remaining Stage 58
   finals before any stronger claim.
4. If an audit finds any deciding Stage 58 number unsupported by its
   named artifact, H024's verdict reverts to OPEN pending re-derivation.

## Links

- `docs/phase5-final-report.md` (Codex's evidence closeout this ADR accepts)
- `docs/phase5-completion-audit.md` · `docs/phase5-developmental-midrun-report.md`
- `docs/hypotheses/024-developmental-acquisition-order.md` (resolved by this ADR)
- `docs/decisions/0015-phase-5-developmental-training-and-open-source-posture.md`
- `docs/figures/phase5/` (fig1 to fig4, `figures_data.json`, `h024_replica_sign_check.json`)
- `experiments/tiny_language_lab/RESULTS.md` (Stage 56 through Stage 58 closeout entries)
- `research/theme_3_training_dynamics_and_curriculums/17_data_mixing_laws_and_doremi.md` · `18_catastrophic_forgetting_and_replay_buffers.md` · `research/theme_2_in_context_learning_and_rag/11_induction_heads_and_zero_shot_failure.md` · `research/theme_4_domain_specialization_and_substrates/03_curriculum_learning_simple_to_complex.md` (Gemini's post-Phase-5 inputs to the Phase 6 intake)
- `docs/open-source-release-and-disk-plan.md` · `docs/phase5-licensing-notes.md` (the user-gated release lane)
