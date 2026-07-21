# ADR 0015 · Phase 5 trains smarter, not bigger: the developmental experiment is the centerpiece, and the lab prepares for open source

- Status: Accepted · forks answered by the user on 2026-07-07
  ("smarter training is more important than training"; "planning this more
  as an open source project maybe"; Recipe v2 and behavior probe: "test,
  evaluate, then decide").
- Date: 2026-07-07
- Author: Claude (hypothesis, ADR, and roadmap role)
- Resolves: `docs/phase5-intake.md` Forks 1 through 4; ADR 0014 D6's
  registered Phase 5 question.
- Builds on: ADR 0014 (flagship verified; evaluation standard; the
  measured `2.07` bits/char specialization gap), Stage 53 (analytic priors
  interfere under full training, `+0.100` NLL at 2000 steps), H022 as
  specced and calibrated (`docs/hypotheses/022-broad-corpus-specialization-gap.md`),
  Gemini theme-4 notes 01 and 02, the Phase 4 retrospective's recipe-debt
  list (`docs/phase4-flagship-midrun-report.md` Section 2).

## Context

Phase 4 ended with a verified 201.6M flagship and one dominant unexplained
number: the domain-specialization gap. The Phase 5 intake proposed
answering it not by scaling but by asking how a fixed budget is best
SPENT: cold broad training, a simple-register childhood first, or a
mixture. The user adopted that centerpiece, redirected the publication
fork toward open source, and put the recipe upgrades and the behavior
probe behind measured gates. This ADR fixes those directives into
commitments with pass lines, so Codex can execute and nothing rests on
taste.

## Evidence

- Specialization gap: in-domain `0.8126` bits/char vs zero-shot text8
  `2.8817` (`runs/stage55_text8_zero_shot.md`,
  `runs/stage55_validation_suite.md`).
- Prior-interference precedent that the centerpiece tests at the learned
  level: Stage 53's paired deltas `+0.072` and `+0.100` at 1000 and 2000
  steps (`runs/stage53_h020_free_accelerator_summary.md`).
- Recipe debt is measured, not assumed: fp32 training on an Ada GPU, a
  constant LR tuned at 2000-step horizons (the Stage 53 LR rerun proved
  sensitivity), keep-all `1.51 GiB` checkpoints, and block 256 against
  750-to-1500-char stories.
- H022 threshold calibration (2026-07-07): converged small char
  transformers on text8 sit at `1.08` to `1.23` bits/char, so the CONFIRM
  line moved from `1.60` to `1.70` before any run.
- Repo state relevant to open source: git history already carries about
  `1.07 GB` of corpus blobs exceeding GitHub's 100 MB per-file limit, and
  `corpus/tinystories_char_seed.txt` (471 MB) is still tracked; the
  repository cannot be pushed cleanly in its current state.

## Decision

1. **D1 · Centerpiece: the developmental experiment (H024, pre-registered
   here, frozen after Stage 56).** Three arms at 85.11M
   (`L12 H12 D768`), ONE fixed compute budget per arm (about 12 GPU-hours
   under Recipe v2, exact steps set by the Stage 57 throughput
   measurement): COLD (random init, broad corpus full budget), CURRICULUM
   (TinyStories for 25 to 30 percent of the budget, then broad,
   continuing from the narrow weights), MIXTURE (interleaved shards,
   fixed ratio, full budget). Primary metric: text8 TEST bits/char under
   the ADR 0014 chunked convention. Secondary: the TinyStories retention
   curve across checkpoints (union 33-char vocab via the new
   `--vocab-chars` flag). Pre-registered tension: curriculum literature
   predicts CURRICULUM at or above COLD; the Stage 53 analog predicts
   learned narrow weights interfere. The full H024 hypothesis doc is
   written only after Stage 56 (H022) reads out; if H022 KILLs, the same
   three arms run on the H023 BPE substrate. Seeds: seed 7 full budget
   per arm, plus reduced replicas for the winning arm.
2. **D2 · Open-source posture: prepare to release; releasing stays the
   user's call.** Phase 5 produces a release-ready artifact set without
   committing to a publish date: (a) git history surgery to purge corpus
   blobs (`git filter-repo` after a bundle backup; REQUIRED regardless of
   release since the repo already exceeds GitHub's limits, and no public
   push happens before it plus explicit user sign-off); (b) untrack the
   471 MB seed corpus before any further commit; (c) license due
   diligence recorded in a doc (code license choice; TinyStories dataset
   license; text8's Wikipedia lineage; what that implies for released
   weights); (d) an fp16 model-only archival export (about 400 MB) beside
   the ONNX; (e) a model card grown from
   `docs/phase4-flagship-evaluation-report.md`; (f) a scrub pass over
   docs and logs for local paths and personal artifacts. The playground
   is the demo artifact. Any release decision additionally gates on the
   human A/B review outcome (ADR 0014 D4).
3. **D3 · Recipe v2 by measured gates (Codex Stage 57), adopt only what
   passes.** Pre-registered acceptance lines: bf16 autocast adopted if
   200-step NLL is within `0.01` of fp32 AND throughput gain is at least
   1.4x, else fp32 stays; cosine warmdown adopted if a 5000-step 25.25M
   pair (constant vs cosine to one tenth of peak LR) shows the cosine arm
   at or below the constant arm's final sampled NLL, else constant stays
   and the question is re-run at H024 scale; `--checkpoint-keep` and
   `--vocab-chars` are plumbing, adopted after smoke tests; block 512
   gets a 200-step timing-and-VRAM row as a D3a-style decision input for
   H024, not an automatic adoption. Whatever passes is LOCKED as Recipe
   v2 for every Phase 5 run.
4. **D4 · Behavior probe, eval-only, then decide.** Codex builds a
   letters-only variant of the memorization-proof copy generator (the
   Stage 42 design emits digits, which the vocab lacks) and scores the
   flagship zero-shot. The behavior axis reopens only if the flagship
   clears chance by the Stage 42 margin (`0.10` above `1/V`) on this
   probe; otherwise the axis stays paused and the result is recorded as
   scale evidence for ADR 0008's capacity reading. No training is spent
   either way.
5. **D5 · Scheduling.** Before 2026-07-20: the remaining P0 items (human
   A/B review by the user, storage cleanup tiers, ADR 0009 draft
   disposition), Stage 56 (H022), and Stage 57 gates. Inside 2026-07-20
   to 2026-08-21: the H024 arms and replicas plus the D2 preparation
   tasks (all autonomy-compatible). After: H024 analysis, the Phase 5
   closeout ADR, and the release decision if the user takes it.
6. **D6 · Evidence and reproducibility standard.** ADR 0014's conventions
   bind all Phase 5 stages (chunked closeout evals, text8 anchor table,
   sampled eval for monitoring only). The open-source posture adds one
   rule: every number in a release-facing document must regenerate from a
   command present in the repo.

## Scope and what this decision does not claim

- No model above 201.6M parameters and no corpus beyond TinyStories,
  text8, and their mixtures in Phase 5.
- "Open source maybe" is a preparation commitment, not a release
  commitment; nothing becomes public inside this ADR.
- The developmental result, whichever way it lands, is a claim about this
  scale, this substrate pair, and this budget; it is not a general law of
  curriculum learning.
- Closed branches stay closed (ADR 0001, 0004, 0005, 0007, and the
  ADR 0006/0008 copy scope). The equal-compute 85M-vs-200M control stays
  registered but unscheduled; H024's COLD arm informs it for free.

## What would reopen or reverse this decision

1. If H022 KILLs and the H023 BPE probe ALSO fails to beat the char
   substrate at matched compute, the developmental centerpiece pauses on
   substrate grounds and D1 returns to intake.
2. If every Stage 57 gate fails its line, Phase 5 runs on Recipe v1 and
   D1's budget arithmetic is re-estimated before any arm launches.
3. If the human A/B review does not favor the flagship (ADR 0014's
   reversal clause), D2's release narrative is rewritten before any
   release preparation continues past the git surgery.
4. If licensing due diligence finds the corpus terms incompatible with
   releasing trained weights, D2 narrows to code-plus-recipe release and
   says so explicitly.

## Implementation status (updated 2026-07-21)

Implementation: the empirical program is COMPLETE. D1's Stage 56 (H022) read
CONFIRM for the data-distribution cause (`1.485740` text8 TEST bits/char at
50k, below the `1.70` line). Stage 57 locked Recipe v2 from measured gates
(fp32 retained, bf16 rejected on throughput, cosine adopted at `-0.068177`
sampled NLL). The Stage 58 centerpiece resolved H024 as E-NULL, seed-robust
in sign (`+0.005096` seed 7 at 42k; `+0.009845` and `+0.007791` on the 20k
replicas; escalation not triggered; all arms passed their instability
guards). The behavior probe kept the copy axis closed (`0.060547` versus
`0.062500` chance). Closeout, evidence verification, and the reopen clauses
are recorded in ADR 0016
(`docs/decisions/0016-phase-5-closeout-developmental-null-recipe-frontier.md`).
D2/D5 release actions (history surgery, license, Hub upload, public push,
Round-2 A/B votes) remain user-gated and open.

## Links

- `docs/phase5-intake.md` (the forks this ADR resolves)
- `docs/decisions/0014-phase-4-closeout-flagship-verified-evaluation-standard.md`
- `docs/hypotheses/022-broad-corpus-specialization-gap.md` (Stage 56, calibrated)
- `docs/phase4-flagship-evaluation-report.md` · `docs/figures/phase4/`
- `research/theme_4_domain_specialization_and_substrates/01_bpe_vs_character_level_small_models.md` and `02_domain_specialization_gap_and_corpus_breadth.md`
- `experiments/tiny_language_lab/RESULTS.md` (Stages 53 to 55)
- `docs/phase5-codex-goal-prompt.md` (execution handoff)
