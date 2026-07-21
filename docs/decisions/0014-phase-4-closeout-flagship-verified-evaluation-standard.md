# ADR 0014 · Phase 4 closes with a verified flagship; closeout claims move to deterministic evaluation with an external anchor

- Status: Accepted
- Date: 2026-07-07
- Author: Claude (hypothesis, ADR, and roadmap role)
- Resolves: ADR 0013 D3's flagship deliverable and its evaluation
  obligations (`docs/decisions/0013-phase-4-free-accelerator-floor-scaling-flagship.md`);
  no standalone hypothesis doc existed for the build itself (H020 and H021
  were resolved inside ADR 0013's implementation record).
- Builds on: Stages 53, 54, 55 (`experiments/tiny_language_lab/RESULTS.md`),
  the Stage 55 closeout run files
  (`runs/stage55_flagship_200m_b50000_seed7.md`,
  `runs/stage55_flagship_200m_b20000_seed11.md`,
  `runs/stage55_flagship_200m_b20000_seed19.md`), the validation suite
  (`runs/stage55_validation_suite.md`), the text8 anchor
  (`runs/stage55_text8_zero_shot.md`), ADR 0011's human-review rule, and
  the mid-run retrospective
  (`docs/phase4-flagship-midrun-report.md`).

## Context

ADR 0013 scheduled three deliverables: the free-accelerator test (Stage 53,
KILL E-interfere), order-5 floor scaling (Stage 54, CONFIRM), and a
flagship build assembled from those verdicts (Stage 55). Codex completed
Stage 55 on 2026-07-07: `random_full`, `201,609,249` parameters, block 256,
RoPE, Muon, 50,000 steps on seed 7 plus 20,000-step replicas on seeds 11
and 19, with checkpoint-resume carrying the run through a OneDrive
checkpoint-rename failure at step 35,000. Codex also produced the Nsight DL
Designer ONNX export from the final checkpoint.

What remained open was verification and placement: whether the recorded
numbers survive a higher-precision re-evaluation, whether the ONNX artifact
is faithful to the checkpoint, and where this model stands against anything
outside its own corpus. Claude ran that evaluation on 2026-07-07 with new
lab tools (`phase4_validate.py`, `eval_text8.py`, `make_phase4_figures.py`,
`playground.py`, all beside the trainer). This ADR records the closeout and
the two standing conventions the evaluation earned.

## Evidence

1. **Recorded results, Stage 55 closeout** (`RESULTS.md`): seed 7 sampled
   report val NLL `0.556410` (`0.802730` bits/char, 50k steps); replicas
   `0.609039` (seed 11) and `0.596864` (seed 19) at 20k steps; proxy
   generation score `4.333/6`
   (`runs/stage55_flagship_generation_quality.md`).
2. **Deterministic re-evaluation** (`runs/stage55_validation_suite.md`,
   8,000,000 chars per model, chunked non-overlapping windows spread
   uniformly over the whole validation split): seed 7 `0.563231` NLL
   (`0.812571` bits/char); seed 11 `0.592702`; seed 19 `0.595205`;
   Stage 51 25.25M reference `0.824691`. All recorded numbers confirmed
   within sampling noise. The two 20k replicas land `0.0025` NLL apart
   under this eval, versus a `0.030` recorded spread, so the sampled-16
   eval's noise is of the same order as the apparent seed spread.
3. **ONNX parity** (`runs/stage55_validation_suite.json`): the exported
   `artifacts/phase4/nsight_dld/stage55_seed7_final_success_b1_s256.onnx`
   matches the source checkpoint on real validation text with max logit
   difference `8.58e-6`, mean `1.01e-6`, top-1 agreement `1.0` across all
   256 positions.
4. **External anchor, zero-shot text8**
   (`runs/stage55_text8_zero_shot.md`, full 5M-char test split, chunked
   convention): flagship `2.8817` bits/char, Stage 51 reference `3.3118`.
   Against the in-domain `0.8126`, the flagship's measured
   domain-specialization gap is `2.07` bits/char. Published sibling
   anchors: GPT-2 117M zero-shot text8 `1.17`, GPT-2 1542M `0.98`
   (Radford et al. 2019, Table 3).
5. **Figures** regenerated from raw artifacts into `docs/figures/phase4/`
   and interpreted in `docs/phase4-flagship-evaluation-report.md`.

## Decision

1. **Phase 4 is CLOSED as delivered.** The flagship artifact set is the
   verified pair: the resume-capable checkpoints under
   `C:\cassandra_runs\stage55_flagship_checkpoints` (seed 7 step-50000
   plus both replica finals) and the parity-verified ONNX under
   `experiments/tiny_language_lab/artifacts/phase4/nsight_dld/`. These are
   the canonical copies; everything else is cache.
2. **Closeout-level claims now require the deterministic chunked
   evaluation** (the `phase4_validate.py` convention: non-overlapping
   block-size windows, uniform coverage, millions of chars). The
   `--eval-mode sampled --eval-batches 16` path remains the in-run
   monitoring tool only. Grounds: evidence item 2 shows sampled-16 noise
   is the size of effects the lab has been interpreting (seed spread,
   late-run drift).
3. **Zero-shot text8 becomes the standing external anchor** for any future
   flagship-class checkpoint, using the same full-test-split chunked
   method so numbers stay comparable. Baselines recorded now: `2.8817`
   (flagship), `3.3118` (Stage 51 25M). The in-domain versus text8 gap is
   the lab's named measure of domain specialization.
4. **The pending human review runs through the blind A/B protocol**
   (`playground.py`, votes appended to `runs/human_ab_votes.jsonl` with
   identities revealed only after each vote). The coherence claim for the
   flagship stays unmade until roughly 20 or more votes across varied
   prompts exist. The protocol's first named question: the proxy scorer
   ranks the flagship BELOW Stage 51 (`4.333/6` vs `5.667/6`) while NLL
   improved by `0.38` bits/char; the votes decide whether that is a rubric
   artifact or a real NLL-versus-quality divergence.
5. **The storage cleanup tiers are unblocked.** The ONNX-before-delete
   condition in `docs/phase4-flagship-midrun-report.md` (Section 3 plus
   the 2026-07-06 addendum) is satisfied by evidence item 3; the
   resume-proof directory, the superseded OneDrive checkpoint directory,
   and the `%TEMP%` intermediates can now be deleted per that plan.
6. **Phase 5 is a registered question, not a scoped phase.** The measured
   breadth gap (`2.88` zero-shot vs GPT-2 117M's `1.17`) plus the
   retrospective's substrate critique make a BPE or broader-corpus
   `random_full` the leading candidate axis, and the equal-compute sizing
   control (85M at matched wall-clock) remains the cheapest missing
   experiment. Both go through a Phase 5 intake with user forks before
   anything is committed.

## Scope and what this decision does not claim

- No coherence or sample-quality claim is made; that stays gated on D4's
  votes, carrying ADR 0011 D3's rule forward unchanged.
- No general-English capability claim: `2.88` bits/char zero-shot text8 is
  far from both dedicated char-LMs and GPT-2's zero-shot line. The
  flagship is a measured domain specialist.
- The GPT-1 comparison in the evaluation report is an anchor exercise:
  the perplexity row passes through a chars-per-word conversion and the
  corpora differ in breadth and register. It supports a compute-story
  claim, not a capability-parity claim.
- The 50k headline number is a single seed; the replicas bound seed
  variance at 20k only.
- Closed branches stay closed: nothing here reopens ADR 0001, 0004, 0005,
  0007, or the ADR 0006/0008 copy scope.

## What would reopen or reverse this decision

1. **D4 outcome reverses the quality reading**: if the A/B votes do not
   favor the flagship over Stage 51 by a clear majority of non-tie votes,
   Phase 4's achievement reverts to an NLL-only result, the proxy anomaly
   is promoted to a behavior-axis hypothesis, and any coherence framing is
   withdrawn.
2. **D2 reopens** if a future checkpoint shows the chunked and sampled
   evaluations disagreeing beyond noise (more than `0.02` NLL on the same
   model and split), which would mean the chunked convention itself is
   biased for that regime.
3. **D3 reopens** if a sliding-window text8 rescore of the same checkpoint
   moves the anchor by more than `0.1` bits/char, in which case the
   convention must be restated before any cross-report comparison is made.
4. **D6's framing weakens** if a broader-corpus retrain at comparable
   compute fails to close a material part of the `2.07`-bit
   specialization gap, which would point at substrate or architecture
   rather than corpus breadth.

## Addendum · D4 gate read (2026-07-07)

The user completed 21 blind A/B votes (`runs/human_ab_votes.jsonl`). The
count gate is met; the varied-prompts condition is not (single prompt,
temperature 0.8). On the named question, the flagship does NOT clear
Stage 51 by a clear majority of non-tie votes: the direct head-to-head is
2 to 2. Consequences, as pre-registered: the coherence framing stays
withdrawn, and the proxy anomaly is promoted toward a behavior-axis
hypothesis, now carrying two independent signals (proxy scorer and blind
human votes both rank the 25M model at or above the 200M on this prompt,
while NLL strongly favors the 200M). Because only 4 direct comparisons
exist and prompts were not varied, this is recorded as a PROVISIONAL fire
of reopen condition 1: the named follow-up is a targeted round of 10 to 15
flagship-vs-Stage-51 votes across at least 5 varied prompts. ADR 0015's
release-narrative caution (its reopen condition 3) applies until that
round lands. Within-family ordering was clean (flagship over seed19 3-0-4),
so the anomaly is size-specific, not a general training failure.

## Links

- `experiments/tiny_language_lab/RESULTS.md` · Stages 53, 54, 55 and the
  Stage 55 Flagship Long Run Closeout
- `experiments/tiny_language_lab/runs/stage55_validation_suite.md` and `.json`
- `experiments/tiny_language_lab/runs/stage55_text8_zero_shot.md` and `.json`
- `experiments/tiny_language_lab/runs/stage55_flagship_200m_b50000_seed7.md`,
  `..._b20000_seed11.md`, `..._b20000_seed19.md`,
  `..._generation_quality.md`
- `experiments/tiny_language_lab/artifacts/phase4/nsight_dld/` (ONNX,
  manifest, sha256)
- `docs/phase4-flagship-evaluation-report.md` and `docs/figures/phase4/`
- `docs/phase4-flagship-midrun-report.md` (retrospective and cleanup plan)
- `docs/decisions/0013-phase-4-free-accelerator-floor-scaling-flagship.md`
- `docs/decisions/0011-phase-3-coherence-checkpoint-and-crossover-scaling-law.md`
- Radford et al. 2018 (GPT-1); Radford et al. 2019 (GPT-2, Table 3);
  Mahoney text8
