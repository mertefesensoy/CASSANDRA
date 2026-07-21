# Hypothesis 001 · Staged correction-then-retrieval curriculum beats simultaneous mixing

- Status: ANSWERED 2026-06-16 by Stage 17 (see Outcome). Primary LoRA claim
  falsified; full-model diagnostic confirmed; result is a capacity-dependent
  split.
- Date: 2026-06-16
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 18 (staged curriculum and external memory experiments)
- Builds on: Stage 13 (prefix corrections), Stage 14 (probe-time retrieval),
  Stage 15 (retrieval-use training), Stage 16 (simultaneous correction plus
  retrieval-use training)

## Outcome (answered 2026-06-16 by Stage 17)

Codex ran this experiment in parallel with the writing of this hypothesis and
recorded it as Stage 17 in `RESULTS.md`. Codex implemented the switch as
`--copy-curriculum-switch-fraction` (values `0.5` and `0.8`) rather than the
`--copy-curriculum-switch-step` knob proposed below, with configs named
`*_corrthenret` rather than `*_corr_then_ret`, and kept
`--copy-sample-fraction 0.1` from Stage 16 rather than the `0.05` proposed here.
The substance of the test is the same.

Measured copy-probe accuracy (mean of seeds 7, 11, 19):

| Switch | Full model | LoRA rank 2 |
| --- | ---: | ---: |
| Stage 16 simultaneous (baseline) | 0.600877 | 0.302632 |
| Stage 17 staged, 0.5 switch | 0.833333 | 0.236842 |
| Stage 17 staged, 0.8 switch | 0.881579 | 0.236842 |

Verdict, against the pass and kill lines written below:

- Primary decision metric FALSIFIED. The hypothesis predicted staged LoRA copy
  accuracy above `0.302632` and above the baseline per-seed maximum. Instead
  staged LoRA fell to `0.236842` at both switch points, below the Stage 16
  simultaneous mean, with per-seed values `0.25, 0.211, 0.25` and
  `0.211, 0.211, 0.289`. For the constrained surface, simultaneous mixing
  remains better than ordering.
- Diagnostic CONFIRMED. The full model recovered from `0.600877` toward the
  Stage 14 ceiling, and the late switch beat the even switch (`0.881579` versus
  `0.833333`), which confirms the step-budget confound flagged below: correction
  phase length drives full-model accuracy. Staging still did not beat the Stage
  14 correction-plus-probe recipe (`0.960526`).
- The pass and kill lines were underspecified for this case. The kill condition
  required both the LoRA failure and a full-model staying at or below
  `0.600877`. The LoRA half triggered but the full-model half did not, so this
  is neither a clean pass nor a clean kill. The honest reading is that "ordering
  versus simultaneity" is not one phenomenon: it interacts with trainable
  capacity. A large trainable surface prefers ordered phases; a tiny rank-2
  surface prefers simultaneous exposure.

Refined question this raises (carried into Hypothesis 002): why does the
constrained surface prefer simultaneous mixing? The working explanation, shared
by Codex's Stage 17 interpretation, is that rank 2 has too little trainable
surface to preserve the correction interface across a phase switch, so the
retrieval phase overwrites it. If that capacity story is right, raising the LoRA
rank should close and then flip the staged-minus-simultaneous gap. That is the
next test. See `docs/hypotheses/002-lora-capacity-curriculum-interference.md`.

Codex result files: `runs/stage17_staged_correction_then_retrieval.md` (0.5
switch) and `runs/stage17_staged_correction_then_retrieval_late.md` (0.8
switch); `RESULTS.md` Stage 17 entry.

## Context

By Stage 16 the lab has two auxiliary training signals for the long-context
copy task, both layered on the gentle recipe (`--copy-loss-weight 10`,
`--copy-sample-fraction 0.05`, 500 steps, three seeds):

1. Generated prefix correction traces. Mining the current model's failed
   verified copy cases, then synthesizing short `prefix`-template strings that
   restate the rule in place. This is the Stage 13 and Stage 14 signal. It
   teaches the task geometry.
2. Retrieval-use examples. Prepending a compact external memory hint
   (`key=e answer=e`) before the original copy prompt and training through the
   answer. This is the Stage 15 signal. It teaches the model to read an external
   memory.

Measured behavior (copy-probe accuracy, the held-out behavior metric; chance is
about `0.125`):

| Recipe | Config | Full-model copy acc | LoRA copy acc |
| --- | --- | ---: | ---: |
| Stage 14 · prefix correction training, compact retrieval at probe time | `*_copycorrmix` | 0.960526 | 0.241228 |
| Stage 15 · compact retrieval-use training | `*_retmix` | 0.776316 | 0.149123 |
| Stage 16 · simultaneous correction plus retrieval-use training | `*_corrretmix` | 0.600877 | 0.302632 |

Source files: `experiments/tiny_language_lab/runs/stage13_prefix_corrections_500.md`,
`stage14_retrieval_probe_compact.md`, `stage15_retrieval_use_training.md`,
`stage16_correction_retrieval_curriculum.md`, and the Stage 13 to 16 entries in
`experiments/tiny_language_lab/RESULTS.md`.

Stage 16 mixed both signals inside every batch. For the full model that was
worse than either parent. For the constrained frozen-prior rank-2 LoRA surface
it was the best result in the retrieval branch. Stage 16 did not test ordering.

## Two competing explanations for the Stage 16 result

- Explanation A · interference from simultaneity. Putting a correction trace and
  a retrieval-use trace in the same batch makes the gradient pull the model
  toward two different input formats at once, muddying the clean
  prefix-correction signal that reached `0.960526` at Stage 14. If this is the
  cause, separating the two signals in time should recover most of the lost
  full-model accuracy and should help the LoRA surface at least as much as
  simultaneous mixing did.
- Explanation B · non-additivity. The full model already solves the task from
  corrections alone (Stage 14 reached `0.960526`), so any retrieval-use training
  only dilutes a solved behavior, regardless of whether it is mixed or ordered.
  If this is the cause, staging will not lift the full model above its
  correction-only ceiling, and retrieval-use training earns its keep only on the
  constrained LoRA surface that cannot reach that ceiling on its own.

The smallest experiment that separates A from B is a two-phase schedule: train
the correction interface first, then switch to retrieval-use examples. Nothing
else changes.

## Hypothesis

On the long-context copy corpus, under the gentle recipe and a fixed 500-step
budget, an ordered curriculum that trains prefix correction traces in phase one
and then compact retrieval-use examples in phase two will produce higher
copy-probe accuracy than the Stage 16 simultaneous mix `*_corrretmix`, for the
frozen-prior rank-2 LoRA surface.

This is the decision claim. The full-model behavior is a secondary diagnostic
(see the confound note below).

## Expected signal

- Frozen-prior rank-2 LoRA (`count_prior_lora_r2_corr_then_ret`): mean
  copy-probe accuracy across seeds 7, 11, 19 in roughly the `0.33` to `0.40`
  range, above the Stage 16 simultaneous value `0.302632` and above that run's
  per-seed maximum.
- Full model (`random_full_corr_then_ret`): copy-probe accuracy recovering from
  the Stage 16 value `0.600877` toward the Stage 13 and Stage 14 range
  (`0.881579` to `0.960526`). Read as diagnostic only, because phase one gives
  corrections fewer steps than Stage 14 did (see confound).

## Baselines it must beat

- Primary baseline (decides the hypothesis): Stage 16 simultaneous mixing,
  LoRA copy accuracy `0.302632` (`count_prior_lora_r2_corrretmix`).
- Diagnostic baselines (context, not the decision):
  - Stage 16 simultaneous mixing, full model `0.600877`.
  - Stage 14 correction training plus compact probe retrieval, full model
    `0.960526` and LoRA `0.241228`. This is the ceiling the full model is being
    asked to recover.

## Primary decision metric and pass or fail line

Metric: copy-probe accuracy (mean across seeds 7, 11, 19), with per-seed minimum
and maximum reported so the margin can be judged against seed noise. Validation
NLL and copy NLL are recorded as secondary, not decisive.

- PASS (supports Explanation A, ordering matters): mean LoRA copy accuracy for
  `count_prior_lora_r2_corr_then_ret` is greater than `0.302632` and greater
  than the per-seed maximum of the Stage 16 LoRA run, so the gain exceeds
  seed-to-seed noise. A full-model recovery above `0.85` is a strong
  confirmation but is not required for the pass.
- KILL (supports Explanation B, reject staging): mean LoRA copy accuracy is at
  or below `0.302632` within the seed range, and the full model does not exceed
  `0.600877`. In that case ordering does not reduce interference. Retire the
  claim that staging beats simultaneity, demote the retrieval-use branch
  relative to the Stage 14 correction-plus-probe recipe, and pivot the next pass
  to the deeper question of why the LoRA surface saturates near `0.30`.

## Risks and confounds

- Step-budget confound on the full model. Stage 14 reached `0.960526` with 500
  steps of correction training. A 250-step phase one gives corrections only half
  that exposure, so a full-model number below `0.96` does not by itself refute
  Explanation A. This is exactly why the LoRA path, not the full model, is the
  decision metric. An optional confound-control run (switch at step 400, keeping
  correction strong) is listed for Codex if CPU budget allows.
- Probe noise. Copy accuracy is measured on a small probe set, so single-seed
  values move in visible quanta. Reporting per-seed min and max guards against
  reading noise as signal. The pass line already requires the gain to exceed the
  baseline's per-seed maximum.
- Mining schedule across the phase boundary. The failed-case miner
  (`--copy-mine-every 100`) runs during phase one to build corrections. After
  the switch, phase two trains retrieval-use examples and does not need fresh
  mining. Codex should confirm the switch does not leave a stale or empty
  correction buffer driving phase two.

## What result would change the plan

- A clean PASS promotes staged curricula over simultaneous mixing for the
  constrained surface and justifies a follow-up that sweeps the switch point
  (for example 250, 350, 400) to find where the correction phase is long enough
  to transfer while leaving room for the retrieval phase.
- A KILL retires the staging idea and reframes the frontier as a capacity
  question on the LoRA surface, for example whether rank 2 is simply too small
  to hold both the copy rule and the retrieval-read interface, which would point
  to a rank sweep rather than more data-construction tricks.

## Handoff to Codex (Stage 18 spec)

This stage is not pure configuration. It needs a small, well-scoped code
addition so that one training run can switch its copy sampler partway through.
Claude specifies the interface; Codex owns the implementation and the run.

Proposed minimal knobs in `cassandra_tiny_transformer.py` `train(args)` and
`cassandra_compare.py`:

- `--copy-sampler-phase2 NAME` · the sampler to use after the switch. Default
  empty, which preserves current single-phase behavior exactly.
- `--copy-curriculum-switch-step S` · the step at which the batch sampler swaps
  from `--copy-sampler` (phase one) to `--copy-sampler-phase2` (phase two).
  Default unset, meaning no switch.
- Two new named configs in `config_args`, registered in the `--configs` choices
  list:
  - `random_full_corr_then_ret`
  - `count_prior_lora_r2_corr_then_ret`
  Each sets phase-one sampler `correction_mixed` and phase-two sampler
  `retrieval_mixed`, correction template `prefix`, train retrieval template
  `compact`, on top of the existing `random_full` and `count_prior_lora_r2`
  bases.

Required run (primary):

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt `
  --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 `
  --copy-probe-marker "answer=" --copy-probe-retrieval-template compact `
  --copy-train-marker "answer=" --copy-loss-weight 10 `
  --copy-sample-fraction 0.05 --copy-mine-every 100 `
  --copy-correction-template prefix --copy-train-retrieval-template compact `
  --copy-curriculum-switch-step 250 `
  --seeds 7 11 19 `
  --configs random_full_corr_then_ret count_prior_lora_r2_corr_then_ret `
  --out .\experiments\tiny_language_lab\runs\stage18_staged_curriculum.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage18_staged_curriculum.md `
  --title "Stage 18 Staged Correction-then-Retrieval Curriculum"
```

Notes for the run:

- The probe uses compact retrieval, matching Stage 16, so the comparison to the
  Stage 16 simultaneous mix is fair.
- `--copy-sample-fraction 0.05` keeps roughly one auxiliary example per batch in
  whichever phase is active, matching the Stage 13 to 15 gentle dose. Phase one
  spends that budget on corrections; phase two spends it on retrieval-use.
- The switch at step 250 splits the budget evenly. This is the symmetric test of
  ordering versus simultaneity.

Optional confound-control run (only if CPU budget allows): rerun with
`--copy-curriculum-switch-step 400` so correction training keeps 400 steps and
the retrieval phase gets 100. This isolates whether a weaker full-model recovery
is caused by ordering or by the shortened correction phase.

Record in `RESULTS.md` and the run summary, per the Codex evidence standard:
command shape, corpus and split, three seeds, trainable parameter counts,
validation NLL and bits per character, copy-probe accuracy and copy NLL with
per-seed min and max, and a short interpretation against the pass or fail line
above.

## Prior-art flag for Gemini

This proposal resembles several established ideas and should not be called novel
until Gemini compares it to the outside world. Specifically:

- Curriculum learning, where examples or tasks are ordered from simpler to more
  complex rather than mixed uniformly.
- Multi-stage or phased fine-tuning, where a model is trained on one objective
  and then a second, common in instruction tuning and alignment pipelines.
- Retrieval-augmented training, where a model is trained to read retrieved
  context (for example REALM, RETRO, Atlas), and the related question of when
  behavior is better stored in weights versus supplied in context.

The question for Gemini: is "teach the retrieval-read interface in a separate
phase, after the task behavior, to avoid interference on a small trainable
surface" an already-reported result in retrieval-augmented training or
parameter-efficient adaptation? If so, Cassandra should cite it and frame
Stage 18 as a small-scale replication rather than a new finding. See the source
anchors in `docs/LOW_HARDWARE_LM_RESEARCH.md`.

## Links

- Codex result files this builds on: `runs/stage13_prefix_corrections_500.md`,
  `runs/stage14_retrieval_probe_compact.md`,
  `runs/stage15_retrieval_use_training.md`,
  `runs/stage16_correction_retrieval_curriculum.md`.
- Roadmap: `README.md` Next ladder, rung 18.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md` Stage 16 entry and source
  anchors.
- Gemini notes: none yet. Prior-art comparison requested above.
