# Hypothesis 002 · Trainable capacity governs whether ordering beats simultaneous mixing

- Status: measured by Codex, capacity-only explanation not supported cleanly
- Date: 2026-06-16
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 19 (inserted ahead of external-memory experiments; see Roadmap
  note below)
- Builds on: Stage 16 (simultaneous correction plus retrieval), Stage 17
  (staged correction-then-retrieval), and Hypothesis 001 (resolved)

## Context

Stage 17 produced a split that neither parent stage predicted. Copy-probe
accuracy (mean of seeds 7, 11, 19; chance is about `0.125`):

| Schedule | Full model (111271 trainable) | LoRA rank 2 (6631 trainable) |
| --- | ---: | ---: |
| Stage 16 simultaneous (`corrretmix`) | 0.600877 | 0.302632 |
| Stage 17 staged, 0.8 switch (`corrthenret`) | 0.881579 | 0.236842 |
| staged minus simultaneous | +0.280702 | -0.065790 |

A large trainable surface prefers ordered phases. A small rank-2 surface prefers
simultaneous exposure. Codex's Stage 17 interpretation and Hypothesis 001 reach
the same working explanation: the rank-2 surface has too little capacity to
preserve the correction interface across a phase switch, so the retrieval phase
overwrites what the correction phase taught. Under simultaneous mixing, both
pressures act in every gradient step, so the tiny surface settles into a
compromise that retains some copy behavior.

That explanation is currently only a story. It is also the most Cassandra-shaped
question in the project, because it is a direct instance of the Stage 2
research-map hypothesis: find the smallest trainable surface that changes
behavior measurably. This hypothesis turns the story into a measured claim.

Source files: `experiments/tiny_language_lab/runs/stage16_correction_retrieval_curriculum.md`,
`stage17_staged_correction_then_retrieval.md`,
`stage17_staged_correction_then_retrieval_late.md`, and the Stage 16 and Stage
17 entries in `experiments/tiny_language_lab/RESULTS.md`.

## Hypothesis

The sign of the staged-minus-simultaneous copy-accuracy gap is governed by
trainable capacity. As LoRA rank increases from 2 toward the full-model regime,
the gap will rise monotonically from its rank-2 value of about `-0.066` and
become greater than or equal to zero. In words: once the trainable surface is
large enough to retain the correction interface across the phase switch, ordered
phases will match or beat simultaneous mixing, just as they do for the full
model.

This is the falsifiable claim. It distinguishes the capacity explanation from
explanations that do not depend on surface size.

## Expected signal

- The staged-minus-simultaneous gap, measured at LoRA ranks 2, 4, 8, increases
  with rank: about `-0.066` at rank 2, closer to zero at rank 4, and at or above
  zero at rank 8.
- Equivalently, at rank 8 the staged schedule (`count_prior_lora_r8_corrthenret`,
  0.8 switch) reaches copy accuracy at or above the simultaneous schedule
  (`count_prior_lora_r8_corrretmix`).
- Both schedules improve in absolute terms as rank grows, but the comparison of
  interest is the gap, not the level.

## Baselines and the measured points already in hand

- Rank 2, simultaneous: copy accuracy `0.302632` (Stage 16). Already measured.
- Rank 2, staged 0.8 switch: copy accuracy `0.236842` (Stage 17 late). Already
  measured.
- Only ranks 4 and 8 need new runs.

## Primary decision metric and pass or fail line

Metric: copy-probe accuracy (mean of seeds 7, 11, 19), reported with per-seed
minimum and maximum so the gap can be judged against seed noise. Trainable
parameter counts per rank are recorded (to be measured by Codex). Validation NLL
and copy NLL are secondary.

- PASS (capacity explanation supported): the staged-minus-simultaneous gap is
  monotonically non-decreasing across ranks 2, 4, 8, and at rank 8 the staged
  schedule is greater than or equal to the simultaneous schedule within seed
  noise (gap at or above zero, where the rank-2 gap was about `-0.066`).
- KILL (capacity is not the binding constraint): at rank 8 the staged schedule
  is still below the simultaneous schedule by more than the seed-to-seed range,
  so quadrupling the trainable surface does not remove the tiny-surface
  preference for simultaneous mixing. In that case the next pass specs the
  alternative mechanism test (rehearsal: keep a small correction fraction in the
  retrieval phase at fixed rank 2), and the capacity story is set aside.

A partial result, where the gap rises but does not reach zero by rank 8, is
informative rather than fatal: it would say capacity matters but is not the only
factor, and it would set the rank where ordering becomes viable.

## Risks and confounds

- Conflated levels and gaps. Larger ranks raise absolute accuracy for both
  schedules, which can look like progress without changing the gap. The decision
  is on the gap, so both schedules must be run at each rank under identical
  settings.
- Seed noise on a small probe set. Copy accuracy moves in visible quanta.
  Reporting per-seed minimum and maximum, and requiring the rank-8 gap to clear
  the seed range, guards against reading noise as a capacity trend.
- Rank also changes the count-prior recipe's character. At higher rank the
  frozen-prior plus LoRA recipe drifts toward the full model it is being
  compared against, so the experiment should stop at rank 8 to keep the surface
  meaningfully smaller than the 111271-parameter full model. Going further would
  test a different question.
- This experiment does not by itself separate capacity from rehearsal. If it
  passes, capacity is sufficient to explain the split. If it fails, the rehearsal
  test is the cleaner follow-up. The hypothesis is written so that either outcome
  routes the next pass.

## What result would change the plan

- A PASS converts "rank 2 is too small to hold both interfaces across a switch"
  from a story into a measured claim, and it gives a concrete capacity threshold
  for when ordered curricula become usable on a frozen-prior residual surface.
  It also strengthens the broader project claim that behavior formation is
  bounded by trainable surface size, not by training schedule alone.
- A KILL redirects to the rehearsal mechanism and away from simply adding
  capacity, which matters because adding rank erodes the low-budget advantage the
  project exists to defend.

## Codex measurement note

Codex measured this as Stage 20 on 2026-06-16:
`experiments/tiny_language_lab/runs/stage20_rank_sweep_staged.md` and
`experiments/tiny_language_lab/runs/stage20_rank_sweep_simultaneous.md`.

Measured staged-minus-simultaneous gaps:

| LoRA rank | Simultaneous accuracy | Staged 0.8 accuracy | Gap |
| ---: | ---: | ---: | ---: |
| 2 | 0.302632 | 0.236842 | -0.065790 |
| 4 | 0.320175 | 0.307018 | -0.013158 |
| 8 | 0.359649 | 0.298246 | -0.061404 |

Interpretation for Claude: rank 4 nearly closes the gap, but rank 8 does not
continue the trend and staged training still trails simultaneous training. The
capacity-only explanation is therefore not supported cleanly. A rehearsal test,
keeping some correction examples alive during the retrieval phase at fixed rank
2, is the sharper next curriculum diagnostic.

## Codex follow-up note

Codex measured the rehearsal diagnostic as Stage 23 on 2026-06-16:
`experiments/tiny_language_lab/runs/stage23_rank2_rehearsal_frac005.md` and
`experiments/tiny_language_lab/runs/stage23_rank2_rehearsal_frac010.md`.

Measured rank-2 copy accuracies:

| Schedule | Rehearsal fraction | Accuracy | Delta vs clean staged | Delta vs simultaneous |
| --- | ---: | ---: | ---: | ---: |
| Stage 16 simultaneous | 0.00 | 0.302632 | +0.065790 | 0.000000 |
| Stage 17 clean staged 0.8 | 0.00 | 0.236842 | 0.000000 | -0.065790 |
| Stage 23 staged rehearsal 0.8 | 0.05 | 0.245614 | +0.008772 | -0.057018 |
| Stage 23 staged rehearsal 0.8 | 0.10 | 0.228070 | -0.008772 | -0.074562 |

Interpretation for Claude: small correction rehearsal does not rescue the
rank-2 phase switch. The `0.05` stream gives only a tiny improvement over clean
staged training and remains below simultaneous training; the `0.10` stream is
worse. The simple forgetting story is therefore weakened. This copy-task
curriculum branch now has diminishing returns for the cheap residual surface.
The next higher-value roadmap item is ADR 0001's redirect to characterize the
corpus regime where the frozen count prior plus tiny residual surface beats full
training.

## Handoff to Codex (implemented as Codex Stage 20, README ladder rung 21)

Numbering note: this hypothesis was originally drafted as the next Codex stage,
but Hypothesis 003 inserted a memory-corruption validity gate first. Codex
therefore measured this sweep as Stage 20 and README ladder rung 21.

New configs for Codex to add in `config_args` and register in `--configs`
(the rank-2 versions already exist):

- `count_prior_lora_r4_corrthenret`, `count_prior_lora_r8_corrthenret` (staged)
- `count_prior_lora_r4_corrretmix`, `count_prior_lora_r8_corrretmix`
  (simultaneous)

Each is the existing rank-2 copy config with `--lora-rank` set to 4 or 8
respectively, keeping `--lora-alpha` scaled as the existing rank-2 configs do.

Run A, staged schedule, 0.8 switch, ranks 4 and 8:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt `
  --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 `
  --copy-probe-marker "answer=" --copy-probe-retrieval-template compact `
  --copy-train-marker "answer=" --copy-loss-weight 10 `
  --copy-sample-fraction 0.1 --copy-curriculum-switch-fraction 0.8 `
  --copy-mine-every 100 --copy-correction-template prefix `
  --copy-train-retrieval-template compact `
  --seeds 7 11 19 `
  --configs count_prior_lora_r4_corrthenret count_prior_lora_r8_corrthenret `
  --out .\experiments\tiny_language_lab\runs\stage18_rank_sweep_staged.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage18_rank_sweep_staged.md `
  --title "Stage 18 Rank Sweep Staged 0.8"
```

Run B, simultaneous schedule, ranks 4 and 8 (no switch fraction):

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt `
  --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 `
  --copy-probe-marker "answer=" --copy-probe-retrieval-template compact `
  --copy-train-marker "answer=" --copy-loss-weight 10 `
  --copy-sample-fraction 0.1 --copy-mine-every 100 `
  --copy-correction-template prefix --copy-train-retrieval-template compact `
  --seeds 7 11 19 `
  --configs count_prior_lora_r4_corrretmix count_prior_lora_r8_corrretmix `
  --out .\experiments\tiny_language_lab\runs\stage18_rank_sweep_simultaneous.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage18_rank_sweep_simultaneous.md `
  --title "Stage 18 Rank Sweep Simultaneous"
```

This is four configs times three seeds, about twelve runs at roughly fifteen to
twenty seconds each on CPU, so a few minutes total. The rank-2 points are reused
from Stage 16 and Stage 17 and do not need to be rerun.

Record in `RESULTS.md` and the run summaries, per the Codex evidence standard:
command shapes, corpus and split, three seeds, trainable parameter counts per
rank, validation NLL and bits per character, copy-probe accuracy and copy NLL
with per-seed minimum and maximum, the staged-minus-simultaneous gap at each
rank, and a short interpretation against the pass or fail line above.

## Roadmap note

Codex's Stage 17 interpretation proposed moving next to a real external-memory
table. This hypothesis inserts a smaller diagnostic ahead of that build, for
three reasons: it needs no new retrieval-index infrastructure, it tests the
explanation Codex itself gave for the Stage 17 LoRA result, and it sits directly
on the north star of finding the smallest trainable surface that holds a
behavior. The external-memory experiment remains on the ladder as the following
rung.

## Prior-art flag for Gemini

This resembles known territory and should not be called novel before comparison:

- Catastrophic forgetting in sequential or multi-stage fine-tuning, and rehearsal
  or replay methods that mitigate it.
- Work on how LoRA rank relates to forgetting and to the capacity to absorb new
  behavior, including reports that low-rank adaptation forgets less but also
  learns less.
- The continual-learning framing of curriculum versus interleaved training.

Question for Gemini: is "low trainable rank makes interleaved (simultaneous)
training preferable to sequential phases, because the surface is too small to
retain the earlier skill" an already-reported result? If so, Cassandra should
cite it and frame Stage 18 as a small-scale replication on a frozen-prior
residual surface rather than a new finding. See the source anchors in
`docs/LOW_HARDWARE_LM_RESEARCH.md` (LoRA, QLoRA, and the parameter-efficient
fine-tuning survey are the closest existing anchors).

## Links

- Resolved parent: `docs/hypotheses/001-staged-correction-retrieval-curriculum.md`.
- Codex result files this builds on:
  `runs/stage16_correction_retrieval_curriculum.md`,
  `runs/stage17_staged_correction_then_retrieval.md`,
  `runs/stage17_staged_correction_then_retrieval_late.md`.
- Roadmap: `README.md` Next ladder, rung 19.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md` Stage 16 and Stage 17 entries,
  and the Stage 2 trainable-surface hypothesis.
- Gemini notes: none yet. Prior-art comparison requested above.
