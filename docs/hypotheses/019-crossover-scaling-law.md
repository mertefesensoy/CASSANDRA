# Hypothesis 019 · The early-compute crossover budget moves with model capacity

- Status: GRADED. Implemented by Codex Stage 52 on 2026-07-02 after Stage 51's
  corpus rescale and checkpoint run. Crossovers were `1000`, `1000`, `1000`,
  and `500` steps for 3.2M, 10.67M, 25.25M, and 85.11M respectively, so the
  result is mixed rather than clean E1, E2, or E3.
- Date: 2026-07-01
- Author: Claude (hypothesis and roadmap role)
- Builds on: ADR 0010 D5 (registered this as candidate `H019`, unformalized,
  `docs/decisions/0010-phase-2-from-scratch-model-build.md`), Stage 45
  (`experiments/tiny_language_lab/runs/phase2_tinystories_modern_b500.md`,
  3.2M-param 500-step crossover not yet reached, prior leads by `0.042194`
  NLL), Stage 48 (`phase2_tinystories_modern_b1000.md`, 3.2M-param 1000-step
  crossover reached, full model leads by `0.070602` mean NLL, positive on all
  three seeds), ADR 0011 (Phase 3 scope, `docs/decisions/0011-phase-3-coherence-checkpoint-and-crossover-scaling-law.md`).

## Why this, and why now

Phase 1 located the crossover, the step count at which full gradient training
catches and passes a frozen-prior head start, near 100 steps at tiny
synthetic char-level scale (`V = 33`, small transformer). Stage 48 relocated
it to somewhere between 500 and 1000 steps at 3.2M params on real TinyStories
text. That is a five-to-tenfold shift, but it conflates two things that moved
at once: the corpus (synthetic to real natural text) and, implicitly, the
training setup (plain AdamW baseline to RoPE plus Muon plus gradient
accumulation plus activation checkpointing). Model capacity itself has never
been varied while holding the corpus and training recipe fixed. The Phase 3
feasibility probe (`docs/phase3-intake.md`) measured that 10.67M, 25.25M, and
even 85.11M-param configs fit the 8 GB card with large headroom (under 1.2 GB
peak VRAM at the largest), so a capacity ladder is now cheap to run. This is
the natural next link in the crossover story ADR 0010 opened, and it is
squarely on the north star: does the frozen-prior recipe's early-compute
advantage change shape as Cassandra's own from-scratch builds get bigger.

## The mechanism (or competing explanations)

Two explanations predict opposite directions for how the crossover step count
moves with model size, holding corpus and recipe fixed:

- E1, capacity speeds catch-up. A bigger model has more expressive gradient
  updates per step, so it needs fewer optimizer steps to match and then beat
  a fixed frozen-prior head start. Prediction: crossover step count decreases
  as model size increases.
- E2, capacity slows convergence. A bigger model has more parameters to move
  away from random initialization, so in step-count terms, not wall-clock
  terms, it takes longer to organize itself past a head start that costs the
  frozen-prior arm nothing to hold. Prediction: crossover step count increases
  as model size increases.
- E3, capacity-invariant crossover. The crossover step count stays within one
  step-budget rung of the 3.2M value across the tested sizes. The Phase
  1-to-Phase 2 shift was a corpus and recipe effect, not a capacity effect,
  and ADR 0010 D5's capacity clause should be scoped down accordingly.

## Hypothesis

The measured crossover step, the smallest tested step budget at which
`random_full`'s mean validation NLL is at or below the frozen order-4-prior
arm's (`count_prior_ng4_lora_r2`) mean validation NLL, is a monotonic function
of trainable parameter count across 3.2M, 10.67M, 25.25M, and, if D5's
confirmation smoke clears it, 85.11M params, on a fixed, rescaled TinyStories
character corpus. E1 is confirmed if the crossover step decreases with size
across the three larger points. E2 is confirmed if it increases. E3, the kill
condition for the capacity clause, holds if every tested size's crossover
falls within the same 500-to-1000-step bracket Stage 48 already measured at
3.2M, with no consistent trend in either direction.

## The reference points

Two arms per size, matching the Stage 45 and 48 pair exactly so the size
ladder is the only new axis:

- `random_full`, full gradient training, no frozen prior, at each size.
- `count_prior_ng4_lora_r2`, the frozen order-4 count prior plus rank-2 LoRA
  residual, at each size (the LoRA wraps the same base transformer regardless
  of `n_embd`, so its trainable parameter count stays small at every size,
  matching Stage 45's `41,249` at 3.2M).

Sizes: 3.2M (`n_layer=4 n_head=4 n_embd=256`, Stage 45 and 48's existing
measurements, remeasure only if ADR 0011's open question 4 requires it after
the corpus rescale), 10.67M (`n_layer=6 n_head=6 n_embd=384`), 25.25M
(`n_layer=8 n_head=8 n_embd=512`), and 85.11M (`n_layer=12 n_head=12
n_embd=768`, conditional on ADR 0011 D5's confirmation smoke).

Step-budget ladder: 200, 500, 1000, and 2000 steps, bracketing Stage 48's
observed 500-to-1000 crossover at 3.2M with room on both sides for the ladder
to move.

Seeds: `7 11 19`, matching project convention.

Corpus: the ADR 0011 D2 rescaled `tinystories_char_seed.txt`, not the current
10.08M-char slice, so a bigger model is not corpus-starved in a way that would
be misread as a capacity effect.

## Primary decision metric and pass or fail line

Primary metric: crossover step per size, the smallest ladder budget where
`random_full` mean val NLL (three-seed mean) is at or below
`count_prior_ng4_lora_r2` mean val NLL.

- CONFIRM E1: crossover step strictly decreases from 10.67M to 25.25M to
  85.11M (or to whichever sizes clear D5), each step at least one ladder rung
  below the previous size's.
- CONFIRM E2: crossover step strictly increases across the same size sequence,
  each at least one ladder rung above.
- E3 (capacity-invariant, scopes down ADR 0010 D5): every size's crossover
  step lands in the same 500-to-1000 bracket as the 3.2M baseline, with no
  monotonic trend across sizes.
- GRADED: a mixed, non-monotonic pattern that is neither a clean trend nor a
  flat bracket. Report the per-size numbers plainly and treat this as
  evidence that the effect is real but not cleanly characterized by size
  alone; do not force it into E1, E2, or E3.

## Risks and confounds

Seed variance: three seeds per cell is the project standard, not exhaustive;
a crossover that flips ladder rungs on one seed but not the others should be
reported as such, not smoothed into a mean. Step-budget granularity: the
ladder can only bracket the true crossover to within one rung, matching how
Stage 48 bracketed the 3.2M crossover between 500 and 1000 rather than pinning
an exact step. Corpus confound: this is why D2's corpus rescale must land
before this matrix runs, not after; running H019 on the current 10.08M-char
slice would let a bigger model's relative data-starvation masquerade as a
capacity effect. Hardware confound at the top size: the 85.11M point is only
included if ADR 0011 D5's longer confirmation smoke clears; a 5-step smoke is
feasibility evidence, not proof against memory growth over hundreds of real
steps. Wall-clock cost: four sizes times four budgets times two arms times
three seeds is 96 runs; estimate total wall-clock from the measured per-size,
per-step timings before launching the full matrix, and consider dropping the
lowest-value cells (for example the 200-step rung once 500 and 1000 are read)
if the budget does not fit the pre-internship or in-window schedule.

## What result would change the plan

CONFIRM E1 argues that, at this hardware ceiling, spending compute on model
size pays off faster than engineering a better frozen prior, since bigger
builds need less of a head start; this would deprioritize further prior-order
or prior-architecture work in favor of the largest model the 8 GB card (or a
future upgrade) can hold. CONFIRM E2 argues the frozen prior's practical value
grows, not shrinks, as builds scale, which would motivate carrying the prior
into any future BPE or larger-vocabulary work rather than treating it as a
small-model curiosity. E3 scopes ADR 0010 D5 down: record that the
Phase-1-to-Phase-2 crossover shift was a corpus and recipe effect, close the
capacity axis of D5 as tested in this range, and redirect any further
crossover work at corpus complexity instead of model size.

## Result

Stage 52 completed all decision cells on the ADR 0011 rescaled corpus
(`494,094,421` normalized characters, `419,980,257` train characters,
`74,114,164` validation characters, `V = 33`). The first non-sharded prior pass
OOMed in prior construction and is preserved as failed evidence. Codex added a
shard-native order-4 prior builder, verified a small equivalence check at
`max_abs_diff = 0.0`, and used the corrected `_sharded` prior rows for the
decision.

Mean crossovers:

| Size | Crossover budget | Pattern |
| --- | ---: | --- |
| 3.2M | 1000 | prior wins at 200 and 500; full wins at 1000 and 2000 |
| 10.67M | 1000 | prior wins at 200 and 500; full wins at 1000 and 2000 |
| 25.25M | 1000 | mean prior narrowly wins at 500; full wins at 1000 and 2000 |
| 85.11M | 500 | prior wins at 200; full wins at 500, 1000, and 2000 |

Decision: `GRADED`. The result is not E1 because the first three sizes do not
strictly decrease. It is not E2 because the largest model crosses earlier. It
is not E3 because the 85M point leaves the smaller sizes' discrete crossover
rung. The local read is a top-end capacity effect, not a smooth monotonic law.

Interruption audit: every Stage 52 decision cell has three successful rows with
the requested `steps`, `formation_steps`, and `--eval-mode sampled`. Two
corrected-prior rows have inflated `seconds` values due to laptop pause or
sleep during the visible run, but their logs reach final step, eval, JSONL
write, markdown write, and exit code `0`. Use their NLLs for the crossover
decision, but do not use those two `seconds` fields for throughput estimates.

## Handoff to Codex (implemented as Codex Stage 52, after Stage 51's D2 to D3
corpus rescale and checkpoint run)

Files: no new code expected. `cassandra_compare.py` and
`cassandra_tiny_transformer.py` already carry every flag this matrix needs
(`--n-layer`, `--n-head`, `--n-embd`, `--pos-encoding rope`,
`--activation-checkpoint`, `--optimizer muon`, `--grad-accum-steps`,
`--residual-base count-ngram --prior-order 4`, `--train-scope lora
--lora-rank 2`), per the ADR 0010 confirmation report's changes list. If a
per-size sweep needs a config-name convention in `cassandra_compare.py`
(for example `count_prior_ng4_lora_r2_10m`, `_25m`, `_85m`), add those
branches following the existing `count_prior_ng4_lora_r2` pattern, varying
only `n_layer`, `n_head`, `n_embd`.

Command shape, one size and budget cell (repeat across the size and budget
ladders, `--eval-mode sampled` is required, the default `full` mode will
stall on this corpus's validation split):

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\tinystories_char_seed.txt `
  --device cuda --steps 1000 --seeds 7 11 19 `
  --configs random_full_25m count_prior_ng4_lora_r2_25m `
  --n-layer 8 --n-head 8 --n-embd 512 --block-size 128 --batch-size 8 `
  --grad-accum-steps 2 --pos-encoding rope --activation-checkpoint `
  --optimizer muon --muon-lr 0.01 --eval-mode sampled --eval-batches 16 `
  --out .\experiments\tiny_language_lab\runs\stage52_crossover_25m_b1000.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage52_crossover_25m_b1000.md `
  --title "Stage 52 Crossover 25M 1000-step"
```

Metric that decides: per-size, per-budget mean val NLL for both arms, read
off into the crossover step per size as defined above. Restated pass or fail
line: E1 if crossover step strictly decreases across sizes, E2 if it strictly
increases, E3 if every size stays in the 500-to-1000 bracket, GRADED
otherwise, reported plainly rather than forced into a category.

## Prior-art flag for Gemini

This resembles compute-optimal and step-budget scaling-law literature
(Chinchilla-style tradeoffs between model size and training steps) and
warm-start or curriculum-acceleration work on how an initialization's benefit
changes with model scale. Gemini should check whether existing scaling-law
work already characterizes how an analytic or distilled warm start's
step-count advantage scales with parameter count, since if that direction is
established elsewhere, H019's contribution is confirming it in Cassandra's
specific frozen-count-prior setting rather than an open question in general.

## Links

- `docs/decisions/0010-phase-2-from-scratch-model-build.md`
- `docs/decisions/0010-confirmation-report.md`
- `docs/decisions/0011-phase-3-coherence-checkpoint-and-crossover-scaling-law.md`
- `docs/phase3-intake.md`
- `experiments/tiny_language_lab/runs/phase2_tinystories_modern_b500.md`
- `experiments/tiny_language_lab/runs/phase2_tinystories_modern_b1000.md`
- `experiments/tiny_language_lab/RESULTS.md` (Stages 44, 45, 48)
