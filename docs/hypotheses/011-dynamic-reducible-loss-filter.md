# Hypothesis 011 · A dynamic reducible-loss filter is the decisive final test of data-side selection for the rank-2 residual

- Status: killed by Stage 34. This was the last data-side selection attempt before
  retiring the branch for the frozen-prior rank-2 residual.
- Date: 2026-06-23
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 37 (data-side curriculum, final attempt)
- Builds on: Stage 33 (Hypothesis 010 kill, the static prior-loss filter), Stage 11
  and Stage 12 (replay and verifier sampling), and Gemini research notes
  `research/theme_3_training_dynamics_and_curriculums/13_data_selection_and_reducible_loss.md`
  and `research/theme_3_training_dynamics_and_curriculums/14_static_vs_iterative_reducible_loss.md`

## Context: why H010 died, and the two competing explanations

Stage 33 killed the static prior-loss filter on its primary steps-to-target metric.
The shape of the kill matters. The mixed filters did essentially nothing: at
`f = 0.25` the validation NLL delta versus uniform was about `+0.0005` to `+0.001`,
and at `f = 0.50` about `+0.002`, both far below seed spread. Only the pure
`f = 1.0` control was clearly worse, about `+0.015`. So oversampling
high-prior-loss windows ranged from inert to harmful.

Two explanations compete, and they predict different futures:

1. Wrong proxy (Gemini note 14). A frozen prior's loss is a static proxy, and the
   highest-static-loss windows are largely irreducible noise: rare characters,
   formatting, chaotic spans the residual can never learn. Oversampling them wastes
   the residual's tiny capacity. The fix is a dynamic reducible-loss signal that
   selects windows the residual is actively learning from, not windows that are
   permanently hard.
2. Capacity bottleneck. A rank-2 residual has about `6,241` trainable parameters and
   fits its best low-rank correction almost regardless of which windows it sees, so
   no sampling scheme helps. The near-zero effect of the mixed filters, not just the
   harm from pure selection, supports this: a pure noise problem would still leave
   the mixed arms slightly helpful, and they were not.

Hypothesis 011 is the decisive test between the two. It implements the best version
of data-side selection, the dynamic reducible-loss filter that note 14 prescribes.
If even that fails, the proxy was not the bottleneck, the failure is attributed to
capacity, and the data-side selection branch is retired.

## Hypothesis

A dynamic reducible-loss filter, which every `K` steps re-scores a fixed candidate
window pool by the live model's NLL and its recent decrease, and oversamples windows
that are hard but actively improving (high current NLL and a positive recent loss
delta), mixed with uniform sampling, lets the rank-2 residual reach the uniform
baseline's target validation NLL in materially fewer optimizer steps, at equal
compute including the re-scoring cost, on the natural-text corpus.

This is the falsifiable claim. It is killed if no mixing fraction reaches the uniform
target faster than a strict uniform baseline within seed noise, at equal wall-clock.
A kill, with the Stage 11, 12, and 33 history, retires data-side selection for the
frozen-prior rank-2 residual and attributes the failure to capacity, not proxy.

## The dynamic reducible-loss score, the one change over H010

H010 scored windows by the static frozen-prior NLL. That number conflates two very
different windows: ones where the prior is wrong but the residual can learn the
structure, and ones where the prior is wrong because the text is noise the residual
also cannot learn. The frozen prior cannot tell them apart, which is exactly why
H010 oversampled noise.

The dynamic signal uses the live residual's loss trajectory to separate them, the
reducible-versus-irreducible decomposition from note 14. A window whose loss is
dropping as training proceeds is reducible and learnable; a window whose loss stays
high despite training is irreducible noise. Concretely, the frozen prior is the
reference and the live residual provides the trajectory:

- Every `K` steps, forward-pass the live model, prior plus residual, with no
  gradient, over a fixed candidate pool, and record each window's mean per-token
  NLL.
- For each window compute the recent delta, previous NLL minus current NLL. A
  positive delta means the residual is actively reducing loss there.
- Score a window high only if its current NLL is high and its delta is positive
  beyond a small threshold. Windows that are high-loss but stuck, delta near zero,
  are treated as irreducible noise and deprioritized.

This is dynamic loss-based data selection in the standard sense, the unsupervised,
frozen-prior-referenced analog of reducible holdout loss selection.

## The control: a strict uniform-sampling baseline at equal compute

The comparison is against uniform random window sampling with the identical model,
the same frozen order-2 prior, the same corpus, the same seeds, and the same total
compute. Equal compute now binds harder than in H010, because the dynamic filter
adds periodic re-scoring forward passes. Report wall-clock as well as steps. A filter
that reaches the target in fewer steps but more wall-clock has not won; that outcome
is a documented partial, not a pass.

## Scope, strictly bounded

The frozen prior order stays at 2, the residual stays rank 2, the corpus stays
`natural_text_seed.txt`, and only the batch sampler and its scoring change. No order
sweep, no rank sweep, no architecture change. The model-side frozen time-series base
is a separate later hypothesis and is out of scope here; it needs its own Gemini
specification note before it can be drafted, because the primitive is not yet
defined.

## Concrete mechanic for Codex

- Candidate pool: sample a fixed pool of `P` legal training windows once,
  deterministically per seed, for example `P = 4096`, so windows are re-identifiable
  across re-scorings.
- Re-score interval: `--curriculum-rescore-every K`, default `K = 25`. At step 0 the
  delta is zero for all windows, so the first selection falls back to uniform.
- Score and select: high-score windows are those with current NLL in the top decile
  and a positive smoothed delta; each batch draws fraction `f` from that set and
  `1 - f` uniform. Test `f = 0.25` and `f = 0.50`. The pure `f = 1.0` control is not
  needed; H010 already showed pure selection harms.
- Determinism under seeds 7 11 19.

## Expected signal and decision metric

Learning curves, validation NLL versus steps, for uniform versus dynamic-filtered at
`f = 0.25` and `f = 0.50`, on the same corpus, order-2 prior, rank-2 residual, and
seeds, plus wall-clock per arm. The decisive number is steps-to-target against the
uniform 200-step mean NLL, read together with wall-clock.

- PASS: a dynamic arm reaches the uniform 200-step target in clearly fewer steps,
  beyond seed noise, without more wall-clock. Dynamic reducible-loss selection then
  helps the tiny residual, unlike the static proxy.
- PARTIAL: a dynamic arm reaches the target in fewer steps but the re-scoring
  overhead erases the win in wall-clock, or it lowers final NLL slightly without a
  steps win. The filter works but is not worth its cost.
- KILL: no dynamic arm reaches the target faster than uniform within seed noise. The
  proxy was not the problem, the rank-2 residual is data-order insensitive, and the
  data-side selection branch is retired with a capacity explanation.

## Risks and confounds

- Capacity is the leading alternative explanation. If H011 kills, do not try a fifth
  data-side variant. Read the kill as a capacity result and pivot, either to a richer
  frozen base, the model-side direction, or a larger residual rank, which is a
  different question.
- Re-scoring overhead. Equal compute must be wall-clock, not only steps, because the
  dynamic filter is not free.
- Delta noise. Per-window NLL deltas on a tiny model are noisy. Use a small
  smoothing window and a threshold so the filter does not chase delta noise.
- Pool staleness. The fixed candidate pool must cover the corpus well enough to be
  representative; report the pool size and its coverage of the training split.

## What result would change the plan

- A PASS keeps data-side selection alive as a dynamic reducible-loss filter and
  motivates a scaled follow-up.
- A KILL retires the data-side selection branch. Four consistent negatives, Stage 11,
  Stage 12, Stage 33, and Stage 34, with a clear mechanism, justify a short ADR that
  retires data-side curriculum for the frozen-prior rank-2 residual and records the
  capacity attribution. The roadmap then turns to the model-side frozen-base
  direction, which first needs a Gemini note defining the time-series or long-range
  primitive, and to the long-deferred non-gradient residual-formation branch.

## Handoff to Codex (next stage; Codex stage number 34, README ladder rung 37)

Files to modify:

1. `experiments/tiny_language_lab/cassandra_tiny_transformer.py`: extend the existing
   `--curriculum-filter` with a `dynamic-reducible` mode and add
   `--curriculum-rescore-every K` and `--curriculum-pool-size P`. Reuse the H010
   mixing path; the only new logic is the periodic live-model re-scoring over the
   fixed pool, the per-window delta, and the high-NLL-and-positive-delta selection.
   No gradients in scoring.
2. `experiments/tiny_language_lab/cassandra_compare.py`: add configs
   `count_prior_ng2_lora_r2_dynfilter_f025` and `count_prior_ng2_lora_r2_dynfilter_f050`,
   register them in `--configs`.

Runs, all on `corpus/natural_text_seed.txt`, order-2 prior, rank-2 residual,
`--block-size 96 --eval-mode sampled --eval-batches 16 --count-alpha 0.1
--ngram-backoff 10 --curriculum-rescore-every 25 --seeds 7 11 19`, at budgets
`50, 100, 200, 500`:

- Arm A, uniform baseline: `count_prior_ng2_lora_r2`, filter off.
- Arm B, dynamic filter: `count_prior_ng2_lora_r2_dynfilter_f050`.
- Arm C, dynamic filter: `count_prior_ng2_lora_r2_dynfilter_f025`.

Smoke-test the dynamic sampler before the full matrix. Record in `RESULTS.md` and the
run summaries to the Codex evidence standard: the per-arm learning curves, the
steps-to-target table versus Arm A, the final NLL at 500 steps, and wall-clock per
arm so the equal-compute comparison is honest. State the result against the pass and
kill lines.

## Codex result (Stage 34)

Codex implemented `--curriculum-filter dynamic-reducible`, deterministic fixed
pools, periodic no-grad live-model scoring, smoothed per-window loss deltas, and
runner configs `count_prior_ng2_lora_r2_dynfilter_f050` and
`count_prior_ng2_lora_r2_dynfilter_f025`.

Result: KILL. The uniform 200-step target was `2.040189` mean validation NLL.
Dynamic `f=0.50` did not reach the target at any measured budget and was worse
than uniform at every matched budget. Dynamic `f=0.25` was closer but also did
not reach the target, including at 500 steps. At 500 steps, uniform was
`2.050701`, dynamic `f=0.50` was `2.060916`, and dynamic `f=0.25` was
`2.053026`. The dynamic arms also used more wall-clock because pool re-scoring
added repeated forward passes.

Evidence: `experiments/tiny_language_lab/runs/stage34_dynfilter_summary.md` and
`experiments/tiny_language_lab/runs/stage34_dynfilter_summary.jsonl`.

Decision: the data-side selection branch is retired for this ladder. The local
explanation is capacity and data-order insensitivity in the rank-2 residual, not
merely a bad static proxy.
## Prior-art, resolved by Gemini notes 13 and 14

The novelty gate is already cleared. Notes 13 and 14 locate this against reducible
versus irreducible loss, dynamic and iterative data scoring, and reducible holdout
loss selection. Use those terms. A positive is a local instance of dynamic
reducible-loss selection accelerating a frozen-prior residual, not a new algorithm.

## Links

- Killed parent: `docs/hypotheses/010-akba-curriculum-filter.md`, Stage 33.
- Gemini notes: `research/theme_3_training_dynamics_and_curriculums/13_data_selection_and_reducible_loss.md`,
  `research/theme_3_training_dynamics_and_curriculums/14_static_vs_iterative_reducible_loss.md`.
- Skeptical prior: `RESULTS.md` Stages 11, 12, and 33.
- Natural-text baseline: `RESULTS.md` Stage 30; `runs/stage30_naturaltext_summary.md`.
- Roadmap: `README.md` Next ladder, rung 37.
