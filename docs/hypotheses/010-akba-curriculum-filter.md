# Hypothesis 010 · A mixed loss-based data-selection filter speeds rank-2 residual convergence on natural text

- Status: measured by Codex as Stage 33; KILL on the primary steps-to-target metric
- Date: 2026-06-22
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 36 (the data-side curriculum branch)
- Builds on: Stage 30 (natural-text finite-order priors), Stage 11 and Stage 12 (the
  verifier-guided replay and sampling results, the skeptical prior), and Gemini
  research notes
  `research/theme_3_training_dynamics_and_curriculums/12_akba_feature_selection_and_time_series.md`
  and `research/theme_3_training_dynamics_and_curriculums/13_data_selection_and_reducible_loss.md`

## Scope, strictly bounded

This hypothesis tests one thing: whether a mixed loss-based data-selection filter,
which scores training windows by the frozen prior's per-token NLL and oversamples the
high-loss ones, lets the rank-2 residual reach a target validation NLL in fewer
optimizer steps than a strict uniform-sampling baseline at equal compute. It does not
sweep prior orders, does not change the architecture, and changes nothing but the
batch sampler. The frozen prior order and the rank-2 residual are held fixed. The
iterative live-model re-scoring and any model-side time-series variant are explicitly
out of scope and noted only as later follow-ups.

## Framing: this is loss-based data selection and hard example mining

Gemini note 13 names the method in standard terms, which Cassandra should use. Scoring
training windows by the frozen prior's loss and oversampling the high-loss ones is
**hard example mining**: concentrate the optimizer on the windows the current
predictor finds hard. It is an unsupervised form of **loss-based data selection** and
**online batch selection**, the unsupervised analog of active learning, with the
frozen prior acting as the reference model that says which windows are already easy.
A window where the frozen prior is accurate has little residual to learn; a window
where the prior is wrong is where the rank-2 residual must work.

One caveat from the reducible-loss literature, carried in note 13, defines the risk.
Raw high-loss selection is dangerous because the highest-loss windows are often
**irreducible noise**, rare characters, formatting, or idiosyncratic spans, not
learnable structure. Modern data pruning prefers *reducible* loss, the gap between a
reference model and a trained model, precisely to avoid chasing noise. Cassandra's
frozen-prior NLL is a cheap **proxy** for that hardness, not true reducible loss: it
marks where the analytic baseline is wrong, which is where the residual could add
value, but it cannot separate learnable residual structure from noise. That limitation
is the reason for the mixed design and the kill line below.

## The skeptical prior, from Cassandra's own history

This idea is close to experiments the lab already ran, and they were not wins for the
small surface:

- Stage 11 failed-case replay selected windows the current model got wrong and
  replayed them. Pure failed-only replay hurt: it raised copy accuracy slightly but
  damaged validation NLL and did not beat static verified sampling. Mixed replay was
  less destructive but still trailed simpler static sampling.
- Stage 12 showed that pure selection off the training distribution fails, while a
  small mixed dose helped.

Gemini note 13 independently reaches the same conclusion from the literature: hard
example mining needs a regularizing uniform mix to prevent instability. So the honest
expectation is skeptical, the filter must be mixed not pure, and the comparison must be
a strict uniform baseline at equal compute.

## Hypothesis

A mixed loss-based filter, reserving a fraction `f` of each batch for windows with the
highest frozen-prior NLL and filling the remainder by uniform sampling, lets the
rank-2 residual reach a target validation NLL in materially fewer optimizer steps than
a strict uniform-sampling baseline at equal compute, on the natural-text corpus, with
the frozen prior order held fixed. Any gain comes from spending the residual's tiny
capacity on windows the frozen prior cannot already predict.

This is the falsifiable claim. It is killed if the mixed filter does not reach the
uniform baseline's target NLL in fewer steps within seed noise, or if it is worse,
which would repeat the Stage 11 pure-replay degradation and show that loss-based
selection does not help the tiny residual on natural text either.

## The control: a strict uniform-sampling baseline at equal compute

The only valid comparison is against uniform random window sampling, run with the
identical model, the same frozen prior, the same corpus, the same seeds, and the same
total compute. The baseline is not `random_full`; it is the same order-2 prior plus
rank-2 residual config with the filter off. Equal compute means the filter's
window-scoring cost is counted: report wall-clock as well as steps. A steps-to-target
win that is erased by scoring overhead is not a win.

## Which frozen prior order to filter on

Stage 32, Hypothesis 009b, did not locate a moderate sweet spot. Under the cross-domain
split, high-order validation-hit coverage dropped only modestly, order 4 from `0.961`
to `0.890`, and order 4 remained the strongest prior, a local kill of the humped-curve
prediction. So there is no sweet-spot order to inherit.

For this filter, fix the prior order at **order 2**. The reason is the filter, not raw
accuracy: order 2 leaves the residual the most headroom and the clearest loss signal to
select on, while a near-perfect high-order prior, order 3 or 4, predicts most windows
well and gives the filter almost nothing to choose between. Order 3 is the documented
alternative only if order 2's residual headroom proves too small to show any effect. Do
not sweep orders in this hypothesis.

## Concrete mechanic, bounded

- Window score: the frozen order-2 prior's mean per-token NLL on each candidate window.
  This uses only the frozen count table that already backs the residual, no gradients,
  so it is cheap. Compute it once over the training split.
- Mixed selection: each batch is fraction `f` high-score windows, drawn from the top of
  the score distribution, plus `1 - f` uniform random windows. Test `f = 0.5` and
  `f = 0.25`, mirroring the mixed samplers that beat pure ones in Stages 9 and 12.
- Negative control: pure selection `f = 1.0`, the Stage 11 trap, reported only to
  confirm that pure selection is unstable.
- Out of scope, do not build now: iterative re-scoring with the live model, and any
  model-side time-series primitive. These are later follow-ups.

## Expected signal

- Learning curves, validation NLL versus steps, for the uniform baseline versus
  mixed-filtered at `f = 0.5` and `f = 0.25`, on the same corpus, same order-2 prior,
  same rank-2 residual, same seeds.
- The decisive number is steps-to-target: take the validation NLL the uniform baseline
  reaches at a reference budget, its 200-step value, and report how many steps each
  filtered arm needs to reach it. A pass is a clear reduction beyond seed noise, at
  equal compute.

## Baselines and points already in hand

- The uniform-sampling order-2 natural-text learning curve from Stage 30,
  `count_prior_ng2_lora_r2` at budgets 10 to 500, is the curve to beat. At 500 steps it
  held `+0.110` NLL over `random_full`, so the prior is useful but far from
  saturating, which is the headroom the filter needs.
- Stage 11 and Stage 12 are the skeptical prior: selection helped little or hurt on the
  copy task.

## Primary decision metric and pass or fail line

Metric: steps-to-target validation NLL for each mixed-filtered arm versus the strict
uniform baseline, mean over seeds 7, 11, 19 with per-seed spread, plus the final
validation NLL at a matched step budget, plus wall-clock. The trainable surface, the
frozen prior order, and the corpus are identical across arms; only the batch sampler
changes.

- PASS: a mixed-filtered arm reaches the uniform baseline's target NLL in clearly fewer
  steps, beyond seed noise, at one or both fractions, without a worse final NLL and
  without the gain being erased by scoring wall-clock. Loss-based selection then speeds
  residual convergence on natural text.
- PARTIAL: a filtered arm reaches a slightly lower final NLL at the same budget but not
  in fewer steps, or it helps only at one fraction. A real but narrow effect.
- KILL: no fraction reaches the target faster than uniform within seed noise, or the
  filter is worse. Loss-based window selection does not help the tiny residual on
  natural text, consistent with the Stage 11 copy-task result.

## Risks and confounds

- Irreducible noise, the reducible-loss risk. The highest frozen-prior-loss windows may
  be noise, not learnable structure. If the filter overfits to them, validation NLL
  worsens, which the kill line catches. The mixed remainder is the regularizing guard.
- Pure selection is the Stage 11 failure mode. Treat `f = 1.0` as a negative control,
  not the headline.
- Equal-compute accounting. Scoring windows costs time. Report wall-clock and steps so a
  steps-to-target win is not an artifact of ignoring scoring overhead.
- Circular evaluation. Evaluate on the standard held-out validation set, never on the
  filtered training distribution.

## What result would change the plan

- A PASS gives the project a second natural-text positive, a cheap loss-based selection
  rule that speeds the frozen-prior residual, framed as a local instance of known data
  selection. It would motivate a later, separate hypothesis on iterative reducible-loss
  scoring.
- A KILL closes the data-side curriculum filter, consistent with Stage 11. The Akba
  branch would then survive only through its model-side option, a frozen time-series
  primitive in place of the count table, a separate later hypothesis.

## Original handoff to Codex, now measured as Stage 33

Files to modify:

1. `experiments/tiny_language_lab/cassandra_tiny_transformer.py`: add the mixed
   loss-based sampler. Add `--curriculum-filter {off,prior-loss}` (default `off`,
   preserving current behavior) and `--curriculum-fraction f`. When `prior-loss`,
   precompute each training window's mean per-token NLL under the frozen prior that
   already backs the residual, once, then each training batch draws `round(f * B)`
   windows from the top-scoring pool and `B - round(f * B)` uniform windows. Keep the
   draws deterministic under seeds 7 11 19.
2. `experiments/tiny_language_lab/cassandra_compare.py`: pass the two flags through, and
   add a named config or config suffix so the filtered order-2 arm is selectable, for
   example `count_prior_ng2_lora_r2_filter`. Register it in the `--configs` choices.

Runs, all on `corpus/natural_text_seed.txt`, order-2 prior, rank-2 residual,
`--block-size 96 --eval-mode sampled --eval-batches 16 --count-alpha 0.1
--ngram-backoff 10 --seeds 7 11 19`, at budgets `50, 100, 200, 500`:

- Arm A, uniform baseline: `count_prior_ng2_lora_r2`, `--curriculum-filter off`.
- Arm B, mixed filter: `count_prior_ng2_lora_r2` filter config,
  `--curriculum-filter prior-loss --curriculum-fraction 0.5`.
- Arm C, mixed filter: same with `--curriculum-fraction 0.25`.
- Arm D, negative control: same with `--curriculum-fraction 1.0`.

Example command, one arm and budget:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\natural_text_seed.txt `
  --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 `
  --count-alpha 0.1 --ngram-backoff 10 `
  --curriculum-filter prior-loss --curriculum-fraction 0.5 `
  --seeds 7 11 19 --configs count_prior_ng2_lora_r2_filter `
  --out .\experiments\tiny_language_lab\runs\stage33_filter_f050_b200.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage33_filter_f050_b200.md `
  --title "Stage 33 Curriculum Filter f0.5 200 steps"
```

Collect and record in `RESULTS.md` and the run summaries, to the Codex evidence
standard: per arm, the validation NLL learning curve across the four budgets, the
steps-to-target table versus Arm A, the final NLL at 500 steps, and wall-clock per arm.
State the result against the pass and kill lines, and report the `f = 1.0` negative
control explicitly. Smoke-test the sampler before the full matrix.

## Codex Stage 33 outcome

Codex measured H010 as Stage 33 on normalized Tiny Shakespeare with the fixed
order-2 frozen prior, rank-2 LoRA residual, budgets `50`, `100`, `200`, and
`500`, seeds `7 11 19`, and four arms: uniform, mixed `f=0.25`, mixed `f=0.50`,
and pure high-loss `f=1.00` as the negative control.

The filter scored all `935,516` legal training windows by mean per-token NLL
under the frozen order-2 prior and used the top 10 percent, `93,552` windows, as
the high-loss pool. The uniform 200-step mean validation NLL target was
`2.040189`. No filtered arm reached that target earlier than uniform. The
`f=0.25` arm reached it only at 200 steps, `f=0.50` did not reach it through 500
steps, and `f=1.00` was worse at every budget. At 500 steps, both mixed filters
were slightly worse than uniform on the three-seed mean.

This kills H010 on its primary metric. The result is a local negative for hard
example mining by frozen-prior NLL, consistent with the Stage 11 and Stage 12
warning that selected hard windows may contain irreducible noise or awkward
formatting rather than learnable residual structure. It does not close the later
iterative reducible-loss or model-side time-series branches.

Links: `experiments/tiny_language_lab/runs/stage33_filter_summary.md` and
`experiments/tiny_language_lab/RESULTS.md` Stage 33.
## Prior-art, resolved by Gemini note 13

The novelty gate is cleared. Gemini note 13 locates this against hard example mining,
loss-based data selection, online batch selection, and reducible-loss data pruning, and
against the project's own Stage 11 result. Use those terms in any writeup. Frame a
positive as a local instance of known data selection accelerating a frozen-prior
residual, not a new algorithm; the only project-specific element is the
budget-constrained frozen-prior residual that the selection runs on top of.

## Links

- Gemini notes: `research/theme_3_training_dynamics_and_curriculums/12_akba_feature_selection_and_time_series.md`,
  `research/theme_3_training_dynamics_and_curriculums/13_data_selection_and_reducible_loss.md`.
- Skeptical prior: `RESULTS.md` Stages 11 and 12.
- Natural-text baseline: `RESULTS.md` Stage 30; `runs/stage30_naturaltext_summary.md`.
- Companion: `docs/hypotheses/009b-ood-validation-true-sweet-spot.md`, Stage 32, which
  found no moderate sweet spot, so this filter fixes order 2 for filtering signal.
- Roadmap: `README.md` Next ladder, rung 36.
