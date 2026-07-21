# ADR 0005 · Gradient forms the tiny frozen-prior residual: retire gradient-free formation and close the formation-side trilogy

- Status: Accepted
- Date: 2026-06-23
- Author: Claude (hypothesis, ADR, and roadmap role)
- Decides: how the tiny frozen-prior residual is formed, and whether the
  "form the residual more cleverly" branch stays open at consumer scale
- Resolves: Hypothesis 013 (kill of the full rank-2 evolution-strategy claim, weak
  partial for rank-1 coordinate search, Stage 36); consolidates the formation-side
  thread with ADR 0004 (data-side retired) and Stage 35 (analytic-base near ceiling)
- Builds on: Gemini research note 16 (zeroth-order optimization and evolution
  strategies), Stage 36 Codex results, ADR 0002 and ADR 0003 (the analytic-prior
  wins), ADR 0004 (the sibling formation-side retirement)

## Context

Hypothesis 013 asked the project's purest question: can the residual on top of the
frozen count prior be formed with no backpropagation at all, leaving the whole recipe
gradient-free, a frozen analytic prior plus a searched residual? Stage 0 already did
this for the `V x V` bigram matrix with count construction plus coordinate search;
H013 tried to complete the arc on the winning transformer recipe. Codex measured it as
Stage 36 on the `25,909`-character structured corpus from Stages 5 and 6, seeds
`7 11 19`, `50` steps, block size `32`, on the laptop RTX 4070.

Three formation methods were compared on the same frozen count-bigram base and the same
rank-2 residual parameterization, about `6,696` parameters, changing only how the
residual is formed:

| Arm | Optimizer | Mean val NLL | Mean seconds | Formation forward passes |
| --- | --- | ---: | ---: | ---: |
| `count_prior_lora_r2_floor` | none (frozen prior) | `2.018509` | `1.1800` | `0` |
| `count_prior_lora_r2` | AdamW (target) | `2.000801` | `1.9344` | `50` |
| `count_prior_lora_r2_es` | evolution strategy | `2.060750` | `4.2399` | `803` |
| `count_prior_lora_r1_coord` | coordinate search | `2.011522` | `1.5655` | `101` |
| `random_full` | AdamW (context) | `2.107505` | `1.6753` | `50` |

The floor-to-target gap, the value the search must recover, is `0.017708` NLL. The full
rank-2 evolution strategy did not recover it. It was worse than the frozen floor by
`0.042242` NLL on the three-seed mean, a recovered-gap value of about `-238%`, while
using `803` formation forward passes against AdamW's `50`. The rank-1 coordinate
feasibility arm recovered `0.006986` NLL, about `39.5%` of the mean gap, but only beat
the floor on one of three seeds.

The diagnostic is important. The ES arm did improve its fixed search objective on seed
`7`, from `1.958517` to `1.823466`, while validation worsened. The failure is therefore
fixed-batch overfitting, not a dead search loop. A four-batch ES diagnostic reduced the
damage but still missed the floor, `2.021374` versus `2.018407`, at `3212` forward
passes and `11.4758` seconds; a smaller search learning rate also missed.

One quantitative fact reframes the whole formation-side question. The frozen prior alone
beats `random_full` by `0.088996` NLL, while the gradient-trained residual adds only
`0.017708` on top. Of the recipe's total `0.106704` NLL edge over the full random
transformer on this corpus, the frozen prior carries about `83.4%` and the trainable
residual about `16.6%`. The residual is a minor refinement on a prior that does the heavy
lifting.

## The mechanism

Two reinforcing reasons explain why gradient-free formation fails here, and why this is
not surprising.

First, geometry and scale. Gemini note 16 places this against MeZO (Malladi et al.,
2023), which fine-tunes models up to `66B` parameters with forward-only zeroth-order
estimates, and against evolution strategies (Salimans et al., 2017). Those methods work
through the blessing of scale: large, redundant models have smooth, high-dimensional loss
landscapes where random perturbations reliably find descent directions. A rank-2 residual
on a frozen prior is the opposite regime, a tiny, sharp, non-redundant surface. Antithetic
Gaussian probes mostly point in destructive directions, and coordinate search is too slow
to thread the narrow valley that AdamW descends with exact gradients and momentum. Non-
gradient parameter guessing does not scale down.

Second, stakes. The surface the search is fighting over is worth only `0.017708` NLL.
Even AdamW, with exact gradients, extracts only that much here. A noisier estimator cannot
reliably capture a sliver that small without overfitting whatever fixed batch defines its
objective, which is exactly the observed failure: search objective down, validation up.

These compound. Gradient-free formation is both ill-suited to the geometry and aimed at a
low-value target. The kill is robust for the immediate version, and the small target value
means the cost-to-benefit of engineering a better search is poor in this regime.

## Decision

Retire unguided gradient-free formation of the frozen-prior rank-2 residual at consumer-
scale budgets. Form the residual with AdamW. Do not spend further stages on antithetic
evolution strategies, hill-climbing, or coordinate search variants of this specific
surface at this scale.

Record the consolidation. With ADR 0004 (data-side selection and curriculum retired on a
capacity argument) and Stage 35 (the analytic frozen base is near its ceiling, the best
base is the highest estimable count order with good backoff), the three cheap levers for
"form the residual more cleverly", choosing its data, enriching its frozen base, and
swapping its optimizer, are now bounded. The validated recipe is a frozen finite-order
count prior that carries most of the edge, plus a small gradient-trained residual that is
a minor refinement, ADR 0002 and ADR 0003.

## Scope and what this decision does not claim

- It does not claim zeroth-order optimization fails in general. It succeeds at scale,
  MeZO being the clearest case. This is a result about a tiny, sharp, low-rank residual
  aimed at a small-value target.
- It does not claim every gradient-free variant is dead even here. Only a simple
  antithetic ES and a rank-1 coordinate search were tested, and the ES carried a known
  fixed-batch overfitting confound. A less-overfit objective could in principle be
  revisited, but only under the reopening gate below, because the target value is small.
- It does not weaken the recipe. The prior-carries-it decomposition is a strength, not a
  disappointment: the residual being minor and gradient-formed is exactly why the recipe
  is cheap, robust, and reproducible at laptop scale.
- It does not retire the Stage 0 bigram coordinate-search result, which stands as the
  analytic-plus-search baseline on the `V x V` logit matrix. This ADR is about the
  transformer residual, a different and harder surface.

## What would reopen this decision

Because both the geometry and the stakes drove the kill, reopening requires changing at
least one of them:

- A capacity or rank increase, for example rank 8 or 16, where the residual both carries
  materially more value and has the degrees of freedom for a noisy estimator to help. This
  ties to ADR 0004's reopening clause, which also gates on capacity.
- A measured regime where the floor-to-target residual gap is large, provisionally at
  least about `0.05` NLL, roughly three times the structured-corpus gap, so that gradient-
  free formation would be fighting for real stakes.
- A less-overfit zeroth-order objective, for example a multi-batch or full-evaluation
  search target, that reliably beats the floor across seeds at matched wall-clock on such
  a large-gap regime.

## Redirect: gate the next move on the residual's marginal value

Stage 36 surfaced a question that should be answered before any further formation-side
investment, whether a richer frozen base, a better search, or a rank increase: is the
residual's small marginal value, `0.017708` NLL, a property of this regime or intrinsic to
the recipe?

The cheap gate probe reuses existing machinery, the `count_prior_lora_r2_floor` and
`count_prior_lora_r2` configs from Stage 36, with no new primitive to build. Measure the
floor-to-target gap across regimes that already favor the recipe:

- natural-text Tiny Shakespeare from Stages 30 to 32 at `200` and `500` steps, count
  orders 2 to 4, where the recipe's advantage over `random_full` is already large; and
- a rank sweep, ranks 1, 2, and 4, on the structured corpus, to see whether more residual
  capacity widens the gap.

The decision the probe feeds:

- If some cheap regime shows a large residual gap, on the order of `0.05` NLL or more with
  a stable sign across seeds, then formation-side work reopens there with real statistical
  power, and that regime becomes the setting for Hypothesis 014.
- If every cheap regime keeps the residual gap near the structured `0.0177`, then the
  recipe is confirmed as prior-dominated, and the roadmap turns away from formation-side
  NLL mechanics toward two higher-value axes: strengthening the frozen prior itself within
  its near-ceiling, the highest estimable count order with good backoff, and the
  behavior axis, the copy probe and verifier machinery from Stages 7 to 23, where the
  north-star goal of useful behavior actually lives and where NLL and task behavior are
  known to diverge.

The model-side richer-base idea from ADR 0004's redirect, a frozen state-space or
convolutional kernel or n-gram cache, remains a candidate but is now lower priority and
explicitly gated: it is a large speculative build whose payoff feeds the same residual the
gate probe is testing, so it is not worth grounding until the residual is shown to carry
large marginal value somewhere cheap.

## Codex follow-up · Stage 37 residual marginal-value gate

Codex ran the ADR redirect as Stage 37 on 2026-06-23. The aggregate artifacts are
`experiments/tiny_language_lab/runs/stage37_residualgap_summary.md` and `.jsonl`.

The gate result is closed. Natural-text Tiny Shakespeare at 200 and 500 steps,
orders 2 to 4, did not show a stable positive residual gap. The gaps were mixed
or negative across seeds even though the floor priors beat `random_full` by large
margins. On the structured corpus, increasing LoRA rank from 1 to 4 widened the
residual gap, but only to `+0.023058` NLL at rank 4, below the `0.03` closed-gate
threshold and far below the `0.05` reopening line.

This follow-up confirms the ADR's concern: the tested cheap regimes are
prior-dominated, and formation-side optimizer work has no large local target. It
does not open Hypothesis 014.

## Prior-art flag for Gemini

Gemini note 16 already covers MeZO, evolution strategies, and block coordinate descent. No
novelty is claimed here. The project's contribution from this branch is a clean documented
negative: unguided gradient-free formation does not help a tiny, sharp, low-value residual,
because non-gradient parameter guessing does not scale down. Frame any future positive as
applying a known zeroth-order method to a larger-stakes residual, not as a novel optimizer.

## Links

- Resolved hypothesis: `docs/hypotheses/013-non-gradient-residual-formation.md`.
- Prior decisions: `docs/decisions/0002-frozen-prior-is-bounded-early-compute-accelerator.md`,
  `0003-graded-source-prior-order-law.md`,
  `0004-retire-data-side-selection-rank2-residual.md` (the sibling formation-side ADR).
- Stage 35 analytic-base ceiling: `experiments/tiny_language_lab/RESULTS.md` Stage 35;
  `docs/hypotheses/012-frozen-recency-base.md`.
- Gemini note: `research/theme_3_training_dynamics_and_curriculums/16_zeroth_order_optimization_and_evolution.md`.
- Codex results: `experiments/tiny_language_lab/RESULTS.md` Stage 36;
  `experiments/tiny_language_lab/runs/stage36_h013.md` and the ES diagnostics
  `runs/stage36_h013_es_searchbatches4_diag.md`, `runs/stage36_h013_es_lr0005_diag.md`.
- Roadmap: `README.md` Next ladder.
