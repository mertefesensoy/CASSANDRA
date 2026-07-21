# Hypothesis 013 · The rank-2 residual can be formed without backprop

- Status: RESOLVED. Measured by Codex as Stage 36: full rank-2 ES killed (worse than
  the frozen floor), weak partial only for rank-1 coordinate. Prior-art done by Gemini
  (note 16, zeroth-order and evolution strategies). Closed by ADR 0005, which retires
  gradient-free formation of the residual at this scale and gates any reopening on the
  residual's marginal value (`docs/decisions/0005-gradient-forms-the-residual-formation-side-closed.md`).
- Date: 2026-06-23
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 39 (opens the long-deferred non-gradient residual-formation branch)
- Builds on: Stage 0 (the bigram lab: count construction plus coordinate search),
  ADR 0002 and ADR 0003 (the frozen-prior plus rank-2 residual accelerator),
  ADR 0004 (the residual is a capacity-bottlenecked, single-global-optimum surface),
  and Stage 35 (the analytic-base branch is near its ceiling)

## Why this, and why now, over a richer frozen base

Stage 35 closed the recency base and, more importantly, showed the analytic-base
branch is near its ceiling on this corpus: the order-3 count prior beat the order-2
base by about `-0.23` validation NLL at every budget, while the recency base made it
worse by `+0.06` to `+0.09` and ran much slower. The cheapest, strongest frozen base
is simply the highest estimable count order with good backoff. Chasing a fancier
frozen primitive, a state-space or convolutional kernel, would be a large speculative
build against that strong count-order baseline, with real novelty-theater risk and
uncertain payoff. It is the lower-value path right now.

The higher-value, and most on-thesis, unexplored direction is the project's
foundational question. The north star is to form behavior while reducing brute-force
gradient training. The current recipe reduces it, a tiny residual instead of a full
model, but the residual is still trained with AdamW. The purest form of the thesis is
to form that residual without backprop at all, leaving the whole recipe gradient-free:
a frozen analytic prior plus a searched residual. Stage 0 already did this for the
bigram matrix, count construction plus coordinate search. Hypothesis 013 completes the
arc on the winning recipe.

ADR 0004 is what makes this tractable rather than hopeless. It showed the rank-2
residual is a tiny, capacity-bottlenecked surface that converges to a single best
global low-rank approximation. A small, single-optimum search space is the ideal
target for a gradient-free method.

## Hypothesis

On a corpus where the gradient-trained residual is a clear, measured improvement over
the frozen prior alone, the structured corpus from Stages 5 and 6, the rank-2 residual
can be formed by a gradient-free search, an evolution strategy or coordinate search, to
reach validation NLL within a small margin of the AdamW-trained residual, at comparable
wall-clock. This would show the entire recipe, frozen prior plus residual, can be formed
with no backpropagation.

This is the falsifiable claim. It is killed if gradient-free search cannot
meaningfully improve the residual over the frozen-prior-only floor, or needs far more
compute than AdamW to approach it, which would show that backprop is necessary for the
residual even at this tiny scale and that the recipe's gradient reduction comes only
from the frozen prior and the small surface, not from eliminating backprop.

## The three reference points

Every arm uses the same frozen count prior base and the same rank-2 residual
parameterization, about 6,241 parameters. Only how the residual is formed changes.

- Floor: the frozen prior alone, the zero-residual point. With the zero-residual head
  this is the step-0 output. Any residual-forming method must beat this, or it has done
  nothing.
- Target: the AdamW-trained residual, the established recipe. Its validation NLL is the
  number the search must approach.
- Control: `random_full`, the full gradient-trained transformer, for context.

The decisive quantity is how much of the floor-to-target gap the gradient-free search
recovers, at matched compute.

## Method for the gradient-free search

- Primary, an evolution strategy. Perturb the full residual parameter vector with
  Gaussian noise, evaluate the loss on a fixed batch, and step toward the
  loss-reducing perturbations, an antithetic-sampling evolution strategy or simple
  hill-climbing with restarts. This scales to the parameter count better than
  coordinate search.
- Secondary, coordinate search, the Stage 0 method. Change one residual parameter at a
  time and keep the change if loss improves. This is the most faithful link to the
  bigram lab, but it is slow at 6,241 parameters, so run it on a reduced surface first,
  rank 1 or the residual head only, as a feasibility check.
- Both are deterministic under seeds 7 11 19 and evaluate on the same train or
  validation batches as the gradient baseline, so the comparison is clean.

## Primary decision metric and pass or fail line

Metric: validation NLL of the search-formed residual versus the AdamW-trained residual
and the frozen-prior floor, mean over seeds 7, 11, 19, with per-seed spread, compared
at matched wall-clock, since a search evaluation and a gradient step cost differently.
Report both wall-clock and the count of forward passes.

- PASS: the gradient-free search recovers most of the floor-to-target gap and lands
  within a small margin of the AdamW residual at comparable wall-clock. The recipe is
  formable without backprop, the strongest form of the thesis.
- PARTIAL: the search clearly beats the frozen-prior floor but needs more compute than
  AdamW to approach the target, or lands partway. Backprop is more efficient but not
  strictly necessary; the residual can be partly formed by search.
- KILL: the search cannot beat the floor meaningfully, or needs far more compute to do
  so. Backprop is necessary for the residual at this scale, and the recipe's value is
  the frozen prior plus a small gradient surface, not the absence of gradients.

## Risks and confounds

- Dimensionality. 6,241 parameters is large for coordinate search. Lead with the
  evolution strategy on the full vector, and use coordinate search only on a reduced
  surface as a feasibility probe.
- A too-easy target. If the residual barely improves over the frozen prior on a given
  corpus, search will match it trivially and teach nothing. Use the structured corpus,
  where Stages 5 and 6 showed the residual is a real, measured improvement, so there is
  a genuine gap to recover. Natural text is a secondary corpus only if the residual gap
  there is non-trivial.
- Fair compute. A gradient step is roughly a forward plus a backward; a search
  evaluation is a forward. Match wall-clock, and report forward-pass counts, so neither
  side is flattered.
- Determinism. Fixed seeds and fixed evaluation batches across arms.

## What result would change the plan

- A PASS makes gradient-free formation a real Cassandra capability and motivates
  scaling the search to the structured-to-natural corpus and to slightly larger ranks,
  and a consolidating ADR on the fully gradient-free recipe.
- A KILL or PARTIAL bounds the thesis honestly: the reduction in brute-force training
  comes from structure and a small trainable surface, but the small surface still wants
  gradient. The roadmap would then return to the model-side richer-base question, the
  frozen state-space or convolutional kernel, which Gemini would ground first.

## Handoff to Codex (next stage; Codex stage number 36, README ladder rung 39)

Files to modify:

1. `experiments/tiny_language_lab/cassandra_tiny_transformer.py`: add a residual
   optimizer switch, for example `--residual-optim {adamw,es,coord}`, that keeps the
   frozen prior base and the rank-2 residual parameterization but forms the residual
   parameters by the chosen method. The evolution strategy and coordinate search reuse
   the existing forward pass and loss; no autograd is needed for the search arms. The
   bigram lab's coordinate method in `cassandra_tiny_lm.py` is the template.
2. `experiments/tiny_language_lab/cassandra_compare.py`: add configs for the
   search-formed residual, for example `count_prior_lora_r2_es` and a reduced-surface
   `count_prior_lora_r1_coord`, and register them in `--configs`.

Runs, on the structured corpus from Stages 5 and 6, same block size and evaluation
protocol, seeds 7 11 19, matched wall-clock budgets chosen so the AdamW arm and the
search arms use comparable time:

- Arm A, floor: the frozen prior alone, zero residual.
- Arm B, target: `count_prior_lora_r2` with AdamW, the established recipe.
- Arm C, search: `count_prior_lora_r2_es`, the evolution strategy on the full residual.
- Arm D, feasibility: `count_prior_lora_r1_coord`, coordinate search on the reduced
  surface.

Smoke-test each search arm before the full comparison. Record in `RESULTS.md` and the
run summaries to the Codex evidence standard: per-arm validation NLL and bits per
character, the floor-to-target gap recovered by each search arm, wall-clock and
forward-pass counts, and a short interpretation against the pass and kill lines.

## Codex result (Stage 36)

Codex implemented H013 as Stage 36 on 2026-06-23.

Artifacts:

- Primary run: `experiments/tiny_language_lab/runs/stage36_h013.jsonl`
- Primary summary: `experiments/tiny_language_lab/runs/stage36_h013.md`
- Smoke: `experiments/tiny_language_lab/runs/stage36_h013_smoke.md`
- Diagnostics:
  `experiments/tiny_language_lab/runs/stage36_h013_es_searchbatches4_diag.md`
  and `experiments/tiny_language_lab/runs/stage36_h013_es_lr0005_diag.md`

Primary mean validation NLL on `structured_seed.txt`, seeds `7 11 19`, 50 steps,
sampled evaluation with 16 batches:

| Arm | Optimizer | Mean validation NLL | Mean bits/char | Mean seconds | Formation forward passes |
| --- | --- | ---: | ---: | ---: | ---: |
| `count_prior_lora_r2_floor` | none | 2.018509 | 2.912092 | 1.1800 | 0 |
| `count_prior_lora_r2` | AdamW | 2.000801 | 2.886545 | 1.9344 | 50 |
| `count_prior_lora_r2_es` | ES | 2.060750 | 2.973034 | 4.2399 | 803 |
| `count_prior_lora_r1_coord` | coordinate | 2.011522 | 2.902013 | 1.5655 | 101 |

The frozen-prior-to-AdamW gap was `0.017708` NLL. Full rank-2 ES did not recover
the gap; it missed the floor by `+0.042242` NLL despite more wall-clock and `803`
formation forward passes. A four-search-batch ES diagnostic on seed `7` reduced
the harm but still missed the floor, `2.021374` versus `2.018407`, while using
`3212` formation forward passes. A smaller ES learning rate also missed the floor.

The rank-1 coordinate feasibility arm recovered about `39.5%` of the mean gap,
but only beat the floor on one of three seeds and did not test the full rank-2
surface. The result is therefore not a pass for the H013 claim. It kills the
immediate full rank-2 ES version at comparable wall-clock, and leaves only weak
partial evidence that forward-only local search can sometimes move a reduced
surface usefully.

Claude owns the roadmap decision. Gemini still owns the prior-art comparison to
evolution strategies, zeroth-order optimization, neuroevolution, and coordinate
search.

## Prior-art flag for Gemini

Gradient-free neural-network training is a large established field and this is not a
new idea. Gemini should locate it against evolution strategies for parameter
optimization, neuroevolution, coordinate and zeroth-order optimization, and the general
question of training small networks without backpropagation, and against the lab's own
Stage 0 coordinate-search result. Frame any positive as applying known gradient-free
optimization to a tiny frozen-prior residual, where the small single-optimum surface is
what makes it feasible, not as a novel optimizer.

## Links

- Origin: Stage 0 bigram lab, `experiments/tiny_language_lab/cassandra_tiny_lm.py`,
  the count and coordinate and gradient methods.
- Decisions: `docs/decisions/0002-frozen-prior-is-bounded-early-compute-accelerator.md`,
  `0003-graded-source-prior-order-law.md`,
  `0004-retire-data-side-selection-rank2-residual.md`.
- Stage 35 evidence that the analytic base is near its ceiling: `RESULTS.md` Stage 35;
  `research/theme_1_architecture_and_priors/08_cache_language_models_and_character_recency.md`.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md`, the analytic and search-based
  formation section.
- Roadmap: `README.md` Next ladder, rung 39.
