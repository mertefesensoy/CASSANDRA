# ADR 0002 · The frozen-prior recipe is a bounded early-compute accelerator, not an asymptotic replacement

- Status: Accepted
- Date: 2026-06-16
- Author: Claude (hypothesis, ADR, and roadmap role)
- Decides: the status of the project's core positive result, the frozen count
  prior plus tiny residual recipe
- Resolves: Hypothesis 005 (boundary characterized) and Hypothesis 006 (surface
  confirmed, mechanism corrected); closes the budget-by-complexity characterization
  thread and opens a new method branch

## Context

The recipe under decision is: build a smoothed bigram count prior from corpus
statistics, freeze it as a residual base, and train only a rank-2 LoRA surface on
top (`count_prior_lora_r2`, about 6.6K trainable parameters versus about 111K for
the full random transformer). Stages 5 and 6 showed it beating full random
training on plain language-model validation NLL under a 50-step budget on a
structured corpus. Stage 7 showed it losing on a long-context corpus. Those two
anecdotes have now been resolved into a measured surface.

Stage 24 swept a corpus-complexity axis (`make_complexity_corpus.py`, mixing
bigram-local and long-context lines at long-fraction `p`) at 50 and 100 steps.
Stage 25 completed the surface at 10 and 25 steps. The decision metric is the
advantage, `random_full` mean val NLL minus `count_prior_lora_r2` mean val NLL,
over seeds 7, 11, 19; positive means the cheap recipe won.

| Long fraction p | 10-step | 25-step | 50-step | 100-step |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | +0.688454 | +0.247912 | +0.059159 | -0.156425 |
| 0.25 | +0.701242 | +0.278451 | +0.098079 | -0.101866 |
| 0.50 | +0.787087 | +0.299924 | +0.078996 | -0.230854 |
| 0.75 | +0.856700 | +0.247387 | -0.066922 | -0.438575 |
| 1.00 | +0.948164 | +0.241261 | -0.162385 | -0.642078 |

Crossover contour `p*(steps)`, the long-fraction where the advantage passes
through zero:

| Steps | p*(steps) |
| ---: | --- |
| 10 | above the measured range, greater than 1.00 |
| 25 | above the measured range, greater than 1.00 |
| 50 | about 0.635343 |
| 100 | below the measured range, less than 0.00 |

Two facts from the surface:

- At every fixed `p`, the advantage decreases monotonically with the step budget
  (10 greater than 25 greater than 50 greater than 100). By 100 steps the full
  model wins at every measured `p`, including the most bigram-predictable `p = 0`.
- The crossover budget falls as long-range dependence rises. On bigram-local text
  the cheap recipe holds its edge longer; on long-range text the full model
  overtakes sooner.

The mechanism, corrected by Stage 25. The original framing was that the full model
is near-random noise at 10 steps. That is false. After 10 AdamW steps the full
model is already far below the uniform floor `ln(vocab_size)` on every corpus
point; it trains fast. The cheap recipe stays within roughly 0.001 to 0.008 NLL of
the pure count prior at 10 steps. So the advantage is not that the random model
stays untrained. It is that the frozen count prior supplies a much better
low-step surface, a head start that the fast-training full model needs roughly 50
to 100 steps to erase. The advantage is a decaying head start, not an asymptotic
property.

A subtlety worth keeping: at 10 steps the advantage grows with `p` (from `+0.688`
at `p = 0` to `+0.948` at `p = 1`), because the full model is furthest from
convergence on the hardest long-range corpus when it has barely trained. Yet that
same high-`p` regime is where the full model overtakes fastest, so by 50 steps the
cheap recipe already loses at `p = 1`. Fixed-low-budget advantage and crossover
budget are different lenses on the same head start.

## Decision

Lock the core Cassandra capability claim in its precise, bounded form, and stop
running further budget-by-complexity sweeps of this recipe. The recipe is
characterized.

The locked claim: a frozen bigram count prior plus a rank-2 residual surface beats
full random transformer training on plain language-model validation NLL only in a
bounded early-compute regime. The size of that regime depends on corpus long-range
fraction. On this generator family and tiny model, the advantage is positive at 10
and 25 steps across the whole complexity axis, has a crossover near long-fraction
0.635 at 50 steps, and is negative everywhere by 100 steps. The effect is an
early-compute inductive-bias head start that decays as the full model trains, not
an asymptotic replacement for full training.

## Scope and what this decision does not claim

- It is one tiny CPU model (about 111K full parameters), one synthetic generator
  family, and one metric (plain next-character validation NLL). It is not a claim
  about natural text, larger models, or any downstream behavior.
- It does not claim the head start is worthless. A decaying early-compute advantage
  can still matter where compute or steps are the binding constraint. That economic
  question is untested here.
- It does not revive the retrieval or copy-behavior branches, which remain closed
  (ADR 0001). This ADR is about the plain-LM core recipe only.

## Consequences

- The frozen-prior recipe is the project's validated positive, now with honest
  bounds. The README and research map should describe it as a bounded early-compute
  accelerator, not as a general win over full training.
- Further sweeps of this exact recipe across budgets or complexity have reached
  diminishing returns. The roadmap should open a new method branch rather than add
  more points to this surface.
- The methodological assets remain: the comparison matrix, the complexity
  generator, the count-prior init and residual machinery, and the determinism
  protocol. The next branch should reuse them.

## What would reopen or change this decision

- A recipe variant that shifts the crossover to materially higher budgets, for
  example a higher-order analytic prior (trigram or longer) or a different frozen
  base, so the advantage is no longer just an early-compute head start.
- The accelerator holding at higher budgets on a different corpus, especially
  natural text, which would widen the regime beyond this synthetic family.
- Evidence that the early-compute head start matters for a downstream behavior or a
  wall-clock-to-threshold target, not only fixed-step NLL.
- Any of these would justify reopening the claim and writing a new hypothesis.

## Redirect: open a new method branch

The count-prior accelerator answered a weak version of the project thesis: analytic
structure can reduce gradient training, but here only as a decaying head start. The
open question for the north star is whether any analytic or search-based formation
method does better than a head start, meaning it stays competitive as the budget
grows rather than being overtaken by step 100.

Recommended next branch, in priority order:

1. Higher-order analytic prior. Extend the frozen prior from bigram to trigram or a
   short n-gram, freeze it as the residual base, and rerun the budget-by-complexity
   surface. The decision is whether a richer analytic prior pushes the crossover
   contour to materially higher budgets, which would be a genuinely stronger result
   than the bigram head start. This is the cheapest direct test, reuses all
   existing machinery, and stays on the project thesis. It should be the next
   hypothesis.
2. Non-gradient residual formation. Form the small residual surface by coordinate
   or evolutionary search rather than backprop, the purest version of the reduce
   backpropagation thesis and the still-unexplored research-map section 5.
3. Natural-text external validity. Test whether the accelerator survives on a tiny
   natural corpus, which tests generalization rather than a new method.

The next pass will formalize the chosen branch, defaulting to the higher-order
analytic prior, as Hypothesis 007.

## Codex follow-up

Codex measured the higher-order branch as Stage 26 from Hypothesis 007 on
2026-06-16. The result qualifies this ADR without overturning it.

On a pure order-2 Markov corpus, the matched frozen trigram prior plus rank-2
LoRA stayed strongly ahead of full random training through 200 steps. Its
cheap-minus-full advantage was `0.637544` NLL at 100 steps and `0.637384` at 200
steps. The mismatched bigram prior on the same order-2 source decayed to tied by
200 steps, with advantage `-0.000367`.

On the pure order-1 control, the matched bigram prior stayed positive at 50 and
100 steps (`0.042389` and `0.029240`), while the over-specified trigram prior was
near tied or slightly negative by 100 steps (`-0.004205`).

Interpretation: the original bigram recipe remains a bounded early-compute
accelerator on the previous synthetic family, but analytic priors are not doomed
to be only head starts. A correctly specified analytic prior can be durable on a
source generated by the same order family. The next branch should map the
source-order versus prior-order match surface before moving to natural text.

Codex then completed that immediate `V = 16` closeout as Stage 27. Across source
orders 1 and 2, prior orders 1 and 2, and budgets 10, 25, 50, 100, and 200, matched
priors stayed positive through 200 steps: order-1 plus bigram retained `0.018835`
NLL advantage, and order-2 plus trigram retained `0.637384`. The order-2 source
with a bigram prior decayed to tied by 200 steps (`-0.000367`), while the
order-1 source with a trigram prior became negative by 100 and 200 steps
(`-0.004205`, then `-0.015480`). This sharpens the ADR consequence: analytic
priors are durable only when their order matches the source in this controlled
lab.

Claude's H008 now formalizes the next test: rerun the surface at `V = 8`, add
order 3 on both the source and prior axes, and diagnose sparsity explicitly.
That is the next decision point for this ADR branch.

Codex measured H008 as Stage 28. The order-3 matched prior passed the sharpest
cell: `A(3,3)=0.630598` at 200 steps with full highest-order context coverage.
The diagonal was positive and increasing: `A(1,1)=0.006804`,
`A(2,2)=0.436673`, and `A(3,3)=0.630598`. The strict lower-triangle prediction
was partial because `A(3,2)=0.106932` stayed meaningfully positive. This ADR
branch now needs a follow-up decision record: analytic priors are not merely
early-compute accelerators when well matched, but the source/prior order law is
graded rather than binary.

## Prior-art flag for Gemini

This ADR should be located against known results before any claim of novelty.

- Warm-start and informed-initialization studies where a strong or structured
  initialization accelerates early training but is overtaken by longer training.
  The crossover contour is a controlled instance of that.
- The bias-variance and prior-strength tradeoff expressed along a compute axis: a
  strongly biased low-variance model wins in the small-compute regime and a richer
  model wins with more budget.
- n-gram versus neural language-model crossovers, where count models are
  competitive on predictable text and at small scale and lose as capacity and
  budget grow.

Question for Gemini: what is the standard name and the closest published result for
this early-compute crossover, so Cassandra cites it and frames the surface as a
careful laptop-scale measurement rather than a discovery. See the source anchors in
`docs/LOW_HARDWARE_LM_RESEARCH.md`, especially nanoGPT, TinyStories, and the
parameter-efficient and distillation entries.

## Links

- Resolved hypotheses: `docs/hypotheses/005-cheap-recipe-corpus-regime.md`,
  `docs/hypotheses/006-time-budget-inductive-bias-boundary.md`.
- Prior decision: `docs/decisions/0001-retire-compact-text-prefix-external-memory.md`.
- Codex result files: `experiments/tiny_language_lab/RESULTS.md` Stages 5, 6, 7,
  24, and 25; `runs/stage24_complexity_summary.md` and
  `runs/stage25_timebudget_summary.md`.
- Roadmap: `README.md` Next ladder, rung 26 (done) and the new method branch.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md`, core-claim Stages 5 to 7 and
  the analytic-formation section 5.
