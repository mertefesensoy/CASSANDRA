# ADR 0004 · Retire data-side selection and curriculum for the frozen-prior rank-2 residual: it is a capacity bottleneck

- Status: Accepted
- Date: 2026-06-23
- Author: Claude (hypothesis, ADR, and roadmap role)
- Decides: the future of the data-side selection and curriculum branch on the tiny
  frozen-prior residual
- Resolves: Hypothesis 010 (kill, Stage 33) and Hypothesis 011 (kill, Stage 34);
  consolidates the curriculum and replay thread, Stages 11, 12, 33, 34
- Builds on: Gemini research notes 13, 14, and 15

## Context

The lab tried four times to make data selection or curriculum help the trainable
surface, and all four failed. The two most recent and most controlled were decisive.

- Stage 33, Hypothesis 010, a static filter scoring training windows by the frozen
  order-2 prior's per-token NLL and oversampling the high-loss ones. The mixed arms
  were inert and the pure arm was harmful: at `f = 0.25` the validation NLL delta
  versus uniform was about `+0.001`, at `f = 0.50` about `+0.002`, and at the pure
  `f = 1.0` control about `+0.015`. No arm reached the uniform target earlier.
- Stage 34, Hypothesis 011, the dynamic reducible-loss fix that note 14 prescribed.
  It re-scored a fixed pool under the live model every 25 steps and oversampled
  windows that were both high-loss and actively improving, the theoretical ideal of
  reducible-loss data selection. It was worse than uniform at every budget, by about
  `+0.008` to `+0.010` NLL at `f = 0.50` and `+0.0017` to `+0.0023` at `f = 0.25`,
  and slower, because the re-scoring forward passes roughly doubled wall-clock.

Stage 33 left two competing explanations: a wrong proxy, or a capacity bottleneck.
Stage 34 resolved them. The dynamic filter was the correct proxy, and it still did
not help, and in fact hurt. The explanation is capacity, not proxy. Full numbers are
in `experiments/tiny_language_lab/RESULTS.md` Stages 33 and 34 and the run summaries.

## The mechanism

A rank-2 LoRA residual has about 6,241 trainable parameters and enforces a strict
low-rank subspace on the update. With that little freedom the optimizer finds a
single best global low-rank approximation of the corpus. The surface is therefore
data-order insensitive: it cannot adapt to a focused curriculum batch without
unlearning the rest, so reweighting the data toward any subset moves the global
approximation off its optimum. That is why Stage 34 was worse, not merely inert.
Gemini note 15 places this against the literature, where data pruning and dynamic
curricula help large models, for example DoReMi and Sheared LLaMA, precisely because
those models have the capacity to assimilate selected examples, and where
parameter-efficient methods such as PRILoRA and LoPrune address the same limit by
aligning selection with the adapter subspace or by growing rank during training.

## Decision

Retire the data-side selection and curriculum branch for the frozen-prior rank-2
residual. Do not test further data-filtering, hard-example-mining, replay, or
curriculum hypotheses on this specific trainable surface. The question is settled at
this scale.

## Scope and what this decision does not claim

- It does not claim data selection is useless in general. Data selection and
  curricula work for higher-capacity models. This is a result about an extremely
  low-rank, parameter-constrained residual.
- It does not weaken the core recipe. The frozen finite-order prior plus rank-2
  residual remains the project's validated accelerator, ADR 0002 and ADR 0003. The
  capacity bottleneck is in fact the reason that recipe is robust and reproducible:
  its result does not depend on data ordering, sampling, or curation, which is a
  feature at laptop scale, not only a limitation.
- It does not retire the verifier and probe machinery, which stays as tooling. It
  retires only the data-side selection hypotheses on this surface.

## What would reopen this decision

Data selection may be revisited only when combined with a capacity increase, because
the bottleneck is the surface, not the data signal. Any of the following would
justify a new hypothesis:

- A materially larger residual rank, for example 8 or 16, where the surface has
  enough degrees of freedom to be data-order sensitive.
- Dynamic rank allocation that grows the adapter where the curriculum demands it,
  the PRILoRA direction.
- A selection metric aligned with the adapter's gradient subspace rather than with
  the frozen prior's loss.

## Redirect: the model-side branch

The roadmap turns to the model side. The count prior's structural weakness is that
it is a fixed-order Markov base: it can only see the last k characters and cannot
capture longer-range context. The model-side question is whether a richer frozen
base, still untrained, gives the same rank-2 residual a durable advantage that the
bounded-order count base cannot, because it captures decaying or unbounded context.

This is the project-thesis-aligned next branch, and it can become Hypothesis 012,
but it is gated. The "time-series matrix" framing from Gemini note 12 is not yet a
falsifiable mechanism, so it needs a Gemini specification note before Claude can
draft H012. Per the role boundary, Claude does not author that research note; Gemini
does. To seed it, three candidate frozen long-range primitives, all parameter-light
and analytic, are worth grounding and comparing:

1. A frozen linear state-space or recurrent kernel, S4 or SSM style, that captures
   decaying unbounded context analytically.
2. A frozen recency or cache feature, a pointer-to-recent-context base, that captures
   local repetition beyond a fixed order.
3. A frozen wider-window co-occurrence or PMI base that captures medium-range
   association.

The Gemini specification note must answer: which primitive maps cleanly onto a frozen
additive base in the residual forward pass, `logits = base[context] + residual`; how
it is computed analytically without training; its parameter and memory cost at the
`V = 33` natural-text vocabulary; and the prior art for frozen long-range or
state-space initializations, so H012 is framed as a controlled measurement of a known
primitive rather than a novel architecture. Once that note exists, Claude drafts H012
with the falsifiable claim above and a strict baseline against the order-2 count base.

The long-deferred non-gradient residual-formation branch, research-map section 5,
remains the alternative if the model-side base also stalls.

## Prior-art flag for Gemini

Gemini notes 13, 14, and 15 already cover the data-selection side, with DoReMi,
Sheared LLaMA, PRILoRA, and LoPrune as anchors. No novelty is claimed here. The
project's contribution from this branch is a clean, documented negative: data-side
selection does not help an extremely low-rank residual, and the bottleneck is
architectural. The next required Gemini note is the model-side primitive
specification described in the redirect.

## Links

- Killed hypotheses: `docs/hypotheses/010-akba-curriculum-filter.md`,
  `docs/hypotheses/011-dynamic-reducible-loss-filter.md`.
- Prior decisions: `docs/decisions/0001-retire-compact-text-prefix-external-memory.md`
  (the earlier retired branch), `0002` and `0003` (the analytic-prior wins).
- Gemini notes: `research/theme_3_training_dynamics_and_curriculums/13_data_selection_and_reducible_loss.md`,
  `14_static_vs_iterative_reducible_loss.md`, `15_capacity_bottlenecks_in_data_selection.md`.
- Codex results: `RESULTS.md` Stages 11, 12, 33, and 34; `runs/stage33_filter_summary.md`;
  `runs/stage34_dynfilter_summary.md`.
- Roadmap: `README.md` Next ladder.
