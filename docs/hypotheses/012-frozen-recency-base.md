# Hypothesis 012 · A frozen recency base extends the count prior past its fixed order on natural text

- Status: killed by Stage 35. Gemini note 08 has resolved the prior-art comparison
  for the tested character-recency cache.
- Date: 2026-06-23
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 38 (the model-side branch opened by ADR 0004)
- Builds on: ADR 0004 (data-side selection retired on a capacity bottleneck, pivot to
  the model side), Stage 30 and Stage 32 (natural-text finite-order priors and the
  phrase-reuse finding), and Gemini note 12 (the Akba model-side branch)

## The strategic through-line from ADR 0004

ADR 0004 established that the trainable rank-2 residual is capacity-bottlenecked: it
fits one global low-rank correction and cannot be improved by feeding it better or
reordered data. The right consequence is not to train the residual harder; it is to
push more capability into the part of the recipe that is free and unbounded, the
frozen analytic base. The frozen base costs no trainable parameters, so enriching it
is the on-thesis way to improve the recipe given a maxed-out residual.

This hypothesis is the first model-side step: enrich the frozen base so it captures
context the fixed-order count prior cannot, then keep the same rank-2 residual on top
and ask whether the richer base alone moves validation loss.

## The count prior's structural weakness, and the proposed primitive

The count prior is a fixed-order Markov base: at order k it conditions only on the
last k characters and is blind to anything earlier. On natural text that is a real
limit, because the thing that makes high-order priors win, per Stage 32, is phrase
reuse: a name, word, or phrase recurs later in the document. A fixed-order count
table cannot copy a character it saw 40 positions ago; it has fallen out of the
window.

The proposed richer base is a frozen recency base, a time-series exponential-smoothing
operator over the character signal in the context. It is the most lab-feasible
concrete instance of the model-side "time-series matrix" direction:

- At position `t`, for each candidate next character `c`, compute a recency score
  `r(c) = sum over i < t of [char_i == c] * exp(-(t - i) / tau)`. Characters seen
  recently in the context score high. This is an exponential-smoothing time series of
  the one-hot character occurrences, with a single decay constant `tau`.
- Convert `r` to a probability `P_recency` by normalizing, and interpolate it with the
  count prior: `P_base = (1 - lambda) * P_count + lambda * P_recency`, then
  `base_logits = log P_base`.
- The base stays frozen and analytic: `tau` and `lambda` are fixed constants, no
  gradients, no training. The rank-2 residual sits on top unchanged.

This is the classic cache or recency mechanism, applied as a frozen additive base
rather than a trained component, and combined with the count prior so it adds
unbounded recency on top of the prior's local conditional structure.

## Hypothesis

On the natural-text corpus, a frozen recency-augmented base, count prior interpolated
with the exponential-recency operator, gives the rank-2 residual a durable validation
NLL advantage over the count-prior-only base at the same prior order, because it
captures unbounded decaying repetition and phrase reuse that the fixed-order count
table cannot.

This is the falsifiable claim. It is killed if the recency-augmented base does not
beat the count-only base on validation NLL beyond seed noise through the measured
budget, which would mean the fixed-order count prior already captures the usable local
structure and unbounded recency adds nothing the residual can exploit on this text.

## Why it might work, and why it might fail

- Might work: Stage 32 showed phrase reuse keeps high-order priors net positive even
  across a domain shift. A recency base captures phrase reuse directly and without a
  fixed order limit, cheaply, so it could give the base the long-range signal that the
  count table is structurally blind to.
- Might fail: natural-text recency on a single small corpus may be weak, or the count
  prior at order 2 to 4 may already capture most reusable local phrases, or the
  interpolation may inject noise. The kill line catches all of these.

Note the capacity bottleneck does not doom this the way it doomed data selection. The
recency capability lives in the frozen base, which has unlimited representational room
because it is not the trainable surface. The residual is unchanged.

## Baseline and points in hand

- The strict baseline is the count-only base at the same order, `count_prior_ng2_lora_r2`
  on `natural_text_seed.txt`, the Stage 30 learning curve. The recency arm must beat
  that curve.
- Stage 30 natural-text numbers at order 2: validation NLL about `2.04` across budgets,
  advantage about `+0.11` over `random_full` at 500 steps. The recency arm should lower
  the order-2 base's own NLL if it adds real signal.

## Primary decision metric and pass or fail line

Metric: mean validation NLL of the recency-augmented base versus the count-only base at
the same prior order, over seeds 7, 11, 19 with per-seed spread, at budgets 50, 100,
200, 500. The trainable rank-2 residual and the corpus are identical; only the frozen
base changes.

- PASS: the recency-augmented base lowers validation NLL beyond seed noise, durably
  through 500 steps. Enriching the frozen base with unbounded recency helps, and the
  model-side branch is alive.
- PARTIAL: the recency base helps at a low prior order, where the count table is weakest,
  but the gain vanishes once the count order is raised, meaning recency and higher count
  order are substitutes rather than complements.
- KILL: no durable validation NLL gain over the count-only base. The fixed-order count
  prior already captures the usable structure on this text, and this particular richer
  base does not help. The branch then turns to a different frozen primitive, for example
  a state-space kernel, or to non-gradient residual formation.

## Risks and confounds

- Two free constants. `tau` and `lambda` are hyperparameters. To avoid an unfair sweep,
  fix sensible defaults first, for example `tau` around the block size and a small
  `lambda` like 0.2 to 0.3, run the headline comparison, and only then do a small
  documented sweep if the headline is borderline. Report the values.
- Cost. The recency base is computed per position from the preceding context, not a
  single table lookup. It is still cheap and analytic, but report wall-clock so the
  base's cost is visible against the count-only base.
- Order interaction. Recency may merely duplicate what a higher count order already
  provides. Run the recency arm at order 2 against the order-2 count baseline, and as a
  diagnostic also report the order-3 count baseline, so a recency gain can be compared
  to simply raising the order.
- Determinism. Fixed seeds 7 11 19 and a fixed corpus split, as in Stage 30.

## What result would change the plan

- A PASS makes a frozen long-range base the project's active model-side direction and
  motivates testing richer primitives, a state-space kernel or a wider co-occurrence
  base, and varying corpus and order.
- A KILL points away from recency specifically. The next model-side candidate would be
  a state-space or convolutional frozen kernel, which Gemini should ground, or the
  long-deferred non-gradient residual-formation branch.

## Handoff to Codex (next stage; Codex stage number 35, README ladder rung 38)

Files to modify:

1. `experiments/tiny_language_lab/cassandra_tiny_transformer.py`: add a recency base
   option, for example `--residual-base count-recency` with `--recency-tau` and
   `--recency-lambda`, or a `--recency-mix` flag layered on the existing
   `count-ngram` base. In the forward pass, for each position compute the
   exponential-recency character distribution over the preceding context in the block,
   interpolate with the count prior at the configured order, and use the log of the
   mixture as `base_logits`. No gradients through the base.
2. `experiments/tiny_language_lab/cassandra_compare.py`: add a config
   `count_prior_ng2_recency_lora_r2` that carries the recency flags on the order-2
   base, and register it in `--configs`.

Runs, on `corpus/natural_text_seed.txt`, order-2 prior, rank-2 residual,
`--block-size 96 --eval-mode sampled --eval-batches 16 --count-alpha 0.1
--ngram-backoff 10 --seeds 7 11 19`, at budgets `50, 100, 200, 500`:

- Arm A, baseline: `count_prior_ng2_lora_r2`, no recency.
- Arm B, recency: `count_prior_ng2_recency_lora_r2` with the default `tau` and `lambda`.
- Diagnostic: also include `count_prior_ng3_lora_r2` so a recency gain can be read
  against the cost-free alternative of raising the count order.

Smoke-test the recency base before the matrix. Record in `RESULTS.md` and the run
summaries to the Codex evidence standard: per-arm validation NLL and bits per character
across budgets with per-seed spread, the recency-minus-count advantage, the `tau` and
`lambda` used, and wall-clock per arm. State the result against the pass and kill lines.

## Codex result (Stage 35)

Codex implemented `--recency-tau`, `--recency-lambda`, and the runner config
`count_prior_ng2_recency_lora_r2`. The default comparison used `tau=96` and
`lambda=0.25`, with the same order-2 count prior and rank-2 LoRA residual as the
baseline.

Result: KILL. The recency arm was worse than the order-2 count-only baseline at
every budget: `+0.085528` NLL at 50 steps, `+0.080251` at 100, `+0.072638` at
200, and `+0.061996` at 500. The order-3 count diagnostic was much better than
order 2 at every budget, by about `-0.230` NLL. Recency also added wall-clock cost
because the base is computed per block position rather than by a table lookup.

Evidence: `experiments/tiny_language_lab/runs/stage35_recency_summary.md` and
`experiments/tiny_language_lab/runs/stage35_recency_summary.jsonl`.

Decision line: this kills the default frozen character-recency interpolation. It
does not kill every model-side frozen primitive. Gemini note 08 frames the result
as the expected character-level failure mode of cache language models: cache helps
most at word or token level, while bag-of-recent-characters injects noise into
next-character syntax.

## Prior-art resolved by Gemini note 08

This is a known primitive and must be framed as a controlled measurement, not a new
idea. Gemini note 08 locates it against cache language models and n-gram plus cache
interpolation. The note explains that Stage 35 tested the wrong granularity for a
cache: character-level recency is structure-blind, unlike word or token caches.
Future model-side primitives should preserve sequence order, for example frozen SSM
kernels, one-dimensional convolutional kernels, or n-gram caches.

The original comparison targets were:

- Neural and continuous cache language models, the recency or cache mechanism that
  upweights recently seen tokens, and pointer or copy mechanisms.
- Classic n-gram plus cache interpolation, which is exactly the count-plus-recency
  mixture proposed here.
- Exponential smoothing and autoregressive time-series operators, and whether this is
  the concrete realization of Dr. Akba's "time-series matrix" or whether Akba's method
  is a different, richer primitive worth a separate hypothesis.

The question for Gemini: is a frozen recency base over characters a documented win or a
documented null on small natural-text language models, and what is the standard name, so
Cassandra cites it and does not claim novelty.

## Links

- Decision that opened this branch:
  `docs/decisions/0004-retire-data-side-selection-rank2-residual.md`.
- Natural-text evidence: `RESULTS.md` Stages 30 and 32; the phrase-reuse note
  `research/theme_1_architecture_and_priors/07_domain_shift_and_phrase_reuse.md`.
- Akba framing: `research/theme_3_training_dynamics_and_curriculums/12_akba_feature_selection_and_time_series.md`.
- Roadmap: `README.md` Next ladder, rung 38.
