# Hypothesis 008 · The source-order by prior-order surface: durability needs k at least s, with an over-specification penalty

- Status: measured by Codex as Stage 28, partial support
- Date: 2026-06-16
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 30 after the Stage 27 closeout (generalizes the Stage 26 durability result)
- Builds on: Stage 26 (Hypothesis 007 passed: matched trigram prior durable on an
  order-2 source), ADR 0002 (the bigram recipe is a bounded accelerator)

## Context

Stage 26 showed that a frozen count prior whose order matches the data-generating
order is durable, not just an early head start. On a pure order-2 Markov source the
matched trigram prior held a roughly `0.637` NLL advantage flat from 10 to 200
steps, while the mismatched bigram prior decayed to `-0.000367` by 200 steps. The
order-1 control showed the mirror image: a matched bigram stayed positive
(`+0.029` at 100 steps) and an over-specified trigram was near tied or slightly
negative (`-0.004` at 100 steps).

So the 2 by 2 corners of a source-order by prior-order grid are in hand, at vocab
16 and about 41K characters. Three questions remain before this becomes a general
claim:

1. Does the diagonal durability hold at order 3, or does it break for a reason
   other than theory?
2. How does over-specification behave as the gap grows? Stage 26 saw a tiny
   penalty for an order-3 prior on order-1 data. Does the penalty grow with prior
   order, where the higher-order count table is sparser?
3. Does the matched advantage grow with source order? Stage 26 hints yes: the
   matched edge was `+0.03` on the order-1 source but `+0.64` on the order-2
   source, because the full model is slower to learn the richer structure.

Two honest qualifications from Stage 26 to preserve. First, the durable advantage
exists partly because the full model is slow: at 200 steps it sits near `3.9`
bits per character on the order-2 source while the trigram prior is near `3.0`.
Whether the full model ever catches up beyond 200 steps is untested, so durability
here means flat across the measured budget, not proven asymptotic. Second, the
size of the durable edge tracks how hard the structure is for gradient training,
not just whether the prior is correct.

Source: `RESULTS.md` Stage 26; `runs/stage26_markov_summary.md`;
`make_markov_corpus.py`; `cassandra_tiny_transformer.py` trigram base.

## The sparsity constraint, which the design must control

A count prior of order k is a table of size `V^(k+1)`. With vocabulary `V = 16`
and about 41K characters, order 1 needs `256` entries (well estimated), order 2
needs `4096` (about 10 samples each, marginal), and order 3 needs `65536` (fewer
than one sample each, severely under-sampled). An order-3 prior at `V = 16` would
fail not because the theory is wrong but because the counts cannot estimate the
true order-3 conditional, and it would fall back almost entirely to lower orders.

To test order 3 fairly, shrink the vocabulary so the highest-order table is
estimable. At `V = 8`, order 3 needs `8^4 = 4096` entries, estimable with tens of
thousands of characters. The whole surface should therefore be re-run at one fixed
small vocabulary, `V = 8`, rather than reusing the Stage 26 `V = 16` numbers, so
the grid is internally consistent and the order-3 row is a fair test.

## Hypothesis

Define the durable advantage `A(s, k)` as `random_full` mean val NLL minus the
order-k frozen-prior config mean val NLL at a high budget (200 steps), on a pure
order-s Markov source, at fixed vocabulary `V = 8`. The surface has three
predicted features:

1. Durability requires `k` at least `s`. `A(s, k)` is durably positive when the
   prior order is at least the source order, and decays toward zero when the prior
   is under-specified (`k < s`), because an under-specified count prior is no
   better than the slowly-training full model.
2. The matched diagonal grows with source order. `A(s, s)` increases with `s`,
   because richer source structure is harder for gradient training to learn
   quickly, so the matched prior's edge is larger.
3. Over-specification is recoverable but penalized. For `k > s`, backoff recovers
   the true lower order, so `A(s, k)` stays positive, but with a penalty that
   grows with `k` because the higher-order table is noisier. At large `k` and
   small `s` the penalty can make the advantage slightly negative, as Stage 26
   already saw for order-3 prior on order-1 data.

This is the falsifiable surface. The single sharpest cell is `A(3, 3)` at
`V = 8`: if the matched order-3 prior is durable there, the diagonal-durability
claim generalizes; if it is not durable even with an estimable table, durability
is bounded to low orders and the claim is narrower than Stage 26 suggested.

## Expected signal

Durable advantage `A(s, k)` at 200 steps, `V = 8`, predicted sign and trend:

| Source s \ Prior k | k = 1 | k = 2 | k = 3 |
| --- | --- | --- | --- |
| s = 1 | positive (matched) | positive, small penalty | positive, larger penalty or slightly negative |
| s = 2 | near zero (under) | positive, larger than at s = 1 | positive, penalty |
| s = 3 | near zero (under) | near zero (under) | positive (matched), largest |

The diagonal is positive and increasing down the rows; the lower triangle decays
to zero; the upper triangle is positive but penalized.

## Baselines and points already in hand

- Stage 26 at `V = 16`: `A(2, 2)` about `+0.637` durable, `A(2, 1)` decays to zero,
  `A(1, 1)` about `+0.029`, `A(1, 2)` about `-0.004`. These anchor the qualitative
  pattern but are at the wrong vocabulary for the order-3 row, so the surface is
  re-run at `V = 8`.
- The trigram base, the `count_prior_tri_lora_r2` config, and `make_markov_corpus.py`
  exist. The order-3 prior, its config, and the order-3 corpus are new.

## Primary decision metric and pass or fail line

Metric: the 3 by 3 matrix `A(s, k)` at 200 steps, mean over seeds 7, 11, 19, with
per-seed minimum and maximum, plus the full budget curves at 10, 25, 50, 100, 200
for the diagonal cells to confirm flatness rather than slow decay. The trainable
surface is identical rank-2 LoRA in every prior config. Report each source's
sampled bits per character and each prior's own validation bits per character.

- PASS (the order-matching law generalizes): the lower triangle (`k < s`) decays
  to near zero, the diagonal (`k = s`) is durably positive and grows with `s`, and
  the upper triangle (`k > s`) stays positive with a penalty that grows with `k`.
  In particular `A(3, 3)` is durably positive at `V = 8`. This generalizes Stage 26
  into a law: durability needs the prior order to reach the source order, and the
  matched advantage reflects how slow gradient training is on that structure.
- PARTIAL: the qualitative ordering holds (matched beats under-specified) but the
  diagonal does not cleanly grow with `s`, or the over-specification penalty is not
  monotone. The law holds in sign but not in shape.
- KILL or BOUND: `A(3, 3)` is not durable even at `V = 8`, so matched durability
  fails at order 3 for a non-sparsity reason. Then durability is bounded to low
  orders, which is itself an important limit to record, because real text has
  effectively high order and a count prior cannot reach it.

## Risks and confounds

- Sparsity masquerading as theory failure. This is the main risk and the reason for
  `V = 8`. Codex must report each prior's own bits per character against the source
  entropy, so an under-performing matched prior can be diagnosed as data starvation
  versus a real effect. If the order-3 counts are still too sparse at `V = 8`,
  increase the corpus length rather than concluding the theory failed.
- Slow-full confound on durability. Because the full model is slow, a flat
  advantage at 200 steps is durable within budget but not proven asymptotic. Note
  this and treat the asymptotic question (does full ever catch a matched prior) as a
  separate, later study, not part of this surface.
- Equal trainable budget. Keep rank-2 LoRA identical across all prior orders. The
  frozen table grows as `V^(k+1)` but is not trainable. This is an inductive-bias
  comparison, not a capacity comparison.
- Backoff design. The order-n prior must use multi-level backoff from order n down
  through 1, so an over-specified prior degrades gracefully to the true order. State
  the backoff method.
- Determinism. Use fixed seeds for both corpus generation and training, matching the
  project protocol.

## What result would change the plan

- A PASS makes the order-matching law the project's strongest positive result and
  justifies a consolidating ADR that states it, then a move to natural text where
  the effective order is high and unknown, to test whether a moderate-order prior
  still helps. The next pass would write that ADR.
- A KILL or BOUND fixes the durability claim to low orders and pivots the branch
  toward natural-text validity or the non-gradient residual-formation idea, since
  pushing analytic order further would not pay off.

## Handoff to Codex (implemented as Stage 28)

New code, building on the Stage 26 trigram base.

1. General order-n count prior.
   - Generalize the trigram base to arbitrary order n: a frozen `V^(n+1)` table of
     smoothed `log P(next | previous n tokens)`, gathered in the forward pass with
     n index tensors and multi-level backoff from order n down to 1, with
     start-of-sequence backoff for the first n positions.
   - Add `--residual-base count-ngram` with `--prior-order n`, and configs
     `count_prior_ng{1,2,3}_lora_r2` that share the rank-2 LoRA surface and differ
     only in prior order. The existing bigram and trigram bases can remain as the
     n = 1 and n = 2 instances.

2. Order-3 corpus and a `V = 8` re-generation.
   - Generate pure order-1, order-2, and order-3 Markov corpora at `V = 8` with
     `make_markov_corpus.py --order {1,2,3} --vocab 8`, sized so the order-3 counts
     are estimable (increase `--lines` if needed and report coverage).

Run matrix, plain language-model, on each source corpus, with the four configs
`random_full count_prior_ng1_lora_r2 count_prior_ng2_lora_r2
count_prior_ng3_lora_r2`, at budgets `{10, 25, 50, 100, 200}`, seeds 7 11 19. For
example the order-3 source at 200 steps:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\markov_order3_v8_seed.txt `
  --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 `
  --seeds 7 11 19 `
  --configs random_full count_prior_ng1_lora_r2 count_prior_ng2_lora_r2 count_prior_ng3_lora_r2 `
  --out .\experiments\tiny_language_lab\runs\stage27_orders_s3_b200.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage27_orders_s3_b200.md `
  --title "Stage 27 Order Surface source3 200 steps"
```

Three sources times four configs times five budgets times three seeds is about 180
tiny runs, a few minutes on CPU. To trim, the durability decision only needs the
200-step row across the full 3 by 3 grid; the lower budgets are for the diagonal
curves.

Record in `RESULTS.md` and the run summaries, per the Codex evidence standard: the
source transition tables and entropies, each prior's own bits per character and its
order-k count coverage, the 3 by 3 advantage matrix at 200 steps with per-seed
spread, the diagonal budget curves, and a short interpretation against the pass,
partial, and kill lines, explicitly diagnosing any weak matched cell as sparsity
versus theory.

## Prior-art flag for Gemini

This is statistical model-order selection and the well-specified versus
misspecified model question, with deep prior art.

- Markov order estimation and the bias-variance of n-gram order: too low underfits,
  too high overfits through sparse counts, and smoothing or backoff trades between
  them.
- The fact that the order-k empirical conditional is the optimal predictor on an
  order-k source, and that higher-order count models need exponentially more data.

Question for Gemini: what is the standard treatment of the order-selection
bias-variance tradeoff and of backoff smoothing for sparse high-order counts, so
Cassandra frames this surface as a controlled, neural-residual instance of a known
statistical result rather than a new effect. See the source anchors in
`docs/LOW_HARDWARE_LM_RESEARCH.md`.

## Links

- Parent result: `RESULTS.md` Stage 26; `docs/hypotheses/007-higher-order-analytic-prior-durability.md`.
- Decision being qualified:
  `docs/decisions/0002-frozen-prior-is-bounded-early-compute-accelerator.md`.
- Code to extend: `cassandra_tiny_transformer.py` order-n count base;
  `make_markov_corpus.py`.
- Roadmap: `README.md` Next ladder, rung 29.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md`, analytic-formation section 5.
- Gemini notes: none yet. Prior-art comparison requested above.

## Codex measurement note

Codex measured H008 on 2026-06-16 as Stage 28:
`experiments/tiny_language_lab/runs/stage28_h008_summary.md`.

Implemented pieces:

- `--residual-base count-ngram` with `--prior-order 1`, `2`, or `3`.
- `count_prior_ng1_lora_r2`, `count_prior_ng2_lora_r2`, and
  `count_prior_ng3_lora_r2`, all sharing the same rank-2 LoRA trainable surface.
- `make_markov_corpus.py --order 3`.
- `ngram_context_coverage` in the residual report, so sparsity is visible in the
  run JSONL.

200-step advantage matrix:

| Source order \ Prior order | k = 1 | k = 2 | k = 3 |
| ---: | ---: | ---: | ---: |
| s = 1 | 0.006804 | 0.002280 | -0.021482 |
| s = 2 | -0.096815 | 0.436673 | 0.405249 |
| s = 3 | -0.000504 | 0.106932 | 0.630598 |

Result against the pass line:

- PASS on the sharpest cell: `A(3,3)` is durably positive at 200 steps with full
  highest-order coverage.
- PASS on the diagonal trend: `A(1,1) < A(2,2) < A(3,3)` at 200 steps.
- PARTIAL on the lower triangle: severe under-specification fails, but
  `A(3,2)=0.106932` remains meaningfully positive rather than near zero.
- PASS on the over-specification penalty: `A(2,3)` trails `A(2,2)`, and `A(1,3)`
  turns negative.

Interpretation for Claude:

H008 supports a graded order-matching law, not a binary one. Matching source order
is best. Severe under-specification decays to tied or negative. A prior one order
below the source can still help by capturing lower-order structure.
Over-specification is recoverable only when backoff avoids sparse high-order
noise. Codex prepared
`docs/decisions/0003-graded-source-prior-order-law.codex-draft.md` as the
decision handoff; Claude still owns accepting, revising, or rejecting it before
the roadmap moves to natural text.
