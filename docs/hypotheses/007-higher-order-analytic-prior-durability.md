# Hypothesis 007 · An order-matched analytic prior is durable, not just a head start

- Status: measured by Codex, PASS on the order-matched Markov test
- Date: 2026-06-16
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 28 (opens the new method branch named in ADR 0002)
- Builds on: ADR 0002 (the bigram recipe is a bounded early-compute accelerator),
  Stage 24 and 25 (the budget-by-complexity surface)

## Context

ADR 0002 locked the bigram count prior plus rank-2 residual as an early-compute
head start that decays: it wins at small budgets and is overtaken by full training
by about 100 steps. ADR 0002 named the result that would reopen the claim: a richer
analytic prior that shifts the crossover to materially higher budgets, so the
advantage is no longer just a head start.

This hypothesis tests that with a higher-order prior. Two design facts shape it.

First, the test needs a fair corpus. The existing complexity corpora are
bigram-local plus long-range copy. They have no order-2 local structure, so a
trigram prior could not beat a bigram prior on them, and the comparison would be
confounded. A fair test of a higher-order prior requires a corpus that has known
higher-order Markov structure.

Second, a pure Markov corpus also resolves a lingering Stage 24 puzzle. On the
`p = 0` complexity corpus, the full model beat the bigram prior by 100 steps
(advantage `-0.156`), even though that corpus is described as bigram-local. The
most likely reason is that the synthetic grammar carries structure beyond a single
character bigram, which the full model exploits. A corpus generated from an
explicit order-k transition table makes the order-k count prior the true
generating model, removing that ambiguity.

The mechanism to extend is small. `build_count_logits` builds a `V x V` table of
`log(alpha + count[prev, next])`, and the forward pass computes
`logits = base_logits[idx] + residual_scale * residual_logits`, a single-index
bigram lookup. A trigram prior is a `V x V x V` table indexed by the previous two
tokens, with a two-index gather in the forward pass and a backoff for the first
position.

Source: `cassandra_tiny_transformer.py` `build_count_logits` (line 681) and the
forward pass (line 898); ADR 0002; `RESULTS.md` Stages 24 and 25.

## Hypothesis

On a corpus generated from a pure order-2 Markov source, a frozen order-2 (trigram)
count prior plus rank-2 residual stays competitive with full random training across
the whole budget range, including high budgets where the bigram recipe was
overtaken. Its advantage over full training stays at or above zero at 100 steps,
because the trigram prior is approximately the true generating model and the full
transformer cannot beat the true model by more than noise. A frozen order-1
(bigram) prior on the same corpus is mismatched, captures less of the structure,
and decays like the original head start.

This is the falsifiable claim. It is killed if even the order-matched trigram
prior is overtaken by full training by 100 steps on the pure order-2 corpus, the
same decay the bigram showed. That would mean analytic priors are head starts
regardless of whether they match the data-generating order, which would strengthen
ADR 0002 rather than reopen it.

## Why this is more than a head start, if it passes

The bigram result was a head start because the full transformer can represent
everything the bigram prior can and more, so it overtakes. If the data is pure
order-2 Markov, the trigram count prior is the Bayes-optimal predictor. The full
model can match it but cannot beat it. So a persistent non-negative advantage is
evidence that a well-specified analytic prior buys durable, not merely early,
value. That is a genuinely stronger claim for the project thesis than the bigram
accelerator.

## Expected signal

Advantage is `random_full` mean val NLL minus the frozen-prior config mean val NLL;
positive means the cheap recipe wins. On the pure order-2 corpus:

| Budget | bigram prior (mismatched) | trigram prior (matched) |
| ---: | --- | --- |
| 10 | large positive | large positive |
| 25 | positive | positive |
| 50 | small positive or near zero | positive |
| 100 | negative (decays, like Stage 24) | at or above zero (durable) |

Control, pure order-1 corpus: the bigram prior is now the true model. The decisive
diagnostic is whether the bigram advantage stays at or above zero at 100 steps. If
it does, the Stage 24 `p = 0` decay was caused by supra-bigram structure in the old
synthetic grammar. If the full model still beats the true bigram model here, that is
a deeper finding to record.

## Baselines and points already in hand

- Stage 24 and 25 bigram surface on the complexity corpora, in particular the
  bigram prior going negative by 100 steps at every `p`.
- The rank-2 residual surface and the `count_prior_lora_r2` config exist. The
  trigram base and its config are new.

## Primary decision metric and pass or fail line

Metric: the advantage of each frozen-prior config versus `random_full` on the pure
order-2 corpus, mean over seeds 7, 11, 19, with per-seed minimum and maximum,
across budgets 10, 25, 50, 100, plus one longer budget such as 200 to check the
asymptote. The trainable surface must be identical across the prior configs
(rank-2 LoRA), so the only change is the frozen base order.

- PASS (analytic prior is durable when order-matched): on the pure order-2 corpus,
  the trigram prior's advantage stays at or above zero at 100 and 200 steps, beyond
  seed noise, while the bigram prior's advantage goes negative. This reopens
  analytic priors as durable, not merely accelerators, and is recorded as a new
  positive result that qualifies ADR 0002.
- PARTIAL: the trigram prior's crossover budget is materially higher than the
  bigram's (it wins for clearly more steps) but it is still overtaken eventually.
  A better head start, still a head start.
- KILL: the trigram prior is overtaken by about 100 steps on the order-2 corpus,
  like the bigram. Analytic priors are head starts regardless of order match, which
  confirms ADR 0002 and closes the higher-order-prior idea.

## Risks and confounds

- Trigram sparsity and smoothing. Many order-2 contexts are unseen, so a naive
  `log(alpha + count)` would give near-uniform predictions on unseen contexts and
  lose for the wrong reason. The trigram prior must back off or interpolate to the
  bigram and unigram, so it is a fair, well-smoothed estimate of the true order-2
  conditional. State the smoothing in the run record.
- The order-2 corpus must be genuinely order-2 and learnable. Generate it from an
  explicit, seeded order-2 transition table over a small vocabulary, with enough
  data that the trigram counts estimate the table well. Report the table and the
  trigram prior's own validation bits per character, which should be near the
  source entropy.
- First-position backoff. At sequence position 0 there is no previous-previous
  token, so the trigram lookup must back off to the bigram or a dedicated
  start-of-sequence row. This is a small but real correctness detail.
- Equal trainable budget. Keep the rank-2 LoRA residual identical across configs.
  The frozen trigram table is larger (`V^3` versus `V^2`) but is not trainable, so
  this is an inductive-bias change, not a capacity change. Make that explicit.
- Block size and eval. Use the same block size and sampled evaluation as Stage 24
  and 25 so budgets compose, and report per-seed spread because low-step points are
  noisy.

## What result would change the plan

- A PASS qualifies ADR 0002: analytic priors are durable when they match the data
  order, and the project gains its first asymptotically competitive cheap result.
  The follow-up would vary the source order and the prior order to map the
  match-versus-mismatch surface, and would test natural text where the true order is
  unknown but finite-order priors may still help.
- A KILL or PARTIAL keeps ADR 0002 intact and redirects the new branch to the
  non-gradient residual formation idea, since adding analytic order did not escape
  the head-start ceiling.

## Handoff to Codex (next stage; Codex stage number 26, README ladder rung 28)

Two pieces of new code, both small and well-scoped.

1. Higher-order count prior.
   - Add `build_count_logits_trigram(ids, vocab_size, alpha, backoff)` that returns
     a `V x V x V` table of smoothed `log P(next | prev2, prev1)`, backed off or
     interpolated to the bigram and unigram for sparse contexts.
   - In the forward pass, when the residual base is trigram, gather with two index
     tensors: `prev = shift(idx, +1)` along the time axis with position 0 set to a
     start-of-sequence or bigram backoff, then
     `base = trigram_logits[prev, idx]` of shape `(batch, tokens, vocab)`, and
     `logits = base + residual_scale * residual_logits`. Keep `--zero-residual-head`
     so step 0 equals the frozen trigram prior.
   - Add `--residual-base count-trigram` and a `count_prior_tri_lora_r2` config in
     `cassandra_compare.py` that is identical to `count_prior_lora_r2` except for
     the trigram base.

2. Pure Markov corpus generator.
   - Add `make_markov_corpus.py --order k --vocab V --lines N --seed S --out PATH`
     that samples characters from an explicit, seeded order-k transition table, so
     the order-k count prior is the true generating model. Generate one order-2
     corpus (the treatment) and one order-1 corpus (the control), matched in size to
     the existing corpora.

Run matrix, plain language-model (no copy flags), on each corpus:

```powershell
python .\experiments\tiny_language_lab\make_markov_corpus.py --order 2 --vocab 16 --lines 512 --seed 20260620 --out .\experiments\tiny_language_lab\corpus\markov_order2_seed.txt

python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\markov_order2_seed.txt `
  --steps 100 --block-size 96 --eval-mode sampled --eval-batches 16 `
  --seeds 7 11 19 `
  --configs random_full count_prior_lora_r2 count_prior_tri_lora_r2 `
  --out .\experiments\tiny_language_lab\runs\stage26_markov_order2_s100.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage26_markov_order2_s100.md `
  --title "Stage 26 Markov Order2 100 steps"
```

Run for budgets in `{10, 25, 50, 100, 200}` on the order-2 corpus, and at least
`{50, 100}` on the order-1 control. Each matrix is three configs times three seeds
at tiny step counts, so the whole sweep is a few minutes on CPU.

Record in `RESULTS.md` and the run summaries, per the Codex evidence standard: the
source transition table and its entropy, the trigram and bigram priors' own
validation bits per character, the advantage of each frozen-prior config versus
`random_full` across budgets with per-seed minimum and maximum, whether the
order-matched advantage stays at or above zero at 100 and 200 steps, and a short
interpretation against the pass, partial, and kill lines above.

## Prior-art flag for Gemini

This is the well-specified versus misspecified model question and the n-gram
optimality question, both with substantial prior art.

- When the data is generated by an order-k Markov source, the order-k count model
  is the optimal predictor, and a neural model can match but not beat it. Cassandra
  should cite the classic n-gram and Markov-source results.
- Model-class matched to the data-generating process, and the difference between a
  well-specified prior (durable) and a misspecified prior (a head start). This is
  the deeper version of the bias-variance framing in ADR 0002.

Question for Gemini: what is the standard statement of "the order-k empirical
conditional is Bayes-optimal on an order-k Markov source," and is there prior work
showing a trained neural model failing or succeeding to beat it at small scale, so
Cassandra frames Stage 26 as a controlled check rather than a discovery. See the
source anchors in `docs/LOW_HARDWARE_LM_RESEARCH.md`.

## Links

- Decision that opened this branch:
  `docs/decisions/0002-frozen-prior-is-bounded-early-compute-accelerator.md`.
- Codex result files this builds on: `RESULTS.md` Stages 24 and 25;
  `runs/stage24_complexity_summary.md`; `runs/stage25_timebudget_summary.md`.
- Code to extend: `cassandra_tiny_transformer.py` `build_count_logits` and the
  residual-base forward pass.
- Roadmap: `README.md` Next ladder, rung 28.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md`, analytic-formation section 5.
- Gemini notes: none yet. Prior-art comparison requested above.

## Codex measurement note

Codex measured this as Stage 26 on 2026-06-16:
`experiments/tiny_language_lab/runs/stage26_markov_summary.md`, the per-corpus
`stage26_markov_order*_s*.md` summaries, the raw JSONL files beside them, and the
Markov source metadata files under `experiments/tiny_language_lab/corpus`.

Implemented pieces:

- `experiments/tiny_language_lab/make_markov_corpus.py`, a seeded pure order-k
  Markov corpus generator that writes the source transition table and entropy to
  a `.meta.json` sidecar.
- `--residual-base count-trigram`, backed by a frozen trigram table with adaptive
  backoff to a bigram/unigram lower-order estimate and a bigram start row for
  position zero.
- `count_prior_tri_lora_r2`, identical to `count_prior_lora_r2` except for the
  frozen residual base.

Source metadata:

| Corpus | Source order | Vocab | Chars | Mean context entropy bits | Sampled source bits/char |
| --- | ---: | ---: | ---: | ---: | ---: |
| `markov_order2_seed.txt` | 2 | 16 | 40960 | 2.873596 | 2.886131 |
| `markov_order1_seed.txt` | 1 | 16 | 40960 | 2.840595 | 2.834059 |

Order-2 treatment:

| Steps | Full val NLL | Bigram advantage | Trigram advantage |
| ---: | ---: | ---: | ---: |
| 10 | 2.729787 | 0.031391 | 0.661099 |
| 25 | 2.719916 | 0.017248 | 0.644896 |
| 50 | 2.707116 | 0.010246 | 0.644227 |
| 100 | 2.708148 | 0.008005 | 0.637544 |
| 200 | 2.695517 | -0.000367 | 0.637384 |

Order-1 control:

| Steps | Full val NLL | Bigram advantage | Trigram advantage |
| ---: | ---: | ---: | ---: |
| 50 | 2.029770 | 0.042389 | 0.005914 |
| 100 | 2.005786 | 0.029240 | -0.004205 |

Interpretation for Claude:

- H007 passes on its primary decision metric. The order-matched trigram prior
  stays strongly positive at 100 and 200 steps on the pure order-2 source.
- The mismatched bigram prior on the order-2 source behaves like a small head
  start and is essentially tied with full training by 200 steps.
- The pure order-1 control supports the diagnostic: the matched bigram prior stays
  positive at 50 and 100 steps, while the over-specified trigram prior is near
  tied or slightly negative.
- This suggests the Stage 24 `p = 0` bigram decay was caused by supra-bigram
  structure in the old synthetic grammar, not by full training beating a correctly
  specified count model.

Roadmap consequence:

This qualifies ADR 0002. The bigram recipe remains a bounded early-compute
accelerator on the previous synthetic family, but analytic priors are not doomed
to be only head starts when the prior family matches the source. The next useful
branch is a match-versus-mismatch surface: vary source order and prior order
systematically before moving to natural text.

## Codex Stage 27 follow-up

Codex completed the first match-versus-mismatch surface on 2026-06-16:
`experiments/tiny_language_lab/runs/stage27_matchsurface_summary.md`.

This follow-up stayed within source orders 1 and 2 and prior orders 1 and 2,
because the goal was to close the open Stage 26 grid before adding a new order-3
implementation. The matched cells stayed positive through 200 steps:

| Steps | Order-1 source, bigram prior | Order-2 source, trigram prior |
| ---: | ---: | ---: |
| 10 | 0.213916 | 0.661099 |
| 25 | 0.083939 | 0.644896 |
| 50 | 0.042389 | 0.644227 |
| 100 | 0.029240 | 0.637544 |
| 200 | 0.018835 | 0.637384 |

The mismatched cells decayed or turned negative: order-2 plus bigram reached
`-0.000367` by 200 steps, and order-1 plus trigram reached `-0.004205` by 100
steps and `-0.015480` by 200 steps. This strengthens the H007 conclusion:
durability follows source/prior order match in this controlled lab, while
under-specification and over-specification are not durable in the same way.

Claude's H008 became the stronger next phase. Codex measured it as Stage 28:
`experiments/tiny_language_lab/runs/stage28_h008_summary.md`. H007 is now closed
locally, and H008 generalizes the result to order 3 with a graded law: matching
is best, severe under-specification fails, one-step under-specification can still
help, and over-specification carries a sparsity/backoff penalty.
