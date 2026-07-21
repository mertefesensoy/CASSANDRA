# Hypothesis 009 · On natural text the finite-order prior advantage is humped, with a coverage-bounded sweet spot

- Status: implemented by Codex as Stage 30, awaiting Claude decision
- Date: 2026-06-16
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 33 (the natural-text external-validity branch chosen in ADR 0003)
- Builds on: ADR 0003 (the graded source/prior order law on synthetic Markov
  sources), Stage 29 (the tiny-prose smoke), and Gemini research note
  `research/theme_1_architecture_and_priors/05_ngram_order_selection_and_bias_variance.md`

## Framing, from the Gemini prior-art note

This stage is a controlled neural-residual measurement of a known statistical
structure, not a novelty claim and not a claim of beating transformers. Natural
language has effectively unbounded Markov order, so every finite count prior is
misspecified. Classical n-gram order selection and the bias-variance tradeoff
predict that a low order is too biased to help much and a high order is too sparse
to be reliable, so an intermediate order should be the sweet spot. Backoff and
Kneser-Ney smoothing exist precisely because low-order statistics are a stable
foundation when high-order counts are sparse. Hypothesis 009 measures where that
sweet spot is for a frozen n-gram residual base under a tiny gradient budget.

## Context

ADR 0003 established a graded order-matching law on pure Markov sources: a matched
prior is durable, the matched advantage grows with source order, and
specification error in either direction degrades the advantage gradually. That law
is synthetic only. The decisive external-validity question is whether any finite
count prior keeps a durable early-compute edge on natural text, where the effective
order is high and unknown.

Stage 29 hinted yes but is untrustworthy. On a 1,129-character prose corpus the
order-3 prior had only about 2 percent coverage and the full model was overfitting,
so the apparent advantage was a two-bad-models artifact. The Gemini note makes the
requirement explicit: the corpus must be large enough that the optimal order is set
by the bias-variance curve, not by artificial starvation, and coverage must be
reported per order.

## Hypothesis

On a natural-text corpus large enough that low-order count tables are well covered,
the advantage of a frozen order-k prior plus rank-2 residual over full random
training, measured as `random_full` mean val NLL minus the prior config mean val
NLL, is a humped function of the prior order k. It rises from order 1 (too biased)
to an intermediate sweet-spot order (likely 2, possibly 3), then falls as higher
orders lose coverage and inject sparsity noise. The sweet-spot prior holds a
positive advantage through the measured budget.

This is the falsifiable claim. It has two distinct failure modes:

- KILL or BOUND: no finite-order prior holds a durable advantage on natural text,
  with all advantages decaying to tied or negative by the high budget. Then the
  ADR 0003 durability law is bounded to synthetic Markov sources, because real
  text's effective order exceeds any estimable count prior. This is an important
  honest limit, not a soft result.
- INCONCLUSIVE through starvation: coverage collapses for every order above 1 even
  on this corpus, so the sweet spot cannot be located. Then the corpus is too small
  or the vocabulary too large, and the fix is more data or a smaller character set,
  not a conclusion about the theory.

## Expected signal

Advantage versus prior order at a durability budget (200 steps), schematic:

| Prior order k | coverage | advantage |
| ---: | --- | --- |
| 1 | full | positive but modest (high bias) |
| 2 | good | largest (the sweet spot) |
| 3 | partial | positive but smaller (variance starts to bite) |
| 4 | poor | near zero or negative (over-specified, sparse) |

The two curves to read together are advantage versus order and coverage versus
order. The sweet spot is where advantage peaks, and it should sit at or just below
the order where coverage starts to collapse.

## Baselines and points already in hand

- Stage 29 tiny-prose smoke: ng1 `+0.218`, ng2 `+0.350`, ng3 `+0.400` at 100
  steps, but coverage was `1.000`, `0.277`, `0.021`, so only ng1 is trustworthy
  there and the corpus is far too small. Treat Stage 29 as a machinery check, not a
  baseline.
- Synthetic ADR 0003 surface: matched priors durable, the comparison anchor for
  what durable looks like.

## Primary decision metric and pass or fail line

Metric: the advantage of each `count_prior_ng{k}_lora_r2` config versus
`random_full` on the natural corpus, mean over seeds 7, 11, 19, with per-seed
minimum and maximum, across budgets 10, 25, 50, 100, 200, and 500, reported with
the highest-order coverage of each prior and each prior's own validation bits per
character. The trainable surface is identical rank-2 LoRA in every prior config.

- PASS (the law usefully transfers to natural text): some order, most likely 2,
  gives a positive advantage that survives to 200 and 500 steps, and the
  advantage-versus-order curve is humped, peaking at an order with good coverage.
  Finite-order priors are then a useful early-compute accelerator on real text with
  a findable sweet spot, which is the project's first natural-language positive.
- PARTIAL: a low-order prior holds a durable advantage but the curve is monotone
  decreasing rather than humped, or the advantage is positive early but decays
  toward tied by 500 steps as in the ADR 0002 bigram case. Useful but only an
  accelerator, not durable, on natural text.
- KILL or BOUND: no order holds a durable advantage to 500 steps. The durability
  law is bounded to synthetic sources.

## Risks and confounds

- Coverage and sparsity are the central risk. Report highest-order coverage for
  every prior. A weak high-order cell must be diagnosed as starvation versus a real
  variance effect using its coverage, exactly as the `V = 8` control did for the
  synthetic surface.
- Smoothing quality matters more on natural text than on synthetic Markov data. A
  weak add-alpha smoother would make a moderate order fail for the wrong reason.
  Use a strong backoff, Katz or Kneser-Ney style interpolation from order k down to
  1, and state the method in the run record. This is the natural-text analog of the
  synthetic coverage control.
- Memory. The frozen table is `V^(k+1)`. At a natural vocabulary of about 65 the
  order-3 table is already tens of millions of entries and order 4 is impractical.
  Normalizing to a moderate character set, for example lowercased letters, space,
  and a little punctuation at about `V = 30`, makes orders 2 and 3 estimable and
  order 4 a coverage-bounded edge case. Document the normalization, because it
  changes the result.
- Durable means flat through the measured budget. On a large natural corpus a tiny
  model at 200 to 500 steps is far from converged, so the advantage again reflects
  early training. Keep the same caveat as ADR 0003 and treat the asymptote as
  untested.
- Split hygiene. Use a deterministic train and validation split with no leakage,
  and build the count tables only from the training split.

## What result would change the plan

- A PASS is the project's first evidence that analytic priors help on real text and
  promotes the analytic-prior thread from a synthetic proof of mechanism to a
  candidate laptop-scale method. The follow-up would vary corpus size to map how the
  sweet spot and its advantage scale with data, and compare smoothing methods.
- A KILL bounds the whole analytic-prior program to synthetic structure and
  redirects the roadmap to the non-gradient residual-formation branch, since
  pushing finite-order priors on natural text would not pay off.

## Handoff to Codex (next stage; Codex stage number 30, README ladder rung 33)

The `count-ngram` base and the `count_prior_ng{1,2,3}_lora_r2` configs already
exist from Stage 28. New work is the corpus and the smoothing.

1. Natural corpus.
   - Add a deterministic natural-text corpus, for example tiny Shakespeare or a
     public-domain text of at least about one million characters, normalized to a
     moderate character set near `V = 30` (lowercase letters, space, sentence
     punctuation). Record the source, the normalization, the vocabulary, the
     character count, and a fixed train and validation split.
2. Smoothing.
   - Ensure the order-n count base uses multi-level backoff with a strong smoother,
     Katz or Kneser-Ney style, and report which one. If only add-alpha is available,
     state it as a limitation so a moderate-order failure is not misread.
3. Optional order 4.
   - Add `count_prior_ng4_lora_r2` if memory allows at the chosen vocabulary;
     otherwise report that order 4 is omitted for coverage and memory reasons, which
     is itself informative.

Run matrix, plain language-model:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\natural_text_seed.txt `
  --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 `
  --seeds 7 11 19 `
  --configs random_full count_prior_ng1_lora_r2 count_prior_ng2_lora_r2 count_prior_ng3_lora_r2 `
  --out .\experiments\tiny_language_lab\runs\stage30_naturaltext_b200.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage30_naturaltext_b200.md `
  --title "Stage 30 Natural Text Finite-Order 200 steps"
```

Run at budgets `{10, 25, 50, 100, 200, 500}`, seeds 7 11 19, four or five configs.
Natural text with a larger vocabulary is heavier than the synthetic runs, but a
single corpus at six budgets and a few configs remains a laptop-scale CPU job.

Record in `RESULTS.md` and the run summaries, per the Codex evidence standard: the
corpus source and normalization, the smoothing method, each prior's highest-order
coverage and own bits per character, the advantage-versus-order curve at each
budget with per-seed spread, the located sweet-spot order, and a short
interpretation against the pass, partial, and kill lines, diagnosing any weak cell
as coverage versus variance.

## Codex implementation note

Codex implemented this as Stage 30 on 2026-06-17. The corpus was normalized Tiny
Shakespeare (`1,100,721` chars, `V = 33`), with recursive add-alpha interpolation
using `count_alpha=0.1` and `ngram_backoff=10`. The matrix ran budgets `10`, `25`,
`50`, `100`, `200`, and `500`, seeds `7`, `11`, and `19`, and configs
`random_full`, `count_prior_ng1_lora_r2`, `count_prior_ng2_lora_r2`, and
`count_prior_ng3_lora_r2`.

Result summary: natural-text transfer is positive, but the humped sweet-spot law
is not closed. At 500 steps, ng2 remained positive at `+0.110081` mean NLL
advantage and ng3 remained positive at `+0.340641`, while ng1 fell to
`-0.266069`. The measured curve is monotone increasing through order 3, so the
descending limb remains unmeasured. See
`experiments/tiny_language_lab/runs/stage30_naturaltext_summary.md` and
`experiments/tiny_language_lab/RESULTS.md` Stage 30.

Codex then ran the optional order-4 extension as Stage 31. Order 4 was feasible
on the normalized corpus (`42,802,056` frozen logits) and became the best
measured prior at every budget. At 500 steps, ng4 advantage was `+0.463572`.
This strengthens the natural-text transfer result but still does not close the
humped sweet-spot law, because order-4 validation hit coverage remained high
(`0.961867`). See
`experiments/tiny_language_lab/runs/stage31_order4_extension_summary.md`.

## Prior-art flag for Gemini

The order-selection and bias-variance framing is covered by Gemini note 05. One
further comparison may be needed before any novelty wording: combining a frozen
n-gram distribution with a neural model is classical, including n-gram and neural
interpolation and residual or backoff hybrids. If a later stage claims the LoRA
residual is specifically learning the high-order complement of the n-gram prior,
Gemini should compare that to existing neural plus n-gram interpolation work so
Cassandra frames it as a small instance, not a new mechanism.

## Links

- Decision that chose this branch:
  `docs/decisions/0003-graded-source-prior-order-law.md`.
- Gemini prior-art note:
  `research/theme_1_architecture_and_priors/05_ngram_order_selection_and_bias_variance.md`.
- Codex results: `RESULTS.md` Stage 29; `runs/stage29_tinyprose_ngram_summary.md`.
- Code: `cassandra_tiny_transformer.py` count-ngram base; `cassandra_compare.py`
  `count_prior_ng{1,2,3}_lora_r2` configs.
- Roadmap: `README.md` Next ladder, rung 33.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md`, analytic-formation section and
  the TinyStories and nanoGPT anchors.
