# ADR 0003 Draft · The source/prior order law is graded, not binary

- Status: ACCEPTED WITH REVISIONS by Claude as ADR 0003. The decision of record is
  `docs/decisions/0003-graded-source-prior-order-law.md`. This file is retained as
  the Codex evidence draft that informed it.
- Date: 2026-06-16
- Author: Codex (execution and evidence role)
- Decision owner: Claude (hypothesis, ADR, and roadmap role)
- Decides: how to interpret the higher-order analytic-prior branch after H008
- Builds on: ADR 0002, Hypothesis 007, Hypothesis 008, Stages 26 to 28

## Context

ADR 0002 locked the original frozen bigram prior plus rank-2 residual surface as
a bounded early-compute accelerator. It was not an asymptotic replacement on the
Stage 24 corpus family, because full random training overtook it by about 100
steps.

Stages 26 to 28 changed the scope of that conclusion. The failure was not
"analytic priors are only head starts" in general. The failure was "the bigram
prior is misspecified for the source being tested."

Stage 26 implemented H007 with a pure order-2 Markov source and a matched frozen
trigram prior. The matched order-2 prior stayed strongly ahead through 200 steps.
Stage 27 filled the missing order-1/order-2 cells at `V = 16`. Stage 28 then
implemented H008 at `V = 8`, adding source order 3 and prior order 3 while
recording sparsity diagnostics.

The H008 decision metric was:

`A(s,k) = random_full mean validation NLL - count_prior_ng{k}_lora_r2 mean validation NLL`

Positive `A(s,k)` means the frozen-prior config beat full random training.

## Evidence

Stage 28 artifacts:

- aggregate summary:
  `experiments/tiny_language_lab/runs/stage28_h008_summary.md`
- aggregate JSONL:
  `experiments/tiny_language_lab/runs/stage28_h008_summary.jsonl`
- hypothesis:
  `docs/hypotheses/008-source-order-prior-order-surface.md`
- durable log:
  `experiments/tiny_language_lab/RESULTS.md`, Stage 28

The full `V = 8` 200-step matrix:

| Source order \ Prior order | k = 1 | k = 2 | k = 3 |
| ---: | ---: | ---: | ---: |
| s = 1 | 0.006804 | 0.002280 | -0.021482 |
| s = 2 | -0.096815 | 0.436673 | 0.405249 |
| s = 3 | -0.000504 | 0.106932 | 0.630598 |

Diagonal budget curves:

| Steps | A(1,1) | A(2,2) | A(3,3) |
| ---: | ---: | ---: | ---: |
| 10 | 0.078849 | 0.555962 | 0.633965 |
| 25 | 0.031066 | 0.542886 | 0.632985 |
| 50 | 0.014747 | 0.539947 | 0.628784 |
| 100 | 0.012231 | 0.532621 | 0.628518 |
| 200 | 0.006804 | 0.436673 | 0.630598 |

Highest-order coverage at 200 steps:

| Source order | Prior order | Frozen params | Initial val NLL | Highest-order coverage | Mean observed count |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 64 | 1.537964 | 1.000000 | 4351.875000 |
| 1 | 2 | 576 | 1.542779 | 0.968750 | 561.516113 |
| 1 | 3 | 5184 | 1.567934 | 0.787109 | 86.384613 |
| 2 | 1 | 64 | 1.924597 | 1.000000 | 4351.875000 |
| 2 | 2 | 576 | 1.385287 | 1.000000 | 543.968750 |
| 2 | 3 | 5184 | 1.416895 | 0.898438 | 75.680435 |
| 3 | 1 | 64 | 2.072102 | 1.000000 | 4351.875000 |
| 3 | 2 | 576 | 1.967223 | 1.000000 | 543.968750 |
| 3 | 3 | 5184 | 1.436815 | 1.000000 | 67.994141 |

## Proposed decision

Accept a graded source/prior order law for the controlled Markov branch:

1. Matching source order is best in the measured range.
2. The matched diagonal can be durable, not merely an early head start, when the
   frozen prior is well specified and its count table is estimable.
3. Severe under-specification decays to tied or negative.
4. One-step under-specification can still help by capturing lower-order structure.
5. Over-specification carries a sparsity and backoff penalty. It is recoverable
   when lower-order backoff is strong enough, but can become harmful when the
   high-order table is noisy or unnecessary.

This should replace the binary phrasing "durability requires k at least s" with a
more careful rule:

> Analytic frozen priors are durable in this lab when they are well specified
> enough for the source. Exact order match is best. Nearby lower-order priors can
> still help, but the advantage weakens as specification error grows.

## What this decision does not claim

- It is not a natural-language result. The evidence comes from synthetic pure
  Markov sources with `V = 8` or `V = 16`.
- It is not proof that a full transformer never catches up. The longest measured
  budget is 200 steps.
- It is not a claim that larger count tables are free. Frozen parameters grow as
  `V^(k+1)`, and count estimation becomes sparse quickly.
- It is not a claim of novelty. Gemini still needs to compare this to Markov
  order estimation, n-gram smoothing and backoff, bias-variance tradeoffs, and
  well-specified versus misspecified statistical models.

## Consequences

- The analytic-prior branch remains alive. Stage 28 is stronger than ADR 0002's
  original "bounded early-compute accelerator" conclusion because a well-matched
  order-3 prior stayed strongly ahead through 200 steps.
- The roadmap should stop repeating pure Markov order-grid cells unless a new
  confound is being tested, such as longer budgets, different smoothing, or data
  scarcity.
- The next executable branch should test external validity: finite-order priors
  on a tiny natural corpus where the true source order is unknown, or a
  non-gradient residual-formation method on top of the frozen prior.
- Before any public-facing novelty claim, Gemini should produce a source-backed
  research note on n-gram order selection and backoff smoothing.

## What would change this decision

- A longer-budget run where full random training catches the matched order-3
  prior would narrow the word "durable" to "durable through 200 steps."
- A natural-text run where all finite-order priors lose quickly would keep the
  law as a synthetic Markov result only.
- A natural-text run where a moderate-order prior remains competitive would
  upgrade this from a controlled proof-of-mechanism to a useful laptop-scale
  formation method.
- A non-gradient residual-formation method that improves the residual path on top
  of the frozen prior would move the project closer to the user's "train without
  training" north star.

## Recommended next steps

For Claude:

- Accept, revise, or reject this draft as ADR 0003.
- If accepted, decide whether the next hypothesis should be natural-text
  finite-order priors or non-gradient residual formation.

For Gemini:

- Research Markov order estimation, backoff smoothing, Katz or Kneser-Ney style
  language-model smoothing, and the bias-variance framing for over-specified
  n-gram models.
- Identify what wording Cassandra should use so Stage 28 is framed as a local
  neural-residual measurement of known statistical structure, not a new theorem.

For Codex:

- Wait for Claude's ADR decision before running another large matrix.
- Codex has already run the smallest exploratory next step as Stage 29:
  `experiments/tiny_language_lab/runs/stage29_tinyprose_ngram_summary.md`.
  Treat it as weak evidence only. A real natural-text stage should wait for
  Claude's ADR decision and Gemini's prior-art comparison.
