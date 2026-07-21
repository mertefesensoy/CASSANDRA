# ADR 0003 · The source-order by prior-order law is graded, and analytic priors are durable when well specified

- Status: Accepted
- Date: 2026-06-16
- Author: Claude (hypothesis, ADR, and roadmap role)
- Accepts with revisions: the Codex evidence draft
  `docs/decisions/0003-graded-source-prior-order-law.codex-draft.md`, which is
  superseded by this record
- Decides: how to state the higher-order analytic-prior result after H008, and
  the next branch
- Builds on: ADR 0002, Hypotheses 007 and 008, Stages 26 to 29

## Context

ADR 0002 locked the original frozen bigram prior as a bounded early-compute
accelerator: on the Stage 24 corpus family the full model overtook it by about 100
steps. Hypothesis 007 then showed that this was not a universal property of
analytic priors but a property of a misspecified one. On a pure order-2 Markov
source a matched trigram prior stayed durable through 200 steps.

Hypothesis 008 generalized the test to a 3 by 3 source-order by prior-order grid at
vocabulary 8, the small vocabulary chosen so the order-3 count table is estimable
rather than data starved. Codex implemented and measured it as Stage 28 and
prepared an evidence draft for this decision. This ADR accepts that evidence and
states the law, with revisions.

## Evidence

Stage 28, `V = 8`, advantage `A(s, k) = random_full` mean val NLL minus
`count_prior_ng{k}_lora_r2` mean val NLL at 200 steps, seeds 7, 11, 19. Positive
means the frozen-prior config won.

| Source s \ Prior k | k = 1 | k = 2 | k = 3 |
| ---: | ---: | ---: | ---: |
| s = 1 | +0.006804 | +0.002280 | -0.021482 |
| s = 2 | -0.096815 | +0.436673 | +0.405249 |
| s = 3 | -0.000504 | +0.106932 | +0.630598 |

Diagonal budget curves confirm flatness, not slow decay, for the higher orders:
`A(2, 2)` is `0.556` at 10 steps and `0.437` at 200; `A(3, 3)` is `0.634` at 10
steps and `0.631` at 200.

Coverage diagnostics matter. The sharpest cell `A(3, 3)` had full highest-order
coverage (`1.000`, mean observed count about 68), so its strong durable advantage
is a fair test and not a backoff artifact. Over-specified cells had partial
coverage (`A(2, 3)` at `0.898`, `A(1, 3)` at `0.787`), which is the source of their
penalty.

Full Stage 28 numbers, coverage, and the tiny-prose Stage 29 smoke are in
`experiments/tiny_language_lab/RESULTS.md` Stages 28 and 29, and in the run
summaries.

## Decision: accept the graded law

Accept the following statement for the controlled Markov branch, replacing the
binary phrasing "durability requires `k` at least `s`" from Hypothesis 008.

> On a pure order-s Markov source, a frozen order-k count prior plus a rank-2
> residual gives an advantage over full training that is largest at exact match
> (`k = s`), grows with the source order `s`, and degrades gradually with
> specification error in either direction. Under-specification (`k < s`) loses
> higher-order structure; over-specification (`k > s`) adds count-sparsity noise
> that backoff only partly absorbs. The matched advantage is durable through the
> measured 200-step budget when the matched prior's high-order table is estimable.

Five measured facts support this:

1. The matched diagonal is positive and grows with source order:
   `A(1,1)=0.0068`, `A(2,2)=0.437`, `A(3,3)=0.631`.
2. The matched advantage is durable through 200 steps where coverage is full, and
   the order-3 result generalizes the Stage 26 finding.
3. Severe under-specification fails: `A(2,1)=-0.097` and `A(3,1)=-0.001`.
4. One-step under-specification still helps: `A(3,2)=0.107`. The law is graded,
   not a hard threshold.
5. Over-specification is penalized but often recoverable: `A(2,3)=0.405` trails the
   matched `A(2,2)=0.437`, while `A(1,3)=-0.021` turns slightly harmful where
   coverage is lowest.

## Revisions and nuances added to the Codex draft

- The advantage is always relative to how slowly the full model learns the source.
  This explains an apparent anomaly: `A(3,1)=-0.001` (tied) is better than
  `A(2,1)=-0.097` (negative), even though the bigram prior is more under-specified
  on the order-3 source. On the harder order-3 source the full model is also slow,
  so even a weak prior ties it; on the order-2 source the full model trains fast
  enough to beat the under-specified bigram. The advantage is a race between the
  prior's specification and the full model's learning speed, not a property of the
  prior alone.
- Coverage is the right diagnostic for whether a weak matched or over-specified
  cell reflects theory or data starvation. The `V = 8` design made the order-3
  diagonal a fair test, which is why this ADR can accept the order-3 result.
- Durable means flat through 200 steps, not proven asymptotic. The full model's
  eventual catch-up is untested beyond 200 steps and is named as a reopening
  condition.

## Scope and what this decision does not claim

- It is a synthetic pure-Markov result at `V = 8` and `V = 16`, on a tiny CPU
  model, measured by plain validation NLL. It is not a natural-language claim. The
  Stage 29 tiny-prose smoke is weak two-bad-models evidence, not validity: the
  order-3 prior there had only about 2 percent coverage and the full model was
  overfitting a 1,129-character corpus.
- It is not proof that a full transformer never catches a matched prior, only that
  it does not within 200 steps.
- It is not free: frozen parameters grow as `V^(k+1)` and count estimation becomes
  sparse quickly, so the usable order is data bounded.
- It is not a novelty claim. This is very likely a neural-residual restatement of
  classic n-gram order selection, backoff smoothing, and the bias-variance of model
  order. Gemini must produce the prior-art note before any external framing.

## Consequences

- The analytic-prior branch is the project's strongest positive thread and stays
  alive, now upgraded from ADR 0002's bounded accelerator to a graded durability
  law on controlled sources.
- Stop adding pure-Markov order-grid cells unless a new confound is being tested
  (longer budgets to probe the asymptote, alternative smoothing, or deliberate data
  scarcity). The grid is characterized.
- The next executable branch is external validity on natural text, not more
  synthetic cells and not yet non-gradient residual formation. Natural text is the
  decisive question: does a moderate-order prior keep a durable edge where the true
  order is high and unknown, or do all finite-order priors lose once the full model
  has real structure to learn?
- The natural-text stage must avoid the Stage 29 trap. Use a corpus large enough
  that the order sweet spot is data-driven, not starvation-driven, and that the full
  model can train without immediately overfitting. Report order-k coverage so the
  bias-variance sweet spot is visible.

## What would reopen or change this decision

- A longer-budget run where full training catches the matched order-3 prior would
  narrow "durable" to "durable through N steps."
- A natural-text run where all finite-order priors lose quickly would confine the
  law to synthetic Markov sources.
- A natural-text run where a moderate-order prior stays competitive would upgrade
  this from a proof of mechanism to a useful laptop-scale formation method, and
  would justify a stronger ADR.

## Next steps and ownership

- Gemini (prerequisite): produce a source-backed note on Markov order selection,
  Katz and Kneser-Ney backoff smoothing, and the bias-variance of over-specified
  n-gram models, so Cassandra frames Stage 28 as a controlled neural-residual
  measurement of known statistical structure rather than a new result.
- Claude (next pass): after the Gemini note, spec Hypothesis 009, natural-text
  finite-order priors, with the corpus-size and coverage requirements above. The
  non-gradient residual-formation idea remains the following branch.
- Codex: hold large new matrices until Hypothesis 009 is specced. Stage 29 already
  served as the smallest exploratory probe.

## Links

- Superseded evidence draft:
  `docs/decisions/0003-graded-source-prior-order-law.codex-draft.md`.
- Prior decisions: `docs/decisions/0002-frozen-prior-is-bounded-early-compute-accelerator.md`.
- Hypotheses: `docs/hypotheses/007-higher-order-analytic-prior-durability.md`,
  `docs/hypotheses/008-source-order-prior-order-surface.md`.
- Codex results: `RESULTS.md` Stages 26 to 29; `runs/stage28_h008_summary.md`;
  `runs/stage29_tinyprose_ngram_summary.md`.
- Roadmap: `README.md` Next ladder.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md`.
