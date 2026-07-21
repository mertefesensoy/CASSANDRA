# Hypothesis 009b · Under an out-of-distribution validation split the natural-text order curve becomes humped

- Status: measured by Codex as Stage 32; local KILL on the implemented split, with source-choice caveat
- Date: 2026-06-22
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 34 (resolves the Stage 31 coverage confound left open by Hypothesis 009)
- Builds on: Stage 30 and Stage 31 (natural-text finite-order priors), Hypothesis
  009, and Gemini research note
  `research/theme_1_architecture_and_priors/06_neural_ngram_hybrids_and_coverage.md`

## Context and the confound

Hypothesis 009 predicted a humped advantage-versus-order curve on natural text, with
a coverage-bounded sweet spot at a moderate order. Stage 30 and Stage 31 measured
the opposite shape: the advantage is monotone increasing through order 4 at every
budget, including 500 steps. Order 4 had the largest advantage at every budget
(`+0.463572` mean NLL at 500 steps).

The coverage table explains why, and it is a confound, not a refutation. Order 4 had
only `0.029713` table coverage but `0.961867` validation-hit coverage. The corpus is
a single author, Tiny Shakespeare, split as a train prefix and a validation suffix.
The suffix reuses local phrases from the prefix, so even a very sparse order-4 table
almost never misses on validation. The high-order prior is acting as a memorized
lookup table for validation n-grams it already saw in training. Gemini note 06 states
the same reading: the 96 percent validation-hit coverage is artificially high, and a
harsher split became the required Stage 32 test.

So the Stage 30 and Stage 31 result is real but friendly-split. The bias-variance
sweet spot that Hypothesis 009 expected is masked, because the descending limb only
appears once high-order n-grams stop transferring from train to validation.

## Hypothesis

Under a validation split where high-order n-grams do not transfer from training,
validation-hit coverage will fall with prior order, and the advantage-versus-order
curve will become humped, with a true sweet spot at a moderate order (likely 2 or 3).
When high-order validation-hit coverage collapses, the order-4 prior injects sparse
backoff noise rather than memorized signal, so its advantage drops below a
moderate-order prior that still generalizes.

This is the falsifiable claim. It is killed if, under the harsher split, the
advantage curve is still monotone increasing in order and high-order validation-hit
coverage stays high. That would mean the high-order prior genuinely generalizes on
this text rather than memorizing, the Stage 31 result is not a split artifact, and
there is no masked sweet spot.

## How to build the harsher split

The point is to lower high-order train-to-validation n-gram transfer while keeping
the language and character set the same, so low-order structure still transfers and
the comparison stays a language-model comparison.

- Preferred: train on one source text and validate on a different source text in the
  same normalized character set, for example train on Tiny Shakespeare and validate
  on a different public-domain author or work. This produces a genuine distribution
  shift in phrasing while keeping orthography and basic syntax shared.
- Acceptable alternative: hold out whole structural units (scenes, acts, or
  documents) chosen to minimize shared high-order phrases, rather than a contiguous
  suffix of the same work.

The diagnostic that the split is actually harsher is the validation-hit-coverage
column: order 3 and order 4 hit coverage must drop well below the Stage 31 values of
`0.996` and `0.962`. If they do not drop, the split is not harsh enough and the test
is inconclusive, not a pass or a kill.

## Expected signal

Two curves at 200 and 500 steps, read together:

- Validation-hit coverage versus order: now decreasing with order, sharply for order
  3 and 4, instead of staying near 1.0.
- Advantage versus order: humped, rising from order 1 to a moderate sweet-spot order,
  then falling for the high orders whose coverage collapsed.

## Baselines and points already in hand

- Stage 30 and Stage 31 friendly-split advantage at 500 steps: ng1 `-0.266`, ng2
  `+0.110`, ng3 `+0.341`, ng4 `+0.464`, monotone increasing. This is the curve the
  harsher split should bend into a hump.
- Stage 31 validation-hit coverage: order 1 `1.000`, order 2 `0.99993`, order 3
  `0.996`, order 4 `0.962`. The harsher split must lower the order 3 and 4 values.

## Primary decision metric and pass or fail line

Metric: for each prior order 1 to 4, the advantage versus `random_full` on the
harsher validation set at 200 and 500 steps, mean over seeds 7, 11, 19 with per-seed
spread, reported alongside validation-hit coverage per order. The training corpus,
the model, and the rank-2 residual stay identical to Stage 30, so only the validation
distribution changes.

- PASS (the confound is real and the sweet spot is located): high-order
  validation-hit coverage drops materially, and the advantage curve is humped, with a
  moderate order strictly above the highest order at 200 and 500 steps. The true
  natural-text sweet spot is then this moderate order, and the Stage 31 monotone
  result is confirmed as a friendly-split artifact.
- PARTIAL: coverage drops and the high orders lose their lead, but the peak is flat
  across two adjacent orders rather than a single clear sweet spot.
- KILL: coverage stays high or the curve stays monotone increasing. The high-order
  prior genuinely transfers, the Stage 31 result is not a split artifact, and
  Hypothesis 009's humped prediction is wrong for this text.

## Risks and confounds

- A distribution shift makes the task harder for every model, including
  `random_full`. The decision is the shape of the advantage curve, full minus prior
  on the same shifted validation set, which is robust to a uniform shift. Report
  absolute NLLs too so a collapse of all configs is visible.
- Too-harsh shift. If the validation text shares almost nothing with training, even
  low-order priors lose and no sweet spot is visible. The hit-coverage column guards
  against this: order 1 and order 2 hit coverage should stay high while order 3 and 4
  fall. Pick a validation text in the same language and genre.
- Keep the normalization and character set identical to Stage 30, or the priors will
  miss for a tokenization reason rather than a generalization reason.

## What result would change the plan

- A PASS gives the project an honest natural-text sweet-spot order and corrects the
  Stage 31 over-reading. It also sets the order to use for any downstream natural-text
  work, including the Hypothesis 010 curriculum filter.
- A KILL means high-order count priors really do transfer on this kind of text, which
  is a stronger and more surprising positive, and would itself be worth a careful
  prior-art comparison with neural plus n-gram interpolation work.

## Handoff to Codex (measured by Stage 32; README ladder rung 34)

- Add a second normalized natural-text validation corpus in the same `V = 33`
  character set as `natural_text_seed.txt`, from a different public-domain source, via
  the existing `make_natural_text_corpus.py` path. Keep training on the Stage 30
  Shakespeare prefix. Build all count tables from the training split only and evaluate
  on the new out-of-distribution validation set.
- Re-run the Stage 30 matrix, configs `random_full count_prior_ng1_lora_r2
  count_prior_ng2_lora_r2 count_prior_ng3_lora_r2 count_prior_ng4_lora_r2`, at budgets
  100, 200, and 500, seeds 7 11 19, with the same `count-alpha 0.1` and
  `ngram-backoff 10` as Stage 30.
- Report per order: validation-hit coverage on the new split, advantage versus
  `random_full`, and each prior's own validation bits per character, with per-seed
  minimum and maximum. Name the sweet-spot order if a hump appears.

## Codex Stage 32 outcome

Codex measured H009b as Stage 32 with the required configs, budgets `100`, `200`,
and `500`, seeds `7 11 19`, `count_alpha=0.1`, and `ngram_backoff=10`. The
implemented corpus kept the Stage 30 Shakespeare training prefix and replaced the
validation suffix with normalized Cassandra project prose in the same `V = 33`
alphabet. This is a cross-domain character split, but it is not the preferred
second public-domain author or work, so the source remains a limitation.

Coverage dropped but did not collapse: order-4 validation-hit coverage moved from
Stage 31's `0.961867` to `0.889803`, and order 3 moved from `0.995997` to
`0.964350`. The advantage curve stayed monotone increasing through order 4 at all
three decision budgets. At 500 steps, ng4 advantage over `random_full` was
`+0.291191` NLL, ahead of ng3 at `+0.199130` and ng2 at `+0.016767`; ng1 was
negative at `-0.203702`.

This locally kills the predicted moderate-order hump for the implemented split.
It does not prove every harsher validation source would keep the monotone curve,
because order-3 and order-4 hit coverage remained high enough for substantial
transfer. The measured handoff to Claude is that no moderate sweet spot was
located, and order 4 remains the strongest local natural-text prior through Stage
32. The handoff to Gemini is to compare this against known n-gram domain-shift and
neural interpolation results before any external claim.

## Prior-art flag for Gemini

The mechanism is train-to-test n-gram overlap and the difference between table
coverage and hit coverage, which Gemini note 06 already covers. One addition: the
general phenomenon of n-gram language models scoring well on in-domain held-out text
because of phrase reuse, and failing under domain shift, is a standard result. Gemini
should cite the canonical statement so this stage is framed as a controlled
measurement of a known overlap effect.

## Links

- Parent hypothesis: `docs/hypotheses/009-natural-text-finite-order-prior-sweet-spot.md`.
- Gemini note: `research/theme_1_architecture_and_priors/06_neural_ngram_hybrids_and_coverage.md`.
- Codex results: `RESULTS.md` Stages 30 to 32; `runs/stage30_naturaltext_summary.md`;
  `runs/stage31_order4_extension_summary.md`;
  `runs/stage32_crossdomain_summary.md`.
- Decision context: `docs/decisions/0003-graded-source-prior-order-law.md`.
- Roadmap: `README.md` Next ladder, rung 34.




