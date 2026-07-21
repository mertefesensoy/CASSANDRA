# Hypothesis 006 · The frozen-prior advantage is largest at minimal compute and traces a crossover contour in the budget-by-complexity plane

- Status: measured by Codex, surface confirmed and mechanism clause complicated
- Date: 2026-06-16
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 26 (the low-compute completion of the Stage 24 surface)
- Builds on: Stage 24 (corpus-complexity sweep at 50 and 100 steps, Hypothesis
  005), Stage 5 and 6 (the original 50-step cheap-wins result)

## Context

Stage 24 measured the cheap-minus-full advantage, defined as `random_full` mean
val NLL minus `count_prior_lora_r2` mean val NLL, across the long-fraction axis at
two budgets. Positive means the cheap recipe won.

| Long fraction p | 50-step advantage | 100-step advantage |
| ---: | ---: | ---: |
| 0.00 | +0.059159 | -0.156425 |
| 0.25 | +0.098079 | -0.101866 |
| 0.50 | +0.078996 | -0.230854 |
| 0.75 | -0.066921 | -0.438576 |
| 1.00 | -0.162385 | -0.642078 |

At 50 steps the cheap recipe wins up to about `p = 0.5`, with the crossover
between `p = 0.5` and `p = 0.75`. At 100 steps the full model wins at every point,
including the most bigram-predictable `p = 0`. The honest reading, which Stage 24's
interpretation states, is that the frozen count prior plus tiny residual is an
early-training accelerator: it gives a better start under a tight budget, and the
full random transformer overtakes it given more steps. It is an efficiency effect
in the low-compute regime, not an asymptotic improvement.

This hypothesis completes the low-compute end of that surface, at 10 and 25 steps,
where the accelerator effect should be strongest and where the project's value
proposition (form behavior cheaply) actually lives.

Source files: `experiments/tiny_language_lab/RESULTS.md` Stage 24;
`runs/stage24_complexity_summary.md`; `make_complexity_corpus.py`.

## Hypothesis

Two linked claims about the advantage surface `advantage(p, steps)`:

1. Compute monotonicity. At every fixed `p`, the advantage decreases as steps
   increase. The cheap recipe's edge is maximal at the smallest budget and decays,
   crossing to negative as the full model catches up. So the 10-step advantage is
   greater than the 25-step advantage, which is greater than the 50-step value
   already in hand.
2. Moving crossover contour. The crossover long-fraction `p*(steps)`, where the
   advantage passes through zero, moves toward higher `p` as steps shrink. At 100
   steps `p*` is below 0; at 50 steps `p*` is near 0.6; at 10 and 25 steps `p*`
   should be higher, plausibly at or beyond `p = 1`, meaning the cheap recipe wins
   across the whole complexity range when compute is tiny.

Mechanism check (the concrete sub-claim from the goal directive): at 10 steps the
full random model is still near its untrained floor, with val NLL close to the
uniform baseline `ln(vocab_size)`, while the cheap recipe sits near its frozen
count-prior NLL. The advantage at 10 steps is therefore large and positive across
all `p`.

## Expected signal

Predicted advantage by budget, filling the two missing rows of the surface:

| Long fraction p | 10-step (predict) | 25-step (predict) | 50-step (have) | 100-step (have) |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | large positive | positive | +0.059159 | -0.156425 |
| 0.25 | large positive | positive | +0.098079 | -0.101866 |
| 0.50 | positive | positive | +0.078996 | -0.230854 |
| 0.75 | positive | near zero | -0.066921 | -0.438576 |
| 1.00 | positive or near zero | negative | -0.162385 | -0.642078 |

The exact magnitudes are to be measured by Codex. The shape claim is that each
column is decreasing left to right in step count, and that the zero-crossing moves
up in `p` as steps fall.

## Baselines and points already in hand

- The full 50-step and 100-step rows are measured (Stage 24, table above).
- The five complexity corpora are on disk. No new corpus code is needed.
- The random model's untrained NLL floor is approximately `ln(vocab_size)`, which
  Codex should report so the 10-step full-model NLL can be compared against it.

## Primary decision metric and pass or fail line

Metric: `advantage(p, steps)` at steps in `{10, 25}`, mean over seeds 7, 11, 19,
with per-seed minimum and maximum, combined with the Stage 24 rows at 50 and 100
to form the full surface and the crossover contour `p*(steps)`. The decision is on
the surface shape, not on any single cell.

- PASS (accelerator interpretation confirmed and mapped): at every `p`, the
  advantage decreases with steps, and `p*(steps)` moves toward higher `p` as steps
  fall, so the cheap recipe wins over more of the complexity range at lower budget.
  The mechanism check holds: at 10 steps the full model NLL is near
  `ln(vocab_size)` while the cheap recipe is near its count-prior NLL. This
  completes the surface and confirms the frozen prior is a low-compute accelerator.
- PARTIAL: the advantage is positive at 10 steps but the surface is non-monotonic
  in steps or noisy. The accelerator picture holds qualitatively but not cleanly.
- KILL or COMPLICATE: the 10-step advantage is not larger than the 50-step
  advantage at matched `p`, or the full model is already competitive at 10 steps
  (its NLL is far below `ln(vocab_size)`). Either would refute the early-training
  accelerator mechanism and mean Stage 24's budget dependence needs a different
  explanation, for example that the cheap recipe's edge is about optimization
  stability rather than a head start.

## Risks and confounds

- The "cheap wins at 10 steps" result is near-trivial on its own, because any
  informative initialization beats a barely-trained random model. The
  non-trivial content is the contour `p*(steps)` and the monotone decay with
  compute, so the decision is on the surface, not on the 10-step column alone.
- Very low step counts are noisy. Ten optimizer steps give a high-variance
  estimate, so per-seed minimum and maximum must be reported and the trend judged
  against that spread rather than from means alone.
- Fixed-step advantage is not the same as a time-to-quality measure. The cleaner
  operationalization of "efficiency" is steps or wall-clock to reach a target NLL.
  This hypothesis keeps the cheaper fixed-step surface that Stage 24 used so the
  rows compose; a time-to-threshold follow-up is noted as a later rung.
- The cheap recipe at 10 steps is near, but not exactly, the frozen prior, because
  the residual head trains from step 0. With the zero-residual-head default the
  step-0 output equals the prior, so 10 steps should stay close to it. Codex should
  confirm by reporting the cheap recipe's 10-step NLL against the count-prior NLL.

## What result would change the plan

- A PASS finishes the budget-by-complexity surface and gives the project its most
  precise honest statement: the frozen-prior recipe is an early-training
  accelerator whose advantage region is the low-budget, low-long-range corner,
  with a mapped crossover contour. That conclusion is strong enough to justify a
  consolidating ADR that fixes the scope of the core claim, which is the natural
  artifact for the next pass once this lands.
- A KILL or COMPLICATE would reopen why the cheap recipe helps at all, pointing to
  an optimization-stability study rather than a head-start story, and would defer
  the consolidating ADR.

## Handoff to Codex (next stage; Codex stage number 25, README ladder rung 26)

The corpora and matrix already exist. Run the Stage 24 plain-language-model matrix
at the two lower budgets. One command per corpus point per budget, for example
`p050` at 10 steps:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\complexity_p050_seed.txt `
  --steps 10 --block-size 96 --eval-mode sampled --eval-batches 16 `
  --seeds 7 11 19 `
  --configs random_full count_prior_head count_prior_lora_r1 count_prior_lora_r2 `
  --out .\experiments\tiny_language_lab\runs\stage25_timebudget_p050_s10.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage25_timebudget_p050_s10.md `
  --title "Stage 25 Time Budget p050 10 steps"
```

Run for each corpus in `{p000, p025, p050, p075, p100}` and for steps in
`{10, 25}`, so ten matrices. Each is four configs times three seeds at tiny step
counts, so the whole sweep is well under a minute of compute plus model setup.

Optional, only if cheap: a denser long-fraction grid near the 50-step crossover
(`p = 0.55, 0.60, 0.65, 0.70`) at 50 steps, to pin `p*(50)` precisely. This is a
nice-to-have, not required for the decision.

Record in `RESULTS.md` and the run summaries, per the Codex evidence standard: the
advantage at each `p` for steps 10 and 25 with per-seed minimum and maximum, the
assembled four-row surface (10, 25, 50, 100), the crossover `p*(steps)` at each
budget, the 10-step full-model NLL against `ln(vocab_size)` and the cheap recipe's
NLL against the count-prior NLL for the mechanism check, and a short interpretation
against the pass, partial, and kill lines above.

## Prior-art flag for Gemini

This is a learning-curve crossover and a warm-start question, both with prior art.
Specifically:

- Warm-starting and informed initialization that accelerate early training but are
  overtaken by models trained longer, including n-gram or statistical
  initialization of neural language models.
- Learning-curve and scaling crossovers, where a simpler or more biased model wins
  in the small-data or small-compute regime and a richer model wins asymptotically,
  which is the bias-variance tradeoff expressed along a compute axis.

Question for Gemini: what is the standard name for "advantage of a strong prior
decays with training budget and crosses zero," so Cassandra frames Stage 25 as a
controlled measurement of a known crossover rather than a new effect. See the
source anchors in `docs/LOW_HARDWARE_LM_RESEARCH.md`, especially nanoGPT and the
parameter-efficient and distillation entries.

## Links

- Parent hypothesis: `docs/hypotheses/005-cheap-recipe-corpus-regime.md`
  (measured; boundary confirmed at 50 steps, gone by 100).
- Codex result files this builds on: `RESULTS.md` Stage 24;
  `runs/stage24_complexity_summary.md`; `runs/stage24_complexity_p*_s50.md` and
  `_s100.md`.
- Roadmap: `README.md` Next ladder, rung 26.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md`, core-claim Stages 5 to 7 and
  the bounded-regime framing.
- Gemini notes: none yet. Prior-art comparison requested above.

## Codex measurement note

Codex measured this as Stage 25 on 2026-06-16:
`experiments/tiny_language_lab/runs/stage25_timebudget_summary.md`, the
per-corpus `stage25_timebudget_p*_s10.md` and `stage25_timebudget_p*_s25.md`
summaries, and the raw JSONL files beside them.

Measured cheap-minus-full advantage, defined as `random_full` mean val NLL minus
`count_prior_lora_r2` mean val NLL:

| Long fraction | Count-bigram bits | 10-step advantage | 25-step advantage | 50-step advantage | 100-step advantage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 2.905610 | 0.688454 | 0.247912 | 0.059159 | -0.156425 |
| 0.25 | 3.105543 | 0.701242 | 0.278451 | 0.098079 | -0.101866 |
| 0.50 | 3.030076 | 0.787087 | 0.299924 | 0.078996 | -0.230854 |
| 0.75 | 2.792105 | 0.856700 | 0.247387 | -0.066922 | -0.438575 |
| 1.00 | 2.483352 | 0.948164 | 0.241261 | -0.162385 | -0.642078 |

Crossover contour:

| Steps | p*(steps) |
| ---: | --- |
| 10 | above the measured range, `> 1.00` |
| 25 | above the measured range, `> 1.00` |
| 50 | about `0.635343` |
| 100 | below the measured range, `< 0.00` |

Interpretation for Claude:

- The main surface claim is supported. At every measured `p`, the advantage
  decreases as steps increase: 10 > 25 > 50 > 100.
- The crossover contour moves in the expected direction. At 10 and 25 steps the
  cheap recipe wins across all measured corpus points; at 50 steps it crosses
  between `p = 0.50` and `p = 0.75`; by 100 steps the full model wins everywhere.
- The mechanism clause is only partial. The 10-step full model starts near
  `ln(vocab_size)`, but after 10 steps it is already about `0.89` to `1.09` NLL
  below the uniform floor across the corpus points. The random model is learning
  quickly, just not quickly enough to catch the frozen-prior surface.
- The cheap recipe stays close to the pure count-prior NLL at 10 steps, within
  about `0.001` to `0.008` NLL in the measured points.

Roadmap consequence:

H006 confirms the budget-by-complexity contour and partially complicates the
simple head-start mechanism. The next best roadmap artifact is a consolidating
ADR from Claude that scopes Cassandra's current core claim as an early-compute
inductive-bias advantage, not an asymptotic replacement for full training.
