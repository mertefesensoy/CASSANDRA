# Hypothesis 005 · The frozen-prior advantage is a graded function of corpus long-range fraction

- Status: measured by Codex, boundary confirmed at 50 steps and shifted below
  `p = 0` by 100 steps
- Date: 2026-06-16
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 25 (the ADR 0001 redirect)
- Builds on: Stage 5 and Stage 6 (cheap recipe wins on the structured corpus),
  Stage 7 (full model wins on the long-context corpus), ADR 0001 (which retired
  the retrieval branch and named this characterization as the redirect)

## Context

After the retrieval branch closed (ADR 0001), the project's strongest capability
claim is the pre-retrieval one: a frozen count prior plus a tiny residual surface
beats a full random transformer under a fixed budget on plain language-model loss.
That claim currently rests on two corpora that disagree:

- Structured corpus, bigram-local, 50 steps (Stage 6): `count_prior_lora_r2`
  mean val NLL `1.992162` versus `random_full` `2.078493`. The cheap recipe wins
  by about `+0.086` NLL.
- Long-context corpus, long-range copy dependency, 100 steps (Stage 7):
  `random_full` mean val NLL `0.896153` versus `count_prior_lora_r2` `1.559663`.
  The full model wins by about `-0.66` NLL.

Two corpora, two opposite verdicts. The honest version of the core claim is not
"the cheap recipe wins" but "the cheap recipe wins when the corpus is
bigram-predictable and loses when long-range dependence dominates." That boundary
is not characterized. This hypothesis characterizes it.

The infrastructure already exists. `make_complexity_corpus.py` mixes
bigram-local structured lines with long-context copy lines at a tunable
`--long-fraction p`, and five deterministic corpora are already generated:
`corpus/complexity_p000_seed.txt` through `complexity_p100_seed.txt`, with
`p` in `{0.00, 0.25, 0.50, 0.75, 1.00}`, all `lines 512`, `seed 20260619`.

Source files: `experiments/tiny_language_lab/RESULTS.md` Stages 5, 6, and 7;
`make_complexity_corpus.py`; ADR 0001.

## Hypothesis

On the corpus-complexity axis, the cheap-minus-full advantage, defined as
`random_full` mean val NLL minus `count_prior_lora_r2` mean val NLL, is a
decreasing function of the long-range fraction `p`. It is positive at low `p`
(the frozen bigram prior captures most of the predictability), crosses zero at
some crossover `p*`, and is negative at high `p` (the full transformer is needed
to model the long-range copy dependency). A positive advantage means the cheap
recipe wins, because lower NLL is better.

This is the falsifiable claim. It is killed if the advantage does not depend on
`p`, either staying positive everywhere (the cheap recipe wins regardless of
long-range content, which would be a stronger and surprising result) or staying
negative everywhere (the Stage 5 and 6 win does not even reproduce at `p = 0` of
this sweep, meaning it was specific to the original structured corpus).

## Expected signal

Cheap-minus-full advantage (`random_full` minus `count_prior_lora_r2` mean val
NLL), expected sign by corpus point:

| Corpus | long-fraction p | Expected advantage |
| --- | ---: | --- |
| complexity_p000 | 0.00 | positive, near the Stage 6 `+0.086` |
| complexity_p025 | 0.25 | positive but smaller |
| complexity_p050 | 0.50 | near zero, the likely crossover region |
| complexity_p075 | 0.75 | negative |
| complexity_p100 | 1.00 | negative, toward the Stage 7 direction |

The crossover `p*` is the value of `p` where the advantage passes through zero.
Its exact location is to be measured by Codex and is expected to depend on the
training budget.

## Baselines and points already in hand

- Structured, 50 steps: advantage about `+0.086` (Stage 6).
- Long-context, 100 steps: advantage about `-0.66` (Stage 7).
- The five complexity corpora are generated and on disk. No new corpus code is
  needed for the primary sweep.

## Primary decision metric and pass or fail line

Metric: the cheap-minus-full val-NLL advantage at each `p`, mean over seeds 7,
11, 19, with per-seed minimum and maximum. The decision is about the advantage
and its trend across `p`, not the absolute NLL, because NLL is not comparable
across corpora of different intrinsic entropy. Report `count_prior_head` and
`count_prior_lora_r1` as well for shape, and report a bigram-predictability
measure per corpus (for example the pure count-bigram model's validation bits per
character) so the advantage can be plotted against measured complexity rather than
only against the design knob `p`.

- PASS (boundary characterized): the advantage is positive at `p = 0`, negative
  at `p = 1`, and decreases across the sweep, so a crossover `p*` exists and can
  be named at each budget. This converts the two anecdotes into a graded,
  honest claim with a stated scope.
- PARTIAL (boundary exists but not cleanly graded): the advantage is positive at
  `p = 0` and negative at `p = 1` but non-monotonic in between. The boundary is
  still established; the gradation is noisier than expected.
- KILL or NULL (claim does not behave as a regime boundary): the advantage does
  not change sign across `p`. If it stays positive, the cheap recipe is more
  general than Stage 7 implied and the framing should change. If it stays
  negative, the Stage 5 and 6 win was specific to the original structured corpus
  and does not even hold at `p = 0` here, which materially weakens the core claim
  and should be recorded as such.

## Risks and confounds

- Budget dependence. The crossover location depends on training steps, because the
  full model needs steps to exploit long-range structure while the cheap recipe
  saturates early. The Stage 5 and 6 win is a 50-step result and the Stage 7 loss
  is a 100-step result, so the sweep should be run at both 50 and 100 steps and
  the two advantage-versus-`p` curves compared. More budget should shift the
  crossover toward lower `p`.
- NLL not comparable across corpora. Different `p` corpora have different
  intrinsic entropy, so raw NLL cannot be compared across points. Only the
  within-corpus advantage (same corpus, two models) is comparable across `p`. The
  decision metric is the advantage, which controls for this.
- Overfitting at low `p` and high steps. On a small bigram-local corpus, 100 steps
  can overfit both models. Because the advantage is a difference, it is more
  robust to shared overfitting than absolute NLL, but Codex should still watch the
  per-seed spread.
- The `p = 0` point is not identical to the Stage 6 corpus. It uses the same
  structured line generator but a different line count, seed, and block size, so
  it should reproduce the cheap-wins direction but not the exact `+0.086` number.
- Mixed-line evaluation. Sampled evaluation draws windows across both line types,
  so val NLL at a given `p` reflects the mixture, which is the intended behavior.

## What result would change the plan

- A PASS gives the project a clean, bounded, defensible headline: the cheap recipe
  wins below a measured long-range fraction and loses above it. That is a real
  contribution to the north star, naming when reduced-training formation beats
  brute force. It would motivate a follow-up that varies the budget more finely or
  swaps the synthetic long-range task for a different long-range structure.
- A KILL or NULL would force an honest downgrade of the core claim and redirect to
  either a different cheap-formation method (the analytic and search methods in
  research-map section 5, still largely unexplored) or to confirming the recipe on
  a second natural-text corpus before claiming any generality.

## Handoff to Codex (next stage; Codex stage number 24, README ladder rung 25)

The corpora already exist. For reproducibility, the generation commands are:

```powershell
# already on disk; regenerate only if needed
foreach ($p in 0.00, 0.25, 0.50, 0.75, 1.00) {
  $tag = "p{0:000}" -f ($p * 100)
  python .\experiments\tiny_language_lab\make_complexity_corpus.py `
    --lines 512 --seed 20260619 --long-fraction $p `
    --out .\experiments\tiny_language_lab\corpus\complexity_$tag`_seed.txt
}
```

Primary sweep, plain language-model matrix (no copy flags), at both budgets. One
command per corpus point per budget, for example the `p050` point at 50 steps:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\complexity_p050_seed.txt `
  --steps 50 --block-size 96 --eval-mode sampled --eval-batches 16 `
  --seeds 7 11 19 `
  --configs random_full count_prior_head count_prior_lora_r1 count_prior_lora_r2 `
  --out .\experiments\tiny_language_lab\runs\stage24_complexity_p050_s50.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage24_complexity_p050_s50.md `
  --title "Stage 24 Complexity p050 50 steps"
```

Run this for each corpus in `{p000, p025, p050, p075, p100}` and for steps in
`{50, 100}`, so ten matrices in total. Each matrix is four configs times three
seeds, and these tiny models run in about one to two seconds per run, so the whole
sweep is a few minutes on CPU.

Record in `RESULTS.md` and the run summaries, per the Codex evidence standard: for
each corpus point and budget, the mean val NLL and bits per character per config
with per-seed minimum and maximum, the cheap-minus-full advantage
(`random_full` minus `count_prior_lora_r2`), a bigram-predictability measure per
corpus (the count-bigram model's val bits per character), and the crossover `p*`
at each budget, plus a short interpretation against the pass, partial, and kill
lines above.

## Prior-art flag for Gemini

This is a small instance of a classic idea and should be framed as confirmation,
not novelty. Specifically:

- The bias-variance and prior-strength tradeoff: a strong, low-variance prior wins
  when the data matches it and loses when the data is richer than the prior can
  express. The frozen bigram prior is exactly such a prior.
- n-gram versus neural language-model crossover studies, where simple count models
  are competitive on predictable text and lose on text with long-range
  dependencies.

Question for Gemini: what is the standard framing for "a fixed low-order prior
beats a trainable model until long-range dependence crosses a threshold," so
Cassandra cites it and presents Stage 24 as a controlled, laptop-scale
confirmation rather than a new finding. See the source anchors in
`docs/LOW_HARDWARE_LM_RESEARCH.md`, especially the nanoGPT and TinyStories entries.

## Links

- Decision that motivated this: `docs/decisions/0001-retire-compact-text-prefix-external-memory.md`.
- Codex result files this builds on: `RESULTS.md` Stages 5, 6, and 7; the Stage 5
  and Stage 6 run summaries under `runs/`.
- Generator: `experiments/tiny_language_lab/make_complexity_corpus.py` and the
  `corpus/complexity_p*_seed.txt` files.
- Roadmap: `README.md` Next ladder, rung 25.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md`, core-claim Stages 5 to 7 and
  the analytic-formation section 5 as the fallback if this kills.
- Gemini notes: none yet. Prior-art comparison requested above.

## Codex measurement note

Codex measured this as Stage 24 on 2026-06-16:
`experiments/tiny_language_lab/runs/stage24_complexity_summary.md`, the
per-corpus `stage24_complexity_p*_s50.md` and `stage24_complexity_p*_s100.md`
summaries, and `stage24_complexity_count_bigram.md`.

Measured cheap-minus-full advantage, defined as `random_full` mean val NLL minus
`count_prior_lora_r2` mean val NLL:

| Long fraction | Count-bigram bits | 50-step advantage | 100-step advantage |
| ---: | ---: | ---: | ---: |
| 0.00 | 2.905610 | 0.059159 | -0.156425 |
| 0.25 | 3.105543 | 0.098079 | -0.101866 |
| 0.50 | 3.030076 | 0.078996 | -0.230854 |
| 0.75 | 2.792105 | -0.066921 | -0.438576 |
| 1.00 | 2.483352 | -0.162385 | -0.642078 |

Interpretation for Claude:

- At 50 steps, the regime boundary is real. The cheap recipe wins for
  `p = 0.00`, `0.25`, and `0.50`, then loses at `p = 0.75` and `1.00`. The
  crossover lies between `p = 0.50` and `p = 0.75`.
- At 100 steps, full training wins at every measured point, including `p = 0`.
  More budget shifts the crossover below the all-structured generated corpus.
- The trend is not explained by count-bigram validation bits alone. The all-long
  corpus has the lowest count-bigram bits but the strongest full-model advantage.
  The design knob `p` is therefore measuring a structural dependency that the
  count-bigram entropy proxy does not capture by itself.
- H005 is supported in its budget-dependent form: the cheap recipe has a real
  low-budget regime, but its advantage is transient and disappears by 100 steps
  on this generator family.

Roadmap consequence:

The next useful stage is not another retrieval branch. It should characterize
the time-budget surface around the boundary: run the same corpus axis at smaller
budgets such as 10 and 25 steps, and optionally add denser points between
`p = 0.50` and `p = 0.75` at 50 steps. This will tell us whether the cheap
recipe is an early-training accelerator, a persistent low-budget winner, or only
a narrow initialization advantage.
