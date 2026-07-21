# Phase 4 Intake Brief · Cassandra

Drafted 2026-07-02 by Claude from the completed Stage 51 and Stage 52 results.
Section 1 holds the forks only the user can answer, each with Claude's
recommendation. Sections 2 to 4 are facts and analysis gathered by Claude from
the recorded artifacts and the code, not estimated. This doc feeds the Phase 4
ADR (next number: 0013) and the next hypothesis (next number: 020).

## Section 0 · Where Phase 3 left the project

Both ADR 0011 deliverables landed on 2026-07-02:

- Stage 51: a 25.25M-param `random_full` checkpoint family at `0.818608` mean
  val NLL, `1.181001` bits/char, deterministic generation proxy `5.667/6`.
  Human review of the saved samples is the one remaining ADR 0011 item and
  still gates any genuine-coherence claim.
- Stage 52: the full H019 matrix, 4 sizes x 4 budgets x 2 arms x 3 seeds, all
  96 decision rows clean. Verdict `GRADED`: discrete crossovers `1000, 1000,
  1000, 500` steps for 3.2M, 10.67M, 25.25M, 85.11M.

Section 3 argues the `GRADED` verdict understates what the table shows, and
Phase 4's candidate directions follow from that re-read.

## Section 1 · The forks · for the user

### Fork 1 · Primary research axis for Phase 4 · ANSWERED 2026-07-02

The user expanded the scope beyond the original recommendation: **Option B
first, then Option A, then a bigger and better flagship build.** Phase 4 is
therefore three sequenced deliverables: the free-accelerator test (H020,
Codex Stage 53), the prior-order floor-scaling test (H021, Codex Stage 54,
behind a feasibility gate), and a flagship checkpoint (Codex Stage 55) whose
recipe is assembled from the verdicts of the first two. Hypothesis numbering
follows execution order, so the free-accelerator test is H020 and the
floor-scaling test is H021, swapping the tentative labels used in the option
text below (kept as originally drafted for the record). See ADR 0013.

The Stage 52 re-read (Section 3) exposes one constant and one curve: the
frozen order-4 prior is a fixed NLL floor near `1.104` on this corpus, and
every full model is a learning curve that crosses that floor sooner the bigger
it is. That gives four candidate axes:

- **Option A · Lower the floor (H020, prior-order scaling).** Stage 35 closed
  the analytic-base branch with "best frozen base = highest estimable count
  order," measured on a corpus of about 10M chars. The corpus is now 494M
  chars, roughly 42x larger, so what is estimable has changed; order 5 has
  about 10.7 observations per context on average now. Test whether an order-5
  (or backoff-smoothed) prior lowers the floor below `1.104` and pushes the
  crossover later at every size. Requires new code and a memory feasibility
  gate first (Section 3, fact 4): the code caps `--prior-order` at 4 and a
  dense order-5 table is about 4.8 GiB fp32.
- **Option B · The free-accelerator test (H021, prior under full training).**
  Stage 52's prior arm caps the trainable surface at rank 2, which is why it
  plateaus at the floor. Nobody has run the frozen prior with `--train-scope
  all` at these sizes and budgets. If prior-plus-full-training matches or
  beats `random_full` at every budget, the prior is a strictly-nonnegative
  warm start and the practical recommendation becomes "always start from the
  prior." If it interferes (the base-interference effect Stage 43 flagged at
  tiny scale), the prior's value stays confined to the tiny-surface regime.
  No new code; one 25.25M column of the Stage 52 matrix, 4 budgets x 3 seeds
  x 1 new arm = 12 runs.
- **Option C · Pin the law (H019b, finer budget ladder).** The gap between
  arms is strictly monotone in capacity at every tested budget, 16 of 16
  cells (Section 3, fact 2), but the discrete 200/500/1000/2000 ladder
  quantized the crossover into `GRADED`. A finer ladder near the crossover,
  for example 350/425/500/650/800 at all four sizes, could convert this into
  a clean continuous statement: crossover step as a smooth decreasing
  function of capacity. Main value is publication-grade cleanliness.
- **Option D · Coherence v2 (checkpoint continuation).** Stage 51's 5000
  steps saw only about 2.4% of the train split (Section 3, fact 3). A 20k to
  50k step run at 25.25M is the natural next quality push, sized by measured
  wall-clock (about 0.14 s/step at 25.25M means 50k steps is roughly 2
  hours per seed). Conditional on the human review verdict for v1.

Claude's recommendation: **B first, then A**, with C only if Fork 2 resolves
toward publication and D conditional on the Stage 51 review. Reasoning: B is
the cheapest (12 runs, no new code) and the most decision-relevant, because it
determines whether the prior graduates from "bounded early-compute
accelerator" (ADR 0002's framing) to "default initialization for every
from-scratch build," which is the strongest practical form of the north-star
claim. A has the biggest scientific upside but needs a Codex feasibility gate
and new sparse/backoff code before any matrix can be sized honestly.

### Fork 2 · Publication intent, revisited

Phase 3 left this undecided. It is now more concrete: the 16/16 monotone-gap
fact, plus a floor-scaling result from Option A, plus the existing
Phase-1-to-Phase-3 crossover narrative (synthetic ~100 steps, 3.2M real text
~1000 steps, 85M real text ~500 steps) reads like the skeleton of a workshop
paper on how an analytic prior's step-count advantage scales. If the answer
becomes "yes, aim for a workshop paper," Option C moves up, because the
discrete-ladder `GRADED` verdict is the weakest link in that story.

### Fork 3 · Stage 51 human review scheduling

Only the user can do this, it gates both the coherence claim and Option D, and
it should land before the internship starts on 2026-07-20. The saved samples
are in `experiments/tiny_language_lab/runs/stage51_checkpoints/` and the score
sheet is `runs/stage51_coherence_25m_b5000_generation_quality.md`. Suggested
form: read the samples against the Stage 46 score-sheet criteria and record a
pass/fail per criterion plus an overall verdict in RESULTS.md under Stage 51.

## Section 2 · Constraints

- External internship 2026-07-20 to 2026-08-21 (hard window, see
  `memory/phase3-internship-window.md`). Same sequencing logic as
  Phase 3: hands-on items before 2026-07-20 (Stage 51 human review, the
  Option A feasibility gate, the cheap Option B column), autonomous matrices
  inside the window (the Option A sweep or Option C ladder, or Option D's
  long run).
- Hardware stays the 8 GB laptop (Phase 3 Fork 3, unchanged). Note the
  Option A dense-table math makes an order-5 prior the first Phase 4 item
  that genuinely presses the VRAM ceiling; sparse, hashed, fp16, or CPU-side
  gather designs are the escape routes to evaluate in the gate, not a
  hardware upgrade.

## Section 3 · Facts gathered

### Fact 1 · The prior arm is a constant, not a curve

Across all 16 Stage 52 cells, `count_prior_ng4_lora_r2` mean val NLL sits in
`[1.103436, 1.115967]`, regardless of model size (3.2M to 85.11M) or budget
(200 to 2000 steps). The rank-2 residual adds nothing measurable on NLL at any
size, which extends ADR 0005's prior-dominance finding from 3.2M synthetic
scale to 85M real-text scale. Consequence: the "crossover" is not two learning
curves racing; it is one learning curve (`random_full`) crossing a horizontal
threshold set entirely by the frozen order-4 prior, about `1.104` at its best.
Whoever controls the floor controls the crossover.

### Fact 2 · The continuous capacity signal is clean even though the discrete verdict was GRADED

Prior-minus-full mean NLL, read across sizes at each fixed budget, is strictly
monotone increasing in capacity in every row:

| Budget | 3.2M | 10.67M | 25.25M | 85.11M | Monotone? |
| ---: | ---: | ---: | ---: | ---: | --- |
| 200 | -0.339331 | -0.281373 | -0.239711 | -0.195772 | yes |
| 500 | -0.062033 | -0.021067 | -0.001908 | +0.014134 | yes |
| 1000 | +0.054721 | +0.090249 | +0.108862 | +0.126825 | yes |
| 2000 | +0.142342 | +0.178514 | +0.190090 | +0.195204 | yes |

16 of 16 cells are consistent with "capacity uniformly speeds catch-up," the
weak form of H019's E1. The 500-step row shows why the discrete verdict came
out `GRADED`: the true crossover for 3.2M, 10.67M, and 25.25M sits between 500
and 1000 steps but compresses toward 500 as size grows (the 25.25M gap at 500
is `-0.0019`, within seed noise, with 1 of 3 seeds already crossed), and only
85M slips under the 500 rung. The ladder granularity, not the physics,
produced the mixed verdict. This is the single most important analytical fact
for Phase 4 planning.

### Fact 3 · Every Stage 52 cell is severely token-limited

Each step consumes `batch_size 8 x block_size 128 x grad_accum 2 = 2,048`
train chars. So a 2000-step run sees about 4.10M chars, roughly 1.0% of the
419,980,257-char train split; Stage 51's 5000 steps saw about 10.2M chars,
roughly 2.4%. The diminishing size returns at fixed budget (2000-step NLLs
`0.969 / 0.934 / 0.922 / 0.918`) are therefore a tokens-seen effect, not a
data ceiling: no run is anywhere near one epoch. This both explains the
saturation and means Option D has enormous headroom before repeating data.

### Fact 4 · Order-5 prior feasibility math (Option A gate inputs)

- Code cap: `--prior-order` raises `ValueError` above 4
  (`cassandra_tiny_transformer.py`, the "currently supports 1, 2, 3, or 4"
  checks near lines 1203 and 1341). Order 5 requires new code, not a flag.
- Dense order-4 table: `33^5 = 39,135,393` floats, about 149 MiB fp32. Fits;
  this is what Stage 52's shard-native builder cached (`stage52_prior_cache/`).
- Dense order-5 table: `33^6 = 1,291,467,969` floats, about 4.8 GiB fp32 or
  2.4 GiB fp16. Does not fit comfortably next to an 85M model plus optimizer
  state on an 8,188 MiB card; fp16 might squeeze at small sizes. Realistic
  designs: sparse or hashed storage keyed by observed contexts only (natural
  text occupies a tiny fraction of `33^5` contexts), a backoff scheme that
  falls to order 4 for unseen contexts, or CPU-side table with per-batch
  gather.
- Estimability: `419,980,257 / 33^5` is about 10.7 observations per order-5
  context on average. Sparse but plausible with smoothing or backoff; the
  Stage 35 ceiling ("best base = highest estimable order") was measured on
  the 1.1M-char Tiny Shakespeare corpus, roughly 380x smaller (the 42x
  figure elsewhere in this doc is against the 10.08M-char Phase 2
  TinyStories seed), so re-testing it here applies that finding rather than
  contradicting it.

### Fact 5 · Stage 51 status

`0.818608` mean val NLL, `1.181001` bits/char, proxy score `5.667/6`, three
seeds, checkpoints and samples saved. Human review pending; it is the only
open ADR 0011 item.

### Fact 6 · Timing anchors for Phase 4 budgeting

From Stage 51 Gate C and the Stage 52 rows: about `0.14` to `0.17` s/step at
25.25M (5000 steps averaged `695.6` s/row), and the 85M 200-step smoke ran
`75.1` s at `1,683.4` MiB peak CUDA. Two Stage 52 rows carry inflated
`seconds` from laptop pause or sleep (25.25M/1000/seed 11 at `3156.4` s;
85.11M/2000/seed 19 at `15858.996` s) and must be excluded from any
throughput estimate. Option B's 12-run column at 25.25M costs on the order of
1.5 to 2 GPU-hours total; Option C's finer ladder (4 sizes x ~5 budgets x 2
arms x 3 seeds, minus reusable Stage 52 cells) is the most expensive
candidate and belongs inside the internship window if chosen.

## Section 4 · Scope confirmation

Closed branches stay closed; no Phase 4 candidate touches them:

- External memory and retrieval (ADR 0001): untouched.
- Data-side selection, static and dynamic (ADR 0004): untouched.
- Frozen recency base (H012): untouched. Option A varies count order, not the
  base family.
- Non-gradient residual formation (Stage 36, ADR 0005): untouched.
- General in-context copy for the frozen-prior family (Stages 40 to 43, ADR
  0006 as narrowed by ADR 0008): untouched. The behavior axis stays paused
  unless a Phase 4 fork deliberately reopens it, which none of A to D does.
- BPE as substrate (Stage 50, ADR 0011 D1): stays a tracked ablation.

One deliberate reopening, argued not to violate its closure: Option A re-tests
the Stage 35 analytic-base ceiling. That ceiling was stated as "best frozen
base = highest estimable count order," and the 42x corpus rescale raises the
highest estimable order. Option A therefore applies the ceiling rule under new
data rather than contradicting it. The Phase 4 ADR should say this explicitly.

## Links

- `docs/decisions/0013-phase-4-free-accelerator-floor-scaling-flagship.md`
- `docs/hypotheses/020-frozen-prior-free-accelerator.md`
- `docs/hypotheses/021-prior-order-floor-scaling.md`
- `docs/decisions/0011-phase-3-coherence-checkpoint-and-crossover-scaling-law.md`
- `docs/hypotheses/019-crossover-scaling-law.md`
- `experiments/tiny_language_lab/runs/stage52_h019_crossover_scaling_summary.md`
- `experiments/tiny_language_lab/RESULTS.md` (Stages 51, 52)
- `docs/phase3-intake.md`
- `memory/phase3-internship-window.md`
