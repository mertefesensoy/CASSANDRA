# Hypothesis 021 · The analytic floor scales with corpus size (order-5 prior)

- Status: OPEN. Written 2026-07-02 for Codex Stage 54, the second Phase 4
  deliverable under ADR 0013, behind an explicit feasibility gate.
  Post-Stage-53 note: H020 resolved KILL E-interfere, so the flagship takes
  NO prior and every "flagship carries the order-5 prior" clause below is
  moot. The hypothesis itself is unaffected (its arms live in the
  tiny-surface regime where the prior is proven safe) and stays scheduled
  on its standalone claim: does the analytic floor scale with corpus size.
- Date: 2026-07-02
- Author: Claude (hypothesis and roadmap role)
- Builds on: Stage 35 / H012 (the analytic-base ceiling: best frozen base =
  highest ESTIMABLE count order, measured on the 1.1M-char Tiny Shakespeare
  corpus), Stage 52 / H019 (the order-4 prior arm is a constant near `1.104`
  on the rescaled corpus and fully determines the crossover), Stage 51 Gate
  A (the corpus is now `494,094,421` chars: roughly 380x the Tiny
  Shakespeare corpus the ceiling was measured on, and 42x the 10.08M-char
  Phase 2 TinyStories seed), ADR 0013 (Phase 4 scope).

## Why this, and why now

Stage 52's re-read (`docs/phase4-intake.md` Section 3) showed the crossover
is one learning curve crossing a horizontal threshold, and the threshold is
the order-4 prior's floor. Whoever controls the floor controls the
crossover: a lower floor means the frozen prior stays ahead of full training
longer at every model size, extending the recipe's useful zone for free at
run time. The only analytic lever left for the floor is prior order, and the
Stage 35 ceiling that closed this branch was an ESTIMABILITY statement, not
an absolute one: the best base is the highest order the corpus can estimate.
The corpus scale has changed by orders of magnitude since that measurement:
roughly 380x Stage 35's 1.1M-char Tiny Shakespeare, and 42x the 10.08M-char
Phase 2 TinyStories seed. Order-5 contexts now average about `10.7`
observations each (`419,980,257 / 33^5`), against about `0.26` on the Phase
2 seed. Re-testing the ceiling under the new data applies the Stage 35 rule
rather than contradicting it. This is also a direct input to the
flagship (ADR 0013 D3): if H020 confirms the prior survives full training,
the flagship should carry the best floor available.

## The feasibility gate (Phase A, decides whether Phase B runs)

Order 5 does not fit the existing dense-table design: the order-4 table is
`33^5` floats, about 149 MiB fp32, but a dense order-5 table is `33^6 =
1,291,467,969` floats, about 4.8 GiB fp32, next to a model, optimizer state,
and activations on an 8,188 MiB card. The code also caps `--prior-order` at
4 (`cassandra_tiny_transformer.py`, the "currently supports 1, 2, 3, or 4"
checks). So Phase A is a design-and-measure gate, not a formality. Candidate
designs, for Codex to choose among with measurements:

- fp16 dense table (about 2.4 GiB): simplest, may fit beside the 25.25M
  model (Stage 52 peak there was under 1 GiB), will not fit beside 85M
  training comfortably.
- Sparse or hashed table keyed by OBSERVED order-5 contexts only: natural
  text occupies a small fraction of the `39.1M` possible contexts, so this
  is likely far smaller than dense; needs a lookup path in the forward pass.
- Backoff design: order-5 counts where the context was observed at least a
  threshold number of times, else fall back to the existing order-4 table.
  This is the design most faithful to the Stage 35 "estimable" rule and the
  recommended default.
- CPU-side table with per-batch gather: unbounded size, pays transfer cost
  per step; measure before dismissing.

Gate deliverables: the builder (shard-native, streaming, like the Stage 52
order-4 builder), measured build wall-clock, measured memory (table size and
peak CUDA during a smoke), and THE KEY NUMBER, a PAIRED floor comparison:
run the frozen order-5 floor arm (`count_prior_ng5_lora_r2_floor`, new,
`residual_optim none`) and the frozen order-4 floor arm
(`count_prior_ng4_lora_r2_floor`, ALREADY registered in
`cassandra_compare.py`) in the SAME invocation with seeds `7 11 19`, so the
per-seed sampled eval windows are identical and the deltas are paired.
Stage 52 never ran a floor arm; its trained rank-2 means (`1.104469` at 500
steps, `1.113529` at 200) are CONTEXT for sanity-checking the measured
order-4 floor, not the baseline.

Gate pass line: mean paired delta, order-5 floor minus order-4 floor, at or
below `-0.02` NLL with all three paired per-seed deltas negative. Paired
floors on shared eval windows are near-deterministic, so `0.02` is a
materiality line (anything smaller is not worth new machinery), not a noise
line. If the gate fails, H021 is KILLED at Phase A for the immediate
version: the Stage 35 ceiling holds even at this corpus scale because
order-5 sparsity noise (or backoff dilution) eats the headroom; record the
measured floors and stop, no matrix.

## Hypothesis (Phase B, only if the gate passes)

On the rescaled corpus, the frozen order-5 (or backoff order-5-over-4)
prior lowers the floor below the order-4 value by at least `0.02` mean NLL,
and as a consequence the `random_full` crossover at 25.25M moves at least
one budget rung later than Stage 52's bracket (from crossing between 500 and
1000 steps to crossing at or after 1000).

## The reference points

Phase B arms, matching Stage 52's protocol exactly except prior order, at
25.25M (`n_layer=8 n_head=8 n_embd=512`), budgets 200, 500, 1000, 2000,
seeds `7 11 19`, modern recipe, sampled eval:

- `count_prior_ng5_lora_r2` (NEW config): frozen order-5 prior, rank-2 LoRA,
  the direct analog of the Stage 52 prior arm.
- `random_full`: REUSE the Stage 52 25.25M rows; identical protocol.
- Order-4 references from Stage 52 need no rerun.

If wall-clock allows inside the internship window, an 85M column
(`n_layer=12 n_head=12 n_embd=768`) is the value-add extension: Stage 52
showed 85M crossing at 500 steps, the tightest squeeze, so a lower floor
should visibly delay it there too. Optional, not decision-bearing.

## Primary decision metric and pass or fail line

Phase A metric: order-5 floor NLL versus order-4 floor NLL (gate line
above). Phase B metric: the 25.25M crossover budget with the order-5 prior,
versus Stage 52's (full model wins at 1000, near-tie at 500).

- CONFIRM: gate passes AND the 25.25M crossover moves at least one rung
  later (full model no longer at-or-below the prior arm at 1000). Be honest
  about the implied magnitude: `random_full`'s 1000-step mean is
  `0.999363`, so holding the crossover past 1000 requires the new floor to
  sit below that, a drop of about `0.105` from the `1.104` order-4 level,
  roughly five times the gate margin. CONFIRM is deliberately a high bar.
  If it lands: the floor scales with data and the Stage 35 ceiling is
  corpus-relative. (The original "flagship carries the order-5 prior"
  consequence is dead: Stage 53 resolved H020 as E-interfere, so no
  flagship prior regardless of this verdict; the beneficiary is the
  low-budget tiny-surface recipe.)
- GRADED: gate passes but the crossover does not move a full rung; in
  floor-drop terms this covers roughly `[0.02, 0.10)` and is the most
  likely positive outcome. Real but small; record the floor gain for the
  tiny-surface recipe, no claim about the crossover law.
- KILL at Phase A: gate fails; the ceiling holds at this corpus scale.
  Record and close the order axis again, this time with the corpus-scaling
  clause tested.

## Risks and confounds

Sparsity is heavy-tailed: `10.7` average observations per context hides a
distribution where most mass sits in frequent contexts and many validation
contexts are unseen; the backoff design absorbs this, a pure order-5 table
may not, so the gate should report the validation backoff rate (fraction of
positions falling back to order 4). Smoothing: the `alpha` in
`log(alpha + counts)` matters more at order 5; Codex should hold the
existing convention rather than tuning it, and note it as a fixed choice.
Memory pressure is the reason the gate exists; do not launch Phase B at any
size whose smoke shows the table plus training within 1 GiB of the card
limit. Wall-clock: counting is one streaming pass over 420M chars (the
order-4 shard builder took minutes, order 5 is the same scan with a wider
key); Phase B is 12 new runs at 25.25M, comparable to Stage 53's hour. Two
Stage 52 prior rows carry sleep-inflated `seconds`; do not reuse Stage 52
timing for Phase B estimates (use Gate C's `0.169` s/step at 25.25M).

## What result would change the plan

CONFIRM turns the frozen-prior recipe into something that scales WITH data
rather than being a fixed small-corpus trick: every future corpus upgrade
buys a lower floor and a longer useful zone, which is the strongest
remaining form of the analytic-prior thesis on the NLL axis (and, after
Stage 53's KILL, the prior's only remaining growth path). KILL at the gate is cheap and clean: the
ceiling is confirmed to be about estimation noise, not corpus size, and the
prior-order axis closes with a tested scaling clause instead of an
assumption. GRADED still improves the flagship's floor while keeping the
crossover law as Stage 52 measured it.

## Handoff to Codex (implement as Codex Stage 54, Phase A first, Phase B
only if the gate passes)

Files: extend the shard-native count-prior builder in
`cassandra_tiny_transformer.py` to order 5 with the backoff design as the
default (order-5 where the context count clears the `--prior5-min-count`
threshold, default 10, else order-4), raise the `--prior-order` cap for the
backoff path only, add `count_prior_ng5_lora_r2` and
`count_prior_ng5_lora_r2_floor` branches in `cassandra_compare.py`
mirroring the ng4 pair (rank 2, `lora_alpha 2.0`), and cache the built
prior to disk beside the Stage 52 cache with a distinct hash. Hold
`count_alpha 0.1` and `ngram_backoff 1.0` at the Stage 52 values; they are
fixed choices, not tuning knobs. Keep the fp32-versus-fp16 and
sparse-versus-dense choice measurement-driven and record it in the stage
notes.

Phase A command shape (the paired floor comparison; budget is nominal since
floor arms never train):

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\tinystories_char_seed.txt `
  --device cuda --steps 200 --seeds 7 11 19 `
  --configs count_prior_ng4_lora_r2_floor count_prior_ng5_lora_r2_floor `
  --n-layer 8 --n-head 8 --n-embd 512 --block-size 128 --batch-size 8 `
  --grad-accum-steps 2 --pos-encoding rope --activation-checkpoint `
  --optimizer muon --muon-lr 0.01 --eval-mode sampled --eval-batches 16 `
  --train-shard-dir .\experiments\tiny_language_lab\corpus\tinystories_char_shards_500mb `
  --prior-cache-dir .\experiments\tiny_language_lab\runs\stage52_prior_cache `
  --out .\experiments\tiny_language_lab\runs\stage54_gateA_floor_pair.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage54_gateA_floor_pair.md `
  --title "Stage 54 Gate A Order-5 vs Order-4 Floor"
```

Alongside the paired floors, report build seconds, table bytes,
kept-context count, and validation backoff rate.

Phase B command shape, one budget cell (repeat for 200, 500, 1000, 2000;
only if the gate passes), reusing the Stage 52 `random_full` 25.25M rows as
the comparison:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\tinystories_char_seed.txt `
  --device cuda --steps 1000 --seeds 7 11 19 `
  --configs count_prior_ng5_lora_r2 `
  --n-layer 8 --n-head 8 --n-embd 512 --block-size 128 --batch-size 8 `
  --grad-accum-steps 2 --pos-encoding rope --activation-checkpoint `
  --optimizer muon --muon-lr 0.01 --eval-mode sampled --eval-batches 16 `
  --train-shard-dir .\experiments\tiny_language_lab\corpus\tinystories_char_shards_500mb `
  --prior-cache-dir .\experiments\tiny_language_lab\runs\stage52_prior_cache `
  --out .\experiments\tiny_language_lab\runs\stage54_ng5_25m_b1000.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage54_ng5_25m_b1000.md `
  --title "Stage 54 Order-5 Crossover 25M 1000-step"
```

Metric that decides: Phase A, mean paired delta (order-5 floor minus
order-4 floor) at or below `-0.02` with all three paired deltas negative;
Phase B, the 25.25M crossover rung versus Stage 52. Restated: CONFIRM =
gate passes and the crossover moves later by a rung (requires a floor drop
near `0.105`); GRADED = gate passes only; KILL = gate fails, stop at Phase
A.

## Prior-art flag for Gemini

Kneser-Ney and Stupid Backoff are the canonical answers to "how high an
n-gram order can a corpus of size N estimate," and there is literature on
character-level n-gram model entropy versus order on large corpora. Gemini
should check what bits/char a well-smoothed order-5 character model achieves
on child-directed or simple English text, since that predicts our gate
outcome directly, and whether n-gram-fused neural LMs saw their fusion
benefit grow or shrink with n-gram order.

## Links

- `docs/decisions/0013-phase-4-free-accelerator-floor-scaling-flagship.md`
- `docs/hypotheses/020-frozen-prior-free-accelerator.md`
- `docs/hypotheses/019-crossover-scaling-law.md`
- `docs/phase4-intake.md`
- `experiments/tiny_language_lab/runs/stage52_h019_crossover_scaling_summary.md`
