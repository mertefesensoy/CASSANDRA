# Hypothesis 020 · The frozen prior is a free accelerator under full training

- Status: RESOLVED · KILLED (E-interfere). Implemented by Codex Stage 53 on
  2026-07-02. The frozen prior under full training won big early (`-0.252`
  mean NLL at 200 steps) but lost late (`+0.072` at 1000, `+0.100` at 2000,
  all paired deltas positive), and the mandated `--muon-lr 0.005` rerun
  shrank but did not erase the interference (`+0.055`, all seeds). The
  Stage 55 flagship initializes from RANDOM per ADR 0013 D3's assembly
  rule. See the Result section.
- Date: 2026-07-02
- Author: Claude (hypothesis and roadmap role)
- Builds on: Stage 52 / H019 (`docs/hypotheses/019-crossover-scaling-law.md`,
  the prior arm is a constant floor near `1.104` while `random_full` is a
  learning curve that crosses it), ADR 0002 (the recipe is a bounded
  early-compute accelerator), ADR 0005 (the residual is NLL-worthless on top
  of a frozen prior at rank 2), Stage 43 (base-interference flagged on the
  behavior axis at tiny scale), ADR 0013 (Phase 4 scope).

## Why this, and why now

Every crossover result so far compares two arms that differ in TWO ways at
once: `random_full` has full capacity and no prior; `count_prior_ng4_lora_r2`
has the prior and almost no trainable capacity (rank 2, about 41k params at
3.2M scale). So the project has never measured the prior's effect in
isolation under full training. Stage 52 makes the missing cell obvious: the
prior arm plateaus at the floor because rank 2 cannot move, and the full
model wins late because it has capacity, but nobody knows what happens when
one model has BOTH the frozen prior and the full trainable body.

This is the single most decision-relevant unknown for the Phase 4 flagship
build (ADR 0013 D3): if the frozen additive prior never hurts and helps
early, the flagship should train on top of it; if it interferes at higher
budgets, the flagship should start from random init and the prior stays a
tiny-surface, low-budget tool. It is also the cheapest Phase 4 item: one new
config branch, twelve runs, and the `random_full` comparison rows already
exist in Stage 52.

## The mechanism (or competing explanations)

The arm under test is `--residual-base count-ngram --prior-order 4
--train-scope all`: logits = frozen order-4 count table (looked up by the
last 4 chars) + the full transformer's output, with `--zero-residual-head`
(default on) making step-0 output exactly the prior. Three explanations
predict different outcomes as steps grow:

- E-accel, free accelerator. The prior hands the model the local n-gram
  statistics for free, so gradients specialize the full body on what the
  table cannot express (long-range structure). Prediction: the prior arm is
  at or below `random_full`'s NLL at every budget, with a large early lead
  (near the `0.24` NLL gap at 200 steps) that decays toward a small or zero
  late gap but never inverts.
- E-wash, transient head start. The full body eventually re-learns everything
  the table provides and the additive base becomes redundant. Prediction:
  large early lead, then the arms are indistinguishable (within seed spread)
  at 1000 and 2000 steps.
- E-interfere, the prior is a handicap under full training. The fixed
  additive bias distorts the loss surface: the body must learn corrections
  AROUND a table it cannot change, and optimizer dynamics tuned for random
  init (Muon LR, zero-init head) mis-scale. Stage 43 saw the full body on a
  frozen base fail to form behavior that the same body formed without the
  base, so interference has precedent on the behavior axis. Prediction: the
  prior arm falls BEHIND `random_full` at 1000 or 2000 steps, beyond seed
  spread.

## Hypothesis

On the rescaled TinyStories character corpus with the modern recipe, a
25.25M-param transformer trained with `--train-scope all` on top of the
frozen order-4 count prior has mean validation NLL at or below `random_full`
at every tested budget (200, 500, 1000, 2000 steps), and below the frozen
floor (`~1.104`) from 500 steps on. That is, the prior is a strictly
nonnegative warm start under full training (E-accel), not a transient
(E-wash) or a handicap (E-interfere).

## The reference points

All at 25.25M params (`n_layer=8 n_head=8 n_embd=512`), rescaled corpus
(`494,094,421` chars, `V = 33`), modern recipe (CUDA, RoPE, Muon
`--muon-lr 0.01`, activation checkpointing, `--block-size 128 --batch-size 8
--grad-accum-steps 2`), `--eval-mode sampled --eval-batches 16`, seeds
`7 11 19`, budgets 200, 500, 1000, 2000:

- `count_prior_ng4_all` (NEW config, the subject): frozen order-4 prior,
  full body trainable, `25,253,921` trainable params plus the frozen table.
- `random_full` (baseline): REUSE the existing Stage 52 25.25M rows
  (`stage52_crossover_25m_b{200,500,1000,2000}_random_full.jsonl`);
  identical corpus, recipe, sizes, seeds, and eval mode, so no rerun is
  required (cross-stage row reuse has precedent: Stage 31 reused Stage 30
  rows). Stage 52 means: `1.353240 / 1.106377 / 0.999363 / 0.922147` at
  200/500/1000/2000.
- `count_prior_ng4_lora_r2` (context anchor, no rerun): Stage 52's frozen
  floor plus rank-2 arm, means `1.113529 / 1.104469 / 1.108225 / 1.112237`.

The order-4 prior tensor must come from the Stage 52 shard-native builder's
disk cache (`runs/stage52_prior_cache/count_ngram_561fefe48ed103be.pt`), same
corpus and order, so prior construction costs nothing and cannot OOM the way
the first Stage 52 prior pass did.

## Primary decision metric and pass or fail line

Primary metric: per-budget validation NLL delta, `count_prior_ng4_all` minus
`random_full`. Two noise units, used for different clauses. PAIRED per-seed
deltas: same seed on the same device draws the same sampled eval windows
(the eval batches come from one generator seeded by `args.seed`, and both
arms consume it identically when steps, grad-accum, and eval interval
match), so pairing cancels eval-window noise and a small margin is honest.
UNPAIRED seed spread: max minus min across the three `random_full` seeds at
that budget, from the Stage 52 rows (`0.0344` at 1000 steps, `0.0467` at
2000); this is the unit for mean-level claims. Verdicts are evaluated in
this order, first match wins, so the bands partition:

1. KILL E-interfere: at 1000 or 2000 steps, all 3 PAIRED deltas are positive
   (prior arm worse) by more than `0.01` NLL. The `0.01` margin rides on
   paired deltas, not the unpaired spread, which is why it can be smaller
   than the spread without being under-powered. The prior stays confined to
   the tiny-surface low-budget regime; the flagship starts from random
   init; ADR 0002's "bounded accelerator" framing hardens.
2. CONFIRM E-accel: mean delta at or below zero at every budget, the 200 and
   500 mean deltas negative by more than the unpaired spread, all 3 PAIRED
   deltas at or below zero at both 1000 and 2000, and `count_prior_ng4_all`
   below the `1.104` floor from 500 steps on. The prior is a free
   accelerator; the flagship (ADR 0013 D3) trains on top of a frozen prior.
3. E-wash: the 200 and 500 mean deltas are negative by more than the
   unpaired spread, and the late budgets matched neither clause above
   (mixed-sign or near-zero paired deltas at 1000 or 2000). Still a positive
   practical result (free early compute, no demonstrated late cost); the
   flagship uses the prior, and the claim is recorded as transient rather
   than durable.
4. GRADED: anything else, for example no clear early advantage. Report
   per-seed rows plainly and defer the flagship init decision to a
   longer-budget follow-up at the flagship's own step count.

## Risks and confounds

Optimizer mis-tuning is the main one: Muon at `--muon-lr 0.01` was tuned on
random init, and a frozen additive base changes early gradient scale, so an
E-interfere read could be an LR artifact rather than a structural fact. If
the KILL line trips, Codex should rerun the 2000-step cell once with
`--muon-lr 0.005` before the verdict is recorded; if the interference
vanishes, record GRADED with an LR-sensitivity note instead of KILL. Eval
comparability: reusing Stage 52 `random_full` rows assumes same-seed sampled
eval windows match across configs on the same device; they are drawn from the
same seeded generator and the corpus is identical, but if Codex prefers
row-adjacent evidence, rerunning the 200 and 500 `random_full` cells costs
under ten minutes and removes the concern. Wall-clock: 12 new runs at
25.25M, about `0.17` s/step, roughly 3,700 steps per seed across the ladder,
on the order of one GPU-hour total.

## What result would change the plan

CONFIRM upgrades the north-star claim to its strongest practical form: an
analytic prior costs nothing at build time (cached, shard-native), never
hurts, and buys early compute, so EVERY from-scratch build in this lab,
including the Phase 4 flagship, should start from one. KILL bounds the
recipe honestly: the prior is only useful when the trainable surface is too
small to interfere with it, which is a publishable negative and redirects
the flagship to random init. E-wash keeps the flagship on the prior (free
early progress) while scoping the durable-advantage claim down. Any verdict
also feeds H021: a lower floor (order 5) only matters for the flagship if
the prior survives full training at all.

## Result

Codex Stage 53 ran all cells cleanly (an interrupted 2000-step partial was
preserved and rerun). The confirm-first items all held: the prior cache hit
(`prior_cache_status=hit`), step-0 sampled NLL of the full-body arm exactly
matched the rank-2 arm at `1.103458` (zero-residual-head works under
`train-scope all`), and Muon grouped the block matrices while Adam took
embeddings, biases, norms, and head.

Mean val NLL, `count_prior_ng4_all` versus the Stage 52 `random_full` rows:

| Budget | Prior-all | Random-full | Mean delta | Paired deltas |
| ---: | ---: | ---: | ---: | --- |
| 200 | 1.101573 | 1.353240 | -0.251667 | all negative |
| 500 | 1.081726 | 1.106377 | -0.024651 | all negative |
| 1000 | 1.071762 | 0.999363 | +0.072398 | all positive |
| 2000 | 1.022622 | 0.922147 | +0.100475 | all positive |

The KILL clause tripped at 1000 and 2000 (all paired deltas above `+0.01`),
so Codex ran the mandated 2000-step `--muon-lr 0.005` cell: mean improved
to `0.976702` but still trailed by `+0.054555` with all three paired deltas
positive. Verdict: KILL E-interfere, LR-sensitive but not LR-explained.

Two reads worth keeping. First, unlike the rank-2 arm's flat floor, the
full body DOES escape the floor on top of the prior (down to `1.0226` by
2000 steps, and already below the floor at 200); it just climbs slower than
a randomly initialized body learns from scratch. The frozen additive table
is scaffolding early and drag late. Second, this is now the SECOND
base-interference measurement: Stage 43 saw the full body on a frozen base
fail to form copy behavior that the same body formed without the base, and
Stage 53 shows the NLL-axis analog at 25.25M real-text scale. The pattern
generalizes: the frozen prior helps only surfaces too small to fight it.
ADR 0002's bounded-accelerator framing hardens, now measured under full
training, not just tiny residuals. Consequence per ADR 0013 D3: the
flagship initializes from random; the prior's home is the low-budget
tiny-surface regime and the crossover law itself (H021).

## Handoff to Codex (implement as Codex Stage 53)

Files: one new branch in `cassandra_compare.py` `config_args`, name
`count_prior_ng4_all`, mirroring `count_prior_ng4_lora_r2` but with
`"train_scope": "all"` and no LoRA keys (see the `count_prior_all_copyw`
branch for the train-scope-all shape, WITHOUT its copy flags and with
`residual_base` `count-ngram` + `prior_order` 4 instead of `count-bigram`).
Register the name in the `--configs` choices list. No trainer changes: the
forward path, zero-residual-head, and the shard-native prior cache already
exist. The prior must load from
`runs/stage52_prior_cache/count_ngram_561fefe48ed103be.pt`: NOTE that the
cache is only consulted when `--prior-cache-dir` is passed
(`count_ngram_prior_cache_path` returns `None` when the dir is unset), so
every command below carries it; without it each budget cell silently
rebuilds the prior from shards.

Command shape, one budget cell (repeat for 200, 500, 1000, 2000):

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\tinystories_char_seed.txt `
  --device cuda --steps 500 --seeds 7 11 19 `
  --configs count_prior_ng4_all `
  --n-layer 8 --n-head 8 --n-embd 512 --block-size 128 --batch-size 8 `
  --grad-accum-steps 2 --pos-encoding rope --activation-checkpoint `
  --optimizer muon --muon-lr 0.01 --eval-mode sampled --eval-batches 16 `
  --train-shard-dir .\experiments\tiny_language_lab\corpus\tinystories_char_shards_500mb `
  --prior-cache-dir .\experiments\tiny_language_lab\runs\stage52_prior_cache `
  --out .\experiments\tiny_language_lab\runs\stage53_prior_all_25m_b500.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage53_prior_all_25m_b500.md `
  --title "Stage 53 Prior-All 25M 500-step"
```

Metric that decides: per-budget val NLL of `count_prior_ng4_all` versus the
Stage 52 `random_full` 25.25M rows, paired per seed. Apply the four verdict
clauses IN THE STATED ORDER (KILL, CONFIRM, E-wash, GRADED), first match
wins, with the one LR-sensitivity rerun before any KILL is recorded.

## Prior-art flag for Gemini

This is close to warm-starting and knowledge-injection literature: additive
logit priors and residual adapters on frozen bases (the reverse direction),
n-gram or cache fusion with neural LMs (interpolation at the output layer),
and the shrink-and-perturb / warm-start-hurts line of work on why
initialization from prior solutions can damage later training. Gemini should
check whether output-layer n-gram fusion under FULL training is already
characterized (help early, hurt late, or neutral), since that would predict
our verdict before we run it.

## Links

- `docs/decisions/0013-phase-4-free-accelerator-floor-scaling-flagship.md`
- `docs/hypotheses/019-crossover-scaling-law.md`
- `docs/hypotheses/021-prior-order-floor-scaling.md`
- `docs/phase4-intake.md`
- `experiments/tiny_language_lab/runs/stage52_h019_crossover_scaling_summary.md`
