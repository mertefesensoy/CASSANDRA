# Hypothesis 024 · At one fixed 85M compute budget, a learned TinyStories childhood does not beat cold broad training on text8 test bits/char; the Stage 53 interference precedent predicts it can hurt, curriculum literature predicts the opposite, and three equal-budget arms decide which

- Status: RESOLVED, E-NULL, seed-robust in sign (2026-07-21, ADR 0016).
  CURRICULUM minus COLD on deterministic text8 TEST bits/char: `+0.005096`
  (seed 7, 42,000 steps), `+0.009845` (seed 11, 20k replica), `+0.007791`
  (seed 19, 20k replica); all three signs favor COLD, all magnitudes sit
  inside the registered `+/-0.05` band, and the escalation rule does not
  fire. MIXTURE read `+0.027977` versus COLD with a `2.73` bits/char
  TinyStories retention advantage (descriptive). Evidence:
  `runs/stage58_dev_*_text8_test.json`, `runs/stage58_dev_*_retention.json`,
  `docs/phase5-final-report.md`, `docs/figures/phase5/`. Closure and reopen
  clauses live in
  `docs/decisions/0016-phase-5-closeout-developmental-null-recipe-frontier.md`.
  (Original pre-registration text follows, frozen: specced for Codex as
  Stage 58, README ladder rung 68, design pre-registered in ADR 0015 D1;
  Stage 56 (H022) read out CONFIRM, so the arms ran on the char substrate,
  not the never-written H023 BPE contingency.)
- Date: 2026-07-09
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 68 (Codex stage number 58)
- Builds on: ADR 0015 D1 (the pre-registration this document expands into a
  runnable hypothesis, `docs/decisions/0015-phase-5-developmental-training-and-open-source-posture.md`);
  the Phase 5 intake P2 (`docs/phase5-intake.md`); the Stage 53 / H020
  analytic-prior interference result (`+0.072` and `+0.100` NLL at 1000 and
  2000 steps, `docs/hypotheses/020-frozen-prior-free-accelerator.md`, ADR 0013,
  `runs/stage53_h020_free_accelerator_summary.md`); Stage 56 / H022, which
  confirmed the char substrate carries broad text (`1.485740` bits/char on
  text8 TEST, `docs/hypotheses/022-broad-corpus-specialization-gap.md`);
  Stage 57 Recipe v2 (`RESULTS.md` Stage 57); TinyStories (Eldan and Li 2023);
  Gemini theme-4 notes 01 and 02.

## Why this, and why now

Phase 4 ended with one dominant unexplained number, the domain-specialization
gap, and H022 resolved its origin: the gap is a data-distribution effect, and
the plain char recipe trained on broad text reaches `1.485740` bits/char on the
text8 TEST split. That closes the "what is the gap" question and opens the
Phase 5 thesis directly: given a fixed compute budget, what is the smartest
ACQUISITION ORDER for breadth? This is the founding Cassandra bet, cheap
starting points versus brute force, reborn one level up. In Phase 1 the cheap
starting point was an analytic count prior. In Phase 5 the lab owns something it
never had before, a set of trained narrow-domain weights (the TinyStories
family), and the question is whether a LEARNED simple-register childhood behaves
like a curriculum that accelerates broad-text learning or like the analytic
prior that interfered with it.

Now is the moment because every prerequisite is measured and locked. H022 fixed
the substrate (char, not BPE). Stage 57 fixed Recipe v2 (fp32, cosine warmdown,
checkpoint retention, the vocab-union override that this stage needs to score
one checkpoint across two corpora). The behavior probe stayed closed, so no
training budget is owed to that axis. The internship window (2026-07-20 to
2026-08-21) is the registered home for exactly this workload: long, autonomous,
GPU-bound arms (ADR 0015 D5). Stage 58 is the last gate, and this document is
what unblocks it.

## The mechanism (or competing explanations)

Three arms grow an 85.11M `L12 H12 D768` model to the SAME fixed compute budget,
and their final text8 TEST bits/char is compared:

- **Arm COLD** · random init, broad text (text8 train split) for the full
  budget. Spends the entire budget on the target distribution. The baseline
  every childhood must beat.
- **Arm CURRICULUM** · the first 25 percent of the budget on TinyStories, then
  the remaining 75 percent on broad text, continuing from the narrow weights
  (checkpoint resume, one continuous LR schedule across the phase boundary). The
  "childhood first" arm.
- **Arm MIXTURE** · TinyStories and broad shards interleaved at a fixed 1:3
  token ratio for the full budget. Same total TinyStories dose as CURRICULUM,
  distributed instead of front-loaded. The "grow up bilingual" arm.

Because the budget is fixed in COMPUTE, the two childhood arms trade broad-text
epochs for a TinyStories phase (see the token accounting below), so a childhood
must EARN its keep by making the remaining broad text more efficient than the
broad text it displaced. The PRIMARY decision is CURRICULUM versus COLD, and
three mechanisms compete for it:

- **E-interfere (the lab's own precedent).** Learned narrow-domain weights are a
  bad initialization for broad text, exactly as the analytic count prior was
  under late full-body training (Stage 53, `+0.100` NLL at 2000 steps). Then
  CURRICULUM lands ABOVE COLD on text8 TEST bits/char (worse), the front-loaded
  TinyStories budget is wasted, and agreement between LEARNED-prior and
  ANALYTIC-prior behavior becomes the cleanest cross-phase law the lab has
  produced.
- **E-curriculum (the outside literature).** A simple register first teaches
  reusable structure that transfers, so a childhood is worth more than the broad
  text it costs. Then CURRICULUM lands BELOW COLD (better), vindicating the
  TinyStories thesis as a developmental STAGE rather than a destination.
- **E-null (order-invariant).** At this budget and dose the model washes out its
  starting point; CURRICULUM lands within seed noise of COLD, and acquisition
  order does not matter here.

MIXTURE is a SECONDARY contrast, reported alongside but never merged into the
primary trichotomy. MIXTURE versus COLD asks whether a distributed childhood
dose helps at all; MIXTURE versus CURRICULUM asks whether ORDER matters at equal
dose (front-loaded versus interleaved), the rehearsal-versus-sequential axis of
the continual-learning literature. One caveat bounds the order reading: under a
single continuous cosine schedule over the full budget, CURRICULUM meets broad
text only in the decayed low-LR tail, while MIXTURE meets broad text across the
whole LR range including the high-LR early steps. So the CURRICULUM-versus-MIXTURE
contrast co-varies the LR regime under which broad text is learned; it is an
order contrast confounded with an LR-exposure contrast, and that limit is
intrinsic to fixed-compute front-loading, not a fixable design flaw.

## Hypothesis

Primary claim (the falsifiable directional bet, following the lab's own Stage 53
precedent): at a fixed budget of about 12 GPU-hours per arm (registered as
50,000 steps, see budget arithmetic), the CURRICULUM arm does NOT beat the COLD
arm on text8 TEST bits/char, and the E-interfere direction is live, meaning
CURRICULUM lands at or above COLD.

The result that would kill it, in both directions (decision on CURRICULUM alone,
`a = CURRICULUM - COLD` in bits/char on the deterministic text8 TEST eval):

- **Kills toward E-curriculum** if `a` at or below `-0.05`. A learned childhood
  then measurably accelerates broad-text acquisition, and the TinyStories
  developmental-stage bet holds at this scale.
- **Confirms E-interfere** if `a` at or above `+0.05` (worse), echoing Stage 53's
  interference, so a learned narrow childhood is a net drag on a fixed broad-text
  budget.
- **E-null** if `|a|` below `0.05`; acquisition order is a non-effect at this
  budget and dose, and the plan records that.

Comparison unit and noise: `a` is a seed-7 delta, so COLD and CURRICULUM share
the same seed 7 and the delta is paired by initialization. The `0.003035`
bits/char figure is H022's UNPAIRED cross-seed spread (seed 11 versus seed 19 at
20k). Because the paired delta shares a seed, seed-specific variance largely
cancels, so the delta's own seed sensitivity is at most a small multiple of
`0.003035`, far under `0.05`. The reduced-budget replicas (seeds 11 and 19) test
whether the SIGN of `a` is seed-robust, not the exact margin. The `0.05` margin
is anchored twice: it is roughly 16 times the `0.003035` cross-seed floor, and
it sits BELOW the expected E-interfere effect. Stage 53's interference is
`+0.072` to `+0.100` NLL in nats/char, which converts to about `0.104` to
`0.144` bits/char (divide by `ln 2`), so a genuine interference clears the
`0.05` line with headroom rather than sitting on it.

## The reference points

Measured or published anchors, no rerun needed:

- Broad-trained 85M char model, text8 TEST, from H022: `1.485740` bits/char at
  50,000 steps seed 7; replicas `1.532627` (seed 11) and `1.529591` (seed 19)
  at 20,000 steps (`docs/hypotheses/022-broad-corpus-specialization-gap.md`
  Result, `RESULTS.md` Stage 56 TEST table). Arm COLD at the registered 50,000
  steps is the SAME configuration and budget as H022's seed-7 Arm A, so its
  expected value is near `1.485740` bits/char (`+/-` the cross-seed spread); the
  three arms all live in the `1.48` to `1.55` neighborhood, and the decision is a
  `0.05` separation inside it.
- Cross-seed spread for a fixed arm and budget under the deterministic chunked
  text8 eval: `0.003035` bits/char (H022 seeds 11 and 19). The noise floor the
  `0.05` decision margin sits far above.
- Stage 53 analytic-prior interference: paired deltas `+0.072` and `+0.100` NLL
  (nats/char) at 1000 and 2000 steps (`runs/stage53_h020_free_accelerator_summary.md`),
  about `0.104` to `0.144` bits/char. The effect size the E-interfere direction
  predicts CURRICULUM will reproduce.
- 85M block-256 SUSTAINED throughput from H022's clean from-scratch replicas:
  `17,502.09` s (seed 11) and `17,323.05` s (seed 19) for 20,000 steps
  (`RESULTS.md` Stage 56 in-run table), that is about `0.87` s/step wall-clock,
  inclusive of periodic eval and checkpoint I/O. Peak CUDA about `1,272` to
  `1,593` MiB, well inside the 8 GB card. This is the number the budget
  arithmetic rests on. The 200-step D3a probe rate `0.524990` s/step
  (`runs/stage55_size_85m_b256.jsonl`) is about `1.66x` faster and is NOT used
  for budgeting: a short probe does not capture sustained throttling plus
  eval/checkpoint overhead.
- Recipe v2 cosine overhead: `2.9843` versus `3.0214` steps/sec at 25M
  (`RESULTS.md` Stage 57), a 1.2 percent slowdown, negligible for step-count
  sizing.

Arms (all `random_full`, `L12 H12 D768` = 85.11M, block 256, batch 8,
grad-accum 2, RoPE, activation checkpointing, Muon `0.01`, Recipe v2 fp32 with
`--lr-schedule cosine --lr-final-frac 0.1`, the 33-char union vocab, CUDA):

- **Arm COLD (baseline that CURRICULUM must beat)** · seed 7, 50,000 steps on
  text8 train shards.
- **Arm CURRICULUM** · seed 7, 12,500 steps on TinyStories shards, then 37,500
  steps resumed on text8 shards with the cosine schedule continued over the full
  50,000-step horizon.
- **Arm MIXTURE (secondary contrast)** · seed 7, 50,000 steps on interleaved
  TinyStories-plus-text8 shards at a 1:3 token ratio.
- **Reduced-budget replicas** · seeds 11 and 19 at 20,000 steps for COLD and for
  the deciding childhood arm (the winner under E-curriculum, or CURRICULUM under
  E-interfere), to confirm the SIGN of the margin is seed-robust (the H022
  honest-package precedent). If the publication fork (intake Fork 2) is answered
  ON, all three arms are promoted to three seeds at the full 50,000-step budget
  instead of this reduced tier.

## Primary decision metric and pass or fail line

Primary metric: **text8 TEST split bits/char** (final 5M chars, never seen in
training or in-run eval), computed by `eval_text8.py`'s chunked full-split
convention (ADR 0014 D3) on each arm's final checkpoint. In-run sampled eval is
monitoring only.

Primary trichotomy (CURRICULUM alone, `a = CURRICULUM - COLD` on seed 7):

- **E-curriculum wins** = `a` at or below `-0.05` bits/char.
- **E-interfere wins** = `a` at or above `+0.05` bits/char.
- **E-null** = `|a|` below `0.05` bits/char.

Replica rule (applies symmetrically to whichever verdict seed 7 selects): the
reduced-budget replicas must reproduce the SIGN of `a` at 20,000 steps. If they
agree, the verdict stands as seed-robust in sign. If they DISAGREE with seed 7,
or if the seed-7 margin is marginal (between `0.05` and `0.10`), the result is
INCONCLUSIVE and COLD plus the deciding arm are escalated to full-budget
(50,000-step) seeds 11 and 19 before any ADR is written. Stated limitation: the
20,000-step replicas test seed-robustness at a REDUCED budget, not the sign at
the 50,000-step decision budget; a childhood can interfere early and pay off
late under fixed compute, so a clean replica agreement confirms robustness of
direction, not of magnitude.

Secondary contrasts (reported, each with its own `0.05` line, never merged into
the primary trichotomy):

- **MIXTURE versus COLD** · `MIXTURE - COLD` at or below `-0.05` means a
  distributed childhood dose helps; at or above `+0.05` means it hurts.
- **MIXTURE versus CURRICULUM** · a separation of at least `0.05` means ORDER
  matters at equal dose (subject to the LR-regime caveat above); within `0.05`
  means front-loaded and interleaved childhoods are equivalent here.

Secondary metric (descriptive): **TinyStories val bits/char across the
checkpoint series** for the seed-7 arms, scored on
`corpus/tinystories_char_shards/val.txt` under the 33-char union vocab. This is
the retention-and-forgetting curve ADR 0015 D1 named. Predicted shape, stated
before running: CURRICULUM forgets TinyStories most sharply after switching to
broad text; MIXTURE retains most through rehearsal; COLD starts and stays worst
on TinyStories. Weakly falsifiable read: if MIXTURE does not retain TinyStories
best, the rehearsal mechanism is not operating as modeled and the order
interpretation must be revisited.

INCONCLUSIVE guard (instability): if any full-budget seed-7 arm's final in-run
sampled val NLL is worse than its own mid-budget (step-25,000) value by more than
`0.05`, that arm destabilized; fix before reading any line. The same guard H022
used. For CURRICULUM, step 25,000 is 12,500 steps into the broad phase, so the
guard compares broad-phase mid to broad-phase final, which is the intended
comparison.

## Risks and confounds

- **Fixed compute means unequal broad-text exposure, and that is the design, not
  a bug.** COLD sees about 2.28 broad epochs; the childhood arms see about 1.71
  broad epochs plus a TinyStories phase (token accounting below). The childhood
  arms must overcome that deficit, which is precisely the "smartest spend"
  question. The alternative framing (equal broad-text exposure, childhood added
  on top) was rejected because it cannot answer whether a childhood is worth its
  cost; it only asks whether more total compute helps.
- **Order contrast is confounded with LR regime.** As stated in the mechanism
  section, a single continuous cosine over the full budget means CURRICULUM
  learns broad text under a lower-LR tail than MIXTURE does. The MIXTURE versus
  CURRICULUM comparison is therefore an order-plus-LR contrast, not pure order;
  this is named as an interpretation limit and is intrinsic to front-loading
  under one schedule.
- **Multi-epoch repetition.** 50,000 steps over a 90M-char text8 split is 2.28
  epochs for COLD; repetition can inflate train-side metrics. The untouched
  TEST split is the defense, and epoch counts must be reported per arm.
- **LR-schedule continuity across the CURRICULUM phase boundary.** The cosine
  warmdown must span the full 50,000-step horizon as if one run; if phase 2
  restarts its own warmup, CURRICULUM gets an unfair second high-LR window.
  Codex must pass a global-step offset or total-step horizon so the schedule is
  continuous. This is a required confirm-first item.
- **Dose-versus-order confound between the childhood arms is controlled** by
  fixing MIXTURE's TinyStories fraction to the CURRICULUM fraction (25 percent),
  so CURRICULUM versus MIXTURE isolates order (modulo the LR caveat).
- **Vocab consistency.** All three arms train with the same 33-char union vocab
  (`\n ,.abcdefghijklmnopqrstuvwxyz!?'`, Stage 57 smoke), even COLD, so the
  three checkpoints share one codec and the TinyStories retention eval and any
  logit comparison are apples to apples. text8 uses only 27 of the 33 chars;
  the six unused embedding rows simply never receive gradients in COLD.
- **Reduced-seed decision.** ADR 0015 D1 registered seed 7 at full budget per
  arm plus reduced replicas for the deciding arm; three full seeds per arm is
  adopted only if the publication fork fires. The deciding read is a single
  full-budget seed guarded by reduced-budget replicas, and that limit is stated
  honestly.
- **Device caveat.** CUDA throughout; the decision eval is deterministic chunked
  scoring of fixed weights, so it carries no sampled-eval noise (unlike the
  CPU-versus-CUDA sampled-row caveat in CLAUDE.md).

Budget arithmetic (the piece ADR 0015 D1 left for this document; Stage 57 did
not produce an 85M block-256 SUSTAINED rate, so the anchor is H022's observed
sustained rate): at the sustained `0.87` s/step, a 12 GPU-hour target is
`43,200 / 0.87` = about `49,655` steps, registered as a clean **50,000 steps**
(about `12.1` GPU-hours), which also divides exactly for the 25 percent
childhood split (12,500 + 37,500). Tokens per arm: `50,000 * 8 * 2 * 256` =
`204.8M`. COLD sees `204.8M / 90M` = `2.28` broad epochs (matching H022's
seed-7 Arm A exactly, which is why COLD is expected near `1.485740`); CURRICULUM
and MIXTURE each see `51.2M` TinyStories tokens (well under one TinyStories
epoch) and `153.6M` broad tokens (`1.71` broad epochs). HARD confirm-first
blocker: Codex re-measures the Recipe v2 (cosine, checkpoint-keep) sustained
throughput at 85M block 256 before locking; the 200-step probe rate is known to
under-estimate by about `1.66x`, so the step count must be set from a sustained
measurement to hold each arm at or under 12 GPU-hours, and the `2.28` / `1.71`
epoch figures move with whatever step count survives that re-measure.

## What result would change the plan

- **E-curriculum wins** (CURRICULUM beats COLD) produces the Phase 5 closeout
  ADR recording that a learned simple-register childhood accelerates broad-text
  acquisition at fixed compute, promotes CURRICULUM to the release recipe, and
  makes it the headline developmental result for the publication fork. If the
  MIXTURE secondary contrast also beats COLD, the ADR reports whether order
  matters (subject to the LR caveat).
- **E-interfere wins** (CURRICULUM at or above COLD by `0.05`) produces the
  closeout ADR recording the cross-phase law: learned narrow weights interfere
  exactly as the analytic prior did (Stage 53), unifying the Phase 1 and Phase 5
  findings under one statement, and it retires the "childhood first" idea at this
  scale.
- **E-null** produces an ADR recording acquisition-order invariance at this
  budget and dose, and schedules a single follow-up: widen the TinyStories dose
  or lengthen the budget to test whether the null is a budget artifact before
  any general claim.
- **INCONCLUSIVE** (replica sign disagreement or a marginal seed-7 margin)
  escalates COLD plus the deciding arm to full-budget seeds 11 and 19 before any
  ADR.
- Any outcome updates the standing text8 anchor table (ADR 0014 D3) with the
  three new points and folds the TinyStories retention curve into the model card
  and, if the fork fires, the paper.

## Handoff to Codex (implemented as Codex stage 58, README ladder rung 68)

Files to create or modify:

1. `make_mixture_shards.py` (new, small): deterministically interleave
   `corpus/tinystories_char_shards/` and `corpus/text8_char_shards/` train
   shards at a 1:3 token ratio into `corpus/mixture_char_shards/`, writing a
   `.meta.json` provenance file (the corpus payload stays gitignored per D2).
   Deterministic ordering; no normalization. This is the only new corpus
   artifact; both source shard sets already exist on disk.
2. CURRICULUM two-phase launch: reuse the trainer's checkpoint-resume path so
   phase 1 (TinyStories, 12,500 steps) saves a durable checkpoint that phase 2
   (text8, 37,500 steps) resumes from. Ensure `--lr-schedule cosine` computes
   its factor over the full 50,000-step horizon across the boundary (a
   `--lr-total-steps 50000` or global-step-offset argument), so phase 2 does not
   restart the warmdown. If this continuity is not already expressible, it is the
   one required trainer change.
3. `corpus/phase5_union_vocab.txt` (new): the exact 33-char union alphabet, fed
   to every arm via `--vocab-chars-file` (the newline in the alphabet makes the
   file form safer than an inline `--vocab-chars` string).
4. Register three named configs in `cassandra_compare.py` (`dev_cold`,
   `dev_curriculum_phase2`, `dev_mixture`) or drive the arms directly through the
   trainer, and add the Stage 58 cell to the ADR 0012 visible-launch script.
5. `eval_text8.py` already accepts an explicit `--checkpoint`; add a
   TinyStories-val chunked scorer (or reuse the existing eval with
   `--corpus corpus/tinystories_char_shards/val.txt`) to produce the retention
   curve across the seed-7 checkpoint series.

Confirm-first before launching any arm: (a) re-measure the Recipe v2 sustained
throughput at 85M block 256 and set the exact step count to hold each arm at or
under 12 GPU-hours (the 200-step probe under-estimates by about `1.66x`);
(b) verify the cosine schedule is continuous across the CURRICULUM phase
boundary; (c) checkpoints go to `C:\cassandra_runs` (ADR 0014 D1; never
OneDrive, never `%TEMP%`), and the seed-7 checkpoint series is kept until the
TinyStories retention curve is scored, then pruned to the finals plus fp16
model-only archives.

Canonical command shape (Arm COLD, the deciding seed-7 run; CURRICULUM and
MIXTURE change only the corpus and shard source, with CURRICULUM split into two
resumed phases):

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\text8_char_seed.txt `
  --device cuda --steps 50000 --eval-interval 5000 --log-every 1000 `
  --seeds 7 --configs random_full `
  --block-size 256 --batch-size 8 --grad-accum-steps 2 `
  --pos-encoding rope --activation-checkpoint `
  --optimizer muon --muon-lr 0.01 `
  --lr-schedule cosine --lr-final-frac 0.1 `
  --vocab-chars-file .\experiments\tiny_language_lab\corpus\phase5_union_vocab.txt `
  --eval-mode sampled --eval-batches 16 --no-copy-train-marker `
  --prompt "the history of " --max-new-tokens 240 `
  --n-layer 12 --n-head 12 --n-embd 768 `
  --train-shard-dir .\experiments\tiny_language_lab\corpus\text8_char_shards `
  --stream-train-eval-chars 200000 `
  --val-fraction 0.05263157894736842 `
  --checkpoint-dir C:\cassandra_runs\stage58_dev_cold_checkpoints `
  --checkpoint-every 5000 `
  --out .\experiments\tiny_language_lab\runs\stage58_dev_cold_85m_b50000_seed7.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage58_dev_cold_85m_b50000_seed7.md `
  --title "Stage 58 Developmental COLD 85M 50000-step seed 7"
```

Arm deltas: CURRICULUM phase 1 uses
`--train-shard-dir ...\corpus\tinystories_char_shards --steps 12500` and saves a
durable checkpoint, then phase 2 resumes that checkpoint with
`--train-shard-dir ...\corpus\text8_char_shards --steps 37500` and the continued
cosine horizon (`--lr-total-steps 50000`). MIXTURE uses
`--train-shard-dir ...\corpus\mixture_char_shards --steps 50000`. Reduced-budget
replicas repeat COLD and the deciding arm with `--seeds 11` then `--seeds 19` at
`--steps 20000`.

The deciding metric, restated: after each arm completes, run the chunked
full-test text8 evaluation on its final checkpoint. **CURRICULUM at or below
`COLD - 0.05` bits/char = E-curriculum wins (childhood helps). CURRICULUM at or
above `COLD + 0.05` = E-interfere wins (childhood hurts, the Stage 53 law).
CURRICULUM within `0.05` of COLD = E-null. MIXTURE is reported as a secondary
contrast, not merged into this trichotomy.**

## Prior-art flag for Gemini

This is a curriculum-learning and continual-pretraining study at a small char
scale, and it inverts the usual domain-adaptive-pretraining order. Before any
claim is recorded, Gemini should place it against:

- **TinyStories (Eldan and Li 2023)** as the source of the "simple data teaches
  structure efficiently" bet, and whether anyone has tested that bet as a
  developmental STAGE (narrow then broad) rather than a destination.
- **Curriculum learning (Bengio et al. 2009)** and its mixed record at scale,
  which predicts E-curriculum.
- **Domain-adaptive pretraining (Gururangan et al. 2020)**, noting that H024 is
  the REVERSE order (narrow childhood then broad target), so DAPT's "continued
  pretraining on the target helps" is not a direct precedent.
- **Catastrophic forgetting and rehearsal (French 1999; Kirkpatrick et al. 2017;
  replay-ratio and data-mixing work such as DoReMi)**, which speaks to the
  CURRICULUM-versus-MIXTURE order contrast and the retention curve.
- The lab's own **Stage 53 analytic-prior interference** result, the internal
  precedent for E-interfere.

The specific things Gemini should check: (a) does published curriculum or
continual-pretraining work at comparable small scale show a narrow-then-broad
childhood HELPING or HURTING broad-text perplexity at FIXED compute; (b) is the
fixed-compute framing (childhood displaces target data and must earn it back)
named anywhere, or is it novel here; (c) is there a known verdict on interleaved
versus sequential exposure at equal dose that would predict MIXTURE beating or
losing to CURRICULUM before the run. If the literature says a small model at
this budget cannot recover a front-loaded childhood, the E-interfere prior
strengthens; if narrow-then-broad reliably helps at this scale, the `0.05`
decision margin and the primary directional bet should be revisited before
Codex launches, not after.

## Links

- `docs/decisions/0015-phase-5-developmental-training-and-open-source-posture.md` (D1, the pre-registration)
- `docs/phase5-intake.md` (P2, the developmental experiment)
- `docs/hypotheses/022-broad-corpus-specialization-gap.md` (Stage 56, the broad-char anchor and CONFIRM that fixed the substrate)
- `docs/hypotheses/020-frozen-prior-free-accelerator.md` and `docs/decisions/0013-phase-4-free-accelerator-floor-scaling-flagship.md` (Stage 53 interference precedent)
- `docs/decisions/0014-phase-4-closeout-flagship-verified-evaluation-standard.md` (evaluation conventions, anchor table)
- `experiments/tiny_language_lab/RESULTS.md` (Stages 53, 56, 57, and the behavior probe)
- `experiments/tiny_language_lab/runs/stage55_size_85m_b256.jsonl` (200-step probe rate, contrasted with the sustained rate)
- `research/theme_4_domain_specialization_and_substrates/01_bpe_vs_character_level_small_models.md` and `02_domain_specialization_gap_and_corpus_breadth.md`
- `README.md` Next ladder rung 68 (to be added, marked specced)
