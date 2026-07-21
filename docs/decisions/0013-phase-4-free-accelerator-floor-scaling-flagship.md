# ADR 0013 · Phase 4 · Free-Accelerator Test, Floor Scaling, then a Flagship Build

## Status

Accepted · scope directed by the user on 2026-07-02 ("B, then A, then a
bigger better flagship build"), expanding the Phase 4 intake's Fork 1 beyond
Claude's two-item recommendation. The Stage 51 sample review (ADR 0011's one
open item) was completed by the user on 2026-07-06 and PASSED (many coherent
little stories that made sense; recorded in the Stage 51 entry of
`RESULTS.md`). The Stage 51 coherence gate is cleared; the flagship still
requires its own human review after the long run (open question 3: the
flagship's bar is now "clearly better than the Stage 51 samples").

Implementation: Stage 53 (D1) COMPLETE on 2026-07-02, verdict KILL
E-interfere (early lead `-0.252` at 200 steps, late loss `+0.072/+0.100` at
1000/2000 on all paired seeds, surviving the mandated `--muon-lr 0.005`
rerun at `+0.055`). D3's assembly rule therefore FIRES toward RANDOM INIT
for the flagship, and H021's flagship-carrying clause is moot; Stage 54's
floor-scaling question stands on its own scientific value (see D2 note).
Stage 53 also confirmed checkpoint RESUME does not exist yet; D3's resume
requirement is a build item, not a verification item. Stages 54 and 55
pending.

Closeout: Stage 54 CONFIRMED (order-5 floor `-0.117` vs order 4) and
Stage 55 delivered the flagship (seed 7 `0.556410` val NLL at 50k steps,
replicas at 20k) with a parity-verified ONNX export. Phase 4 is CLOSED by
ADR 0014, which also upgrades the closeout evaluation standard and adopts
the zero-shot text8 external anchor.

## Context

Phase 3 (ADR 0011) completed on 2026-07-02: Stage 51 delivered the 25.25M
coherence checkpoint (`0.818608` mean val NLL, proxy `5.667/6`, human review
pending) and Stage 52 delivered the H019 crossover matrix (verdict `GRADED`,
crossovers `1000/1000/1000/500` steps at 3.2M/10.67M/25.25M/85.11M).

The Phase 4 intake (`docs/phase4-intake.md`) re-read Stage 52 and found three
facts that shape this phase:

1. The prior arm is a CONSTANT, not a curve: all 16 cells sit in
   `[1.103, 1.116]` regardless of size or budget, so the crossover is one
   learning curve crossing a fixed floor near `1.104` set entirely by the
   order-4 prior. Whoever controls the floor controls the crossover.
2. The continuous capacity signal is clean: prior-minus-full is strictly
   monotone in capacity at every budget, 16 of 16 cells (weak E1). The
   discrete budget ladder, not the physics, produced the `GRADED` verdict.
3. Every Stage 52 cell is token-limited: at `2,048` train chars per step, a
   2000-step run sees about 1% of the 420M-char train split, so there is
   enormous headroom for longer training before any data repeats.

Fact 1 exposes the two measurement gaps Phase 4's first two deliverables
fill: nobody has measured the prior UNDER full training (every prior arm so
far was capacity-starved at rank 2), and nobody has re-tested the Stage 35
analytic-order ceiling since the corpus grew 42x. Fact 3 motivates the
flagship: the lab has never trained anything close to its data or its
hardware, and the first two deliverables decide what the strongest cheap
recipe for such a build actually is.

## Decision

### D1 · Stage 53, the free-accelerator test (H020)

Run `docs/hypotheses/020-frozen-prior-free-accelerator.md`: a 25.25M model
with `--residual-base count-ngram --prior-order 4 --train-scope all` versus
the existing Stage 52 `random_full` rows, budgets 200/500/1000/2000, seeds
`7 11 19`. One new config branch (`count_prior_ng4_all`), the prior loads
from the Stage 52 disk cache, about one GPU-hour. Verdict vocabulary:
E-accel (prior never hurts, helps early), E-wash (transient head start),
E-interfere (prior is a handicap under full training, with one
LR-sensitivity rerun before that verdict is recorded). This is first
because it is the cheapest and it decides the flagship's initialization.

### D2 · Stage 54, prior-order floor scaling (H021), behind a feasibility gate

Run `docs/hypotheses/021-prior-order-floor-scaling.md` in two phases.
Phase A gate: build a shard-native order-5 prior (backoff-over-order-4 as
the default design; dense order-5 is about 4.8 GiB fp32 and the code caps
`--prior-order` at 4 today, so this is new code plus measurement), report
build cost, memory, validation backoff rate, and the order-5 FLOOR versus a
PAIRED order-4 floor arm run in the same invocation (Stage 52 ran no floor
arm, so its trained rank-2 means are context, not the baseline); pass only
if the mean paired floor delta is at or below `-0.02` NLL with all three
paired per-seed deltas negative. Phase B (gate-conditional): the 25.25M crossover
column with `count_prior_ng5_lora_r2`, testing whether the crossover moves
at least one rung later. A gate failure is a cheap, clean KILL that closes
the prior-order axis with its corpus-scaling clause tested rather than
assumed.

Post-Stage-53 note: with D1's KILL, the flagship no longer inherits any
prior, so D2's original flagship-feeding rationale is moot. D2 stays
scheduled on its standalone merits: whether the analytic floor scales with
corpus size is the strongest remaining NLL-axis claim of the thesis, the
gate bounds its cost, and it is the second pillar (with the Stage 52
monotone-gap fact) of the potential workshop paper under D5. Its CONFIRM
and GRADED consequences now read "the low-budget tiny-surface recipe gets a
better floor," not "the flagship gets a better base."

### D3 · Stage 55, the flagship build

Train the strongest checkpoint this laptop and corpus support, with the
recipe ASSEMBLED FROM EVIDENCE rather than chosen by taste:

- Initialization and base: frozen count prior if D1 returns E-accel or
  E-wash; random init if D1 returns E-interfere. Prior order 5 if D2's gate
  passed, else order 4. DECIDED by Stage 53: D1 returned E-interfere, so
  the flagship is `random_full`, no prior base, and the order clause is
  moot for this deliverable.
- Sizing gate (D3a, before the long run): a short timing-and-VRAM pass over
  the candidate axes, model size (85.11M default; one larger candidate such
  as `n_layer=16 n_head=16 n_embd=1024`, about 200M params, may be probed
  since the 85M point showed large headroom), block size (128 versus
  256; doubling block doubles chars seen per step for the same step count),
  and step budget. The 85M headroom evidence is Stage 51 Gate B's 200-step
  confirmation smoke (`1,683.4297 MiB` peak CUDA), a smoke-scale number,
  which is exactly why D3a re-measures at the flagship's own configuration
  rather than assuming. The gate picks the largest configuration whose
  measured wall-clock fits the internship window comfortably and whose peak
  CUDA stays at least 1 GiB under the card limit.
- Token target: at least `200M` training characters seen (about half the
  train split), with one full epoch (`420M`) as the stretch goal. For
  reference, block 256 at batch 8 and grad-accum 2 is `4,096` chars per
  step, so 50k steps sees about 205M chars.
- Seeds: 3 seeds (`7 11 19`) if the sizing gate's projected total stays
  under about 36 GPU-hours; otherwise one flagship seed at full budget plus
  two replicate seeds at a reduced budget, stated plainly in the results.
- Evaluation: val NLL and bits/char, the Stage 46 generation score sheet,
  saved samples and checkpoints, and REQUIRED human review before any
  coherence or quality claim (carrying ADR 0011 D3's rule forward).

### D4 · Scheduling around the internship window

Before 2026-07-20: the Stage 51 sample review (user action, also calibrates
the flagship's quality bar), Stage 53, and Stage 54 Phase A (the gate).
Inside 2026-07-20 to 2026-08-21: Stage 54 Phase B if gated in, the D3a
sizing gate, and the flagship long run, which is the most
autonomy-compatible item this project has ever queued. After 2026-08-21:
flagship human review and the Phase 4 close-out.

### D5 · Publication posture

Still undecided (intake Fork 2). H019b, the finer step ladder that would
convert Stage 52's `GRADED` into a clean continuous law, is REGISTERED as an
optional deferred item, activated only if the user decides to aim for a
workshop paper; the monotone-gap fact plus D2's floor-scaling result would
be its backbone. All Phase 4 runs keep three-seed rigor so nothing needs
rerunning if publication is chosen later.

### D6 · Evaluation and evidence standard

Every stage reports the standard schema (command shape, corpus and split,
seeds, trainable and frozen parameter counts, val NLL and bits/char, wall
clock, interpretation of what it does and does not prove) into `RESULTS.md`
and the run summaries. Stage 53 and 54 reuse Stage 52 rows where protocols
match instead of re-spending compute, and say so explicitly in their
summaries.

## Consequences

### Positive

Phase 4 resolves the two cheapest high-leverage unknowns the Stage 52
re-read exposed, then spends the internship window's unattended compute on
the first build that actually uses this lab's data and hardware, with every
recipe choice in the flagship traceable to a measured verdict. The
free-accelerator test in particular upgrades or bounds the north-star claim
in a single GPU-hour.

### Accepted costs

The order-5 builder is real new code with a real chance its gate fails;
that cost is capped by running the gate before the matrix. The flagship is
the project's first multi-hour-per-seed run, so a mid-run interruption
(sleep, move, OOM late in training) is a live risk; the visible-launch
protocol (ADR 0012) plus periodic checkpointing must be used, and Codex
should confirm checkpoint-resume works at the sizing gate before the long
run starts.

### Out of scope

All closed branches stay closed: external memory and retrieval (ADR 0001),
data-side selection (ADR 0004), non-gradient residual formation (ADR 0005),
seen-content copy scope (ADR 0006 as narrowed by ADR 0008), the held-out
token probe (ADR 0007), BPE as substrate (ADR 0011 D1). D2 deliberately
re-tests the Stage 35 ceiling and is argued in the intake (Section 4) to
apply that ceiling's own estimability rule under 42x more data rather than
contradict it. The behavior axis stays paused through Phase 4.

## Open questions

1. Does Muon's learning rate need retuning on top of a frozen additive base?
   H020 carries a one-cell LR-sensitivity rerun to keep an E-interfere
   verdict honest.
2. The flagship's exact configuration (size, block, budget, seed split) is
   deliberately deferred to the D3a sizing gate's measurements.
3. Whether the Stage 51 review's verdict raises or lowers the flagship's
   coherence bar is unknown until the user reviews the samples.

## References

- `docs/phase4-intake.md`
- `docs/hypotheses/020-frozen-prior-free-accelerator.md`
- `docs/hypotheses/021-prior-order-floor-scaling.md`
- `docs/hypotheses/019-crossover-scaling-law.md`
- `docs/decisions/0011-phase-3-coherence-checkpoint-and-crossover-scaling-law.md`
- `docs/decisions/0012-visible-terminal-experiment-launches.md`
- `experiments/tiny_language_lab/runs/stage52_h019_crossover_scaling_summary.md`
- `experiments/tiny_language_lab/RESULTS.md` (Stages 51, 52)
