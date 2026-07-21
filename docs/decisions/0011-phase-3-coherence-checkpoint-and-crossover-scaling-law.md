# ADR 0011 · Phase 3 · Coherence Checkpoint, then the Crossover-Scaling Law

## Status

Accepted · implemented on 2026-07-02 as Codex Stage 51 and Stage 52. Human
review of Stage 51 samples was completed by the user on 2026-07-06 and PASSED
(many coherent little stories that made sense; recorded in the Stage 51 entry
of `RESULTS.md`), closing this ADR's last open item. The genuine-coherence
claim stands at the reviewed scope: fluent micro-stories within the block-128
window, not long-range narrative.

Implementation summary: Gate A rebuilt the TinyStories character corpus from a
`500,000,000` byte slice, yielding `494,094,421` normalized characters,
`419,980,257` train characters, `74,114,164` validation characters, and
`V = 33`. Gate B cleared the 85.11M point with a 200-step CUDA smoke at
`1,683.4297 MiB` peak CUDA allocation. Gate C chose a `5000`-step Stage 51
budget for the 25.25M checkpoint. Stage 51 completed at `0.818608` mean
validation NLL and `5.667/6` deterministic generation proxy score. Stage 52
completed the H019 matrix with crossovers `1000`, `1000`, `1000`, and `500`
steps for 3.2M, 10.67M, 25.25M, and 85.11M respectively, making H019 `GRADED`.

## Context

Phase 2 (ADR 0010) is complete on its own terms. Stage 44 built the first
TinyStories character bridge. Stage 45 adapted `modded-nanogpt` pieces, RoPE,
Muon, gradient accumulation, and activation checkpointing, onto the tiny
transformer and showed the frozen order-4 prior still leads at 500 steps
(`1.102748` versus `1.144942` mean val NLL, `41,249` versus `3,176,481`
trainable params). Stage 46 added a generation-quality score sheet. Stage 47
made the trainer shard-native. Stage 48 doubled the budget to 1000 steps and
found the crossover: the full model passes the frozen prior by then
(`1.052559` versus `1.123161`, prior-minus-full `+0.0706` mean NLL, positive on
every seed). Stages 49 to 50 gave BPE a feasibility smoke and a 500-step
decision surface; the BPE bigram prior lost to `random_full`
(`3.344760` versus `2.404960`), so BPE stays a future ablation, not the Phase 3
substrate.

ADR 0010 D5 registered a live research question as candidate `H019`: does the
early-compute crossover budget move as a function of model capacity and corpus
complexity. Phase 1 located the crossover near 100 steps at tiny synthetic
scale. Stage 48 relocated it to between 500 and 1000 steps at 3.2M params on
real TinyStories text. That is one data point on a corpus axis (synthetic to
real) at one model size. `H019` was never formalized as its own hypothesis
document or measured across a size ladder; this ADR does both.

The 2026-07-01 Phase 3 intake brief (`docs/phase3-intake.md`) resolves the
three open forks. Fork 1, model target: char-level, 10M to 30M params,
chasing genuine TinyStories coherence first, keeping the order-4 char prior
apparatus intact rather than committing to BPE now. Fork 2, primary
deliverable: both, sequenced, a coherent checkpoint first and the
crossover-scaling law second. Fork 3, hardware: stay on the 8 GB laptop.
Publication intent is undecided, which argues for keeping the crossover sweep
at the project's usual three-seed rigor so it can be upgraded to a workshop
paper later without rerunning it.

Two measured facts change the shape of Phase 3 versus what the intake brief
assumed going in.

Feasibility is not the constraint the brief expected. A 5-step CUDA smoke at
each candidate size, on `tinystories_char_seed.txt`, with the Stage 45 modern
recipe (RoPE, Muon, activation checkpointing, `block_size=128 batch_size=8
grad_accum_steps=2`), used `269.2 MiB` peak VRAM at 10.67M params, `459.1 MiB`
at 25.25M params, and only `1194.0 MiB` at 85.11M params, against an 8188 MiB
card. The brief's assumption that the ~100M point would most likely not fit
does not hold at this batch and block size; the headroom comes from the small
effective batch (16) and activation checkpointing, not from the models being
small. This is feasibility evidence from a 5-step probe, not a real-run
clearance: optimizer state and memory growth over hundreds of steps are not
exercised by 5 steps, so a longer confirmation smoke at the top size is still
warranted before committing a full matrix there (see Decision D5).

The corpus is thinner than the target model sizes want. The current
`tinystories_char_seed.txt` is about 10.08M characters, itself a
`download_tinystories.py --byte-cap` slice of a 50 MB (`50,000,000` byte) head
of the official `TinyStories-train.txt`, which is itself far short of the full
upstream file. Training an 8.5M-training-character split against a 25M to
30M-parameter model, the top of Fork 1's range, is a params-to-training-chars
ratio that invites memorization over the genuine coherence the checkpoint
deliverable wants. Phase 3 needs a larger corpus pull before the checkpoint
run, not just a bigger model on the same slice.

An external constraint also shapes sequencing: the user has an external
internship from 2026-07-20 to 2026-08-21
(`memory/phase3-internship-window.md`). Hands-on steps, corpus
rescale, the launch-gate confirmation, and generation-sample review, should
land before 2026-07-20. The larger, more autonomous crossover-scaling matrix
suits the internship window itself, where Codex can run a multi-seed sweep
with less supervision.

## Decision

### D1 · Tokenization stays character-level

Continue the character-level TinyStories path from ADR 0010 D3. The order-4
frozen prior apparatus (`--residual-base count-ngram --prior-order 4`) ports
unchanged, and Stage 50 already showed the BPE bigram prior losing to
`random_full` at the small v256 vocabulary tested. BPE remains a tracked,
open ablation, not the Phase 3 substrate.

### D2 · Corpus rescale before the checkpoint run

Rerun `download_tinystories.py` with a materially larger byte cap (or no cap)
than the current 50 MB slice, then rebuild `tinystories_char_seed.txt` and its
metadata with `make_tinystories_corpus.py` at that larger scale, keeping the
deterministic seed and the train and validation split convention. Measure the
resulting character count and, if it is large enough to stress in-memory
loading, switch the checkpoint run onto the Stage 47 shard-streaming path
rather than assuming one-shot loading still holds. This is a confirmation-pass
item, not a launch blocker for the intake brief itself.

### D3 · Coherence checkpoint, the first deliverable

Train `random_full` at the upper end of the char-level range, matching the
measured `n_layer=8 n_head=8 n_embd=512` config, `25,253,921` trainable
params, on the rescaled corpus, for a budget long enough to plausibly clear
Stage 48's 500-to-1000-step crossover bracket at a larger size, for example
2000 to 5000 steps, sized by the confirmation pass against real wall-clock
measurement rather than assumed. Extend the Stage 46 generation-quality score
sheet with an explicit pass bar for calling this checkpoint "genuinely
coherent," since Stage 45's 500-step, 3.2M samples were rough by the scorer's
own read. Human review of saved samples is required before any external
coherence claim, continuing the ADR 0010 confirmation report's still-pending
item.

### D4 · Crossover-scaling law, the second deliverable, formalized as H019

Register `docs/hypotheses/019-crossover-scaling-law.md`, testing whether the
early-compute crossover step count (the point where `random_full`'s mean
validation NLL passes the frozen order-4-prior arm's) is a function of model
capacity, holding the rescaled TinyStories corpus fixed. Sweep model size
across the measured points, 3.2M (Stage 45 and 48's existing baseline), 10.67M,
25.25M, and 85.11M if the D5 confirmation smoke clears it, at a step-budget
ladder bracketing the Stage 48 crossover, for example 200, 500, 1000, and 2000
steps, three seeds `7 11 19` per cell, matching project convention. Sequence
this after D3 so the checkpoint deliverable is not blocked on the larger
matrix, and so it can reuse the rescaled corpus and any shard-streaming
plumbing D3 already exercised.

### D5 · Hardware stays the 8 GB laptop, with one explicit gate

Remain on the current hardware per Fork 3. Before including the 85M point in
the D4 matrix, run a longer confirmation smoke, 150 to 200 steps, at that size
on the rescaled corpus and record peak VRAM and wall-clock. Only escalate to a
24 GB upgrade or hourly rented compute (ADR 0010 D7's stated fallback path) if
that confirmation smoke OOMs or if a size point beyond what the laptop can
hold becomes necessary to resolve H019's direction.

### D6 · Scheduling around the internship window

Front-load D1 through D3, corpus rescale, the checkpoint run, and its human
review, before 2026-07-20. Use the 2026-07-20 to 2026-08-21 window for D4's
larger, multi-size, multi-seed matrix, which Codex can run with less active
supervision. Resume steering pace after 2026-08-21.

### D7 · Evaluation

Both deliverables report validation NLL and bits per character as before.
D3 additionally reports the extended generation-quality score sheet with a
named coherence bar. D4 additionally reports, per size, the measured crossover
step (the smallest tested budget where `random_full` first passes the prior
arm) and the trainable-parameter and frozen-prior-parameter counts already in
the report schema.

### D8 · Publication posture

Since publication intent is undecided, run D4 at the project's standard
three-seed rigor rather than cutting seeds for speed, so the result is usable
as repo-only evidence now and upgradable to a workshop-paper-grade claim later
without rerunning.

## Consequences

### Positive

Phase 3 produces a demoable artifact, the coherence checkpoint, before the
scarcer-attention window opens, and hands Codex a well-specified, largely
autonomous matrix, the crossover-scaling sweep, for the window itself. The
feasibility probe removes an assumed hardware blocker from the plan. The
corpus rescale fixes a real params-to-data mismatch before it can quietly cap
checkpoint quality.

### Accepted costs

The corpus rescale and the D5 confirmation smoke add a step before the main
runs can launch; neither is optional without risking a checkpoint trained on
too little data or a matrix cell that silently OOMs mid-sweep. The 85M point
in D4 is conditional, not guaranteed, on the confirmation smoke.

### Out of scope

Everything Section 4 of the intake brief reconfirmed stays closed: external
memory and retrieval (ADR 0001), data-side selection (ADR 0004), the frozen
recency base (H012), non-gradient residual formation (Stage 36), and general
in-context copy for the frozen-prior family beyond seen-content (ADR 0006,
narrowed by ADR 0008). BPE as the Phase 3 substrate (Fork 1) and a hardware
upgrade absent a triggering OOM (D5) are also out of scope for now.

## Resolved and Remaining Questions

1. Resolved: D2 rebuilt the corpus at `494,094,421` normalized characters and
   switched downstream runs onto shard-streaming where needed.
2. Resolved: D3 chose `5000` steps after the 25.25M 500-step timing pass.
3. Resolved: D5 kept the 85.11M point after the 200-step smoke completed with
   `1,683.4297 MiB` peak CUDA allocation.
4. Resolved: Stage 52 remeasured the 3.2M baseline on the rescaled corpus. Its
   crossover remained at the `1000` tested budget.
5. Remaining: Mert still needs to review Stage 51 saved samples before the repo
   can claim genuine TinyStories coherence.

## References

- `docs/decisions/0010-phase-2-from-scratch-model-build.md`
- `docs/decisions/0010-confirmation-report.md`
- `docs/phase3-intake.md`
- `docs/hypotheses/019-crossover-scaling-law.md`
- `experiments/tiny_language_lab/RESULTS.md` (Stages 44 to 50)
- `experiments/tiny_language_lab/runs/phase2_tinystories_modern_b500.md`,
  `phase2_tinystories_modern_b1000.md`
- `memory/phase3-internship-window.md`
