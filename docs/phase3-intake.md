# Phase 3 Intake Brief · Cassandra

Filled 2026-07-01. Section 1 answered directly by the user. Sections 2 to 4
facts gathered by Claude via measurement (nvidia-smi, torch, and 5-step CUDA
smokes), not estimated. This doc feeds ADR 0011.

## Section 1 · The three forks · answered

- Fork 1 · Model target: **Option A**, char-level 10M to 30M params, chasing
  genuine TinyStories coherence first. Keeps the order-4 char prior apparatus
  intact.
- Fork 2 · Primary deliverable: **Option C**, both, sequenced. Checkpoint
  first, crossover-scaling law (H019) second.
- Fork 3 · Hardware: **Option A**, stay on the 8 GB laptop.
- Publication intent: **undecided**. Keep the crossover sweep rigorous enough
  to upgrade to a workshop paper later without rerunning everything.

## Section 2 · Constraints only the user knows

- Timeline, effort appetite, budget ceiling: not specified this round.
- Hard external constraint: an external internship runs 2026-07-20 to
  2026-08-21. See `memory/phase3-internship-window.md`. ADR 0011
  phases work so hands-on steps land before 2026-07-20 and Codex-autonomous
  matrix runs fill the internship window.

## Section 3 · Facts gathered

### Hardware and environment

- GPU model and VRAM: NVIDIA GeForce RTX 4070 Laptop GPU, **8188 MiB** total
  (measured via `nvidia-smi`).
- Driver version: **591.59**.
- Torch version: **2.12.1+cu126**. CUDA version (torch build): **12.6**.
  `torch.cuda.is_available()` is `True`.
- Peak VRAM, current 3.2M modern config (`n_layer=4 n_head=4 n_embd=256
  block_size=128 batch_size=8 grad_accum=2`, RoPE, Muon, activation
  checkpointing, `tinystories_char_seed.txt`, 5-step CUDA smoke,
  `torch.cuda.max_memory_allocated()`): **157.9 MiB**.

### Data on hand

- TinyStories characters available locally right now:
  `corpus/tinystories_char_seed.txt` is **10,078,510 chars** (`wc -c`). Stage
  44's quoted `10,000,001` is the same build, off by a small header/count
  convention; treat both as "about 10M chars."
- Raw source: `corpus/tinystories_raw/TinyStories-train.head50mb.txt` is
  exactly **50,000,000 bytes**, a `download_tinystories.py --byte-cap` capped
  slice of the official `roneneldan/TinyStories` `TinyStories-train.txt` on
  Hugging Face (`DEFAULT_URL` in `download_tinystories.py`).
- Full local availability if more is pulled: **not verified locally**. The
  full upstream `TinyStories-train.txt` is publicly documented at roughly 1
  to 2 GB; this repo has never downloaded past the 50 MB cap, so treat any
  larger figure as unconfirmed until `download_tinystories.py` is rerun with a
  larger or no byte cap and the result is measured.

### Timing anchors for the compute budget

- Modern 500-step wall-clock per row at 3.2M (`random_full`, seed-mean, three
  seeds `7 11 19`): **39.8650 s**, confirmed against Stage 45's
  `phase2_tinystories_modern_b500.md`.
- Modern 1000-step wall-clock per row at 3.2M (`random_full`, seed-mean):
  **68.7135 s**, confirmed against Stage 48's
  `phase2_tinystories_modern_b1000.md`.
- Both numbers are exact matches to the values already recorded in
  `RESULTS.md`; no discrepancy found.

### Feasibility probe (5-step CUDA smokes, `tinystories_char_seed.txt`, RoPE,
Muon, activation checkpointing, `block_size=128 batch_size=8
grad_accum_steps=2`, sampled eval, seed 7)

| Target | Config | Trainable params (measured) | Peak VRAM | Seconds/step (5-step, noisy) | Fits? |
| --- | --- | ---: | ---: | ---: | --- |
| ~10M | `n_layer=6 n_head=6 n_embd=384` | 10,672,929 | 269.2 MiB | ~0.93 s | Yes |
| ~25-30M | `n_layer=8 n_head=8 n_embd=512` | 25,253,921 | 459.1 MiB | ~0.59 s | Yes |
| ~85-100M | `n_layer=12 n_head=12 n_embd=768` | 85,106,721 | 1194.0 MiB | ~0.82 s | Yes, no OOM |

None of the three candidate sizes came close to the 8 GB ceiling; the largest
used about 15% of available VRAM. This contradicts the brief's working
assumption that the ~100M point would most likely OOM on this card. The
headroom comes from the small batch size (effective batch 16 via
`batch_size=8` times `grad_accum_steps=2`) and activation checkpointing, not
from the model being small. A 5-step smoke does not stress optimizer-state
memory or longer-run memory growth the way a real 500 to 1000 step run does,
so this is feasibility evidence, not a final clearance; ADR 0011 recommends a
longer (150 to 200 step) confirmation smoke at the top end before committing
a full crossover matrix there.

First-probe caveat: an initial probe run hung because `--eval-mode` defaults
to `full` (a full pass over the ~1.5M-char validation split at step 0), not
the `sampled` mode the real Phase 2 matrices use. It was killed and rerun
with `--eval-mode sampled --eval-batches 2`, which produced the numbers
above in seconds. Any future smoke or matrix launcher must pass
`--eval-mode sampled` explicitly; do not rely on the default.

## Section 4 · Scope confirmation

All confirmed unchanged for Phase 3; none of Phase 3's two deliverables
(char-level coherence checkpoint, H019 crossover-scaling law) touch these
branches:

- External memory and retrieval (ADR 0001): confirmed.
- Data-side selection, static and dynamic (ADR 0004): confirmed.
- Frozen recency base (H012): confirmed.
- Non-gradient residual formation (Stage 36): confirmed.
- General in-context copy for the frozen-prior family (Stages 40 to 43, ADR
  0006 scoped to seen-key identity copy): confirmed.

## Links

- `docs/decisions/0011-phase-3-coherence-checkpoint-and-crossover-scaling-law.md`
- `docs/hypotheses/019-crossover-scaling-law.md`
- `docs/decisions/0010-phase-2-from-scratch-model-build.md`
- `docs/decisions/0010-confirmation-report.md`
