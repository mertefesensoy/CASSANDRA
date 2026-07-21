# Phase 5 Intake · How a Young Model Grows Up

Date: 2026-07-07 · drafted by Claude under ADR 0014 D6's intake gate.
Status: DRAFT · Section 6 forks await the user. Nothing here is committed
until the forks are answered and an ADR is written.

## 1 · Where Phase 4 left us

Phase 4 delivered a verified 201.6M char-level flagship (in-domain `0.8126`
bits/char, parity-verified ONNX, resume-proven training) and, more
importantly, one huge unexplained number: the **`2.07` bits/char
specialization gap** between in-domain quality and zero-shot text8
(`2.8817` vs GPT-2 117M's `1.17`). H022 (Codex Stage 56, specced, threshold
calibrated 2026-07-07) will factor that gap into data versus substrate.

The lab also carries known recipe debt: fp32 training on a bf16-capable
GPU, a constant LR tuned at 2000-step horizons, block 256 on a corpus of
750-to-1500-char stories, and keep-everything checkpoints. Each is a
multiplier on every future GPU-hour, unexploited.

## 2 · The Phase 5 thesis: not more data · smarter acquisition

The user's directive: the models are new and young; be smart instead of
throwing more data and training at them. Three measured facts make one
question the obvious centerpiece:

1. Stage 53 measured that **analytic priors interfere** with late
   full-body training (`+0.100` NLL at 2000 steps). That closed the
   analytic-prior-as-initialization branch.
2. The lab now owns something it never had before: **trained narrow-domain
   weights** (the TinyStories family). Whether a LEARNED simple-domain
   initialization behaves like the analytic prior (interferes), or like a
   curriculum (accelerates), has never been tested here, and it is the
   founding thesis of this lab · cheap starting points versus brute force ·
   reborn one level up.
3. TinyStories itself exists (Eldan and Li 2023) as a bet that SIMPLE data
   teaches structure efficiently. Phase 5 can test whether that bet holds
   as a developmental STAGE rather than a destination: does a model that
   spends part of its budget in a simple register learn broad text better
   or worse than one that never had a childhood?

So Phase 5 asks: **given one fixed compute budget, what is the smartest
acquisition order for breadth?** That is a curriculum and transfer
question, answerable on this laptop, novel at this scale point, and it
converts the flagship from an endpoint into an instrument.

## 3 · Pillars

### P0 · Prerequisites (this week, before 2026-07-20)

- **H022 threshold calibration · DONE 2026-07-07.** CONFIRM line
  recalibrated from `1.60` to `1.70` bits/char against the published
  converged record (Al-Rfou 2018 12L/44M `~1.11` to `1.18`; Transformer-XL
  `1.08`; AWD-LSTM `1.23`). Gemini asked to confirm or supersede.
- **Human A/B review** (user action): about 20 votes in the playground's
  blind A/B tab. Gates any coherence claim (ADR 0014 D4) and directly
  informs P2's retention metric design.
- **Storage cleanup execution** (Codex): the ADR 0014 D5 unblocked tiers,
  about 35 GB back across resume-proof, superseded OneDrive checkpoints,
  and `%TEMP%` intermediates.
- **ADR 0009 draft disposition** (Claude): accept the orphaned codex-draft
  with revisions or fold its evidence into ADR 0008, so the numbering gap
  is documented.

### P1 · Recipe v2 gate (Codex Stage 57, cheap, multiplies everything)

D3a-style 200-step measured gates, then LOCK a Recipe v2 for all Phase 5
runs: bf16 autocast (acceptance: within `0.01` NLL of fp32 at 200 steps
with at least 1.4x throughput, else stay fp32), cosine LR warmdown flags,
`--checkpoint-keep N`, a `--vocab-chars` override (needed by P2's
cross-domain evals and fixes H022's 27-char limitation), and one optional
block-512 gate row. Rationale: every P2 arm costs 10 to 16 GPU-hours;
a 1.5x recipe speedup pays for an entire extra arm.

### P2 · The developmental experiment (H024 candidate · the centerpiece)

Three ways to grow an 85M model, ONE fixed total budget per arm (target
about 12 GPU-hours under Recipe v2), same final evaluation:

- **Arm COLD**: random init, broad corpus (text8 train split) for the full
  budget. The baseline every story must beat.
- **Arm CURRICULUM**: the same budget split · first a TinyStories phase
  (about 25 to 30 percent), then broad text for the remainder, continuing
  from the narrow weights. The "childhood first" arm.
- **Arm MIXTURE**: interleaved TinyStories and broad shards at a fixed
  ratio for the full budget. The "grow up bilingual" arm.

Metrics, all under ADR 0014 conventions: text8 TEST bits/char (primary,
chunked full split), TinyStories val bits/char across the checkpoint
series (the retention and forgetting curve, union vocab via
`--vocab-chars`), and the standing anchor table. Mechanism tension, stated
before running: curriculum literature and TinyStories' own thesis predict
CURRICULUM at or above COLD; the Stage 53 analog predicts learned narrow
weights interfere just like analytic priors did. Either outcome is a
finding; agreement between LEARNED-prior behavior and ANALYTIC-prior
behavior would be the cleanest cross-phase law this lab has produced.

Gating: P2's exact arms are FROZEN only after H022 reads out. If H022
KILLs (substrate story), P2 runs on the H023 BPE substrate instead of
char, same three-arm design. The developmental question survives either
substrate; only the alphabet changes.

### P3 · Behavior probe at scale (optional, eval-only, zero training cost)

The behavior axis (paused since ADR 0008) asked whether cheap surfaces
form general in-context copy circuits; the full model formed a weak one at
3.2M. Nobody has measured whether the 200M flagship formed a strong one
for free. A letters-only variant of the Stage 42 memorization-proof copy
probe (the current generator emits digits, which the vocab lacks) scored
zero-shot against flagship and P2 checkpoints would reopen the axis for
the cost of an evaluation script.

### P4 · Publication fork (Fork 2, decision now due)

The paper skeleton now exists without further experiments: the crossover
law (Stage 52), floor scaling (Stage 54), prior interference (Stage 53),
the measured specialization gap (ADR 0014), plus P2's developmental result
as the headline. Target class: an efficient-ML or science-of-scale
workshop. Deciding NOW matters because P2 keeps three-seed rigor only if
publication is on.

## 4 · Scheduling around the internship (2026-07-20 to 2026-08-21)

- Before the window: P0 items, H022's runs (about 13 GPU-hours), and the
  Stage 57 recipe gate. All hands-on-friendly.
- Inside the window: P2's long arms (3 arms x 12 h, plus replicas if
  Fork 2 says yes), the most autonomy-compatible workload the lab has ever
  queued, matching how Stage 52 used the Phase 3 window. Gemini note
  cycles and paper drafting (if P4 fires) also fit here.
- After the window: P2 analysis, the Phase 5 close-out ADR, and the paper
  submission if chosen.

## 5 · What Phase 5 does NOT do

No model above 201.6M. No new corpus beyond text8 (and its mixtures) until
the developmental question is answered. No reopening of closed branches:
retrieval (ADR 0001), data-side selection for tiny surfaces (ADR 0004),
non-gradient formation (ADR 0005), the held-out-token probe (ADR 0007),
seen-content copy scope (ADR 0006 as narrowed by 0008). The equal-compute
85M-versus-200M control stays registered but unscheduled unless a fork
promotes it; P2's COLD arm partially informs it for free.

## 6 · Forks for the user

- **Fork 1 · Centerpiece.** Adopt P2 (the three-arm developmental
  experiment) as Phase 5's flagship question? Alternatives: run only
  H022/H023 and defer P2; or promote a different centerpiece.
- **Fork 2 · Publication.** Aim for a workshop paper this cycle (P2 gets
  three-seed rigor, drafting happens inside the window), or defer again.
- **Fork 3 · Recipe v2 scope.** Minimal (bf16 + LR schedule + vocab flag),
  or also the block-512 gate row. Recommendation: all of it; the gates are
  200 steps each.
- **Fork 4 · Behavior probe.** Include P3 (eval-only) or keep the behavior
  axis paused through Phase 5.

## 7 · Links

- `docs/decisions/0014-phase-4-closeout-flagship-verified-evaluation-standard.md`
- `docs/hypotheses/022-broad-corpus-specialization-gap.md` (Stage 56, calibrated)
- `docs/phase4-flagship-evaluation-report.md` (the gap measurement)
- `research/theme_4_domain_specialization_and_substrates/01_...` and `02_...`
- Eldan and Li 2023 (TinyStories); Gururangan et al. 2020 (DAPT);
  Al-Rfou et al. 2018; Radford et al. 2018, 2019
- `memory/phase3-internship-window.md` (scheduling constraint)
