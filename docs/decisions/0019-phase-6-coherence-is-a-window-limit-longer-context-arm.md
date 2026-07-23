# ADR 0019 · The flagship's topical drift is a context-window limit, not a substrate failure: Phase 6's coherence intervention is a longer-context char arm, not a subword swap

- Status: Accepted
- Date: 2026-07-23
- Author: Claude (hypothesis, ADR, and roadmap role)
- Resolves: H027
  (`docs/hypotheses/027-context-utilization-and-coherence.md`) and the
  Phase 6 coherence-intervention fork it registered (Workstream 3).
- Builds on: Stage 62 (the context-utilization probe,
  `runs/stage62_context_util_flagship.json`, `RESULTS.md` Stage 62), the
  Stage 61 flagship and its sample review (`runs/stage61_user_samples.md`,
  the user's "off-topic, like ADHD" reading), ADR 0018 (the pure-broad
  flagship), and Stage 57's block-512 timing row (`RESULTS.md` Stage 57).

## Context

The Stage 61 pure-broad 200M flagship writes locally fluent but topically
drifting text. The user asked how to make it understand more of what it is
writing. The honest first problem was that the drift is invisible to NLL
(validation `1.288` bits/char, still drifts), so H027 built an instrument to
measure whether the model actually USES its context, with a pre-registered rule
that the answer picks between two differently-priced interventions: a longer
context window (if the window is the binding constraint) or a subword substrate
(if the model ignores the window it already has).

Stage 62 ran that probe on the flagship at N=4096 held-out text8 passages. It
scores the NLL of a target segment under its TRUE preceding context versus a
RANDOM other context, resolved by distance into the target, with a null control,
a synthetic sensitivity anchor, and an `L_c` dose curve.

## Evidence

All verified against `runs/stage62_context_util_flagship.json`.

1. **Deep context use is strong and significant.** Deep-bucket utilization
   (target chars 33 to 64, spanning nearly the full 256-char window) is
   `U = +0.205289` bits/char, CI `[+0.198204, +0.212034]`. The CI lower bound
   is about four times the `0.05` E-uses-context line.
2. **The reading is trustworthy.** The double-random control collapses to
   `U = -0.004342` bits/char, CI `[-0.008509, -0.000098]` (no manufactured
   signal), and the synthetic deep-copy anchor gives `U = +0.857891` bits/char
   (the deep bucket registers strongly when deep context is informative, so a
   null would have indicted the model, not the probe).
3. **The model wants more context than it has.** The deep-bucket dose curve
   RISES with available context: `+0.136498` at `L_c=64`, `+0.179659` at
   `L_c=128`, `+0.205289` at `L_c=192`. Utilization is still climbing at the
   window edge, so a larger window would be used, not wasted.
4. **The decay shape is textbook.** Near-boundary buckets are strongly positive
   and decay monotonically (`+5.72`, `+1.36`, `+0.64`, `+0.37`, `+0.21`), the
   sharp-nearby, fuzzy-far-away signature of a model that uses context across
   its whole window.

## Decision

1. **H027 resolves E-uses-context.** The flagship uses its context strongly out
   to the far edge of its 256-character window. The topical drift is a
   context-WINDOW-SIZE limit, not a failure to understand: the model uses
   everything it can see (about 45 words) and drifts precisely when a longer
   generation runs past the window and the opening topic falls out of view.
2. **Phase 6's coherence intervention is the LONGER-CONTEXT char arm, not the
   subword substrate.** Per H027's pre-registered rule, E-uses-context selects
   the block-512 char arm. This is also the cheaper arm: it reuses the entire
   character-level eval, probe, and training stack unchanged, whereas the
   subword arm would have required a BPE-aware codec, eval, and probe. The
   probe averted building the wrong, more expensive thing.
3. **The substrate axis stays closed for coherence at this stage.** Stage 56
   (H022) closed substrate for the specialization GAP; Stage 62 now finds the
   coherence deficit is not a substrate problem either, at this scale. A BPE
   arm is not justified by this evidence.
4. **The block-512 training arm itself is user-gated compute.** Like every
   flagship-class run, a longer-context 200M training arm is a multi-hour
   commitment on the 8GB card and is the user's call to spend. The next spec
   (an H028-class hypothesis) will carry: a confirm-first block-512 sizing and
   throughput gate at 200M (attention is O(n^2), so activation memory and
   step time both rise; Stage 57's block-512 timing row is the starting
   estimate), a matched-or-honest compute budget versus the block-256 flagship,
   the context-utilization probe at both the fixed `(192, 64)` setting and a
   STRETCH setting only the larger window can run, deterministic text8 TEST,
   and a fresh sample grid for the user's coherence judgment.

## Scope and what this decision does not claim

- E-uses-context does not prove a block-512 model will read as more coherent to
  a human. A larger window raises the ceiling; the model must still learn to
  use the extra span, and the sample grid remains the ground truth for "reads
  better."
- The probe measures context USE on text8, a proxy for topical coherence, not a
  direct coherence judgment, and not reasoning in general.
- This says nothing about the copy-circuit question (H026) or the specialization
  gap (Stage 56), though all three concern long-range attention machinery.
- No claim about the absolute achievable coherence at laptop scale: GPT-2 117M
  reaches `1.17` bits/char and still drifts, so the honest target remains
  measurably less drift, not human-level coherence.

## What would reopen or reverse this decision

1. If the block-512 arm launches and its context-utilization STRETCH setting
   (context beyond 256) shows near-zero added use, the window was not the
   binding constraint after all, and the substrate arm reopens under a new
   hypothesis.
2. If the block-512 sizing gate prices even a reduced budget beyond what the
   user will spend, the coherence workstream pauses for a size-versus-budget
   fork (the ADR 0017 reopen pattern), not a silent downscope.
3. If a future probe variant on a subword model shows dramatically higher deep
   utilization per unit compute than the char block-512 arm, the substrate
   question reopens for coherence on cost grounds.

## Links

- `docs/hypotheses/027-context-utilization-and-coherence.md` (resolved by this ADR)
- `experiments/tiny_language_lab/eval_context_utilization.py` · `runs/stage62_context_util_flagship.json` · `.md`
- `RESULTS.md` Stage 62 (the evidence entry) · Stage 57 (block-512 timing) · Stage 56 / H022 (substrate closed for the GAP)
- `docs/decisions/0018-phase-6-redesign-circuit-mapping-and-instrumented-flagship.md` · `runs/stage61_user_samples.md`
- `research/theme_2_in_context_learning_and_rag/11_induction_heads_and_zero_shot_failure.md` (the long-range-attention prior art; Gemini to compare the probe against Sun and Iyyer 2021, Khandelwal et al. 2018, O'Connor and Andreas 2021)
