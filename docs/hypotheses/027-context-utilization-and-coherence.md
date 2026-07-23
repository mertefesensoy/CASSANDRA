# Hypothesis 027 · The Stage 61 flagship's topical drift is a long-range-context-use failure that plain NLL cannot see, and a context-utilization probe decides whether the binding constraint is the window SIZE (fix: longer context) or the model not using the window it already has (fix: the substrate)

- Status: OPEN. Specced for Codex as Stage 62 (README ladder rung 73).
  Audited 2026-07-23 (hypothesis-auditor pass; all four required fixes
  applied: a deep-bucket sensitivity anchor so a null is interpretable, an
  exhaustive precedence-ordered partition, a pinned runnable handoff with
  concrete N/seed/derangement/paths, and a Gemini prior-art flag). The probe
  is eval-only and runs on existing checkpoints, so it is independent of and
  may run in parallel with the H026 circuit-mapping stage (Stage 60); neither
  gates the other.
- Date: 2026-07-23
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 73 (Codex stage number 62)
- Builds on: the Stage 61 pure-broad 200M flagship
  (`RESULTS.md` Stage 61 closeout, `runs/stage61_text8_test.json`
  `1.336059` bits/char, `runs/stage61_user_samples.md`), whose generations
  are locally fluent but topically drifting (the user's "off-topic, like
  ADHD" review, 2026-07-23); ADR 0018 (the pure-broad flagship and its
  instrumentation contract); H026
  (`docs/hypotheses/026-diverse-data-circuit-formation.md`), the sibling
  long-range-attention measurement; the copy-probe lineage (Stages 7 to 42,
  the lab's precedent for measuring a behavior that NLL hides); and Gemini
  notes 11 and 12 on induction heads and long-range circuits.

## Why this, and why now

The flagship's samples show the signature clearly: every roughly fifteen-word
span is grammatical and even has real texture ("the treaty of stuttgart", "the
terrorist attack in two zero zero one"), but the subject changes every clause,
with no thread across a paragraph. The user's diagnosis is exact: this is a
long-range-dependency failure.

The load-bearing observation is that this is INVISIBLE to the metric the lab
has optimized all along. The flagship's sampled validation is `1.288` bits/char
(`0.893` nats, `RESULTS.md` Stage 61 closeout, the sampled not the deterministic
figure) and its deterministic text8 TEST is `1.336059` bits/char, both good, and
it still drifts, because next-character cross-entropy is an almost purely local
objective: most of the predictable bits live in the next few characters. The lab has no instrument
for the thing the user cares about. This mirrors exactly the moment in Phase 1
when plain NLL could not see copy behavior and the copy-probe had to be built
(Stages 7 onward). We cannot optimize coherence before we can measure it.

There are three candidate root causes, and they imply DIFFERENT fixes, so
guessing is expensive:

- The window is only 256 characters (about 45 words); the model cannot attend
  to a topic it wrote 300 characters ago. If this is the binding constraint,
  the fix is a LONGER context window.
- The model does not actually use the context it can already see; character
  level spreads each concept across many timesteps, so the long-range
  attention circuits never formed (the same deficit H026 studies for copy).
  If this is the binding constraint, a longer window will not help, and the
  fix is the SUBSTRATE (a subword tokenization that packs more concepts per
  position and makes them atomic).
- The training data (text8: lowercase a-z-plus-space Wikipedia, every period
  and paragraph break stripped) never taught where a thought ends. This is a
  contributing cause but is not separately tested here; it is noted as a limit.

This hypothesis builds the instrument that DISTINGUISHES the first two, because
they point at different and differently-priced interventions.

## The instrument: a context-utilization probe

The probe isolates "uses long-range context" from "locally fluent" by asking a
single question: does the model predict a target segment better when its TRUE
preceding context is present than when a RANDOM unrelated context is present?

### Construction (deterministic, held-out)

On the text8 TEST split (held out, never trained), draw `N` disjoint passages,
each of length `L_c + L_t` characters, with a fixed seed. For passage `i`:

- TRUE condition: feed `[context_i ++ target_i]` where `context_i` is the
  passage's own first `L_c` characters and `target_i` is its next `L_t`
  characters.
- RANDOM condition: feed `[context_j ++ target_i]` where `context_j` is a
  DIFFERENT passage's first `L_c` characters (`j` a fixed derangement of `i`,
  deterministic), with the SAME `target_i`.

In both conditions, score cross-entropy (NLL) on the TARGET positions only,
under teacher forcing, in a single forward pass. Set `L_c + L_t <= B` (the
model's block size) so the whole sequence fits one window with no truncation
and any target position can in principle attend back over the entire context.

Per-passage utilization: `U_i = NLL_random_i - NLL_true_i` (nats; report also
in bits/char via division by `ln 2`). Positive means the true context helped;
`U ~= 0` means the model ignored which context it was given.

### The confound this design must defeat, and how

A model that only uses LOCAL boundary continuity (the first target char follows
naturally from the true context's last char, but not from a random context's
last char) would show positive `U` concentrated entirely in the first few
target characters, even though it carries no topic. That is not the capability
the user wants. So the deciding quantity is not the pooled mean but `U`
RESOLVED BY DISTANCE INTO THE TARGET: report `U` for target-position buckets
(chars 1 to 4, 5 to 8, 9 to 16, 17 to 32, 33 to `L_t`), a decay curve, not one
scalar. A model that only uses boundary continuity shows `U > 0` in the first
buckets and `U ~= 0` in the deep bucket. A model that genuinely carries context
shows `U > 0` persisting, and ideally decaying smoothly rather than dropping to
a cliff, into the deep bucket. The DEEP-BUCKET `U` (chars 33 to `L_t`) is the
coherence signal; the near-boundary buckets are reported to expose the confound
and as the built-in positive control (any fluent model must show them clearly
positive; if it does not, the probe is broken).

### The deep-bucket sensitivity anchor (required before a null is trusted)

A null deep-bucket `U` on text8 is ambiguous on its own: it could mean the model
ignores deep context, OR that text8's deep context simply carries little
predictable signal at chars 33 to 64 (encyclopedic prose is locally dense but
its topic may not sharply constrain a specific character thirty positions
later). Because a null `U_deep` is what launches the EXPENSIVE substrate arm,
the probe must first prove the deep bucket CAN register signal when signal
exists. Two anchors, both run in the smoke:

- **Synthetic upper anchor.** A constructed passage set over the text8 alphabet
  where a rare marker string placed in the context block MUST be copied into the
  deep target (an induction-style forced dependency). A model with any working
  long-range attention shows a large positive `U_deep` here. If even this anchor
  gives `U_deep ~= 0`, the probe or the model's deep attention is broken, and no
  text8 null is interpretable.
- **Dose curve.** Report text8 `U_deep` as a function of `L_c` (for example 64,
  128, 192). A `U_deep` that RISES with more true context is internal evidence
  of genuine long-range use even before the synthetic anchor.

A text8 `U_deep` near zero is read as E-local ONLY when the synthetic anchor is
clearly positive on the same model, proving the deep bucket is sensitive.

### Prior art (flagged for Gemini, not claimed novel)

The true-context-versus-shuffled-context NLL-delta, resolved by distance into
the target, is an established long-range-context evaluation pattern, not a lab
invention. Gemini should compare this probe against, at minimum: Sun and Iyyer
2021 "Do Long-Range Language Models Actually Use Long-Range Context?", Khandelwal
et al. 2018 "Sharp Nearby, Fuzzy Far Away: How Neural Language Models Use
Context", and O'Connor and Andreas 2021 "What Context Features Can Transformer
Language Models Use?", and warn if this design rediscovers a known method under
a new name.

### Fixed and stretch settings

Run one FIXED setting comparable across every model regardless of block size:
`L_c = 192`, `L_t = 64` (fits `B = 256`). Report the deep bucket (chars 33 to
64) as primary. For any longer-context arm (`B = 512` or more), ALSO run a
STRETCH setting (`L_c` up to `B - L_t`) that only the larger window can
execute, to measure whether the extra window is used. The fixed setting is the
apples-to-apples ruler; the stretch setting is the window-size question.

### Reuse, do not reinvent

New file `experiments/tiny_language_lab/eval_context_utilization.py`, reusing
`flagship_eval_lib.load_model` and `encode_fast`, a target-only NLL scorer
patterned on `chunked_nll` (`flagship_eval_lib.py:156`), and the deterministic
held-out-passage discipline of `copy_answer_probe`
(`cassandra_tiny_transformer.py:3067`). Output: a matrix JSON plus a Markdown
summary, one row per model per setting, carrying `N`, the seed, the checkpoint
SHA-256, mean `U` per bucket with bootstrap confidence intervals, and the raw
NLL_true / NLL_random means.

### Runnable handoff

- `N = 4096` passages at the fixed setting (raise to `8192` if E-gray fires,
  per the decision rule); passage seed `20260723`; the RANDOM condition uses
  a single fixed derangement of `0..N-1` seeded from the same value (Sattolo's
  algorithm, so every passage gets a context that is not its own).
- Fixed setting `L_c = 192`, `L_t = 64`; bucket edges (target chars)
  `[1-4, 5-8, 9-16, 17-32, 33-64]`; deep bucket is `33-64`.
- Primary model, the flagship (`B = 256`):
  `C:\cassandra_runs\stage61_pure_broad_200m_checkpoints\stage61_pure_broad_200m_seed7_random_full_seed7.pt`
  (SHA-256 `4e5c0c0540b7b019f7fb6a53636a8963cffae145e6182e2e41aa463b2f8bacd5`,
  the probe re-records it per row).
- Size anchor, the Stage 58 COLD 85M seed-7 final (`B = 256`), confirmed
  retained on disk 2026-07-23:
  `C:\cassandra_runs\stage58_dev_cold_checkpoints\stage58_dev_cold_85m_b42000_seed7_random_full_seed7.pt`.
- Example invocation:

```powershell
python .\experiments\tiny_language_lab\eval_context_utilization.py `
  --checkpoint C:\cassandra_runs\stage61_pure_broad_200m_checkpoints\stage61_pure_broad_200m_seed7_random_full_seed7.pt `
  --split test --n 4096 --context-len 192 --target-len 64 `
  --bucket-edges 4 8 16 32 --seed 20260723 --device cuda `
  --synthetic-anchor `
  --out .\experiments\tiny_language_lab\runs\stage62_context_util_flagship.json `
  --summary .\experiments\tiny_language_lab\runs\stage62_context_util_flagship.md
```

The synthetic sensitivity anchor and the `L_c` dose curve are produced by the
same script under `--synthetic-anchor` and `--context-len-sweep 64 128 192`.

## Pre-registered classification and decision lines

Run the probe on the Stage 61 flagship (`B = 256`) and, as a size anchor, the
Stage 58 COLD 85M seed-7 final (`B = 256`), at the fixed `(192, 64)` setting.
Let `U_deep` be the mean deep-bucket (chars 33 to 64) utilization in bits/char
with its bootstrap 95 percent confidence interval (CI) over the `N` passages,
and let `U_near` be the near-boundary buckets (chars 1 to 8). The `0.05` and
`0.02` bits/char thresholds echo the lab's recurring practical line; they are
PROVISIONAL until the smoke measures the probe's own between-passage spread and
confirms they sit clear of noise (H025:137 precedent for a smoke-confirmed
provisional bound), AND until the synthetic sensitivity anchor confirms the deep
bucket can be positive on this model at all (without which a text8 null is
uninterpretable).

The four classes are evaluated in the following precedence order, with E-gray as
the explicit catch-all, so every outcome lands in exactly one:

1. **E-ignores-context (the strong-substrate case, checked first)** = BOTH
   `U_near` and `U_deep` have their whole CI within `0.02` of zero: NLL_true and
   NLL_random are indistinguishable everywhere. The model ignores context
   entirely (and, being fluent, does so despite local continuity being
   available). Longer context is DEFINITIVELY not the lever; the SUBSTRATE arm
   launches with an added note that the training signal itself (local NLL) may
   need changing.
2. **E-uses-context (window is the binding constraint)** = `U_deep`'s CI LOWER
   bound is at or above `0.05` bits/char. The flagship carries context tens of
   characters deep; it drifts because the window runs out, not because it
   ignores the window. Workstream 3 launches the LONGER-CONTEXT char arm
   (`B = 512`), which reuses the entire char eval and probe stack unchanged and
   is the cheapest arm to execute; Stage 57 has a block-512 timing row to size
   it.
3. **E-local (the substrate is the binding constraint)** = `U_deep`'s CI upper
   bound is at or below `0.02` bits/char (deep use within noise of zero) WHILE
   `U_near`'s CI lower bound is above `0.02` (the model is fluent and uses local
   continuity) AND the synthetic anchor is clearly positive on this model. More
   window will not help a model that does not use the window it has; Workstream 3
   launches the SUBSTRATE arm (a subword-tokenized flagship, roughly 4x
   effective context per unit attention), which needs a BPE-aware codec, eval,
   and probe variant, a larger engineering lift because `flagship_eval_lib` is
   char-only. This reopens the substrate question for COHERENCE specifically,
   which Stage 56 (H022) never tested; H022 closed substrate only for the
   specialization GAP.
4. **E-gray (inconclusive, the catch-all `else`)** = anything the first three
   did not claim: a `U_deep` whose CI straddles `0.02` or `0.05`, an ambiguous
   `U_near`, or a synthetic anchor that fails to register (which would indict the
   probe, not the model). Raise `N` to `8192` and rerun; if it stays gray, or
   the synthetic anchor stays null, neither training arm is justified on this
   evidence and the workstream returns to intake with the probe as its record.

Verdict is read on the flagship; the Stage 58 COLD 85M anchor tells us whether
`U_deep` is a SIZE effect (differs between 85M and 200M) or a WINDOW effect
(similar at both sizes, same `B`). A window effect strengthens E-uses-context;
a near-zero `U_deep` at both sizes strengthens E-local.

## Expected signal, baseline, and honesty constraints

- Baseline: the raw NLL_random and NLL_true means are reported so `U` is
  auditable, not just the difference. The near-boundary bucket is the built-in
  positive control (it should be clearly positive for any fluent model; if it
  is not, the probe is broken).
- Expected if the drift is a pure window-size effect: `U_deep` clearly positive
  at the fixed setting AND the stretch setting on a longer-context arm shows
  further gains. Expected if it is a substrate/circuit effect: `U_deep` near
  zero despite a strongly positive near-boundary bucket.
- The probe measures context USE on text8, a proxy for topical coherence, not a
  human coherence judgment. A model can score well here and still read as dull
  or repetitive; the sample sheet and the user's review remain the ground truth
  for "reads coherently." Workstream 2 (nucleus sampling plus repetition
  penalty added to `sample_text`, then a regenerated sample grid) runs in
  parallel to separate genuine drift from the temperature-0.8, no-top-k
  sampling artifact before any training compute is spent.
- `U` conflates syntactic agreement, entity tracking, and topic; this stage
  does not decompose them.
- Determinism is load-bearing: fixed seed, fixed derangement, checkpoint and
  probe SHA-256 per row, and a re-probe of one anchor to confirm exact
  reproduction (H026 precedent).

## Risks

- **Boundary confound** (a fluent local model faking context use): defeated by
  the distance-resolved deep bucket, which is the whole reason the probe
  reports buckets rather than a pooled mean.
- **The ceiling is real.** A 200M char model on an 8GB laptop will not write
  coherent essays; GPT-2 117M reaches `1.17` bits/char with BPE and far more
  training and still drifts. The honest target is measurably higher `U_deep`
  and less drift, not human-level coherence. If both Workstream 3 arms leave
  `U_deep` near zero, the honest conclusion is that laptop-scale char (and even
  subword) coherence is bounded, recorded as a negative, not spun.
- **BPE arm cost.** The substrate arm needs a BPE-native probe (tokenize
  context and target), so its engineering cost is higher than the
  longer-context arm's; this is stated so the E-local branch is not mistaken
  for a cheap flag flip.
- **Proxy, not judgment.** Even E-uses-context does not guarantee the samples
  will read better to the user; the sample grid is the check.

## What result would change the plan

- **E-uses-context** sends Workstream 3 to the longer-context char arm and
  makes "the window was the wall" a measured claim for the release narrative.
- **E-local or E-ignores-context** sends Workstream 3 to the substrate arm and
  reopens the char-versus-subword question for coherence (distinct from Stage
  56's GAP closure), with the BPE-eval-stack cost acknowledged up front.
- Either training arm, when it lands, is measured on this probe, on
  deterministic text8 TEST, and on a fresh sample grid, and read back against
  the lines above; a new ADR (0019 class) records the intervention choice once
  the probe reads out.
- The probe's connection to H026 is explicit: if H026 finds diverse data forms
  a copy circuit while this probe finds the flagship ignores deep context, the
  two together localize the deficit to long-range attention machinery that broad
  data builds only weakly, which is a unified mechanism claim worth a note.

## Links

- `RESULTS.md` Stage 61 closeout · `runs/stage61_user_samples.md` · `runs/stage61_text8_test.json`
- `docs/decisions/0018-phase-6-redesign-circuit-mapping-and-instrumented-flagship.md`
- `docs/hypotheses/026-diverse-data-circuit-formation.md` (the sibling long-range-attention probe)
- `experiments/tiny_language_lab/flagship_eval_lib.py` (`load_model`, `encode_fast`, `chunked_nll`, `sample_text`) · `eval_text8.py` · `cassandra_tiny_transformer.py` (`copy_answer_probe`)
- `experiments/tiny_language_lab/make_bpe_corpus.py` (the substrate-arm starting point) · `RESULTS.md` Stage 57 (the block-512 timing row) · Stage 56 / H022 (the GAP-only substrate closure)
- `research/theme_2_in_context_learning_and_rag/11_induction_heads_and_zero_shot_failure.md` · `12_forgetting_of_structural_circuits.md`
