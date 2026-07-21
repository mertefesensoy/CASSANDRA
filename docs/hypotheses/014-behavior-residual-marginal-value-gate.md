# Hypothesis 014 · The prior-dominance law inverts on the behavior axis

- Status: RESOLVED. Measured by Codex as Stage 38: H014 confirmed. The frozen-prior floor copied at chance (`0.118421`, near `1 / 8 = 0.125`), while the rank-2 residual arms reached `0.320176` (`copyw`) and `0.307017` (`copymix`) mean copy accuracy, with positive per-seed gaps above `0.10`. Codex prepared ADR 0006 as a draft for Claude review.
- Date: 2026-06-24
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 43 (behavior-axis pivot; Codex stage number 38)
- Builds on: Stage 37 (the residual marginal-value gate closed on NLL,
  `experiments/tiny_language_lab/runs/stage37_residualgap_summary.md`), ADR 0005
  (`docs/decisions/0005-gradient-forms-the-residual-formation-side-closed.md`,
  which retires formation-side NLL mechanics and pre-registers the behavior axis
  as a pivot), the Stage 7 to 16 copy-probe and verifier machinery, and the
  project's foundational methodological finding that validation NLL and copy
  behavior diverge. Gemini prior-art: note 09
  (`research/theme_1_architecture_and_priors/09_peft_capacity_in_prior_dominated_regimes.md`).

## Why this, and why now

Stages 33 to 37 spent five stages establishing that the formation side is
bounded, and ADR 0004 and ADR 0005 retired the three cheap "form the residual
more cleverly" levers (data selection, richer analytic base, gradient-free
search). The capstone number is the decomposition: on the structured corpus the
frozen count prior carries about `83%` of the recipe's edge over `random_full`
and the trainable residual only about `17%`. Stage 37 then confirmed this is not
a one-corpus artifact: across natural-text orders 2 to 4 at 200 and 500 steps,
and across a rank 1/2/4 sweep, the floor-to-target gap never reached the `0.05`
NLL reopening line. The residual looks nearly worthless.

But every one of those stages measured **validation NLL**. The project's own
earliest behavior finding, from Stages 7 onward, is that NLL and task behavior
diverge: a model can drive next-character loss down while copy-probe accuracy
sits at chance. So the formation-side closure may be a property of the metric,
not of the recipe. ADR 0005 pre-registered exactly this pivot, naming the
behavior axis as the place "where the north-star goal of useful behavior actually
lives." The north star is to form useful behavior cheaply, not to lower NLL, so
this is the most on-thesis unexplored direction and it reuses machinery that
already exists.

## The mechanism that makes this falsifiable

A frozen count or n-gram prior is a local object. At the copy-task answer
position the conditioning context is the fixed marker string that ends each line,
for example `...; answer=`, and that context is identical across every case
regardless of the line's key. A prior keyed on local context can therefore only
emit the marginal distribution over keys at that position, which is chance copy
accuracy, about `1 / num_keys`.

Copying the correct key requires reading it from earlier in the same line, an
in-context retrieval or induction operation. Only the attention surface can do
that. In the frozen-prior recipe the only trainable attention surface is the
LoRA residual, the very surface Stage 37 found worthless on NLL. So copy
behavior, if it forms at all, must come from the residual. The floor-to-target
gap that was about zero on NLL should be large on copy accuracy.

## Hypothesis

On the long-context copy corpus, the frozen-count-prior recipe is
**residual-dominated on copy behavior**, inverting Stage 37's NLL result.
Concretely, with the copy probe enabled:

- the frozen-prior floor (`--residual-optim none`) copies at chance,
  `copy_acc ~= 1 / num_keys`; and
- the same recipe with an AdamW-trained rank-2 residual under copy training
  copies materially above chance.

The behavior gap, `copy_acc(AdamW target) - copy_acc(floor)`, is large and
stable in sign where the Stage 37 NLL gap was about zero. This would show the
formation-side closure is metric-specific: the residual is the behavior-forming
surface, and the prior-dominance law holds only for NLL.

This is the falsifiable claim. It is killed two ways, both informative:

- Premise wrong: the floor already copies well above chance. The prior is not
  copy-blind, the behavior axis is also prior-influenced, and the clean inversion
  is false. Mechanistically unlikely, so a positive here would itself be a
  notable finding about what the prior captures, probably a corpus artifact to
  harden against.
- Behavior not cheaply formable: both residual arms stay near the floor at
  chance. The cheap residual does not form copy behavior at this budget, so
  behavior needs heavier machinery, the Stage 11 to 16 verifier, correction, and
  retrieval samplers, or a larger surface. The closure then holds on behavior
  too, at least at this budget, and the roadmap escalates accordingly.

## The reference points

The matched comparison mirrors Stage 37, swapping the metric from NLL to copy
accuracy. Every count-prior arm uses the same frozen count base and the same
rank-2 residual parameterization; only how the residual is formed changes, plus a
full-model control.

- Arm A, floor: `count_prior_lora_r2_copyw_floor`, the frozen prior with the
  residual not trained, copy probe on. Expected copy_acc at chance. This is the
  bar every other arm must beat.
- Arm B, target (weighted): `count_prior_lora_r2_copyw`, AdamW rank-2 residual
  with copy-position loss weighting.
- Arm C, target (verified sampler): `count_prior_lora_r2_copymix`, AdamW rank-2
  residual with the verifier-guided mixed sampler that Stage 9 showed moves copy
  behavior. This gives the residual its fair best shot, so a chance-level result
  here is a genuine "not formable cheaply" signal rather than a too-weak training
  signal.
- Arm D, control: `random_full_copymix`, the full transformer under the same copy
  training, as a behavior ceiling and for context against the cheap recipe.

The decisive quantity is the behavior gap, `copy_acc(target) - copy_acc(floor)`,
for arms B and C against arm A.

## Primary decision metric and pass or fail line

Metric: copy-probe accuracy, mean over seeds `7 11 19` with per-seed spread.
Report copy-probe NLL and plain validation NLL alongside, for continuity with
Stage 37 and to confirm the divergence directly. Chance is `1 / num_keys`, about
`0.125` for the eight-key corpus; Codex confirms `num_keys` from
`make_long_context_corpus.py`.

- CONFIRM INVERSION: the floor sits near chance, within about `0.05` of
  `1 / num_keys`, and at least one residual arm reaches `copy_acc >= floor + 0.10`
  with a positive sign on all three seeds. The behavior gap is large where the
  NLL gap was about zero. Stage 37's prior-dominance is metric-specific, and the
  formation side reopens on the behavior axis.
- KILL, premise wrong: the floor copy_acc is already at least `chance + 0.10`.
  The prior is not copy-blind. Investigate what structure it is exploiting and
  harden the probe.
- KILL, not cheaply formable: both residual arms stay within `0.05` of the floor,
  near chance. The cheap residual does not form copy behavior at this budget.
  Escalate the behavior axis to heavier machinery or a larger surface.

The `0.10` copy-accuracy threshold is provisional and deliberately larger than
the per-seed copy-probe spread seen in Stages 8 to 12, so a confirm is a real
behavior signal, not sampling noise.

## Risks and confounds

- Copy-probe seed variance. Copy accuracy is a sampled metric. Use the three
  fixed seeds and require a stable sign, as in the earlier copy stages.
- Fair shot for the residual. Include the verified mixed-sampler arm C, not only
  the weighted arm B, so a chance-level outcome is a real kill, not an artifact
  of a weak training signal. If both B and C are near chance, the kill is clean.
- The floor must measure the true frozen prior. Arm A must use
  `--residual-optim none` with the zero-residual head, so its output is exactly
  the frozen count prior. The copy-training flags are no-ops there because
  nothing trains; they are passed only so the probe setup matches the target
  arms. The probe is eval-only, so there is no train-time leakage.
- Device and comparability. GPU is the project default. The GPU transition audit
  notes that large behavior-gap kills survive the device switch, and a copy-accuracy
  inversion is a large-signal result, but keep seeds fixed and the corpus and
  block size matched across arms.
- A too-easy or too-hard budget. Stages 8 to 16 formed measurable copy behavior
  in roughly 200 to 500 steps. Use 500 steps as primary so the residual has a
  fair chance to form behavior; a 200-step diagnostic is optional.

## What result would change the plan

- CONFIRM: write a re-scoping ADR. The formation-side closure becomes "closed on
  NLL, open on behavior," the residual is named the behavior-forming surface, and
  the roadmap re-centers on the copy and verifier axis. Possible follow-up behavior-axis stages become the
  behavior analogues of Stages 33 to 37: which sampler, rank, and data most
  cheaply form copy behavior on the frozen prior. This is the north-star
  direction.
- KILL, premise wrong: harden the copy probe and audit the corpus for a local cue
  the prior could exploit, then re-run.
- KILL, not formable: escalate to the Stage 11 to 16 verifier, correction, and
  retrieval samplers, or a rank increase, on the behavior axis, and re-ask whether
  the frozen-prior recipe is the right substrate for behavior at this budget.

## Original Handoff to Codex (implemented as Codex stage 38, README ladder rung 43)

Files to modify:

1. `experiments/tiny_language_lab/cassandra_compare.py`: register a
   copy-probe-enabled floor config `count_prior_lora_r2_copyw_floor`, a mirror of
   `count_prior_lora_r2_copyw` with `--residual-optim none` so the residual is
   frozen at zero, copy probe and `copy_probe_marker` left on. This is the same
   one-line mirror pattern used to add the Stage 37 `_floor` configs. No new
   primitive is needed; arms B, C, and D already exist.

Runs, on the long-context copy corpus from Stages 7 to 16
(`make_long_context_corpus.py` output), the block size and copy-probe protocol
from those stages, sampled evaluation, seeds `7 11 19`, 500 steps, GPU:

- Arm A, floor: `count_prior_lora_r2_copyw_floor`.
- Arm B, target (weighted): `count_prior_lora_r2_copyw`.
- Arm C, target (verified sampler): `count_prior_lora_r2_copymix`.
- Arm D, control: `random_full_copymix`.

Command shape (Codex fills the exact long-context corpus filename, block size,
and copy flags from the Stage 8 to 9 command shapes; ensure the copy probe is on
for every arm including the floor):

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\<long_context_corpus>.txt `
  --device cuda --steps 500 --eval-batches 16 --seeds 7 11 19 `
  --configs count_prior_lora_r2_copyw_floor count_prior_lora_r2_copyw count_prior_lora_r2_copymix random_full_copymix `
  --out .\experiments\tiny_language_lab\runs\stage38_behaviorgap.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage38_behaviorgap.md `
  --title "Stage 38 Behavior Residual Marginal-Value Gate"
```

Smoke-test with a short step count first. Record in `RESULTS.md` and the run
summary to the Codex evidence standard: per-arm copy-probe accuracy and NLL,
plain validation NLL, the behavior gap `copy_acc(target) - copy_acc(floor)` for
arms B and C, the measured chance level `1 / num_keys`, per-seed spread, and a
short interpretation against the confirm and kill lines above. The metric that
decides the stage is copy-probe accuracy, not validation NLL.

Pass/fail line, restated for the handoff: CONFIRM if the floor is near chance and
some residual arm reaches at least `chance + 0.10` copy accuracy stably across
seeds; KILL otherwise, distinguishing "floor already copies" from "no arm copies."

## Prior-art flag for Gemini

This is a measurement, not a new mechanism. The mechanism, that a fixed-order
n-gram or count prior cannot perform in-context copy because its conditioning
context at the answer position is invariant across cases, while attention forms
the copy or induction behavior, connects to the induction-head and in-context
learning literature (Olsson et al., 2022) and to the classical limitation of
Markov and n-gram models. Gemini should locate this against induction heads and
in-context learning, against any work measuring the divergence between perplexity
and task behavior under PEFT or adapters, and against the lab's own Stage 7 to 16
copy results. Frame any positive as a clean measurement, that the prior-dominance
law established on NLL inverts on a behavior metric in this specific cheap
frozen-prior plus small-residual recipe, not as a novel architecture or a novel
optimizer.

## Links

- Gate it follows: `experiments/tiny_language_lab/RESULTS.md` Stage 37;
  `experiments/tiny_language_lab/runs/stage37_residualgap_summary.md`.
- Decision that pre-registered this pivot:
  `docs/decisions/0005-gradient-forms-the-residual-formation-side-closed.md`.
- Behavior machinery: `experiments/tiny_language_lab/RESULTS.md` Stages 7 to 16;
  the `--copy-*` flag surface in
  `experiments/tiny_language_lab/cassandra_tiny_transformer.py`.
- Gemini prior-art:
  `research/theme_1_architecture_and_priors/09_peft_capacity_in_prior_dominated_regimes.md`;
  in-context learning and induction heads to be added by Gemini.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md`, behavior axis.
- Roadmap: `README.md` Next ladder, rung 43.
```
