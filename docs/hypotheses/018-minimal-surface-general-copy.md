# Hypothesis 018 · The minimal trainable surface on the frozen prior that forms a general copy circuit

- Status: TESTED by Codex Stage 43. KILL, capacity wall under the registered line:
  every frozen-prior arm stayed within about `0.05` of chance while the full no-prior
  model cleared chance on all three seeds.
- Date: 2026-06-24
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 51 (Codex stage number 43)
- Builds on: ADR 0008 (the surface and rank question reopened on the general task,
  `docs/decisions/0008-general-copy-needs-more-than-current-rank2-residual.md`),
  Stage 42 (cheap rank-2 at chance, full model `0.226087`,
  `experiments/tiny_language_lab/runs/stage42_random_payload_copy.md`), Stage 39 (rank
  closed only on the memorizable task,
  `experiments/tiny_language_lab/runs/stage39_behavior_rank.md`). Gemini prior-art:
  note 11 (`research/theme_1_architecture_and_priors/11_lora_rank_saturation_and_intrinsic_dimension.md`,
  rank saturation and the alpha-over-rank coupling), note 10
  (`research/theme_1_architecture_and_priors/10_nll_divergence_and_behavior_forming_residuals.md`),
  and note 12
  (`research/theme_1_architecture_and_priors/12_induction_circuit_intrinsic_dimension_and_capacity_walls.md`,
  written for Stage 42: the induction-head intrinsic-dimension capacity wall and the
  alpha-over-rank control this hypothesis uses).

## Why this, and why now

Stage 42 gave the first clean separation on the behavior axis: on the
memorization-proof random-payload probe the cheap rank-2 residual sits at chance
(`0.063768`, `+0.001268` over chance), while the full model forms a real, if weak,
general copy circuit (`0.226087`, `+0.163587` over chance, on every seed). ADR 0008
narrowed ADR 0006 to seen-content behavior and reopened the capacity and surface
question on the general task, because Stage 39's rank closure was measured on the
seen-key memorizable task, where rank-2 already met the memorization ceiling. The
intrinsic dimension of a general induction circuit, query-key matching plus an
output-value copy, plausibly exceeds rank 2.

The on-thesis question is now sharp and bounded. A general copy circuit is formable
at this budget: the full model proves it at about 112k trainable parameters. The
cheap rank-2 residual fails at about 7k. The gap is roughly 16 times the trainable
surface. What is the smallest trainable surface on the frozen prior that closes it?
If a still-small surface (a higher-rank LoRA) forms the circuit, the cheap-recipe
thesis survives for general behavior. If nothing short of the full body forms it,
that is an honest capacity wall for this family. Either answer advances the north
star.

## The competing explanations

A single surface ladder on the Stage 42 corpus distinguishes three explanations for
why the cheap recipe failed:

- E1, capacity in rank. Rank 2 is too low to express the query-key match and the
  output-value copy on top of the frozen-random transformer body. Prediction: copy
  accuracy rises with LoRA rank toward the full-model ceiling.
- E2, surface structure. Low-rank LoRA on a frozen-random body cannot host an
  induction head at any practical rank; the attention body itself must be trainable.
  Prediction: every LoRA rank stays at chance, while training the full body on the
  frozen base clears chance.
- E3, base interference. The frozen count-bigram base biases toward local statistics
  and impedes the long-range circuit. Prediction: even the full body on the frozen
  base fails, while the full model with no frozen base clears.

## Hypothesis

On the Stage 42 random-payload corpus, a trainable surface larger than rank-2 LoRA
but smaller than the full model forms a general copy circuit: copy accuracy rises
above the Stage 42 chance plateau toward the full-model ceiling as the trainable
surface grows. Concretely, with the rank sweep run at alpha matched to rank so the
test is capacity and not update magnitude, at least one of the higher-rank LoRA arms
clears chance materially and exceeds the rank-2 baseline.

If true, the cheap-recipe thesis survives for general copy: a small trainable surface
on a frozen prior forms a content-agnostic copy circuit, and ADR 0006's behavior
claim recovers in a capacity-aware form.

This is the falsifiable claim, killed cleanly in the informative direction:

- KILL, capacity wall: every frozen-prior arm, all LoRA ranks and the full body on
  the frozen base, stays within about `0.05` of chance while the full model with no
  base clears it. General copy at this budget needs full-model capacity, and the
  frozen-prior-plus-small-residual family does not reach it for this behavior. ADR
  0008 hardens into a capacity-wall conclusion.

## The reference points

The same corpus, protocol, seeds, and budget as Stage 42, so only the trainable
surface changes. Chance is `1 / 16 = 0.0625`; the full model is the ceiling at about
`0.226087`.

- Arm A, baseline: `count_prior_lora_r2_copyw`. The Stage 42 failing arm, at chance.
- Arm B, rank up: `count_prior_lora_r8_copyw`.
- Arm C, rank up more: `count_prior_lora_r16_copyw`. Arms A, B, C give the capacity
  trend, run at alpha matched to rank.
- Arm D, full body on the frozen base: `count_prior_all_copyw`. Trains the whole
  transformer body but keeps the frozen count base added in the forward pass. This
  separates a LoRA-surface limit (E2) from a frozen-base limit (E3).
- Arm E, ceiling: `random_full_copymix`. The full model with no frozen base, Stage
  42's `0.226087`.

Arms B and C test the cheap-recipe thesis. Arms D and E are diagnostics: D against E
isolates whether the frozen base, rather than the trainable surface, is the blocker.

## Primary decision metric and pass or fail line

Metric: copy-probe accuracy on the random-payload probe, mean over seeds `7 11 19`
with per-seed spread, against chance `1 / 16` and against the rank-2 baseline, with
the full model as the ceiling. Report validation NLL for the standing dual-axis rule
and the rank trend explicitly.

- CONFIRM: at least one of Arm B or Arm C clears chance by at least `0.10` on all
  three seeds and materially exceeds the rank-2 baseline, ideally approaching the
  full-model ceiling. Report the cheapest surface that clears it.
- KILL, capacity wall: Arms B, C, and D all stay within about `0.05` of chance while
  Arm E clears chance on all three seeds. No surface short of the full no-prior model
  forms the circuit.
- GRADED: a higher-rank arm lifts above chance but below `+0.10`, or below the
  ceiling. Report the trend and the cheapest surface that helps; a longer-budget
  rerun is the natural follow-up.

Two secondary reads, for mechanism: whether copy accuracy rises monotonically with
rank across A, B, C (capacity trend), and whether Arm D matches Arm E or fails (base
interference).

## Risks and confounds

- Alpha-over-rank coupling. Standard LoRA scales the update by alpha over rank, so
  sweeping rank at fixed alpha shrinks the effective update at higher rank and can
  mask a real capacity gain. This is the coupling Gemini notes 11 and 12 flag, and it
  is concrete here: the existing `count_prior_lora_rN_copyw` branch in
  `cassandra_compare.py` hardcodes `lora_alpha = 2.0` for every rank, so the Stage 39
  rank-4 copy arm itself ran at alpha 2.0, not 4, and its rank-4 point was already
  alpha-mismatched. Run this sweep with alpha matched to rank (`lora_alpha =
  lora_rank`, holding alpha over rank at 1) so the comparison tests capacity, not
  update magnitude. The ladder therefore jumps 2 to 8 to 16 with matched alpha for
  clean geometric coverage; rank 4 is skipped because Stage 39 already probed it (and
  did so alpha-mismatched). State the alpha convention in the run.
- Noisy ceiling. The full-model ceiling is high variance (Stage 42 seed 19 only
  `+0.059239` over chance). Keep the three fixed seeds and report per-seed values. If
  the ceiling and any rank lift are borderline at 500 steps, a 1000-step rerun is the
  follow-up; 500 steps stays primary for Stage 42 comparability.
- Diagnostic versus candidate. Arm D, the full body on the frozen base, is about 112k
  trainable parameters; it is a diagnostic for the base-interference question, not a
  cheap-recipe candidate. The cheap-recipe candidates are Arms B and C only. Keep that
  distinction in the interpretation: a CONFIRM requires a small surface (B or C), not D.
- Determinism. Seeds `7 11 19`, the same Stage 42 corpus and split, fixed generator
  seed.

## What result would change the plan

- CONFIRM: write a strengthening ADR. The cheap-recipe thesis holds for general copy
  at a named rank, and the behavior branch turns to how robustly and how cheaply that
  circuit forms (sampler and verifier signal, longer contexts, larger alphabets).
  Flag the capacity-aware recovery for a fresh Gemini prior-art pass.
- KILL, capacity wall: write the ADR that records general copy as a capacity wall for
  the frozen-prior-plus-small-residual family at this budget, and decide whether the
  behavior branch pivots to a different cheap mechanism (a trainable attention prior, a
  retrieval interface measured on this probe) or pauses in favor of the prior or NLL
  axis. Use Arm D against Arm E to say whether the frozen base was the blocker.
- GRADED: pick the cheapest helping surface and run the longer-budget follow-up.

## Codex Stage 43 result

Codex implemented and ran the registered surface ladder on 2026-06-24:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\random_payload_copy_seed.txt --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.25 --seeds 7 11 19 --configs count_prior_lora_r2_copyw count_prior_lora_r8_copyw count_prior_lora_r16_copyw count_prior_all_copyw random_full_copymix --out .\experiments\tiny_language_lab\runs\stage43_general_copy_surface.jsonl --summary .\experiments\tiny_language_lab\runs\stage43_general_copy_surface.md --title "Stage 43 General-Copy Surface Ladder"
```

Result against the registered line: KILL, capacity wall for the frozen-prior family under this protocol. Chance is `0.062500`. Rank 8 reached `0.049275` mean copy accuracy and rank 16 reached `0.049276`, both below the rank-2 baseline at `0.063768`; no higher-rank LoRA arm cleared chance by `+0.10` on any seed. The full-body-on-frozen-base diagnostic stayed at `0.043478` on all seeds. The no-prior full model cleared chance on every seed and reached `0.226087` mean copy accuracy.

The alpha confound was controlled as registered: rank 8 used `lora_alpha = 8.0`, rank 16 used `lora_alpha = 16.0`, and the rank-2 baseline kept `lora_alpha = 2.0`. Trainable counts were `6,956` for rank 2, `19,244` for rank 8, `35,628` for rank 16, and `111,916` for both full-body diagnostics.

Mechanistic read: the capacity trend across ranks is absent, and Arm D failing while Arm E succeeds points to frozen count-base interference under this protocol rather than LoRA rank alone. Validation NLL still moved separately from behavior: rank 16 had the best frozen-prior NLL (`1.642052`) while remaining at chance on copy accuracy.
## Handoff to Codex (implemented as Codex stage 43, README ladder rung 51)

Files to modify:

1. `experiments/tiny_language_lab/cassandra_compare.py`: register three new configs.
   `count_prior_lora_r8_copyw` and `count_prior_lora_r16_copyw` are rank variants of
   `count_prior_lora_r2_copyw`, but they must set `lora_alpha = float(rank)` (rank 8
   gets alpha 8, rank 16 gets alpha 16). This is a deliberate change, not a mirror: the
   existing `count_prior_lora_rN_copyw` branch hardcodes `lora_alpha = 2.0` for every
   rank, which is exactly the alpha-over-rank confound this stage controls for. Arm A
   `count_prior_lora_r2_copyw` keeps `lora_alpha = 2.0`, which already equals its rank.
   `count_prior_all_copyw` is a genuinely new config branch, `train_scope = all` on the
   frozen count base plus the copy flags, not a one-line mirror. Add all three new names
   to the `--configs` choices list. No new model primitive is needed.

Run, on the Stage 42 corpus and protocol so the comparison is clean:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\random_payload_copy_seed.txt `
  --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 `
  --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 `
  --copy-sample-fraction 0.25 --seeds 7 11 19 `
  --configs count_prior_lora_r2_copyw count_prior_lora_r8_copyw count_prior_lora_r16_copyw count_prior_all_copyw random_full_copymix `
  --out .\experiments\tiny_language_lab\runs\stage43_general_copy_surface.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage43_general_copy_surface.md `
  --title "Stage 43 General-Copy Surface Ladder"
```

Smoke-test with a short step count first, and confirm the rank-2 arm reproduces the
Stage 42 chance result and the full model reproduces about `0.226`. Record in
`RESULTS.md` and the run summary to the Codex evidence standard: per-arm copy
accuracy and the rank trend, trainable-parameter count per arm, the alpha convention
used, validation NLL for dual-axis tracking, per-seed spread, Arm D against Arm E for
base interference, and a short interpretation against the confirm, kill, and graded
lines. The metric that decides the stage is copy accuracy on the random-payload probe.

Pass or fail line, restated: CONFIRM if a higher-rank LoRA arm (B or C) clears chance
by at least `0.10` on all three seeds and beats the rank-2 baseline; KILL, capacity
wall if every frozen-prior arm including the full body on the frozen base stays within
about `0.05` of chance while the full no-prior model clears it; GRADED otherwise.

## Prior-art flag for Gemini

This is a measurement on a standard question, not a new mechanism. The specific claim,
how much LoRA rank or trainable surface a frozen-prior small model needs before it
forms (not merely steers) a general in-context copy or induction circuit, connects to
the intrinsic-dimension and rank-saturation literature (Gemini note 11), the
alpha-over-rank and rank-stabilized LoRA work, and the induction-head formation
literature (Olsson et al., 2022). Gemini should locate whether PEFT or LoRA is known
to form, versus only unlock, an induction circuit, and at what rank, before any
external wording. Frame any positive as a clean capacity measurement in the cheap
frozen-prior plus small-residual recipe, not a new architecture.

## Links

- Decision that reopened this: `docs/decisions/0008-general-copy-needs-more-than-current-rank2-residual.md`.
- Clean reversal it follows: `experiments/tiny_language_lab/RESULTS.md` Stage 42;
  `experiments/tiny_language_lab/runs/stage42_random_payload_copy.md`.
- Rank closure it reopens on the general task: `experiments/tiny_language_lab/RESULTS.md`
  Stage 39; `experiments/tiny_language_lab/runs/stage39_behavior_rank.md`.
- Behavior machinery, the `--train-scope`, `--lora-rank`, `--lora-alpha`, and `--copy-*`
  flag surface: `experiments/tiny_language_lab/cassandra_tiny_transformer.py`;
  `experiments/tiny_language_lab/cassandra_compare.py`.
- Gemini prior-art:
  `research/theme_1_architecture_and_priors/11_lora_rank_saturation_and_intrinsic_dimension.md`;
  `research/theme_1_architecture_and_priors/10_nll_divergence_and_behavior_forming_residuals.md`;
  `research/theme_1_architecture_and_priors/12_induction_circuit_intrinsic_dimension_and_capacity_walls.md`.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md`, behavior axis.
- Roadmap: `README.md` Next ladder, rung 51.
