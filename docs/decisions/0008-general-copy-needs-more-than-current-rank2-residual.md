# ADR 0008 · General in-context copy needs more than the current rank-2 cheap residual

- Status: Accepted
- Date: 2026-06-24
- Author: Claude (hypothesis, ADR, and roadmap role)
- Accepts with revisions: the Codex evidence draft
  `docs/decisions/0008-general-copy-needs-more-than-current-rank2-residual.codex-draft.md`,
  which is superseded by this record
- Resolves: Hypothesis 017 (Codex Stage 42), and fires the reversal clause of ADR 0006
- Scopes and partially reverses: ADR 0006
  (`docs/decisions/0006-behavior-axis-reopens-residual-formation.md`)
- Builds on: ADR 0007
  (`docs/decisions/0007-heldout-token-copy-probe-cannot-measure-generalization.md`),
  Stages 38 to 42, Gemini note 10
  (`research/theme_1_architecture_and_priors/10_nll_divergence_and_behavior_forming_residuals.md`,
  NLL versus behavior divergence) and Gemini note 12
  (`research/theme_1_architecture_and_priors/12_induction_circuit_intrinsic_dimension_and_capacity_walls.md`,
  the induction-circuit intrinsic-dimension and capacity-wall reading of Stage 42)

## Context

ADR 0006 reopened the behavior axis. In its strong reading it named the rank-2
residual the behavior-forming surface, after Stage 38 showed it forms seen-key
identity-copy behavior above a copy-blind frozen count-prior floor, even when
validation NLL barely moved. ADR 0006 carried an explicit reversal clause: reverse
if cheap surfaces plateau near the floor on a harder or generalizing probe while
only full-model capacity forms the behavior. Stages 40 and 41 could not test that
clause, because held-out answer tokens were unemittable by construction, and ADR
0007 retired that probe. Hypothesis 017 built the valid test: a memorization-proof
random-payload copy probe where the copied value is drawn uniformly per line from a
fully-seen alphabet, so there is no fixed mapping or line-index regularity to
memorize and every token is emittable. Above-chance accuracy on it can only come
from a general in-context copy circuit.

## Evidence (Stage 42)

Payload alphabet `abcdefghijklmnop`, `V = 16`, chance `1 / 16 = 0.0625`, 500 steps,
block size 96, sampled evaluation, seeds `7 11 19`, 115 validation cases per seed.

| Arm | Trainable params | Mean val NLL | Mean copy acc | Gap vs chance | Per-seed chance gaps |
| --- | ---: | ---: | ---: | ---: | --- |
| `count_prior_lora_r2_copyw_floor` | 0 | 1.760430 | 0.043478 | -0.019022 | -0.019022 each |
| `count_prior_lora_r2_copyw` | 6956 | 1.698173 | 0.063768 | +0.001268 | +0.015761, -0.010326, -0.001630 |
| `random_full_copymix` | 111916 | 0.587600 | 0.226087 | +0.163587 | +0.294022, +0.137500, +0.059239 |

Two facts decide it. The cheap rank-2 residual stays within `0.05` of both the floor
and chance on every seed, missing the `+0.10` confirm line by a wide margin: it is at
chance. The full model clears chance on every seed and by `+0.163587` on the mean, so
the learnability gate is open. This is a clean local reversal, not the Stage 40 and 41
situation where the full model also collapsed.

The result is also a dual-axis split in the opposite direction from Stage 38. The
cheap residual improved validation NLL over the floor by `0.062257`, yet that NLL gain
did not become any general copy accuracy. Stage 38 was behavior without NLL; Stage 42
is NLL without behavior. The dual-axis rule cuts both ways. Gemini note 12 places
this mirror against the grokking and emergent-abilities literature, where a
continuous loss metric and a discrete behavior metric are decoupled.

## Decision

1. Fire ADR 0006's reversal clause, narrowly. Stage 42 meets it exactly: the cheap
   surface is at chance while the full model clears, on a valid generalizing probe.
2. Narrow the ADR 0006 claim. The rank-2 residual forms seen-content copy behavior
   (Stage 38, seen-key identity copy) but does not form a general, content-agnostic
   in-context copy circuit. Do not upgrade ADR 0006 to general copy. The strong
   reading, that the rank-2 residual is the behavior-forming surface in general, is
   reversed to: it forms seen-content behavior only.
3. This is not a needs-more-than-this-hardware conclusion. The full model forms a
   real, if weak (`0.226087`), general copy circuit at the same 500-step laptop
   budget. General copy is formable here; the current cheap rank-2-on-frozen-base
   recipe is what does not form it.
4. Reopen the capacity and trainable-surface question, but only on the general task.
   Stage 39 closed the rank lever on the seen-key memorizable task, where rank-2
   already met the memorization ceiling. The intrinsic dimension of a general
   induction circuit, query-key matching plus an output-value copy, plausibly exceeds
   rank 2, so the rank and surface levers reopen specifically on the memorization-proof
   probe. Gemini note 12 grounds this in the induction-head intrinsic-dimension view: a
   query-key match plus output-value copy circuit has a minimum representational rank,
   and a LoRA update below it hits a hard capacity wall regardless of optimizer or
   budget. This does not reopen the NLL-side rank closure of Stage 37 and ADR 0005,
   which is a different metric.
5. Keep the dual-axis tracking rule, now shown to cut both ways. A behavior-branch
   method is judged on the behavior probe, and an NLL gain is not credited as behavior.
6. Leave the NLL-side closures (ADR 0004, ADR 0005) and the held-out-token probe
   retirement (ADR 0007) untouched. This ADR adds no NLL claim.

## Scope and what this decision does not claim

- It does not claim that no low-hardware method can form general copy. It claims the
  frozen count prior plus rank-2 LoRA residual plus the Stage 38 weighted signal does
  not, at 500 steps on the `V = 16` random-payload probe.
- The full-model ceiling is real but noisy: seed `19` clears chance by only
  `+0.059239`. Read Stage 42 as a local reversal of the cheap recipe, not as a solved
  high-entropy copy benchmark.
- The probe is a tiny character-level copy task with 115 validation cases per seed.

## What would reopen or reverse this decision

- Reverses back toward ADR 0006's stronger reading if a still-small trainable surface,
  a higher-rank LoRA, a small trainable attention surface, or another small residual on
  the frozen prior, forms a general copy circuit that clears chance by at least `0.10`
  on all seeds and approaches the full-model ceiling. Hypothesis 018 specifies the first
  test, a surface and rank ladder on the memorization-proof probe.
- Hardens into a capacity-wall conclusion for this family if no trainable surface short
  of the full body forms the circuit, in which case general copy at this budget requires
  full-model capacity and the cheap frozen-prior-plus-small-residual recipe does not
  reach it for this behavior.

## Links

- Scoped and partially reversed decision:
  `docs/decisions/0006-behavior-axis-reopens-residual-formation.md`.
- Probe-design ADR it follows:
  `docs/decisions/0007-heldout-token-copy-probe-cannot-measure-generalization.md`.
- Resolved hypothesis: `docs/hypotheses/017-memorization-proof-copy-probe.md`.
- Next test: `docs/hypotheses/018-minimal-surface-general-copy.md`.
- Codex evidence draft superseded by this record:
  `docs/decisions/0008-general-copy-needs-more-than-current-rank2-residual.codex-draft.md`.
- Stage 42: `experiments/tiny_language_lab/RESULTS.md` Stage 42;
  `experiments/tiny_language_lab/runs/stage42_random_payload_copy.md` and `.jsonl`.
- Gemini prior-art:
  `research/theme_1_architecture_and_priors/10_nll_divergence_and_behavior_forming_residuals.md`
  (NLL versus behavior divergence);
  `research/theme_1_architecture_and_priors/12_induction_circuit_intrinsic_dimension_and_capacity_walls.md`
  (induction-circuit intrinsic dimension and capacity walls, written for Stage 42).
- Roadmap: `README.md` Next ladder, rung 50.
