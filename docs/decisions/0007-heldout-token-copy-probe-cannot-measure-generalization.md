# ADR 0007 · The held-out-key identity-copy probe cannot measure generalization; scope ADR 0006 to seen keys and retire readout-hardening

- Status: Accepted
- Date: 2026-06-24
- Author: Claude (hypothesis, ADR, and roadmap role)
- Consolidates: Hypotheses 015 and 016 (Stages 40 and 41), the held-out-key
  generalization sub-branch of the behavior axis
- Scopes, does not reverse: ADR 0006
  (`docs/decisions/0006-behavior-axis-reopens-residual-formation.md`)
- Builds on: Stage 38 (seen-key behavior formation,
  `experiments/tiny_language_lab/runs/stage38_behaviorgap.md`), Stage 39 (the rank
  lever closed, `experiments/tiny_language_lab/runs/stage39_behavior_rank.md`),
  Stage 40 (free-vocabulary held-out null,
  `experiments/tiny_language_lab/runs/stage40_heldout_copy.md`), Stage 41
  (forced-choice held-out null,
  `experiments/tiny_language_lab/runs/stage41_forcedchoice_heldout.md`), Gemini
  note 10 (`research/theme_1_architecture_and_priors/10_nll_divergence_and_behavior_forming_residuals.md`)

## Context

ADR 0006 reopened residual formation on the behavior axis. It named the rank-2
residual the behavior-forming surface for seen-key identity copy, and it carried an
explicit reversal clause: re-close the behavior axis if cheap surfaces plateau near
the floor on a generalizing probe (held-out keys) while only full-model capacity
forms the behavior. Hypotheses 015 and 016 built the held-out-key probe to test
exactly that clause, and Codex ran it as Stages 40 and 41. This ADR records what
those two stages settled, because the answer is not the simple confirm or kill the
reversal clause anticipated.

## Evidence

Both stages used the held-out identity-copy corpus: six training keys `a` to `f`,
held-out keys `g` and `h` appearing as `key=X ... answer=X` pairs only in
validation, with the train-split invariant verified and `g h` in vocabulary. Each
seed had 97 seen and 18 held-out validation copy cases.

- Stage 40 (free-vocabulary argmax). Every arm scored exactly `0.000000` held-out
  accuracy on every seed: the frozen floor, the rank-2 residual
  `count_prior_lora_r2_copyw` (seen `0.202749`), and the full control
  `random_full_copymix` (seen `0.364261`, up to `0.670103` on seed 19). Seen-key
  formation was weak: the residual gained only `+0.027491` over the floor, far below
  Stage 38's `+0.20`.
- Stage 41 (forced choice restricted to the key alphabet `a..h`, plus mean
  reciprocal rank). The readout designed to bypass output-head suppression did not
  rescue transfer. Held-out forced-choice accuracy stayed `0.000000` for all arms on
  all seeds. The held-out choice MRR was about `0.134` for every arm, including the
  full model, indistinguishable from the blind floor's `0.133929` and at or below the
  uniform-random level of about `0.34` over eight keys. The correct held-out key is
  therefore ranked near last among the eight, not merely second.

The diagnosis is structural. In identity copy the answer equals the key, so a
held-out key token never appears at the answer position during training. The
output computation at that position assigns near-zero, indeed near-worst, mass to
held-out tokens, independent of any attention-mediated copy circuit. Both the cheap
recipe and the full model do this, and restricting the readout to the key alphabet
does not undo it. The probe cannot express a held-out copy as an output, so it
cannot measure whether a copy circuit generalizes. Stages 40 and 41 are confounded
nulls, not evidence about generalization.

## Decision

1. Scope ADR 0006, do not reverse it. The behavior-forming claim stands as written
   for seen-key identity copy at the Stage 38 budget: the rank-2 residual forms
   seen-key copy behavior above the copy-blind frozen floor. Generalization to unseen
   keys is untested at this budget, neither confirmed nor cleanly refuted.
2. ADR 0006's reversal clause does not fire. It requires the cheap surface to fail on
   a probe where the full-model control succeeds. Here the full model fails
   identically (`0.000000` held-out, floor-level MRR), so the failure belongs to the
   probe, not to the cheap surface.
3. Retire the held-out-key-token identity-copy probe as a generalization instrument.
   Two readouts, free-vocabulary at Stage 40 and forced choice at Stage 41, hit the
   same structural wall, so more readout variants on this corpus are not worth
   running. The wall is upstream: held-out tokens are unemittable at the answer
   position by construction.
4. Keep the rank lever closed (Stage 39) and the NLL-side levers closed (ADR 0004,
   ADR 0005). This ADR adds no NLL claim and reopens none of them.

## Scope and what this decision does not claim

- It does not claim the behavior is memorization. The seen-key circuit may or may not
  generalize; this lab has not validly measured it.
- It does not claim the cheap recipe is worse than the full model on generalization.
  Both scored zero held-out; there is no separation to read.
- It retires a probe design, not the generalization question. That question still
  gates the behavior branch and is taken up by Hypothesis 017.
- A second, smaller reason these stages could not host the test: seen-key formation
  on the six-key held-out corpus was weak (`+0.027491` over the floor), so even a
  valid readout would have had a thin circuit to measure.

## What would reopen or reverse this decision

- Reopens with a generalization probe that has a valid readout: an answer alphabet
  that is fully seen during training, so the copied content is always emittable, with
  generalization stressed along an axis other than answer-token identity (a novel
  in-context binding, or a per-line random payload that makes memorization
  impossible by construction). The precondition for trusting any cheap-recipe result
  on such a probe is a full-model control that demonstrably generalizes on it, the
  ceiling Stages 40 and 41 lacked. Hypothesis 017 specifies the first such test.
- Reverses ADR 0006, folding the behavior axis toward a needs-scale conclusion, only
  if on such a valid probe with a working full-model ceiling the cheap surface
  plateaus near the floor while the full model generalizes.

## Links

- Scoped decision: `docs/decisions/0006-behavior-axis-reopens-residual-formation.md`.
- Consolidated hypotheses: `docs/hypotheses/015-held-out-key-copy-generalization.md`,
  `docs/hypotheses/016-forced-choice-heldout-copy-circuit.md`.
- Next test: `docs/hypotheses/017-memorization-proof-copy-probe.md`.
- Stages: `experiments/tiny_language_lab/RESULTS.md` Stages 38 to 41; run summaries
  `stage40_heldout_copy.md` and `stage41_forcedchoice_heldout.md`.
- Gemini prior-art:
  `research/theme_1_architecture_and_priors/10_nll_divergence_and_behavior_forming_residuals.md`.
- Roadmap: `README.md` Next ladder, rung 48.
