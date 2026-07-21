# Hypothesis 016 · A forced-choice probe separates the copy circuit from the output-emission pathway, so the held-out generalization question can actually be measured

- Status: RESOLVED. Measured by Codex as Stage 41 (README ladder rung 47). H016
  did not confirm: forced choice over `abcdefgh` left the rank-2 residual at
  `0.000000` held-out choice accuracy on every seed, tied with the floor and
  below `1 / 8` chance. The clean memorization kill also does not fire because
  Arm B's seen choice accuracy was only `0.202749`, below the `+0.10` seen-power
  clause. The full control also held-out collapsed under forced choice, so the
  result points to a task or budget failure under this protocol rather than a
  cheap-surface-only reversal. Consolidated with Stage 40 by ADR 0007
  (`docs/decisions/0007-heldout-token-copy-probe-cannot-measure-generalization.md`),
  which retires the held-out-token probe and reopens the question via Hypothesis 017.
- Date: 2026-06-24
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 47 (Codex stage number 41)
- Builds on: Stage 40 (the held-out-key probe returned a confounded null,
  `experiments/tiny_language_lab/runs/stage40_heldout_copy.md`,
  `docs/hypotheses/015-held-out-key-copy-generalization.md`), Stage 38 (the seen-key
  behavior the generalization question is about,
  `experiments/tiny_language_lab/runs/stage38_behaviorgap.md`), ADR 0006 and its
  Stage 40 follow-up (`docs/decisions/0006-behavior-axis-reopens-residual-formation.md`).
  Reuses the candidate-restriction primitive already in the trainer,
  `build_choice_tensors` and `--copy-choice-weight`, in eval form. Gemini prior-art:
  note 10 (`research/theme_1_architecture_and_priors/10_nll_divergence_and_behavior_forming_residuals.md`,
  induction-head reading); a forced-choice and logit-pathway prior-art pass is
  requested below.

## Why this, and why now

Hypothesis 015 asked the question ADR 0006 says decides the whole behavior branch:
is the formed copy behavior a generalizing in-context circuit or seen-key
memorization? Stage 40 ran it and returned a result that looks like a kill but is
not a clean one, and on inspection is not a kill at all. On the held-out corpus,
with `97` seen and `18` held-out validation cases per seed and the train-split
invariant verified, every arm scored exactly `0.000000` held-out accuracy on every
seed:

- frozen-prior floor: seen `0.175258`, held-out `0.000000`;
- rank-2 residual `count_prior_lora_r2_copyw`: seen `0.202749`, held-out `0.000000`;
- full control `random_full_copymix`: seen `0.364261` (up to `0.670103` on seed 19),
  held-out `0.000000`.

Two things in that table say the probe, not the behavior, produced the zero.

First, the full model is the diagnostic. `random_full_copymix` copied seen keys at
up to `0.670103` on seed 19 yet still scored exactly `0.000000` on held-out keys.
A full transformer that copies seen keys two thirds of the time but never once emits
a held-out key, across 54 held-out cases, is not failing to form a copy circuit. It
is failing to emit a token that never appeared as a target at the answer position.

Second, the zero is exact and universal, not near-chance. A genuine no-transfer
result would scatter a little above zero. Exactly `0.000000` for all three arms on
all three seeds is the signature of a hard structural block, not a noisy floor.

The mechanism is separable and the probe conflated it. Copying requires two things:
an attention-mediated route that reads the token after `key=` (the in-context copy
circuit), and an output projection that, at the `answer=` position, can place that
token on top of the full vocabulary (the emission pathway). During training the
`answer=` position is only ever followed by seen keys, so the position-conditioned
output head learns its target is one of `a..f` and drives the logits of `g` and `h`
down, independent of whatever the attention circuit computes. Free-vocabulary argmax
at the answer position therefore measures the product of circuit and emission
pathway. A held-out key is unemittable by construction, so the probe reads zero even
for a perfect content-agnostic copy circuit. Stage 40's null is uninterpretable for
the generalization question.

There is a second, smaller confound to control: on this corpus the seen-key behavior
itself barely formed. The rank-2 residual's seen-bucket gain over the floor was only
`+0.027491`, far below Stage 38's `+0.20`, partly because six training keys raise the
floor toward `1 / 6` and partly because formation looks weaker on this split. A clean
generalization test needs a real circuit to measure, so the re-test must verify seen
behavior is present before reading anything into the held-out number.

We cannot pivot to sampler or verifier levers yet. Those ask how much of the behavior
to form. The branch-gating question, is the behavior a generalizing circuit, is still
open because Stage 40 measured it through a degenerate probe. The goal-loop discipline
for a premise-wrong or artifact result is to harden the probe and re-run before
concluding, which is exactly this hypothesis.

## The hardened probe

Decouple the circuit from the emission pathway with a forced-choice readout. At each
answer position, restrict the prediction to the key alphabet and ask whether the
model ranks the true key first among keys:

- Forced-choice copy accuracy: among the logits of the key characters only, the top
  key equals the true key. This measures whether the copy route selects the correct
  key, with the rest of the vocabulary removed, so a suppressed-but-present circuit
  is visible.
- Correct-key rank, as graded support: the rank of the true key among the key
  characters, reported as mean reciprocal rank, so partial circuits that rank the key
  second or third still register rather than reading as a hard zero.

The forced-choice candidate set must be the full key alphabet `a..h`, including the
held-out keys, otherwise the readout still cannot express a held-out copy. The probe
already extracts each line's key with `extract_key_value`, so the candidate set can be
built automatically as the distinct `key=` characters seen across the validation
lines, which on the held-out corpus naturally includes `g` and `h`. The
training-side `build_choice_tensors` primitive is the same idea applied to the loss;
this is its eval analogue, restricted to the key alphabet rather than to verified
training answers.

The frozen-prior floor stays a clean blind anchor under this readout. At the answer
position the floor emits the marginal over training answers `a..f`, so its
forced-choice held-out accuracy is still about zero, it will pick some seen key, while
its forced-choice seen accuracy is about the modal-key rate. A generalizing circuit is
the only thing that can lift forced-choice held-out accuracy above the floor.

## Hypothesis

Under a forced-choice readout restricted to the key alphabet, the rank-2 residual
that formed seen-key copy behavior **also selects the correct held-out key above
chance**, revealing a content-agnostic copy circuit that Stage 40's free-vocabulary
argmax could not see because the held-out tokens are unemittable at the answer
position.

Concretely, with forced choice over `a..h`:

- the floor's held-out forced-choice accuracy stays near zero, at or below the
  `1 / 8` key-chance line; and
- `count_prior_lora_r2_copyw` reaches held-out forced-choice accuracy materially
  above the floor and above `1 / 8`, with a positive sign across seeds, provided its
  seen forced-choice accuracy confirms a real circuit is present.

If true, the cheap residual formed a generalizing copy circuit, the Stage 40 zero was
an emission artifact, and ADR 0006 strengthens with a documented caveat that free
generation of held-out tokens is a separate, unsolved pathway.

This is the falsifiable claim, killed cleanly in either informative direction:

- KILL, memorization (now fires ADR 0006's reversal clause cleanly): seen
  forced-choice accuracy is clearly above chance, so a circuit did form for seen
  keys, but held-out forced-choice accuracy sits within `0.05` of the floor and of
  `1 / 8`. The circuit is seen-key specific, not content-agnostic. Stage 40's
  conflation is removed and the reversal clause fires on a genuine no-transfer
  result, not an artifact.
- INCONCLUSIVE, underpowered: seen forced-choice accuracy is itself near chance, so
  no circuit formed even on seen keys at this budget on this corpus. The held-out
  number then says nothing; escalate to a stronger-formation re-test, more keys to
  pressure a general circuit over memorization, or more steps, before judging
  generalization.

## The reference points

The same matched arms as Stage 40, on the same held-out corpus, so the only change is
the readout. This isolates the emission confound with everything else fixed.

- Arm A, floor: `count_prior_lora_r2_copyw_floor`. Forced-choice held-out near zero,
  the blind anchor.
- Arm B, target: `count_prior_lora_r2_copyw`. The primary arm.
- Arm C, full control: `random_full_copymix`. Under the emission-artifact reading its
  forced-choice held-out accuracy should rise well above zero, since a full model on a
  trivial identity-copy task is expected to form a content-agnostic copy circuit. Arm
  C is now the strongest test of the artifact hypothesis: if even forced choice leaves
  the full model at chance on held-out keys, the task genuinely fails to induce a
  general circuit at this budget, and a cheap-recipe failure is a task property, not a
  cheap-surface indictment. If Arm C generalizes under forced choice but Arm B does
  not, that is a clean cheap-surface-specific failure.

The decisive quantities are forced-choice held-out accuracy against the floor and
against `1 / 8`, the within-arm forced-choice `seen - heldout` gap, and the seen
forced-choice accuracy as the power check that a circuit exists to test.

## Primary decision metric and pass or fail line

Metric: forced-choice held-out copy accuracy, mean over seeds `7 11 19` with per-seed
spread, reported alongside forced-choice seen accuracy, the correct-key mean
reciprocal rank for both buckets, and, for continuity, the original free-vocabulary
accuracies from Stage 40. Chance under forced choice over eight keys is `1 / 8`.

- CONFIRM GENERALIZATION: Arm B forced-choice held-out accuracy clears the larger of
  the floor's forced-choice held-out accuracy and `1 / 8` by at least `0.10`, positive
  on all three seeds, while Arm B forced-choice seen accuracy is clearly above chance
  so the circuit is real. The residual formed a generalizing copy circuit; the Stage
  40 zero was an emission artifact.
- KILL, memorization: Arm B forced-choice seen accuracy is above chance by at least
  `0.10`, but Arm B forced-choice held-out accuracy stays within `0.05` of the floor
  and of `1 / 8`. No content-agnostic circuit; ADR 0006's reversal clause fires
  cleanly.
- INCONCLUSIVE: Arm B forced-choice seen accuracy is itself within `0.05` of chance.
  No circuit to test; escalate to stronger formation before judging.

The `0.10` thresholds match the earlier copy stages and are larger than the per-seed
spread; with only `18` held-out cases per seed the per-seed numbers must be shown, and
the mean reciprocal rank is the tie-breaker if argmax forced choice is noisy.

## Risks and confounds

- Residual suppression inside forced choice. If training drove `g` and `h` logits far
  enough below the seen keys, forced choice over `a..h` could still rank them last even
  when attention reads them. The correct-key mean reciprocal rank mitigates this, and a
  held-out MRR well above the floor's is itself evidence of a present-but-suppressed
  circuit even if argmax forced choice is at chance. Report both.
- Underpowered formation. The seen circuit was weak on this corpus, `+0.027` free-vocab.
  The seen forced-choice power check is mandatory; if it is near chance, the result is
  inconclusive by construction, not a kill.
- Forced choice changes the metric, not the model. Be explicit in the writeup that
  forced choice is a diagnostic readout of the circuit, and that free-generation of
  held-out tokens remains a separate unsolved pathway. A CONFIRM does not claim the
  cheap recipe can generate held-out keys unprompted; it claims the copy route
  generalizes.
- Small held-out bucket. Eighteen held-out cases per seed. Keep the three fixed seeds,
  require stable sign, and report spread. A larger held-out corpus is the follow-up if
  the signal is borderline.
- Device and comparability. Reuse the exact Stage 40 corpus, block size, seeds, and
  steps so the only change from Stage 40 is the readout, per the GPU transition audit.

## What result would change the plan

- CONFIRM: append a strengthening note to ADR 0006, the cheap residual forms a copy
  circuit that generalizes to unseen keys under a forced-choice readout, with the
  explicit caveat that unprompted emission of held-out tokens is a separate limitation.
  Open a follow-up on whether the emission pathway can be cheaply unlocked, for example
  by a tiny output-side adjustment, and flag the stronger claim for a fresh Gemini
  prior-art pass before any external wording.
- KILL, memorization: write the scoping or reversing ADR that Stage 40 could not
  justify, now on a clean no-transfer result. The behavior axis does not generalize
  cheaply at this budget; escalate to the Stage 11 to 16 verifier and correction
  samplers measured on held-out forced choice, or to a larger-key corpus that pressures
  a general circuit, and re-ask whether the frozen-prior plus small-residual recipe is
  the right substrate for transferable behavior.
- INCONCLUSIVE: re-run with stronger formation, more keys or more steps, before
  judging generalization at all.

## Handoff to Codex (implemented as Codex stage 41, README ladder rung 47)

Files to modify:

1. `experiments/tiny_language_lab/cassandra_tiny_transformer.py`: in the copy probe,
   add a forced-choice readout alongside the existing free-vocabulary argmax at line
   1960. Build a candidate id tensor from the distinct `key=` characters across the
   probe's validation lines, using the existing `extract_key_value`, so the set is the
   full key alphabet including held-out keys. For each case, also compute the argmax
   restricted to those candidate ids and whether it equals the target, and the rank of
   the target among the candidates. Accumulate into the existing seen and heldout
   buckets and report `copy_probe_seen_choice_accuracy`,
   `copy_probe_heldout_choice_accuracy`, and the corresponding mean reciprocal ranks,
   mirroring the existing `copy_probe_seen_accuracy` and `copy_probe_heldout_accuracy`
   fields. This reuses the candidate idea from `build_choice_tensors`; it does not
   change training or any existing metric.
2. `experiments/tiny_language_lab/cassandra_compare.py`: surface the new forced-choice
   fields in the summary table, the same one-line-per-field pattern already used for
   the seen and heldout split.

No new comparison configs and no new corpus are needed. Reuse the Stage 40 corpus
`experiments/tiny_language_lab/corpus/long_context_holdout_seed.txt` and the three
existing arms, so Stage 41 is Stage 40 with a hardened readout:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\long_context_holdout_seed.txt `
  --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 `
  --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 `
  --copy-sample-fraction 0.25 --copy-probe-holdout-keys g h --seeds 7 11 19 `
  --configs count_prior_lora_r2_copyw_floor count_prior_lora_r2_copyw random_full_copymix `
  --out .\experiments\tiny_language_lab\runs\stage41_forcedchoice_heldout.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage41_forcedchoice_heldout.md `
  --title "Stage 41 Forced-Choice Held-Out Copy Circuit"
```

Smoke-test with a short step count first, confirming the new fields appear and the
free-vocabulary numbers reproduce Stage 40 exactly, since the model is unchanged.
Record in `RESULTS.md` and the run summary to the Codex evidence standard: per-arm
forced-choice seen and held-out accuracy, the correct-key mean reciprocal ranks, the
unchanged free-vocabulary numbers for reference, per-seed spread, and a short
interpretation against the confirm, kill, and inconclusive lines. The metric that
decides the stage is forced-choice held-out accuracy, gated by the seen forced-choice
power check.

Pass or fail line, restated for the handoff: CONFIRM if Arm B forced-choice held-out
accuracy clears the floor and `1 / 8` by at least `0.10` on all three seeds with a
real seen forced-choice circuit; KILL, memorization if the seen circuit is present but
held-out forced choice stays within `0.05` of the floor and `1 / 8`; INCONCLUSIVE if
seen forced choice is itself near chance.

## Prior-art flag for Gemini

This is a measurement refinement, not a new mechanism. Two threads to locate. First,
forced-choice or candidate-restricted evaluation, where a model is scored on ranking
the correct option within a constrained set rather than on free generation, a standard
technique in classification-style probing of language models. Second, the mechanistic
separation between an in-context circuit and the output or unembedding pathway, the
logit-lens and induction-head literature, where a model can compute a representation it
does not surface at the output. The specific claim a CONFIRM would make, that a tiny
PEFT residual on a frozen prior forms a copy circuit that generalizes to held-out keys
under forced choice while free emission of those keys stays blocked, should be placed
against work on capability-versus-calibration gaps and on induction-circuit
generalization in small models before any external wording. Frame any positive as a
clean controlled measurement, not a novel architecture or optimizer.

## Links

- Confounded null this hardens: `experiments/tiny_language_lab/RESULTS.md` Stage 40;
  `experiments/tiny_language_lab/runs/stage40_heldout_copy.md`;
  `docs/hypotheses/015-held-out-key-copy-generalization.md`.
- Seen-key behavior under test: `experiments/tiny_language_lab/RESULTS.md` Stage 38;
  `experiments/tiny_language_lab/runs/stage38_behaviorgap.md`.
- Decision whose reversal clause this finally lets us test cleanly:
  `docs/decisions/0006-behavior-axis-reopens-residual-formation.md`, Stage 40 follow-up.
- Candidate-restriction primitive reused in eval form: `build_choice_tensors` and the
  `--copy-choice-weight` path in
  `experiments/tiny_language_lab/cassandra_tiny_transformer.py`; the copy probe argmax
  at line 1960.
- Held-out corpus and the `--copy-probe-holdout-keys` flag surface:
  `experiments/tiny_language_lab/corpus/long_context_holdout_seed.txt`;
  `experiments/tiny_language_lab/make_long_context_corpus.py`;
  `experiments/tiny_language_lab/cassandra_compare.py`.
- Gemini prior-art:
  `research/theme_1_architecture_and_priors/10_nll_divergence_and_behavior_forming_residuals.md`;
  forced-choice evaluation and logit-pathway separation to be added by Gemini.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md`, behavior axis.
- Roadmap: `README.md` Next ladder, rung 47.
