# Hypothesis 017 · A memorization-proof copy probe measures whether the cheap residual forms a general in-context copy circuit

- Status: KILLED by Codex Stage 42 on 2026-06-24. The cheap rank-2 residual stayed
  at chance on the random-payload copy probe while the full control cleared chance,
  firing the registered reversal line. See
  `experiments/tiny_language_lab/runs/stage42_random_payload_copy.md`. Accepted with
  revisions as ADR 0008
  (`docs/decisions/0008-general-copy-needs-more-than-current-rank2-residual.md`), which
  narrows ADR 0006 to seen-content behavior and reopens the minimal-surface question as
  Hypothesis 018 (`docs/hypotheses/018-minimal-surface-general-copy.md`).
- Date: 2026-06-24
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 49 (Codex stage number 42)
- Builds on: ADR 0007 (the held-out-token probe was retired as structurally unable
  to measure generalization, and this is the reopening test,
  `docs/decisions/0007-heldout-token-copy-probe-cannot-measure-generalization.md`),
  Stages 40 and 41 (the confounded nulls), Stage 38 (seen-key behavior formation,
  `experiments/tiny_language_lab/runs/stage38_behaviorgap.md`). Gemini prior-art:
  note 10 (`research/theme_1_architecture_and_priors/10_nll_divergence_and_behavior_forming_residuals.md`);
  a high-entropy-copy and induction-head prior-art pass is requested below.

## Why this, and why now

ADR 0007 retired the held-out-key-token identity-copy probe. Two readouts, free
vocabulary at Stage 40 and forced choice at Stage 41, hit the same structural wall:
a held-out key token never appears at the answer position in training, so every
model, including the full control, suppresses it there (held-out choice MRR about
`0.134`, ranked near last among eight keys), independent of any copy circuit. The
generalization question that gates the whole behavior branch, is the formed copy
behavior a general in-context circuit or seen-key memorization, is therefore still
open. It needs a probe whose readout is valid.

The fix is to remove the memorization shortcut by construction rather than to hold
out tokens. If the token to be copied is drawn uniformly at random for every line
from a fully-seen alphabet, then there is no fixed mapping to memorize and no
line-index regularity to exploit, so copying correctly requires reading the token
from context. Above-chance copy accuracy then is a general in-context copy circuit,
directly, with no held-out tokens and no emission confound, because every token in
the alphabet appears as an answer throughout training and is freely emittable.

This also closes a latent confound in Stage 38 itself. There the key was
`KEYS[index % 8]`, determined by line index, so a model could in principle score
above chance by tracking position rather than by reading the key. A per-line random
payload removes that path too, making this a strictly cleaner copy test than the one
that first reopened the behavior axis.

## The mechanism that makes this falsifiable

Each line is the Stage 38 format with one change: the copied value is a per-line
uniform random draw from a payload alphabet of size `V`, not a cycling key. For
example `case NNNN key=X noise=Y ...; answer=X`, where `X` is independent and
uniform over `V` tokens per line.

- A frozen prior (the floor) sees the fixed `answer=` context and can only emit the
  marginal over the `V` payload tokens, which is chance, about `1 / V`.
- A model that ignores the payload scores at chance.
- The only way to score above chance is an attention-mediated route that reads `X`
  after `key=` and reproduces it after `answer=`. That route is content-agnostic by
  construction: it is the general copy circuit. There is nothing to memorize, because
  `X` changes every line and is not a function of position.

So copy accuracy on this corpus is a direct measure of whether a general copy
circuit formed, and the cheap-versus-full comparison answers the gating question.

## Hypothesis

On the memorization-proof copy corpus, the cheap frozen-prior plus rank-2 residual
recipe (`count_prior_lora_r2_copyw`) forms a general in-context copy circuit: its
copy accuracy is materially above the `1 / V` chance line and above the frozen
floor, with a positive sign on all three seeds, while the full-model control clears
chance by a wide margin (confirming the task is learnable at this budget).

If true, ADR 0006's behavior claim strengthens from "forms seen-key copy behavior"
to "forms a general in-context copy circuit", because memorization is excluded by
the corpus design.

This is the falsifiable claim, killed cleanly in either informative direction:

- KILL, fires ADR 0006's reversal clause: the full model clears chance by a wide
  margin, so a general copy circuit is formable at this budget and there is a real
  ceiling, but the cheap residual stays within about `0.05` of chance and the floor.
  The cheap surface forms seen-content behavior but not a general copy circuit, and
  the behavior axis folds toward a needs-scale conclusion.
- INCONCLUSIVE: even the full model stays near chance. The task is too hard at this
  budget or `V` is too large; reduce `V` or raise the step budget and re-run before
  judging the cheap recipe. The full-model arm is the instrument that tells these
  apart.

## The reference points

The same three matched arms as Stages 38, 40, and 41, on the new corpus, so only the
corpus changes.

- Arm A, floor: `count_prior_lora_r2_copyw_floor`. Expected at chance, about `1 / V`,
  the blind anchor.
- Arm B, subject: `count_prior_lora_r2_copyw`, the cheap rank-2 residual. The arm the
  hypothesis is about.
- Arm C, ceiling instrument: `random_full_copymix`, the full transformer. It must
  clear chance by a wide margin, otherwise the stage is inconclusive rather than a
  cheap-surface kill. Arm C is what Stages 40 and 41 lacked: a control that actually
  performs the behavior under test.

Decisive quantities: Arm B copy accuracy against `1 / V` and against the floor; Arm
C copy accuracy as the learnability gate.

## Primary decision metric and pass or fail line

Metric: copy-probe accuracy, mean over seeds `7 11 19` with per-seed spread. Report
the forced-choice copy accuracy and MRR too (they coincide with free vocabulary now
that all tokens are emittable, and give continuity with Stage 41), plus validation
NLL for the standing dual-axis rule. Chance is `1 / V`; Codex states `V` and the
measured floor accuracy as the empirical chance anchor.

- CONFIRM: Arm B copy accuracy clears the larger of the floor and `1 / V` by at least
  `0.10`, positive on all three seeds, while Arm C clears chance by a wide margin.
- KILL, reversal: Arm C clears chance by a wide margin but Arm B stays within `0.05`
  of the floor and `1 / V`.
- INCONCLUSIVE: Arm C is itself near chance.

The `0.10` threshold matches the earlier copy stages and exceeds the per-seed spread.

## Risks and confounds

- Determinism. The per-line random payload must be drawn from the corpus generator's
  seeded RNG so the corpus is reproducible, preserving the project's determinism
  rule. The payload draw must not depend on line index or any in-context constant.
- Choice of `V`. Too large and even the full model may not clear chance at 500 steps
  (inconclusive); too small and the chance line is high. Recommend `V = 16` (chance
  `0.0625`); Codex may sanity-check that Arm C clears it, and lower `V` or raise steps
  if not. Even `V = 8` is memorization-proof because the payload is per-line random,
  but a larger `V` gives a cleaner separation from chance.
- Emittability. Every payload token must appear as an answer many times in training;
  this holds by construction since every line's answer is a payload token. Confirm the
  payload alphabet is in the codec.
- Floor sanity. The floor should sit near `1 / V`. If the floor is well above `1 / V`,
  the corpus has an unintended regularity (for example a non-uniform payload draw or a
  position cue); audit the generator before trusting Arm B.
- Budget comparability. Reuse the Stage 38 and 41 protocol (block size `96`, `500`
  steps, sampled evaluation with 16 batches, `copy_loss_weight 200`,
  `copy_sample_fraction 0.25`), changing only the corpus, per the GPU transition
  audit.

## What result would change the plan

- CONFIRM: append a strengthening note to ADR 0006, the cheap residual forms a
  general in-context copy circuit, not just seen-key behavior. Flag the stronger
  claim for a fresh Gemini prior-art pass, then turn the behavior branch to how
  cheaply and how robustly the circuit forms (samplers, verifier signal, longer
  contexts), now on a probe known to be valid.
- KILL, reversal: write the ADR that reverses or further scopes ADR 0006 on a clean
  result, the cheap surface does not form a general copy circuit while the full model
  does, and escalate the behavior axis to heavier machinery or a larger surface.
- INCONCLUSIVE: re-run with a smaller `V` or a larger step budget until Arm C clears
  chance, then read Arm B.

## Handoff to Codex (implemented as Codex stage 42, README ladder rung 49)

Files to modify:

1. `experiments/tiny_language_lab/make_long_context_corpus.py`: add a memorization-
   proof mode, for example a `--random-payload` flag with `--payload-alphabet-size`
   (default 16). In that mode each line keeps the existing format but draws the
   copied value uniformly at random per line from the payload alphabet using the
   seeded RNG, independent of line index, with `answer=` equal to that value. Write to
   a new file (for example `corpus/random_payload_copy_seed.txt`) so existing corpora
   are untouched. No probe change is needed: the existing copy probe and the Stage 41
   forced-choice metrics work as is, and no held-out keys are used.

No new comparison configs are needed. Arms A, B, and C already exist. Generate the
corpus, sanity-check it, then run the matched matrix:

```powershell
python .\experiments\tiny_language_lab\make_long_context_corpus.py `
  --random-payload --payload-alphabet-size 16 --lines 768 --seed 20260617 `
  --out .\experiments\tiny_language_lab\corpus\random_payload_copy_seed.txt

python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\random_payload_copy_seed.txt `
  --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 `
  --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 `
  --copy-sample-fraction 0.25 --seeds 7 11 19 `
  --configs count_prior_lora_r2_copyw_floor count_prior_lora_r2_copyw random_full_copymix `
  --out .\experiments\tiny_language_lab\runs\stage42_random_payload_copy.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage42_random_payload_copy.md `
  --title "Stage 42 Memorization-Proof Copy Probe"
```

Smoke-test with a short step count first, and confirm the floor sits near `1 / V` and
the full control clears chance. Record in `RESULTS.md` and the run summary to the
Codex evidence standard: per-arm copy accuracy, forced-choice accuracy and MRR, the
measured `1 / V` chance line and the floor accuracy, validation NLL for dual-axis
tracking, per-seed spread, and a short interpretation against the confirm, kill, and
inconclusive lines. The metric that decides the stage is copy accuracy, gated by the
full-model learnability check.

Pass or fail line, restated: CONFIRM if Arm B clears the floor and `1 / V` by at
least `0.10` on all three seeds while Arm C clears chance by a wide margin; KILL,
reversal if Arm C clears chance widely but Arm B stays within `0.05` of the floor;
INCONCLUSIVE if Arm C is itself near chance.

## Prior-art flag for Gemini

This is a measurement on a standard construction, not a new mechanism. The
memorization-proof copy task, copying a high-entropy or randomized in-context token
stream so the model must read context rather than recall a fixed mapping, is exactly
the setup the induction-head literature uses to elicit and measure in-context copying
(Olsson et al., 2022, random-token copying). Gemini should locate the specific claim,
that a tiny PEFT residual on a frozen prior forms a general copy or induction circuit
under a memorization-proof readout, against that literature and against any work
measuring induction-circuit formation under parameter-efficient fine-tuning, before
any external wording. Frame any positive as a clean controlled measurement, not a
novel architecture or optimizer.

## Links

- Decision that retired the prior probe and set this reopening:
  `docs/decisions/0007-heldout-token-copy-probe-cannot-measure-generalization.md`.
- Confounded nulls this replaces: `experiments/tiny_language_lab/RESULTS.md` Stages 40
  and 41; `runs/stage40_heldout_copy.md`, `runs/stage41_forcedchoice_heldout.md`;
  `docs/hypotheses/015-held-out-key-copy-generalization.md`,
  `docs/hypotheses/016-forced-choice-heldout-copy-circuit.md`.
- Seen-key behavior under test: `experiments/tiny_language_lab/RESULTS.md` Stage 38;
  `runs/stage38_behaviorgap.md`.
- Behavior machinery and the `--copy-*` flag surface plus forced-choice metrics:
  `experiments/tiny_language_lab/cassandra_tiny_transformer.py`;
  `experiments/tiny_language_lab/cassandra_compare.py`.
- Gemini prior-art:
  `research/theme_1_architecture_and_priors/10_nll_divergence_and_behavior_forming_residuals.md`;
  induction-head random-token copying to be added by Gemini.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md`, behavior axis.
- Roadmap: `README.md` Next ladder, rung 49.
