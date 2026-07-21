# Hypothesis 015 · The formed copy behavior is a generalizing induction circuit, not seen-key memorization

- Status: RESOLVED. Measured by Codex as Stage 40 (README ladder rung 46). No
  held-out-key generalization was measured at this budget: the rank-2 residual
  tied the floor at `0.000000` held-out accuracy on every seed. This does not
  cleanly fire the memorization reversal because the residual's mean seen-key
  gain over the floor was only `+0.027491`, and the full control also collapsed
  to `0.000000` held-out accuracy. ADR 0006 should be scoped to seen-key identity
  copy at the Stage 38 budget rather than reversed outright. Consolidated with
  Stage 41 by ADR 0007
  (`docs/decisions/0007-heldout-token-copy-probe-cannot-measure-generalization.md`),
  which retires this held-out-token probe as structurally unable to measure
  generalization and reopens the question via Hypothesis 017.
- Date: 2026-06-24
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 46 (Codex stage number 40)
- Builds on: Stage 38 (the behavior axis reopened; the rank-2 residual forms copy
  behavior above the frozen-prior floor,
  `experiments/tiny_language_lab/runs/stage38_behaviorgap.md`), ADR 0006
  (`docs/decisions/0006-behavior-axis-reopens-residual-formation.md`, which names
  the residual the behavior-forming surface and pre-registers a generalizing probe
  as the reversal test), Stage 39 (the behavior rank sweep, which closed the simple
  capacity-limited rank lever, `experiments/tiny_language_lab/runs/stage39_behavior_rank.md`),
  and the held-out-key machinery already proven in Stages 22 to 23 on the
  non-identity mapping corpus. Gemini prior-art: note 10
  (`research/theme_1_architecture_and_priors/10_nll_divergence_and_behavior_forming_residuals.md`,
  the induction-head reading of the Stage 38 result) and note 11
  (`research/theme_1_architecture_and_priors/11_lora_rank_saturation_and_intrinsic_dimension.md`,
  why rank saturated at 2).

## Why this, and why now

Stage 38 confirmed the inversion: on the identity copy corpus the frozen count
prior copies at chance (`0.118421`, near `1 / 8 = 0.125`), while a rank-2 AdamW
residual reaches about `0.31` copy accuracy. ADR 0006 accepted the rescoping and
named the residual the behavior-forming surface. Stage 39 then asked the obvious
follow-up, is the surface capacity-limited at rank 2, and closed it: rank 4 did not
beat rank 1 with stable sign, the rank-4 over rank-1 mean gap was only `+0.021930`,
and Gemini note 11 grounds this as ordinary rank saturation. So the rank lever is
spent. More capacity is not the next question.

That leaves the question ADR 0006 itself flagged as the one that decides the whole
branch, and the one its reversal clause is written against: **is the formed `0.31`
copy behavior a genuine, content-agnostic in-context copy circuit, or is it
seen-key memorization?** Stage 38 and Stage 39 both ran on the original corpus where
all eight keys appear in training, so `seen_acc` equals `copy_acc` and
`heldout_acc` is `None` on every row. The behavior was never tested on a key the
model did not train on. Until it is, "the residual forms copy behavior" and "the
residual memorizes a router over the eight seen keys" are observationally
identical, and the second is not behavior worth chasing.

This is the most valuable open question right now because it is logically prior to
every other behavior-branch lever. Sampler choice, verifier weighting, and
optimization stability are all questions of how much of this behavior we can form.
They are worth running only if the behavior is real. The generalization test is
also the highest-leverage kind of experiment available: it directly exercises a
standing decision's reversal clause, so a KILL reverses ADR 0006 rather than just
adding a data point. And it is on the north star: forming a copy circuit that
transfers to unseen content is useful behavior; memorizing eight answers is not.

## The two competing explanations and the mechanism that separates them

The copy line is `case NNNN key=X noise=Y ...; answer=X`. The model must emit `X`
after `answer=`. Two mechanisms produce identical scores when every key is seen,
and opposite scores on a held-out key:

- Induction circuit (genuine copy). The residual wires an attention-mediated route
  that reads whatever token follows `key=` and reproduces it after `answer=`. This
  is content-agnostic by construction: it copies any in-vocabulary character,
  including a key the model never trained on. This is the induction-head reading of
  Gemini note 10. Prediction: held-out-key copy accuracy is materially above the
  floor and close to seen-key accuracy.
- Seen-key memorization (router). The residual instead learns a set of
  context-to-output associations specialized to the training key characters, with
  no output pathway to emit a character that never appeared at the answer position
  during training. Prediction: held-out-key copy accuracy collapses toward zero
  while seen-key accuracy stays near `0.31`.

The frozen-prior floor is the blind anchor in both readings. At the answer position
its conditioning context is the fixed `answer=` marker, and in training that marker
is only ever followed by seen keys, so the floor assigns almost no mass to a
held-out key. Floor held-out accuracy is therefore expected near zero, even below
the `1 / 8` seen-corpus chance. This is what makes the test sharp: the gap between
"copies any key" and "cannot emit an unseen key" is close to the full accuracy
range, not a few points.

This is a different and cleaner test than the already-killed Stages 22 to 23. Those
held out keys on the **non-identity** mapping corpus, where the held-out answer is
an arbitrary table entry the model cannot infer from context, so a collapse there
(full model held-out accuracy about `0.19`, ADR 0001) only showed that a compact
text prefix does not carry absent facts. Identity copy is the opposite situation:
the held-out answer is the key itself, visible in the same line, so a genuine
induction circuit can produce it with no external memory. Hypothesis 015 is the
identity-copy generalization test the project has not yet run.

## Hypothesis

On an identity copy corpus where a held-out key set never appears as a
`key=X ... answer=X` pair in training but does appear in validation, the rank-2
residual that formed copy behavior in Stage 38 **generalizes to held-out keys**.
Concretely:

- the frozen-prior floor copies at or below chance on both the seen and the
  held-out buckets, held-out near zero; and
- the AdamW rank-2 residual under copy-position weighting (`count_prior_lora_r2_copyw`)
  copies held-out keys materially above the floor, with a held-out accuracy close to
  its own seen-key accuracy.

If true, the cheap residual formed a content-agnostic copy circuit, the strongest
on-thesis behavior result so far, and ADR 0006's behavior claim strengthens from
"forms behavior on seen keys" to "forms a generalizing copy circuit."

This is the falsifiable claim. It is killed in the informative direction by a clean
seen-versus-held-out split:

- Memorization (reverses ADR 0006): held-out accuracy stays within `0.05` of the
  floor's held-out accuracy while seen accuracy clears floor-seen `+ 0.10`. The
  Stage 38 behavior was a seen-key router, not in-context copying. ADR 0006's
  reversal clause fires and the behavior axis folds toward "needs heavier machinery
  or scale," matching the verifier and retrieval samplers of Stages 11 to 16.
- Partial circuit (middle): held-out accuracy is above the floor but well below
  seen accuracy. Report and interpret as a partially generalizing circuit; the next
  rung becomes hardening or strengthening it.

## The reference points

The matched comparison mirrors Stage 38, swapping only the corpus for one with
held-out keys and turning on the held-out split in the probe. Every count-prior arm
uses the same frozen count base and the same rank-2 residual parameterization.

- Arm A, floor: `count_prior_lora_r2_copyw_floor`, the frozen prior with the
  residual not trained. Expected near zero on the held-out bucket, at or below
  chance on the seen bucket. This is the blind anchor.
- Arm B, target: `count_prior_lora_r2_copyw`, the AdamW rank-2 residual that formed
  the strongest copy behavior in Stage 38. This is the primary arm: it had the best
  seen-corpus behavior, so if it does not generalize, weaker arms will not either.
- Arm C, context control: `random_full_copymix`, the full transformer. This answers
  whether held-out identity copy is formable at all at this budget. If even the full
  model collapses on held-out keys, a cheap-recipe collapse is a task or budget
  property, not a cheap-recipe indictment, and must not be read as a clean KILL of
  the cheap surface alone.

The decisive quantities, per arm, are the held-out copy accuracy against the floor's
held-out accuracy, and the within-arm generalization gap `seen_acc - heldout_acc`.

## Primary decision metric and pass or fail line

Metric: held-out-key copy-probe accuracy, mean over seeds `7 11 19` with per-seed
spread, reported alongside seen-key copy accuracy, copy NLL, and plain validation
NLL for the standing dual-axis rule. Codex confirms the per-seed held-out case count
from the generated corpus and reports it; the seen-corpus chance was `1 / 8`, but on
this corpus the meaningful anchors are the floor's measured seen and held-out
accuracies, not a nominal `1 / num_keys`.

- CONFIRM GENERALIZATION: Arm B held-out accuracy clears the floor's held-out
  accuracy by at least `0.10` with positive sign on all three seeds, and the
  within-arm gap `seen_acc - heldout_acc` is small, provisionally at most about
  `0.10` on the mean. The residual formed a generalizing copy circuit. Write a
  strengthening note to ADR 0006 and turn the branch to hardening the probe.
- KILL, memorization (reverses ADR 0006): Arm B held-out accuracy stays within
  `0.05` of the floor's held-out accuracy while Arm B seen accuracy clears the
  floor's seen accuracy by at least `0.10`. The formed behavior does not transfer.
  Write a scoping or reversing ADR; escalate the behavior axis.
- KILL, premise wrong: the floor itself copies held-out keys above chance. The
  prior is doing something unexpected at the answer position. Audit the corpus for a
  local cue and re-run.

The `0.10` thresholds are provisional and deliberately larger than the per-seed
copy-probe spread seen in Stages 8 to 12 and 38 to 39, so a confirm or a kill is a
real signal, not sampling noise. Final lines are fixed when Codex reports the
held-out case count, since a very small held-out bucket widens the spread.

## Risks and confounds

- Small held-out bucket. With two held-out keys out of eight and the validation
  region covering roughly the last sixth of the corpus, the held-out bucket is small.
  Mitigation: generate the corpus with more lines, `--lines 768` or `1024`, to put
  at least about twenty held-out cases per seed, and report per-seed spread rather
  than hiding it. Codex confirms the count and the final pass line scales to it.
- The invariant must actually hold. The test is only valid if the held-out keys
  never appear as a `key=X ... answer=X` pair in the trainer's train split.
  Mitigation in the handoff: a verification step that greps the train split for
  `answer=g` and `answer=h` and asserts zero matches, mirroring how the Stage 22 to
  23 mapping corpus already places held-out keys only after the train region.
- In-vocabulary held-out characters. A held-out key the model cannot emit at all is
  not a fair test of copying. The keys `g` and `h` already appear in the fillers
  `gentle update` and `hidden copy`, so they are in vocabulary and the model can
  produce them; the question is only whether the copy route does. Codex confirms `g`
  and `h` are in the codec before trusting a held-out zero.
- Fewer training keys. Holding out `g h` leaves six training keys, so seen-bucket
  accuracy on this corpus need not equal Stage 38's `0.31`. The decisive quantity is
  internal to each run, seen versus held-out, not a comparison back to Stage 38.
- Full-model contextualization. Read a cheap-recipe held-out collapse against Arm C.
  Only a result where the cheap surface collapses while the full model generalizes
  is a clean cheap-recipe KILL; a shared collapse is a task or budget statement.
- Device and comparability. GPU is the project default and a generalization gap of
  this size is a large-signal result, but keep seeds, corpus, and block size matched
  across arms, per the GPU transition audit.

## What result would change the plan

- CONFIRM: append a strengthening note to ADR 0006. The behavior claim becomes "the
  cheap residual forms a copy circuit that generalizes to unseen keys," and the next
  rungs turn to hardening the probe and to the genuinely harder generalization
  surfaces, longer contexts, more keys, or distractor pressure, rather than to more
  capacity. Flag the stronger claim for a fresh Gemini prior-art pass before any
  external wording.
- KILL, memorization: write a scoping or reversing ADR against ADR 0006's reversal
  clause. The behavior axis does not generalize cheaply at this budget; escalate to
  the Stage 11 to 16 verifier, correction, and retrieval samplers measured on the
  held-out bucket, and re-ask whether the frozen-prior plus small-residual recipe is
  the right substrate for transferable behavior.
- KILL, premise wrong: harden the corpus and re-run before drawing a conclusion.

## Handoff to Codex (implemented as Codex stage 40, README ladder rung 46)

Files to modify:

1. `experiments/tiny_language_lab/make_long_context_corpus.py`: add held-out-key
   support, porting the construction already proven in
   `make_memory_mapping_corpus.py`. Add a `--holdout-keys` argument; when set, draw
   training-region keys only from the seen set `KEYS` minus held-out, and place
   held-out keys only in a trailing validation region that falls inside the
   trainer's validation suffix. Keep the mapping identity, `answer=key`, so this is
   the Stage 38 substrate with a held-out split, not the non-identity mapping
   corpus. Write the corpus to a new file so the original
   `long_context_seed.txt` is untouched. No change is needed to the trainer, the
   probe, or `cassandra_compare.py`: the probe already buckets `seen` versus
   `heldout` by `--copy-probe-holdout-keys`, and the comparison runner already
   forwards that flag and prints the split.

No new comparison configs are needed. Arms A, B, and C are
`count_prior_lora_r2_copyw_floor`, `count_prior_lora_r2_copyw`, and
`random_full_copymix`, all of which already exist.

Generate the corpus, verify the invariant, then run the matched matrix on the same
protocol as Stage 38:

```powershell
# 1. Generate the identity copy corpus with g and h held out of training.
python .\experiments\tiny_language_lab\make_long_context_corpus.py `
  --holdout-keys g h --lines 768 --seed 20260617 `
  --out .\experiments\tiny_language_lab\corpus\long_context_holdout_seed.txt

# 2. Verify the invariant before trusting any number: the train split must contain
#    zero held-out key answers. (Codex computes the trainer split point and asserts
#    no `answer=g` or `answer=h` appears in the train region; confirm g and h are in
#    the codec via the fillers.)

# 3. Run the held-out generalization matrix.
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\long_context_holdout_seed.txt `
  --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 `
  --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 `
  --copy-sample-fraction 0.25 --copy-probe-holdout-keys g h --seeds 7 11 19 `
  --configs count_prior_lora_r2_copyw_floor count_prior_lora_r2_copyw random_full_copymix `
  --out .\experiments\tiny_language_lab\runs\stage40_heldout_copy.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage40_heldout_copy.md `
  --title "Stage 40 Held-Out-Key Copy Generalization"
```

Smoke-test with a short step count first. Record in `RESULTS.md` and the run summary
to the Codex evidence standard: per-arm seen and held-out copy accuracy and NLL, the
held-out case count per seed, the within-arm `seen_acc - heldout_acc` gap, plain
validation NLL for dual-axis tracking, per-seed spread, and a short interpretation
against the confirm and kill lines. The metric that decides the stage is held-out-key
copy accuracy, not validation NLL and not seen-key accuracy.

Pass or fail line, restated for the handoff: CONFIRM if Arm B held-out accuracy
clears the floor's held-out accuracy by at least `0.10` on all three seeds with a
small within-arm gap; KILL, memorization if Arm B held-out stays within `0.05` of the
floor while its seen accuracy clears the floor by at least `0.10`; KILL, premise
wrong if the floor itself copies held-out keys above chance.

## Prior-art flag for Gemini

This is a measurement, not a new mechanism. The claim under test, that an
attention-mediated copy or induction circuit generalizes to tokens unseen in
training while a frozen n-gram prior and a memorized router do not, connects
directly to induction heads and in-context learning (Olsson et al., 2022, already
cited in Gemini note 10) and to the classical generalization-versus-memorization
distinction. A CONFIRM would make a stronger external claim than Stage 38, that a
tiny PEFT residual on a frozen prior forms a copy circuit that transfers to held-out
content. Gemini should locate that specific claim against work on whether
PEFT-formed or LoRA-formed in-context abilities generalize to held-out tokens, and
against any measurement of induction-circuit generalization in small models, before
any external wording. Frame any positive as a clean controlled measurement in the
cheap frozen-prior plus small-residual recipe, not as a novel architecture or
optimizer.

## Links

- Behavior gate it extends: `experiments/tiny_language_lab/RESULTS.md` Stage 38;
  `experiments/tiny_language_lab/runs/stage38_behaviorgap.md`.
- Rank lever it follows: `experiments/tiny_language_lab/RESULTS.md` Stage 39;
  `experiments/tiny_language_lab/runs/stage39_behavior_rank.md`.
- Decision it tests the reversal clause of:
  `docs/decisions/0006-behavior-axis-reopens-residual-formation.md`.
- Held-out-key precedent (non-identity mapping, retired branch): README ladder rungs
  22 to 23; `experiments/tiny_language_lab/make_memory_mapping_corpus.py`;
  `docs/hypotheses/004-external-memory-carries-absent-facts.md`;
  `docs/decisions/0001-retire-compact-text-prefix-external-memory.md`.
- Behavior machinery and the `--copy-*` and `--copy-probe-holdout-keys` flag surface:
  `experiments/tiny_language_lab/cassandra_tiny_transformer.py`;
  `experiments/tiny_language_lab/cassandra_compare.py`.
- Gemini prior-art:
  `research/theme_1_architecture_and_priors/10_nll_divergence_and_behavior_forming_residuals.md`;
  `research/theme_1_architecture_and_priors/11_lora_rank_saturation_and_intrinsic_dimension.md`;
  induction-head generalization to be added by Gemini.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md`, behavior axis.
- Roadmap: `README.md` Next ladder, rung 46.
