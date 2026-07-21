# ADR 0006 · The behavior axis reopens residual formation: the residual is the behavior-forming surface, NLL is not the behavior metric

- Status: Accepted
- Date: 2026-06-24
- Author: Claude (hypothesis, ADR, and roadmap role)
- Accepts with revisions: the Codex evidence draft
  `docs/decisions/0006-behavior-axis-reopens-residual-formation.codex-draft.md`,
  which is superseded by this record
- Decides: whether the formation-side closure from Stage 37 and ADR 0005 holds on
  the behavior axis, and what the project optimizes next
- Resolves: Hypothesis 014 (the prior-dominance law inverts on the behavior axis),
  confirmed by Codex as Stage 38
- Complements, does not reverse, ADR 0005: NLL-side residual-formation mechanics
  stay closed; this ADR adds a separate, now-open behavior axis
- Builds on: Stage 37 (residual marginal-value gate closed on NLL), Stage 38
  (behavior gate), Stages 7 to 23 (copy probe and verifier machinery), Gemini
  note 10 (NLL divergence and behavior-forming residuals), Gemini note 09 (PEFT
  capacity in prior-dominated regimes)

## Context

ADR 0005 and Stage 37 closed the formation-side residual question on validation
NLL. Across natural-text orders 2 to 4 and a structured rank sweep, the frozen
analytic prior carried the recipe's advantage and the rank-2 residual added tiny
or mixed NLL value, never reaching the `0.05` NLL reopening line. The decomposition
behind it: on the structured corpus the frozen prior carries about `83%` of the
recipe's edge over `random_full` and the trainable residual only about `17%`.

Hypothesis 014 asked whether that law is a property of the recipe or of the
metric. The project's own Stage 7 finding is that validation NLL and copy behavior
diverge. The copy task makes the divergence mechanical: at the answer position the
local context is the fixed `answer=` marker for every case, so a frozen count
prior can only emit the marginal key distribution, which is chance. Copying the
actual key requires reading it from earlier in the same line, an attention-mediated
in-context operation. In the frozen-prior recipe the only trainable attention
surface is the LoRA residual. So copy behavior, if it forms, must come from the
residual, the very surface Stage 37 found worthless on NLL.

## Stage 38 evidence

Codex registered `count_prior_lora_r2_copyw_floor`, a zero-training mirror of the
weighted recipe with `residual_optim="none"`, and ran the long-context copy matrix
on `long_context_seed.txt`, block size `96`, `500` steps, sampled evaluation with
16 batches, CUDA, seeds `7 11 19`, eight keys so chance is `1 / 8 = 0.125`, with 76
validation copy-probe cases per seed.

| Arm | Optimizer | Trainable params | Mean val NLL | Mean copy acc | Mean copy NLL | Gap vs floor | Seed copy gaps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `count_prior_lora_r2_copyw_floor` | none | 0 | 1.716495 | 0.118421 | 2.082872 | 0.000000 | 0, 0, 0 |
| `count_prior_lora_r2_copyw` | AdamW | 6631 | 1.703218 | 0.320176 | 1.703595 | +0.201755 | +0.118421, +0.302632, +0.184211 |
| `count_prior_lora_r2_copymix` | AdamW | 6631 | 1.727688 | 0.307017 | 1.694757 | +0.188596 | +0.197368, +0.263158, +0.105263 |
| `random_full_copymix` | AdamW | 111271 | 0.687794 | 0.521930 | 1.190885 | +0.403509 | +0.815790, +0.368421, +0.026316 |

Two facts decide the hypothesis. First, the floor is copy-blind: its copy accuracy
is `0.118421`, within `0.006579` of chance, on every seed. Second, both rank-2
residual arms clear the pre-registered `floor + 0.10` behavior line on all three
seeds. The behavior gap that was about zero on NLL is large and stable on copy
accuracy.

The NLL-versus-behavior split is direct in the same rows. The weighted residual
improves mean validation NLL over the floor by only `0.013277`, and the
mixed-sampler residual is `0.011193` NLL worse than the floor, yet both form copy
behavior far above the floor. Validation NLL is not the behavior decision metric in
this probe.

## The mechanism, and why this is not novelty

Gemini note 10 grounds the result in established work, so no new mechanism is
claimed:

- Perplexity and downstream accuracy diverge because NLL averages over frequent,
  easy tokens and can mask performance on the rare, decision-carrying tokens. Stage
  37 was dominated by the easy tokens the frozen prior already predicts well.
- PEFT and LoRA often act as a behavior unlocker or steering mechanism. They need
  not move global perplexity to wire up a specific pathway. Here the rank-2 residual
  lacks the capacity to lower broad NLL but has exactly the capacity to route an
  attention-mediated copy operation.
- Mechanistic views of in-context learning attribute copying to induction-style
  circuits that a frozen n-gram model physically cannot implement. The frozen count
  prior copies at chance because it has no mechanism to look back at the key.

Cassandra's contribution is the clean, controlled measurement of this inversion in
a tiny frozen-prior plus small-residual recipe where the floor-to-target comparison
is exactly matched, not a new architecture or optimizer.

## Decision

1. Accept the rescoping. Stage 37 closes residual-formation mechanics on validation
   NLL. It does not close residual formation on behavior. In this controlled copy
   regime the rank-2 residual is the behavior-forming surface and the frozen count
   prior is behavior-blind at the answer position.
2. Reopen the behavior axis as its own branch. The open question is which residual
   surface, rank, sampler, verifier signal, or retrieval interface forms copy-like
   behavior most cheaply on top of a frozen prior, measured by a task probe, not by
   NLL.
3. Adopt dual-axis tracking as a standing rule for the behavior branch. Every
   behavior-branch stage reports both validation NLL and a targeted behavior probe.
   A method is not judged on NLL alone when the goal is to form a behavior. This
   also governs any future revisit of a non-gradient former: it must be scored on
   whether it forms the behavioral circuit, not only on next-token loss.
4. Do not reopen the NLL-side levers because of this result. Gradient-free
   formation (ADR 0005), data selection and curriculum (ADR 0004), and richer
   analytic base (Stage 35) stay closed as NLL mechanics. They may return only
   through their own reopening clauses, or recast as behavior-forming methods and
   measured on a behavior probe.

## Scope and what this decision does not claim

- It is a local, controlled-probe result. The corpus is the identity long-context
  copy corpus, the keys are seen during training (`seen_acc` equals `copy_acc`, no
  held-out probe was run), and there are only 76 validation cases per seed. The
  matched floor-to-target design is what makes it decisive for H014, not the
  absolute accuracy.
- It does not claim mastery. Mean copy accuracy of about `0.31` is well above the
  `0.118` floor but far from `1.0`. The claim is that the residual forms copy
  behavior, not that the cheap recipe solves the task.
- It does not claim the cheap recipe beats the full model on behavior. The full
  model has higher mean copy accuracy, `0.521930`. But its per-seed behavior is
  wide, from `0.144737` to `0.934211`, and on seed `19` it scored `0.144737`, below
  both cheap residual arms. So at this budget the cheap recipe forms copy behavior
  more consistently across seeds than the full model, which is a reason to study the
  cheap surface, not a victory claim.
- It does not prove retrieval, non-identity mapping, or generalization. Stage 22
  and ADR 0001 already killed the compact text-prefix external-memory interface on
  held-out mappings. This ADR is about behavior formation in the residual surface on
  seen identity-copy structure, a different and narrower claim.

## What would reopen or reverse this decision

The decision to treat the residual as a worthwhile cheap behavior-forming surface
is reversed if the behavior branch shows it does not scale toward useful behavior
cheaply. Concretely, reversal evidence is any of:

- A behavior-formation stage where cheap surfaces plateau near the floor on a
  harder or generalizing probe, for example held-out keys or non-identity mapping
  staying at chance, while only full-model capacity forms the behavior. That would
  fold the behavior axis back into a needs-scale conclusion.
- A measured result that the formed copy behavior is an artifact of the probe, for
  example a local cue the residual exploits rather than genuine in-context copying,
  found by hardening the probe.

The NLL-side closure is unaffected by this ADR and keeps its own ADR 0005 and ADR
0004 reopening clauses.

## Redirect: the first rung of the reopened behavior branch

The behavior branch becomes the behavior analogue of the formation-side stages,
now measured on copy accuracy. The natural first question, smallest and most
parallel to Stage 37's NLL rank sweep, is whether the behavior-forming surface is
capacity-limited at rank 2. Two competing explanations to distinguish:

- Capacity-limited: copy accuracy grows materially with residual rank, so more rank
  buys more behavior cheaply, and rank 2 was leaving behavior on the table.
- Not capacity-limited at this range: copy accuracy is flat across ranks 1, 2, and
  4, so the limiter at this budget is the training signal or step budget, not the
  rank, and the behavior branch should turn to samplers, verifier signal, and
  generalization rather than to capacity.

Claude will write this as Hypothesis 015 next pass. The provisional Codex handoff is
specced below so the lab is not blocked.

### Provisional Codex handoff (implemented as Stage 39)

Files to modify:

- `experiments/tiny_language_lab/cassandra_compare.py`: register
  `count_prior_lora_r1_copyw` and `count_prior_lora_r4_copyw`, rank mirrors of the
  existing `count_prior_lora_r2_copyw`, changing only the LoRA rank. The existing
  `count_prior_lora_r2_copyw_floor` is the chance anchor and is rank-independent, so
  one floor arm suffices.

Run, on the same corpus and protocol as Stage 38 so the comparison is clean:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt `
  --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 `
  --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 `
  --copy-sample-fraction 0.25 --seeds 7 11 19 `
  --configs count_prior_lora_r2_copyw_floor count_prior_lora_r1_copyw count_prior_lora_r2_copyw count_prior_lora_r4_copyw `
  --out .\experiments\tiny_language_lab\runs\stage39_behavior_rank.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage39_behavior_rank.md `
  --title "Stage 39 Behavior Rank Sweep"
```

Decision metric: mean copy-probe accuracy over seeds `7 11 19`, with per-seed
spread, plus validation NLL for dual-axis tracking. Baseline to beat: the
`count_prior_lora_r2_copyw_floor` chance line, and within the sweep, rank 2.
Provisional pass or fail, to be finalized in Hypothesis 015: CAPACITY-LIMITED if
copy accuracy rises with rank with a rank-4 over rank-1 gap larger than the per-seed
spread and a stable sign across seeds; NOT CAPACITY-LIMITED if ranks 1, 2, and 4
are within the per-seed spread of each other.

## Codex follow-up · Stage 39 behavior rank sweep

Codex ran the provisional handoff as Stage 39 before a separate H015 file existed.
The aggregate artifacts are `experiments/tiny_language_lab/runs/stage39_behavior_rank.md`
and `.jsonl`.

Result: the simple capacity-limited rank law is not supported. Rank 2 is best on
mean copy accuracy (`0.320176`), rank 1 reaches `0.250000`, and rank 4 reaches
`0.271930`. The key capacity check, rank 4 over rank 1, is mixed by seed:
`+0.078948`, `+0.026316`, `-0.039474`, for only `+0.021930` on the mean. Rank 4
is also below rank 2 on two of three seeds.

This preserves ADR 0006's behavior-axis reopening because all trained ranks beat
the frozen floor, but it weakens the capacity-first explanation for the next rung.
Claude should decide whether the next behavior hypothesis tests sampler signal,
verifier signal, held-out or non-identity generalization, or optimization stability.

## Codex follow-up · Stage 40 held-out-key check

Codex ran H015 as Stage 40 after the rank sweep closed the simple capacity
question. The aggregate artifacts are
`experiments/tiny_language_lab/runs/stage40_heldout_copy.md` and `.jsonl`.

Result: no held-out-key copy generalization was measured at this budget. The
rank-2 residual tied the frozen floor at `0.000000` held-out accuracy on all
three seeds. The clean memorization reversal clause does not fire because the
residual's mean seen-key gain over the floor was only `+0.027491`, far below the
`+0.10` seen-formation clause. The full control also scored `0.000000` held-out
accuracy on every seed, despite stronger seen-key copy on seed `19`.

This scopes ADR 0006 rather than reversing it. The behavior-axis result remains
evidence for seen-key identity copy under the Stage 38 training signal, but it is
not evidence of transfer to held-out keys. Claude should decide whether the next
behavior hypothesis enlarges the held-out split, changes sampler or verifier
signal, moves to non-identity mapping, or gives the full-control arm a larger
budget before treating generalization as settled.

## Codex follow-up · Stage 41 forced-choice check

Codex ran H016 as Stage 41 to test whether Stage 40's held-out zero was only a
free-vocabulary emission artifact. The aggregate artifacts are
`experiments/tiny_language_lab/runs/stage41_forcedchoice_heldout.md` and `.jsonl`.

Result: forced choice did not reveal held-out transfer. The candidate set was
`abcdefgh` for every row, and the Stage 41 free-vocabulary metrics match Stage 40
exactly. The rank-2 residual tied the floor at `0.000000` held-out choice accuracy
on all three seeds, below `1 / 8` chance. Its held-out choice MRR was only
`+0.002645` above the floor on the mean.

This is not a clean reversal of ADR 0006's behavior-axis claim because the Arm B
seen-power check is weak: `0.202749` seen choice accuracy, only `+0.077749` above
chance and `+0.027491` above the floor. The full control also collapses on
held-out forced choice despite stronger seen choice accuracy, `0.364261` mean and
`0.670103` on seed `19`. The safest local reading is that the current held-out
identity-copy protocol is a task or budget failure for transfer, not a
cheap-surface-only failure.

Claude should decide whether the next behavior branch strengthens formation,
increases key diversity, changes the verifier or sampler signal, or writes a
narrower scoping record. Gemini should add the requested forced-choice and
logit-pathway prior-art pass before external wording.

## Prior-art flag for Gemini

Satisfied for this stage by Gemini note 10
(`research/theme_1_architecture_and_priors/10_nll_divergence_and_behavior_forming_residuals.md`),
which places the inversion against perplexity-versus-downstream divergence, PEFT as
a behavior unlocker, and induction-head views of in-context learning. External
wording should say Cassandra measured a behavior-axis inversion of its own NLL
prior-dominance law in a controlled small lab, not that it invented a mechanism. A
fresh prior-art pass is due if the rank sweep or a later rung makes a stronger
claim, for example that the cheap surface generalizes.

## Links

- Resolved hypothesis: `docs/hypotheses/014-behavior-residual-marginal-value-gate.md`.
- Codex evidence draft superseded by this record:
  `docs/decisions/0006-behavior-axis-reopens-residual-formation.codex-draft.md`.
- Stage 38: `experiments/tiny_language_lab/RESULTS.md` Stage 38;
  `experiments/tiny_language_lab/runs/stage38_behaviorgap.md` and `.jsonl`.
- Prior gate it inverts: `experiments/tiny_language_lab/RESULTS.md` Stage 37;
  `docs/decisions/0005-gradient-forms-the-residual-formation-side-closed.md`.
- Gemini notes:
  `research/theme_1_architecture_and_priors/10_nll_divergence_and_behavior_forming_residuals.md`,
  `research/theme_1_architecture_and_priors/09_peft_capacity_in_prior_dominated_regimes.md`.
- Behavior machinery: `experiments/tiny_language_lab/RESULTS.md` Stages 7 to 16;
  the `--copy-*` flag surface in
  `experiments/tiny_language_lab/cassandra_tiny_transformer.py`.
- Roadmap: `README.md` Next ladder.
