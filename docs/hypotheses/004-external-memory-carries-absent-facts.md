# Hypothesis 004 · External memory can supply a mapping the weights never learned

- Status: measured, not supported by the current compact retrieval interface
- Date: 2026-06-16
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 23 (this subsumes and sharpens the queued "non-identity
  retrieval-interface training" rung; see Roadmap note)
- Builds on: Stage 15 (retrieval-use training on the identity task), Stage 19
  (memory corruption ablation, Hypothesis 003), Stage 21 (non-identity memory
  mapping probe)

## Context

Stage 21 removed the identity shortcut that inflated stages 14 to 18. On a fixed
non-identity permutation `a->h b->e c->g d->a e->c f->b g->d h->f`, the
correction-trained full model does better with no hint (`0.722944`) than with a
correct compact memory hint (`0.450216`), and corrupted memory is worse again
(`0.398268`). The retrieval prefix is disruptive once it stops being a restatement
of the answer.

That result is decisive about one thing and silent about another:

- Decisive: the stages 14 to 18 retrieval gains were largely an identity-hint
  artifact, because the compact hint `key=a answer=a` contained the answer.
- Silent: it does not test whether external memory has any genuine value, because
  on this corpus the model already learned the mapping in its weights (0.722944
  with no hint). When the weights already know the answer, memory is redundant at
  best and disruptive at worst. So Stage 21 cannot separate "memory is useless"
  from "memory was unnecessary here."

Codex's queued next stage trains the retrieval interface on the same mapping with
all keys seen, then retests. That is a necessary learnability gate, but it remains
confounded for the value question: even if correct memory then beats no hint, the
model could be answering from weights rather than from memory, because both carry
the mapping.

The project's hypothesis H4 in `docs/LOW_HARDWARE_LM_RESEARCH.md` states the real
claim: retrieval can substitute for memorized facts, freeing parameters from
storing them. The only clean test of H4 is a setting where the weights cannot
possibly know the answer, so any correct answer must come from memory. That means
holding out mappings from training.

Source files: `experiments/tiny_language_lab/runs/stage21_memory_mapping_no_hint.md`,
`stage21_memory_mapping_correct.md`, `stage21_memory_mapping_corrupt.md`,
`stage15_retrieval_use_training.md`, `stage19_memory_corruption_ablation.md`, and
the Stage 15, 19, and 21 entries in `experiments/tiny_language_lab/RESULTS.md`.

## The decisive design: hold out mappings from training

Partition the eight keys into seen keys and held-out keys. Seen-key mappings
appear in training as ordinary `key=X ... answer=Y` lines and in retrieval-use
training examples. Held-out-key mappings never appear in any training line, so the
weights cannot learn them. The external memory table contains all eight mappings.
At probe time the model is asked for held-out keys, and the answer is available
only through the retrieved memory.

Two phases, one corpus, one set of runs:

- Phase 1, learnability gate (this is Codex's queued rung, made rigorous). Train
  with the `retrieval_mixed` sampler and `--copy-verify-mode key-answer` so the
  model learns to read the compact memory interface on seen keys. Confirm on
  seen keys that correct memory is at least as good as no hint, and that
  corrupted memory drops below no hint. If this gate fails, the interface cannot
  be learned at all on a non-identity mapping, and Phase 2 is moot.
- Phase 2, value test (the decisive H4 measurement). On held-out keys, compare no
  hint, correct memory, and corrupted memory. Held-out-key accuracy is the
  decision metric.

## Hypothesis

After retrieval-interface training on seen keys, the model will answer held-out
keys correctly only when correct external memory is supplied. Specifically,
held-out-key copy accuracy with correct memory will be well above chance and well
above the no-hint held-out accuracy, while corrupted memory will push held-out
accuracy back down toward or below chance. This would show that the retrieved
memory carries information the weights never stored, which is H4 at lab scale.

This is the falsifiable claim. It is killed if held-out-key accuracy with correct
memory stays at chance, which would mean the model cannot use memory to answer a
fact it never trained on, even after learning the interface.

## Expected signal

Held-out keys, full model, three conditions:

| Condition | Held-out-key accuracy |
| --- | --- |
| no hint | near chance (`~0.125`), since weights never saw the mapping |
| correct memory | well above chance, ideally above `0.5` |
| corrupted memory | at or below chance, since a confident reader is misled |

Seen keys (Phase 1 gate) should reproduce the qualitative pattern from the
identity task after interface training: correct memory at least matches no hint,
corrupted memory drops below it.

## Baselines and points already in hand

- Stage 21 no-hint full model on the mapping corpus, all keys seen: `0.722944`.
- Stage 21 correct memory, no interface training: `0.450216`. Corrupted:
  `0.398268`.
- Stage 15 identity retrieval-use training full model: `0.776316`, evidence that
  the compact interface is learnable when the task is identity.
- Held-out-key accuracies under all three conditions, Codex Stage 22 full model:
  no hint `0.194444`, correct memory `0.166667`, corrupted memory `0.166667`.
- Chance on eight answer symbols: about `0.125`.

## Primary decision metric and pass or fail line

Metric: held-out-key copy-probe accuracy, mean of seeds 7, 11, 19, with per-seed
minimum and maximum, reported separately from seen-key accuracy. The full model is
the decision surface; the LoRA path is reported for completeness.

- PASS, memory carries absent facts (H4 supported at lab scale): held-out-key
  correct-memory accuracy is above chance by more than the seed range, clearly
  above held-out-key no-hint accuracy, and corrupted memory collapses it. The
  retrieval branch is vindicated: external memory genuinely supplies facts the
  weights lack.
- KILL, memory cannot carry the mapping (H4 not supported here): held-out-key
  correct-memory accuracy is at chance within the seed range, so even a trained
  interface and a correct memory entry do not let the model answer an untrained
  fact. The retrieval branch is then closed as a fact-substitution mechanism for
  this lab, and the research map H4 is downgraded with this evidence.
- GATE FAILURE (inconclusive for H4): if Phase 1 shows the interface cannot be
  learned even on seen keys (correct memory does not reach no hint), report that
  and stop; Phase 2 is not run, and the next pass addresses interface training
  before any held-out claim.

A partial Phase 2 result, correct-memory above chance but below the no-hint
seen-key level, still answers H4 in the affirmative at reduced strength and is
worth recording as the first positive evidence that memory substitutes for
weights in this lab.

## Codex measurement, Stage 22

Codex implemented H004 as Stage 22 on 2026-06-16.

Added code and flags:

- `make_memory_mapping_corpus.py --holdout-keys`, accepting both `gh` and
  `g h`.
- `--copy-probe-memory-scope train|all`, so the probe memory table can include
  mappings absent from the train split.
- `--copy-probe-holdout-keys`, so copy accuracy and copy NLL are reported
  separately for seen-key and held-out-key cases.

The generated corpus used this mapping and partition:

```text
a->h b->e c->g d->a e->c f->b g->h h->e
seen_keys: abcdef
holdout_keys: gh
```

The exact trainer split had zero training cases for `key=g` and `key=h`. The
full memory table still contained all eight entries. In both memory probe
conditions, the table produced 77 hits and 0 misses; the corrupted condition
corrupted all 77 hints.

Full-model result, the decision surface:

| Probe condition | Overall accuracy | Seen-key accuracy | Held-out-key accuracy | Held-out min | Held-out max |
| --- | ---: | ---: | ---: | ---: | ---: |
| no hint | 0.454545 | 0.502564 | 0.194444 | 0.000000 | 0.500000 |
| correct memory | 0.450216 | 0.502564 | 0.166667 | 0.000000 | 0.500000 |
| corrupted memory | 0.441558 | 0.492308 | 0.166667 | 0.000000 | 0.500000 |

LoRA completeness result:

| Probe condition | Overall accuracy | Seen-key accuracy | Held-out-key accuracy |
| --- | ---: | ---: | ---: |
| no hint | 0.155844 | 0.169231 | 0.083333 |
| correct memory | 0.207792 | 0.220513 | 0.138889 |
| corrupted memory | 0.212121 | 0.220513 | 0.166667 |

Decision:

H004 is not supported under this interface. Correct full-corpus memory did not
improve held-out-key accuracy over no hint, and it did not separate from
corrupted memory. The seen-key gate was also weak: correct memory merely matched
no hint, and corruption produced only a small drop. This is best read as a local
kill for the current compact `key=x answer=y` fact-substitution mechanism, not a
global proof against retrieval.

Roadmap consequence:

Do not scale the external-memory table yet. The next memory attempt needs a new
interface hypothesis, likely informed by Gemini's prior-art brief on
in-context-versus-parametric knowledge. Until then, the strongest already
measured follow-up remains the Stage 20 curriculum branch: rank-2 rehearsal
during the retrieval phase.

## Risks and confounds

- Held-out answer coverage. Held-out keys must map to answer symbols that also
  occur as answers for seen keys, so the model is capable of emitting them. With a
  strict permutation each answer appears once, so a held-out key's answer would
  never appear in training and the model could not produce it. Codex should use a
  non-bijective mapping, or otherwise guarantee that every held-out key's answer
  symbol is produced by at least one seen key during training. State the mapping
  and the seen and held-out partitions in the run record.
- Chance baseline. With eight answer symbols chance is about `0.125`, but if the
  model develops an answer-frequency prior the effective no-hint held-out accuracy
  could sit above `0.125`. Reporting no-hint held-out accuracy directly, rather
  than assuming chance, controls for this.
- Small held-out set. Two held-out keys is a small probe. Report per-seed minimum
  and maximum and, if cheap, widen to three held-out keys so the held-out probe
  set is not tiny.
- Gate dependency. Phase 2 is only interpretable if Phase 1 passes. Keep them in
  one corpus and one run so the seen-key gate and the held-out value test use the
  identical trained model.

## What result would change the plan

- A PASS is the most important positive result the project could produce: it would
  be direct laptop-scale evidence for H4, that retrieval substitutes for
  memorized facts. It would justify investing in larger key spaces, multiple
  retrieved candidates, and noisy or conflicting memories.
- A KILL honestly closes the external-memory branch as a fact-substitution
  mechanism and redirects effort to the curriculum and capacity threads, in
  particular the rank-2 rehearsal diagnostic that Stage 20 pointed to.

## Handoff to Codex (next stage; Codex stage number 22, README ladder rung 23)

Numbering note: the last recorded stage is Codex Stage 21, so this is Codex
Stage 22. In the README ladder it is rung 23. Name the Codex stage number in the
run files.

Corpus and code, building on Stage 21 machinery:

- Extend `make_memory_mapping_corpus.py` (or add a sibling generator) to accept a
  held-out key set, for example `--holdout-keys g h`. Held-out keys must never
  appear as a key in any training line, and the mapping must be non-bijective so
  each held-out key's answer symbol is still produced by a seen key. Emit the
  mapping and the seen and held-out partitions into the run record.
- Reuse `--copy-verify-mode key-answer`, the `retrieval_mixed` sampler, the
  compact retrieval template, `--copy-probe-retrieval-source memory`, and
  `--copy-probe-retrieval-corrupt none|wrong-answer` from Stages 18, 19, and 21.
- Add probe reporting that splits copy accuracy into seen-key and held-out-key
  subsets, with per-seed minimum and maximum for each subset.

Phase 1 and Phase 2 are the same trained models probed two ways, so the run set
is: for each of `random_full_retmix` and `count_prior_lora_r2_retmix`, train once
per seed, then probe under three conditions (no hint, correct memory, corrupted
memory), reporting seen-key and held-out-key accuracy separately. Suggested
command skeleton for the correct-memory probe (mirror for no-hint and corrupt):

```powershell
python .\experiments\tiny_language_lab\make_memory_mapping_corpus.py `
  --lines 512 --seed 20260618 --holdout-keys g h `
  --out .\experiments\tiny_language_lab\corpus\memory_mapping_holdout_seed.txt

python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_holdout_seed.txt `
  --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 `
  --copy-probe-marker "answer=" --copy-probe-retrieval-template compact `
  --copy-probe-retrieval-source memory --copy-verify-mode key-answer `
  --copy-train-marker "answer=" --copy-loss-weight 10 `
  --copy-sample-fraction 0.05 --copy-train-retrieval-template compact `
  --seeds 7 11 19 `
  --configs random_full_retmix count_prior_lora_r2_retmix `
  --out .\experiments\tiny_language_lab\runs\stage22_holdout_memory_correct.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage22_holdout_memory_correct.md `
  --title "Stage 22 Holdout Memory Correct"
```

Compute is two configs times three seeds times three probe conditions, all 500
steps on CPU, a handful of minutes. The corpus generator and the seen versus
held-out split reporting are the only new code.

Record in `RESULTS.md` and the run summaries, per the Codex evidence standard:
the mapping and partition, memory coverage on held-out keys, trainable parameter
counts, validation NLL, and copy accuracy and copy NLL split into seen-key and
held-out-key subsets with per-seed minimum and maximum, plus a short
interpretation against the pass, kill, and gate-failure lines above.

## Prior-art flag for Gemini

This is the in-context-versus-parametric-knowledge question, which has substantial
prior art. Specifically:

- Retrieval-augmented generation answering facts that are not in the model's
  parameters, and studies measuring whether the answer came from the retrieved
  passage or from memorized training data.
- Knowledge injection and knowledge editing through context, where a fact unseen
  in training is supplied at inference.
- In-context learning of input-output mappings, where a model applies a rule
  presented only in the prompt.

Question for Gemini: what is the standard framing and metric for "the model must
answer a held-out fact that exists only in retrieved context," so Cassandra cites
the established work and presents Stage 22 as a small-scale instance rather than a
new idea? See the source anchors in `docs/LOW_HARDWARE_LM_RESEARCH.md`, especially
the H4 statement.

## Links

- Prior hypotheses: 001 (resolved), 002 (resolved, capacity not supported), 003
  (resolved, partial reliance).
- Codex result files this builds on:
  `runs/stage21_memory_mapping_no_hint.md`,
  `runs/stage21_memory_mapping_correct.md`,
  `runs/stage21_memory_mapping_corrupt.md`,
  `runs/stage19_memory_corruption_ablation.md`,
  `runs/stage15_retrieval_use_training.md`.
- Roadmap: `README.md` Next ladder, rung 23.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md` hypothesis H4 and the Stage 21
  entry.
- Gemini notes: none yet. Prior-art comparison requested above.
