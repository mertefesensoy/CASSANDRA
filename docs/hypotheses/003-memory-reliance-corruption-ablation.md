# Hypothesis 003 · The retrieval gain is genuine memory use, testable by hint corruption

- Status: measured by Codex, partial memory reliance observed
- Date: 2026-06-16
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 20 (inserted as a validity gate ahead of the capacity sweep and
  the memory-generalization build; see Roadmap note)
- Builds on: Stage 13 (prefix corrections, no retrieval), Stage 14 (held-out
  target retrieval probe), Stage 18 (train-split memory retrieval probe)

## Context and the confound

Stages 14 through 18 report that a compact retrieval hint lifts the
correction-trained full model from `0.881579` copy accuracy (Stage 13, no hint)
to `0.960526` (Stage 14 and Stage 18, with hint). Stage 18 then showed the same
`0.960526` whether the hint came from the held-out target answer or from a
train-split memory table.

Two facts make that gain suspect as evidence of memory use:

1. The copy task is identity: the answer character equals the key character. The
   memory table is therefore the identity map `{a->a, ..., h->h}`, with Stage 18
   reporting 8 entries, 0 conflicts, and 0 probe misses. The lookup is a perfect
   oracle that carries no information beyond "the answer equals the key."
2. The compact hint `key=e answer=e` literally contains the character the probe
   is about to ask for. So a model can reach `0.960526` by reading the answer
   off the hint, with no need to use the probe line's own `key=e` and no need to
   treat the hint as a retrieved memory.

That the number did not move when Stage 18 changed the hint source is consistent
with this reading: in both cases the hint is a faithful copy of the answer, so
the model behaves identically. Five stages of retrieval results rest on this, so
the branch needs a validity gate before any further memory work.

Two competing explanations for the `+0.079` retrieval gain:

- Explanation A (genuine use): the model attends to the retrieved hint and
  follows it. On this identity task that means it copies the hint's answer.
- Explanation B (decorative): the model ignores the hint's answer content and
  solves from the probe line's own `key=`, so the hint's correctness does not
  matter and the gain is some weaker effect such as the hint reinforcing format.

The smallest experiment that separates A from B is a probe-time corruption
ablation: replace the retrieved answer with a wrong but well-formed character and
see whether accuracy collapses (the model was following the hint) or holds (the
model was ignoring it).

Source files: `experiments/tiny_language_lab/runs/stage13_prefix_corrections_500.md`,
`stage14_retrieval_probe_compact.md`,
`stage18_memory_retrieval_probe.md`, and the Stage 13, 14, and 18 entries in
`experiments/tiny_language_lab/RESULTS.md`.

## Hypothesis

The Stage 14 and Stage 18 retrieval gain reflects genuine use of the retrieved
hint (Explanation A). A corrupted hint, in which the retrieved answer character
is replaced by a different valid key character while the probe line is left
unchanged, will reduce full-model copy accuracy materially below the no-hint
correction-only level of `0.881579`, demonstrating that the model follows the
retrieved memory rather than ignoring it.

This is the falsifiable claim. It establishes whether the memory is used at all.
It does not establish that the memory carries non-local information; that is the
separate generalization test on the next rung.

## Expected signal

Three probe conditions for the same correction-trained full model (configs and
training identical; only the probe changes):

| Probe condition | Source | Full-model copy accuracy |
| --- | --- | ---: |
| no hint | none | 0.881579 (Stage 13, in hand) |
| correct hint | train-split memory | 0.960526 (Stage 18, in hand) |
| corrupted hint | memory, wrong answer | to be measured by Codex |

Under Explanation A, corrupted-hint accuracy falls well below `0.881579`, ideally
toward chance `0.125`, because a confident hint-follower is actively misled.
Under Explanation B, corrupted-hint accuracy stays at or above about `0.881579`,
because the model never depended on the hint's answer.

The decisive quantity is the drop from the correct-hint `0.960526` to the
corrupted-hint accuracy. A large drop means high memory reliance. A small drop
means the gain was decorative.

## Baselines and points already in hand

- No-hint correction-only: full `0.881579`, LoRA `0.285088` (Stage 13, prefix
  template).
- Correct compact memory hint: full `0.960526`, LoRA `0.241228` (Stage 18).
- Only the corrupted-hint condition needs a new run. The LoRA path already lost
  accuracy when the correct hint was added (`0.285088` to `0.241228`), which is
  early evidence that LoRA does not use the memory; the corruption test will
  confirm or refute that for both surfaces.

## Primary decision metric and pass or fail line

Metric: full-model copy-probe accuracy under the corrupted hint, mean of seeds
7, 11, 19, with per-seed minimum and maximum. The LoRA path is reported for
completeness but the full model is the decision surface, because it is the one
that showed the retrieval gain.

- PASS, memory is genuinely used (Explanation A): corrupted-hint full-model copy
  accuracy is below `0.881579` by more than the seed range, ideally approaching
  chance. The retrieval branch is validated as real hint-following, and the next
  rung tests whether memory can carry information not present in the local line
  (held-out key to answer mappings).
- KILL, the gain is decorative (Explanation B): corrupted-hint full-model copy
  accuracy is at or above `0.881579` within the seed range, so wrong memory does
  not hurt. The stages 14 to 18 retrieval claim is downgraded in `RESULTS.md`
  from "external memory carries behavior" to "a correct answer hint in context
  helps slightly, but the model does not depend on retrieval," and the
  memory-generalization build is paused until a corpus is designed where the
  answer is not present in the probe line.

A partial drop, below `0.960526` but not below `0.881579`, means the model
partly follows the hint and partly falls back on the in-line key. That is still
informative: it would put a number on how much behavior actually lives in memory
versus weights, which is the project's central question.

## Risks and confounds

- Identity task ceiling. Because the answer equals the key, "genuine use" here
  still means copying the answer from the hint, which is trivial. This stage
  proves use, not value. The value question (does memory carry information the
  local line lacks) is the next rung and needs a non-identity or
  answer-absent corpus. This hypothesis is deliberately scoped to the cheaper,
  prior question.
- Corruption design. The wrong answer must be a valid, in-distribution key
  character so the hint stays well formed; otherwise a drop could be explained by
  the model reacting to malformed text rather than to a wrong answer. Codex
  should map each true answer to a deterministic different key character, for
  example the next key in sorted order with wraparound, seeded for
  reproducibility.
- Probe-set noise. Copy accuracy moves in visible quanta on the small probe set.
  Report per-seed minimum and maximum and judge the drop against the seed range.

## What result would change the plan

- A PASS makes the retrieval claim honest and unlocks the generalization stage
  with confidence that the model attends to memory.
- A KILL is the more valuable outcome for project integrity: it would mean the
  retrieval branch's headline numbers are an in-context-answer artifact, and it
  would redirect effort toward a corpus where external memory must supply
  information the model cannot read locally. Either way the branch becomes more
  honest.

## Codex measurement note

Codex measured this as Stage 19 on 2026-06-16:
`experiments/tiny_language_lab/runs/stage19_memory_corruption_ablation.md`.

Result:

- full model, correct memory: `0.960526` copy accuracy,
- full model, corrupted memory: `0.824561` copy accuracy,
- full-model drop: `0.135965`,
- LoRA rank 2, correct memory: `0.241228`,
- LoRA rank 2, corrupted memory: `0.166666`.

Interpretation for Claude: the result is neither a clean PASS nor a KILL. Wrong
memory hurts, so the hint is behaviorally relevant. But the full model remains
above chance and only moderately below the no-hint Stage 13 value of `0.881579`,
so the model is partly using memory and partly falling back to the local key.
The next memory hypothesis should remove the identity shortcut by using a
non-identity or answer-hidden mapping.

Follow-up: Codex measured the non-identity mapping probe as Stage 21. Correct
compact memory hurt the full model relative to no hint (`0.450216` versus
`0.722944` copy accuracy), while corrupted memory reached `0.398268`. This
means the memory prefix is active but the compact memory interface does not
transfer to non-identity mappings without direct interface training.

## Handoff to Codex (implemented as Codex Stage 19, README ladder rung 20)

Numbering note: the last recorded stage is Codex Stage 18, so this is Codex
Stage 19. In the README ladder it is rung 20. Name the Codex stage number in the
run files, per the project numbering convention.

New knob for Codex to add in `cassandra_tiny_transformer.py` and pass through
`cassandra_compare.py`:

- `--copy-probe-retrieval-corrupt none|wrong-answer` (default `none`, preserving
  current behavior). When `wrong-answer`, the retrieved answer character in the
  probe hint is replaced by a deterministic different valid key character, while
  the probe line itself is unchanged. Reuse the Stage 18 memory machinery
  (`--copy-probe-retrieval-source memory`); corruption acts on the retrieved
  value after lookup. Extend the existing memory-probe reporting so the run
  records how many probes had their hint corrupted.

Required run (corrupted hint):

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt `
  --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 `
  --copy-probe-marker "answer=" --copy-probe-retrieval-template compact `
  --copy-probe-retrieval-source memory --copy-probe-retrieval-corrupt wrong-answer `
  --copy-train-marker "answer=" --copy-loss-weight 10 `
  --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix `
  --seeds 7 11 19 `
  --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix `
  --out .\experiments\tiny_language_lab\runs\stage19_memory_corruption_ablation.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage19_memory_corruption_ablation.md `
  --title "Stage 19 Memory Corruption Ablation"
```

This is two configs times three seeds, about six runs at roughly twenty seconds
each on CPU. The no-hint and correct-hint points are reused from Stage 13 and
Stage 18 and do not need rerunning.

Record in `RESULTS.md` and the run summary, per the Codex evidence standard: the
three-way comparison (no hint, correct hint, corrupted hint) for both configs,
trainable parameter counts, validation NLL, copy accuracy and copy NLL with
per-seed minimum and maximum, the drop from correct to corrupted, and a short
interpretation against the pass or kill line above.

## Prior-art flag for Gemini

This is a faithfulness probe for retrieval-augmented generation, which has
established prior art. Specifically:

- Studies of whether retrieval-augmented models actually use retrieved passages
  versus relying on parametric knowledge, often via counterfactual or corrupted
  context.
- Context-reliance and context-faithfulness metrics, where a model is fed wrong
  context to measure how much it follows it.
- Knowledge-conflict work, where retrieved context contradicts what the model
  would otherwise answer.

Question for Gemini: what is the standard name and method for "feed corrupted
retrieved context and measure the accuracy drop" so Cassandra uses the
established term and cites it, rather than presenting the corruption ablation as
a new technique? See the source anchors in
`docs/LOW_HARDWARE_LM_RESEARCH.md`.

## Links

- Prior hypotheses: `docs/hypotheses/001-staged-correction-retrieval-curriculum.md`
  (resolved), `docs/hypotheses/002-lora-capacity-curriculum-interference.md`
  (open, deferred behind this rung).
- Codex result files this builds on:
  `runs/stage13_prefix_corrections_500.md`,
  `runs/stage14_retrieval_probe_compact.md`,
  `runs/stage18_memory_retrieval_probe.md`.
- Roadmap: `README.md` Next ladder, rung 20.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md` Stage 14 and Stage 18 entries.
- Gemini notes: none yet. Prior-art comparison requested above.
