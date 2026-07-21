# Hypothesis 022 · The flagship's 2.07-bit specialization gap is a data-distribution effect: the same char recipe trained on broad text closes most of it

- Status: CONFIRMED by Codex Stage 56 on 2026-07-09.
- Date: 2026-07-07
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 64 (Codex stage number 56)
- Builds on: Stage 55 flagship closeout and evaluation (ADR 0014,
  `runs/stage55_validation_suite.md`, `runs/stage55_text8_zero_shot.md`),
  Stage 55 D3a sizing gate (`runs/stage55_size_85m_b256.jsonl`), Stages 49
  and 50 (the tiny-vocab BPE kill this hypothesis deliberately does NOT
  re-test), Gemini notes
  `research/theme_4_domain_specialization_and_substrates/01_bpe_vs_character_level_small_models.md`
  and `02_domain_specialization_gap_and_corpus_breadth.md`.

## Why this, and why now

ADR 0014 measured the flagship's domain-specialization gap: `0.8126`
bits/char in-domain against `2.8817` zero-shot on text8's test split, a
`2.07` bits/char gap, with GPT-2 117M's zero-shot `1.17` as the published
sibling anchor. That gap is now the lab's single largest unexplained
number, and two explanations compete for it. Gemini's theme-4 notes frame
the fork from the literature: narrow-corpus SLM training predicts exactly
this failure mode (Gururangan et al. 2020 by inversion), while the
ByT5/Charformer line argues char-level substrates burn small-model capacity
on word composition and could bottleneck broad-text learning regardless of
data. The user directed this axis on 2026-07-07 after reading both notes.
This stage isolates the DATA explanation while holding the substrate, the
optimizer, and the evaluation constant, because text8's alphabet is a
strict subset of the lab codec and its training split is already on disk
(`corpus/text8/text8`, downloaded and verified by Stage 55's evaluation).
It is the cheapest decisive experiment on the gap: no tokenizer work, one
corpus-prep script, existing trainer flags end to end.

## The mechanism (or competing explanations)

- **E-data (data distribution).** The char recipe is healthy; the gap
  exists because TinyStories is a low-entropy register with no lexical
  breadth. Training the same recipe on broad text (text8's own 90M-char
  train split) produces a model whose text8 test bits/char lands far below
  the flagship's zero-shot `2.8817`, approaching the published range for
  small text8-trained transformers.
- **E-substrate (substrate or recipe limit).** Char-level at block 256
  (and/or the constant-LR Muon recipe tuned on TinyStories) cannot model
  broad text well at this scale even when trained ON it; capacity goes to
  word composition (Gemini note 01). Then even in-domain text8 training
  stalls far above the published range, and the gap was never about data.

## Hypothesis

An 85.11M `random_full` transformer (the D3a-measured `L12 H12 D768`
config), trained with the unchanged Stage 55 recipe on text8's train split
for 50,000 steps (204.8M chars, about 2.28 epochs), reaches **text8 test
bits/char at or below 1.70** (chunked full-test convention, ADR 0014 D3).
Kill direction: if it lands **at or above 2.10** despite in-domain
training, and the LR-sensitivity rerun confirms it, the data explanation
dies and the substrate explanation takes the lead.

## The reference points

No rerun needed for any of these; they are measured or published anchors:

- Flagship 201.6M, TinyStories-trained, text8 test zero-shot: `2.8817`
  bits/char (`runs/stage55_text8_zero_shot.md`).
- Stage 51 25.25M, TinyStories-trained, text8 test zero-shot: `3.3118`.
- GPT-2 117M zero-shot text8: `1.17`; GPT-2 1542M: `0.98` (Radford et al.
  2019, Table 3).
- Dedicated text8-trained models at convergence: roughly `1.0` to `1.2`
  bits/char in the published record (Gemini to sanity-check, see prior-art
  flag).
- 85M throughput and memory at this exact config: `0.524990` s/step,
  `1,799.76` MiB peak CUDA (`runs/stage55_size_85m_b256.jsonl`).

New arms (all `random_full`, block 256, batch 8, grad-accum 2, RoPE,
activation checkpointing, Muon `0.01`, CUDA):

- **Arm A (decides):** seed 7, 50,000 steps on text8 train shards.
- **Arms B and C (stability replicas):** seeds 11 and 19, 20,000 steps,
  mirroring the Stage 55 honest-package precedent.

## Primary decision metric and pass or fail line

Primary metric: **text8 TEST split bits/char** (final 5M chars, never seen
in training or in-run eval), computed by `eval_text8.py`'s chunked
full-split convention on the final checkpoint. In-run sampled eval is
monitoring only (ADR 0014 D2).

- **CONFIRM (E-data)** = Arm A at or below `1.70` bits/char, with Arms B
  and C within `0.10` of each other at 20k (the validation suite showed
  same-budget replicas land `0.0025` NLL apart under tight eval, so a wide
  replica split flags instability, not seed physics).
- **KILL (E-substrate)** = Arm A at or above `2.10` after the
  LR-sensitivity guard: one 20,000-step rerun at `--muon-lr 0.005`
  compared against Arm A's own step-20,000 eval; if the lower-LR arm does
  not beat it by more than `0.05` NLL, the KILL stands as a substrate
  verdict rather than a tuning artifact (same guard pattern as H020).
- **GRADED** = strictly between `1.70` and `2.10`: data breadth helps but
  substrate costs are real; both axes stay live and the BPE-times-broad
  cell becomes the next decider.
- **INCONCLUSIVE guard** = if Arm A's final in-run val NLL is worse than
  its own step-20,000 value by more than `0.05`, training destabilized;
  fix before reading any line.

Threshold anchoring: the distance from the flagship's `2.8817` down to
GPT-2 117M's `1.17` is `1.71` bits. The CONFIRM line at `1.70` requires
closing about 69 percent of that distance; the KILL line at `2.10`
means less than 46 percent closed while training directly on the target
distribution, which no data-side story survives.

Calibration note (Claude, 2026-07-07, interim; Gemini to confirm or
supersede): the prior-art check requested below was run against the
published record before handoff. Converged small char transformers on
text8 sit at `1.08` to `1.23` bits/char (Al-Rfou et al. 2018, 12-layer
44M with auxiliary losses and long training, `~1.11` to `1.18`;
Transformer-XL `1.08`; AWD-LSTM `1.23`). Those are convergence numbers at
budgets far beyond 2.3 epochs, so the original `1.60` CONFIRM line risked
killing a healthy substrate for being merely under-trained. The line was
recalibrated to `1.70` before any run. The KILL line at `2.10` stays: it
is `0.87` bits above the converged record, and no under-training story
explains missing it while training on the target distribution.

## Risks and confounds

- **Multi-epoch training is new.** 50k steps over 90M chars is 2.28
  epochs; every prior lab run was sub-epoch. Repetition can inflate
  train-side metrics; the untouched test split is the defense, and the
  epoch count must be reported.
- **Not compute-matched to the flagship.** 85M at 50k is about 7.3 GPU-h
  versus the flagship's ~26. This stage explains the GAP; it does not
  claim flagship parity. The equal-compute question stays registered in
  ADR 0014 D6.
- **Vocab shrinks to 27 chars** (text8 has no punctuation), so the
  reverse-gap eval (broad model scored on TinyStories) is impossible
  without a vocab-superset flag; that eval is an optional stretch, not a
  decision input.
- **Contamination guard:** the shard builder must slice text8 as train =
  chars `[0, 90M)`, in-run val = `[90M, 95M)`, test = `[95M, 100M)` and
  assert the test range appears nowhere in shards or seed file.
- **LR was tuned on TinyStories.** Broad text is harder; hence the
  explicit LR-sensitivity guard on the KILL line.
- Device caveat: CUDA throughout; the decision eval is deterministic
  chunked scoring of fixed weights, so it carries no sampled-eval noise.

## What result would change the plan

- **CONFIRM** produces an ADR directing the Phase 5 intake toward corpus
  breadth on the char substrate (BPE demoted to an efficiency question,
  per Gemini note 01's compute framing), with the broad-trained 85M as the
  new external-anchor baseline.
- **KILL** produces H023, the real-BPE probe (5k to 10k merges, not Stage
  50's 256), and an ADR rescoping ADR 0011 D1's char-level commitment as
  corpus-bounded.
- **GRADED** schedules the BPE-times-broad cell as the single follow-up
  arm before any ADR.
- Any outcome updates the standing text8 anchor table (ADR 0014 D3).

## Handoff to Codex (implemented as Codex stage 56, README ladder rung 64)

Files to create or modify:

1. `experiments/tiny_language_lab/make_text8_shards.py` (new, small):
   slice `corpus/text8/text8` into `corpus/text8_char_shards/`
   (`train_000NN.txt` files of about 10MB covering chars `[0, 90M)`, plus
   `val.txt` = `[90M, 95M)`), and write
   `corpus/text8_char_seed.txt` = chars `[0, 95M)` for `--corpus` (vocab
   and split source). Assert the test range `[95M, 100M)` is excluded
   everywhere. Deterministic byte slicing; no normalization (text8 is
   already lowercase-plus-space and is a subset of the lab alphabet).
2. `run_phase4_visible.ps1` sibling mode (or a `run_phase5_visible.ps1`)
   adding `stage56-cell` under the ADR 0012 visible protocol.
3. `eval_text8.py`: accept an explicit `--checkpoint` path (or register
   the Stage 56 finals in `flagship_eval_lib.FINAL_CHECKPOINTS`).

Primary command shape (Arm A; Arms B and C repeat with `--seeds 11` or
`--seeds 19` and `--steps 20000`):

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py `
  --corpus .\experiments\tiny_language_lab\corpus\text8_char_seed.txt `
  --device cuda --steps 50000 --eval-interval 5000 --log-every 1000 `
  --seeds 7 --configs random_full `
  --block-size 256 --batch-size 8 --grad-accum-steps 2 `
  --pos-encoding rope --activation-checkpoint `
  --optimizer muon --muon-lr 0.01 `
  --eval-mode sampled --eval-batches 16 --no-copy-train-marker `
  --prompt "the history of " --max-new-tokens 240 `
  --n-layer 12 --n-head 12 --n-embd 768 `
  --train-shard-dir .\experiments\tiny_language_lab\corpus\text8_char_shards `
  --stream-train-eval-chars 200000 `
  --val-fraction 0.05263157894736842 `
  --checkpoint-dir C:\cassandra_runs\stage56_broadchar_checkpoints `
  --checkpoint-every 5000 `
  --out .\experiments\tiny_language_lab\runs\stage56_broadchar_85m_b50000_seed7.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stage56_broadchar_85m_b50000_seed7.md `
  --title "Stage 56 Broad-Corpus Char 85M 50000-step seed 7"
```

Confirm-first items for Codex: (a) `--val-fraction 0.05263157894736842` must
land the internal split at char `90,000,000` of the 95M seed so the in-run val
equals the `[90M, 95M)` slice; the shorter decimal `0.05263158` landed one
character early in preflight. (b) Checkpoints go
directly to `C:\cassandra_runs` (ADR 0014 D1 canonical location; never
OneDrive, never `%TEMP%`). (c) Expected wall-clock from the D3a
measurement is about 7.3 h for Arm A and 2.9 h per replica; report actuals.

The deciding metric, restated: after Arm A completes, run the chunked
full-test text8 evaluation on the final checkpoint. **At or below 1.70
bits/char = CONFIRM E-data. At or above 2.10 (surviving the 20k lower-LR
guard) = KILL, E-substrate takes the lead. Between = GRADED.**

## Prior-art flag for Gemini

This is a domain-shift study inverted from Gururangan et al. 2020 (train
narrow, test broad, then retrain broad) crossed with the char-substrate
capacity question from ByT5 and Charformer (Gemini notes 01 and 02,
theme 4). Before any claim is recorded, Gemini should sanity-check the
pass and kill thresholds against published text8 learning curves at
comparable scale and budget: specifically, what bits/char a roughly 85M
char-level transformer reaches after only about 2.3 epochs (not at
convergence), whether `1.70` is reachable inside that budget in the
published record, and whether any known small-model result contradicts
the `2.10` kill line. If the literature says sub-2.3-epoch text8 training
plateaus above `1.70` even for healthy models, the CONFIRM line must be
recalibrated before Codex runs, not after.

## Result

Codex Stage 56 CONFIRMED H022 on 2026-07-09. Seed 7 at 50,000 steps scored
`1.485740` bits/char on the deterministic chunked text8 TEST split, below the
registered `1.70` CONFIRM line. The 20,000-step replicas scored `1.532627`
bits/char for seed 11 and `1.529591` for seed 19, a `0.003035` spread, below
the `0.10` stability guard. The contamination assertion was present and the
final sampled NLLs did not trip the instability guard. See
`experiments/tiny_language_lab/RESULTS.md`, Stage 56 closeout.

## Links

- `docs/decisions/0014-phase-4-closeout-flagship-verified-evaluation-standard.md`
- `experiments/tiny_language_lab/runs/stage55_text8_zero_shot.md` and `.json`
- `experiments/tiny_language_lab/runs/stage55_validation_suite.md`
- `experiments/tiny_language_lab/runs/stage55_size_85m_b256.jsonl`
- `experiments/tiny_language_lab/RESULTS.md` (Stages 49, 50, 55)
- `research/theme_4_domain_specialization_and_substrates/01_bpe_vs_character_level_small_models.md`
- `research/theme_4_domain_specialization_and_substrates/02_domain_specialization_gap_and_corpus_breadth.md`
- `docs/phase4-flagship-evaluation-report.md` (Sections 5 and 8)
- `README.md` Next ladder rung 64
