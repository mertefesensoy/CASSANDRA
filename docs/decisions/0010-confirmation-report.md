# ADR 0010 Confirmation Report

Date: 2026-07-01

## Verdict

ADR 0010 is directionally right for Phase 2. Cassandra should move to a
from-scratch TinyStories-scale build, keep character-level tokenization first,
preserve the frozen n-gram prior as a controlled ablation, and treat
modded-nanoGPT ideas as architecture and optimizer craft rather than pretrained
model outsourcing.

The repository was not ready to launch that run before this confirmation pass.
At the start of the pass, the trainer was still the Phase 1 small GPT: learned
absolute position embeddings, AdamW only, no RoPE, no Muon, no gradient
accumulation, no activation checkpointing, and no TinyStories ingest path. Stage
44 established the TinyStories bridge baseline. Stage 45 implemented and tested
the main modern-baseline training pieces.

## Changes Made

- Added `experiments/tiny_language_lab/make_tinystories_corpus.py`.
- Added local-source TinyStories prep for `.txt`, `.jsonl`, and `.json` files.
- Added deterministic character normalization, metadata, split recording, and
  optional train/validation text shards.
- Added terminal progress output to `cassandra_tiny_transformer.py`, gated by
  `--log-every`.
- Added matrix-run visibility to `cassandra_compare.py`, including run counts,
  start/done lines, validation bits, trainable parameters, and elapsed seconds.
- Added model-size and training-surface CLI overrides to `cassandra_compare.py`:
  `--batch-size`, `--n-layer`, `--n-head`, `--n-embd`, `--dropout`, `--lr`,
  `--weight-decay`, `--eval-interval`, `--log-every`, `--prompt`,
  `--max-new-tokens`, and `--temperature`.
- Added RoPE positional encoding, gradient accumulation, activation
  checkpointing, and a single-device Muon optimizer path to
  `cassandra_tiny_transformer.py`.
- Added modern-baseline CLI overrides to `cassandra_compare.py`:
  `--grad-accum-steps`, `--pos-encoding`, `--activation-checkpoint`,
  `--optimizer`, Adam beta/epsilon overrides, and Muon hyperparameters.
- Added visible `modern-smoke` and `modern500` modes to
  `run_phase2_visible.ps1`.
- Added `score_generation_samples.py` for deterministic local scoring of saved
  prompt completions across coherence, grammaticality, and on-prompt relevance.
- Added `--train-shard-dir` and `--stream-train-eval-chars` to the trainer and
  compare harness, plus a visible `stream-smoke` launcher mode for plain LM
  shard-backed training.
- Added a visible `modern1000` launcher mode for the first longer-budget
  crossover measurement.
- Added `make_bpe_corpus.py`, `decode_bpe_samples.py`, and visible `bpe-smoke`
  mode for the first BPE-token training and BPE-token n-gram prior smoke.
- Added visible `bpe500` mode for the first multi-seed BPE-token decision
  surface.
- Corrected ADR 0010 bookkeeping: title number, confirmation report link, and
  the live hypothesis placeholder from occupied `H013` to candidate `H019`.

## Ready Now

The repo can now prepare a character-level TinyStories corpus once raw
TinyStories files are locally available. It can run visible CUDA smokes and
visible comparison matrices on that corpus using both the Stage 44 bridge
baseline and the Stage 45 modern baseline.

The modern character baseline now covers the hardware-relevant ADR 0010 pieces:
RoPE, Muon, gradient accumulation, and activation checkpointing. The BPE branch
also has a v256 feasibility smoke and a 500-step multi-seed decision surface.
The current default remains the character-level TinyStories baseline, not a
public TinyStories benchmark.

## Still Pending

- Human review of generated samples before any external quality claim.
- Extending shard-backed training beyond plain LM batches to frozen-prior,
  copy-aware, and curriculum modes.
- A tighter crossover interval than the current 500 to 1000 step bracket.
- Larger-vocab BPE follow-up if the BPE branch is revived beyond the v256 smoke
  and b500 decision surface.

## Launch Gate

For future longer Phase 2 runs, keep this launch gate:

1. TinyStories raw files are present locally.
2. `make_tinystories_corpus.py` has produced `tinystories_char_seed.txt` and
   metadata.
3. A 5 to 20 step visible CUDA smoke passes.
4. The user confirms the full training command.

## Recommended First Matrix

The first confirmed matrix should compare the existing full random baseline
against the best Phase 1 frozen-prior family on the same TinyStories character
corpus:

- `random_full`
- `count_prior_ng3_lora_r2`
- `count_prior_ng4_lora_r2`

This preserves the Phase 1 decision surface while moving only one major dial,
the corpus. The modded-nanoGPT architecture and optimizer pass should follow as
the next implementation stage.

## Execution Update

Codex executed the bridge pass on 2026-07-01 as Stage 44. The run used a bounded
official TinyStories train slice, normalized to
`experiments/tiny_language_lab/corpus/tinystories_char_seed.txt` with
`10,000,001` characters and `V = 33`. Visible CUDA smokes passed through
`run_phase2_visible.ps1`, and the b500 bridge matrix completed across seeds
`7 11 19`.

At 500 steps, `count_prior_ng4_lora_r2` reached `1.139715` mean validation NLL
versus `2.352297` for `random_full`, with `41,249` trainable parameters versus
`3,209,249`. The generated samples are still rough, but the frozen-prior arms
produce recognizable TinyStories-like fragments from `once upon a time ` while
the random baseline remains much noisier.

Conclusion at the end of Stage 44: Phase 2 had a successful bridge result, while
the modern-baseline items still stood.

Codex then executed the modern-baseline pass on 2026-07-01 as Stage 45. The
visible `modern-smoke` passed, and `modern500` completed six CUDA rows across
seeds `7 11 19` for `random_full` and `count_prior_ng4_lora_r2`. The run used
RoPE, Muon, gradient accumulation of `2`, activation checkpointing, and the same
TinyStories character corpus.

At 500 steps, the modern `random_full` baseline reached `1.144942` mean
validation NLL, improving sharply over the Stage 44 bridge `random_full` value
of `2.352297`. The modern `count_prior_ng4_lora_r2` arm reached `1.102748`,
retaining a `0.042194` NLL lead while training `41,249` parameters instead of
`3,176,481`.

Conclusion after Stage 45: Phase 2 had a successful modern character-level
TinyStories baseline. Streaming, BPE, formal generation scoring, and
longer-budget crossover measurement remained open at that point.

Codex also executed the generation-quality scoring pass on 2026-07-01 as Stage
46. The scorer wrote
`experiments/tiny_language_lab/runs/phase2_tinystories_modern_b500_generation_quality.md`
from the Stage 45 JSONL artifact. The deterministic proxy score gives
`count_prior_ng4_lora_r2` a `5.667/6` mean total and `random_full` a `3.000/6`
mean total.

Conclusion after Stage 46: the minimum formal generation-quality scoring sheet
is implemented for saved prompt completions. Streaming, BPE, longer-budget
crossover measurement, and human review remain open.

Codex then executed the shard-consumption smoke on 2026-07-01 as Stage 47. The
visible `stream-smoke` mode completed a 20-step RoPE/Muon CUDA run from the five
TinyStories `train_*.txt` shard files. The JSONL reports `train_chars =
8,500,000`, `train_eval_chars = 200,000`, `val_chars = 1,500,001`, and
`train_shard_files` listing the five shard paths.

Conclusion after Stage 47: the trainer can consume TinyStories train shards for
plain LM batches. BPE, longer-budget crossover measurement, human review, and
making frozen-prior/copy/curriculum modes shard-native remain open.

Codex then executed the 1000-step crossover matrix on 2026-07-01 as Stage 48.
The visible `modern1000` mode completed six CUDA rows across seeds `7 11 19` for
`random_full` and `count_prior_ng4_lora_r2`. At 1000 steps, `random_full`
reached `1.052559` mean validation NLL, while `count_prior_ng4_lora_r2` reached
`1.123161`. The prior-minus-full NLL gap was positive on every seed: `+0.071768`,
`+0.079357`, and `+0.060680`.

Conclusion after Stage 48: the first modern Phase 2 crossover is measured. The
order-4 prior leads at 500 steps and loses by 1000 steps, so the current
character-level crossover interval is between 500 and 1000 steps. BPE, human
review, shard-native frozen-prior/copy/curriculum modes, and a tighter crossover
bracket remain open.

Codex then executed the BPE feasibility smoke on 2026-07-01 as Stage 49. A local
256-token BPE tokenizer encoded `1,000,000` TinyStories characters into
`446,694` BPE tokens. The visible `bpe-smoke` mode completed two CUDA rows:
`random_full` at `3.847587` validation NLL and `count_prior_lora_r2` at
`3.405228`. In this setup, `count_prior_lora_r2` is a frozen BPE-token bigram
prior plus rank-2 LoRA residual. Decoded sample artifacts were written with
`decode_bpe_samples.py`.

Conclusion after Stage 49: BPE tokenization and n-gram priors over BPE tokens are
mechanically live in the harness. The durable BPE-vs-character decision remains
open pending larger, multi-seed BPE runs.

Codex then executed the BPE 500-step matrix on 2026-07-01 as Stage 50. The
visible `bpe500` mode completed six CUDA rows across seeds `7 11 19` for
`random_full` and `count_prior_lora_r2` on the v256 BPE artifact. At 500 steps,
`random_full` reached `2.404960` mean validation NLL, while the BPE-token
bigram-prior LoRA arm reached `3.344760`. Approximate source-normalized bits were
`1.549860` for full BPE and `2.155508` for the prior arm.

Conclusion after Stage 50: the BPE path is implemented and has a multi-seed
decision surface. For the current laptop-scale Phase 2 default, keep the
character-level TinyStories baseline. The v256 BPE bigram-prior branch should
not displace it; larger-vocab or higher-order BPE work is future research, not a
launch blocker.
