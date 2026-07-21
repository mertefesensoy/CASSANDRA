# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read `INSTRUCTIONS.md` first.** It covers how the operator works: stack and canonical paths, tone and writing rules, the weekly tasks and which `.claude/skills/` file covers each, the bar for a good day's output, and the standing hazards. This file covers the codebase itself.

## Repository overview

Cassandra is a research workspace with two connected tracks that share one ethos: keep claims small, falsifiable, and honestly recorded.

1. **`AGENT.md`** is a working reference for accountable protective AI systems, framed through *Person of Interest*. It is a writing/research artifact (a framework, not a system to build), with a research layer mapping how frontier labs actually train and govern models. It is prose, not code.
2. **`experiments/tiny_language_lab/`** is the executable track: a laptop-scale lab asking whether useful language-model behavior can be formed by *reducing* brute-force gradient training (through analytic initialization, frozen priors, small trainable surfaces, verifier-guided data, and retrieval) rather than scaling it. `docs/LOW_HARDWARE_LM_RESEARCH.md` is its research map; `experiments/tiny_language_lab/RESULTS.md` is its staged evidence log.

Most code work happens in track 2.

## Claude's role on the Cassandra team

Cassandra is run by three AI roles, each with its own instruction file. Stay in your lane:

- **Claude (this file)** owns hypothesis design, ADRs, roadmaps, and experiment prioritization. Claude turns broad goals into crisp, falsifiable research questions and decides what should be tested next and why. Claude does not need to run the lab, but must understand it well enough to reason about it.
- **Codex (`CODEX.md`)** owns executable progress: implements experiment code, runs comparison matrices, and records every stage (including failures) in the README/RESULTS docs. Treat Codex's measured results as ground truth for the roadmap.
- **Gemini (`GEMINI.md`)** owns research awareness: prior art, outside baselines, and warnings when Cassandra is rediscovering a known method under a new name.

Every hypothesis or ADR Claude writes should carry: context, the decision/hypothesis, expected signal, baseline, risk, what result would change the plan, and links to Codex result files or Gemini notes when available. Separate dream-level goals ("form behavior cheaply") from stage-level claims ("rank-2 LoRA on a frozen count prior beats X under this budget"). Keep the project ambitious without letting it go vague: the best roadmap item is small enough for Codex to run, meaningful enough to teach something, and connectable to public work by Gemini.

## The tiny language lab: architecture

Understanding this lab requires reading three programs together, because the orchestrator imports the model trainer and reuses its config surface.

**Central thesis under test:** build a cheap analytic prior from corpus statistics, *freeze* it, and train only a tiny residual surface on top. The recurring question for every stage is not sample quality but: did validation loss drop, how many parameters were trainable, how long did it take, and (for the copy task) did task behavior actually change?

### Three programs and the data flow

- **`cassandra_tiny_lm.py`** is the character-level bigram lab. The entire parameter surface is one `vocab_size x vocab_size` logit matrix, which lets the same parameters be formed three ways: `--method count` (build logits directly from smoothed bigram counts, no gradients), `coordinate` (change one parameter at a time, keep it if val loss improves), or `gradient` (AdamW, the control). This is the "make the core question measurable" baseline.
- **`cassandra_tiny_transformer.py`** is the workhorse. Its `train(args)` function builds a small GPT-style causal transformer (`TinyTransformer`), runs the training loop, computes metrics, and returns one flat dict report. Everything else (samplers, verifier, copy probe, LoRA/adapter modules, count-prior initializers) lives here. **This `train` function is the single source of truth** that the orchestrator calls.
- **`cassandra_compare.py`** is the multi-seed comparison matrix. It does `from cassandra_tiny_transformer import train`, defines a set of *named configs* (each a bundle of args), then runs every config × every seed, writing raw `runs/<stage>.jsonl` (one report dict per line) and a `runs/<stage>.md` summary (per-config means across seeds, plus raw rows).

Data flow: `make_synthetic_corpus.py` / `make_long_context_corpus.py` generate deterministic corpora into `corpus/*.txt` → `cassandra_compare.py` trains over them → `runs/*.jsonl` + `runs/*.md` → findings hand-written into `RESULTS.md`.

### The count prior: two distinct mechanisms

The smoothed bigram table (`build_count_logits`: `log(alpha + transition_counts)`) enters the transformer in one of two ways, and the distinction is the heart of the early stages:

- `--init count-bigram` (`apply_count_bigram_init`) bakes the prior *into the weights* (identity-style token embedding, zeroed blocks, output head solved by least squares to reproduce the count logits). Finding: the optimizer treats it as just another trainable surface and overwrites it.
- `--residual-base count-bigram` keeps the prior as a **frozen** `base_logits` table added in the forward pass: `logits = base_logits[idx] + residual_scale * residual_logits`. With `--zero-residual-head` (default on), step 0 output exactly equals the frozen prior, and the prior can never be overwritten. This is the winning direction.

On top of a frozen prior, `--train-scope` chooses how small the trainable surface is: `all`, `head` (output head only), `adapters` (a bottleneck `ResidualAdapter` per block, `--adapter-rank`), or `lora` (low-rank updates on attention/MLP via `LoRALinear`, `--lora-rank`/`--lora-alpha`). The current best low-budget recipe is `--residual-base count-bigram --train-scope lora --lora-rank 2`.

### The copy task, verifier, and behavior probe (stages 7+)

The long-context corpus emits lines like `case 0000 key=a noise=f ...fillers...; answer=a`; the model must copy the key character after `answer=`. This exposes the lab's most important methodological point: **plain validation NLL and task behavior diverge.** A model can drive next-character loss down while copy accuracy stays at chance (about 0.125 for eight keys). So there are two metrics: ordinary `val_nll` and the `copy_probe` accuracy/NLL (the behavior metric).

The stages then build a verifier-guided correction toolkit, all configured through `--copy-*` flags on `train`:
- `--copy-train-marker "answer="` plus `--copy-loss-weight` upweight loss at answer positions.
- `find_verified_copy_targets` is the verifier: it accepts an answer position only when the char after `answer=` genuinely equals the char after `key=`, so training uses verified-correct examples.
- `--copy-sampler` selects the strategy, in roughly the order the stages introduced them: `random` then `answer`/`mixed` (oversample verified windows), then `failed`/`failed_mixed` (`mine_failed_copy_starts` replays cases the current model gets wrong), then `correction`/`correction_mixed` (synthesize compact correction strings for failures), then `retrieval`/`retrieval_mixed` (prepend retrieval context). `--copy-choice-weight` adds an auxiliary loss restricted to the verified candidate set.
- Template flags (`--copy-correction-template`, `--copy-train-retrieval-template`, `--copy-probe-retrieval-template`) choose the shape of synthesized strings: `compact`/`focus`/`prefix`/`full`.

### Reading and extending config names in `cassandra_compare.py`

Config names compose predictably. Base family: `random_full` (full random transformer, all params trainable) versus `count_prior_{head,adapter_r4,lora_r1,lora_r2}` (frozen prior plus small surface). Copy-task variants are suffixes layered onto `random_full` and `count_prior_lora_r2`: `copyw` (weighted), `copys` (answer sampler), `copymix` (mixed), `copyfail`/`copyfailmix` (failed replay), `copycorr`/`copycorrmix` (generated corrections), `retmix` (retrieval-use), `corrretmix` (correction plus retrieval), with `_choice` adding the choice loss. To add an experiment, add a branch in `config_args` and register the name in the `--configs` choices list.

## Commands

GPU-first, run from the repo root in PowerShell. The laptop has an NVIDIA GeForce RTX 4070 visible to the driver, and long comparison matrices should use CUDA. If `torch.__version__` ends in `+cpu` or `torch.cuda.is_available()` is false, install or activate a CUDA-enabled PyTorch build before running real matrices. Use CPU only for tiny diagnostics.

```powershell
# Bigram lab: the three ways to form one logit matrix
python .\experiments\tiny_language_lab\cassandra_tiny_lm.py --method count
python .\experiments\tiny_language_lab\cassandra_tiny_lm.py --method coordinate --steps 300
python .\experiments\tiny_language_lab\cassandra_tiny_lm.py --method gradient --steps 300

# Transformer: fast smoke test (run before any long job)
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --device cuda --steps 20 --eval-interval 10 --max-new-tokens 80

# Transformer: the current best low-budget recipe (frozen count prior + rank-2 LoRA)
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --residual-base count-bigram --train-scope lora --lora-rank 2 --lora-alpha 2 --steps 50

# Regenerate the deterministic corpora (seeds are fixed; output goes under corpus/)
python .\experiments\tiny_language_lab\make_synthetic_corpus.py --lines 320 --seed 20260616
python .\experiments\tiny_language_lab\make_long_context_corpus.py --lines 512 --seed 20260617

# The main workflow: a multi-seed comparison matrix -> runs/*.jsonl + runs/*.md
# `cassandra_compare.py` defaults to `--device cuda`; pass `--device cpu` only for diagnostics.
python .\experiments\tiny_language_lab\cassandra_compare.py --steps 50 --eval-batches 16 --seeds 7 11 19 `
  --configs random_full count_prior_head count_prior_lora_r2 `
  --out .\experiments\tiny_language_lab\runs\stageN.jsonl `
  --summary .\experiments\tiny_language_lab\runs\stageN.md --title "Stage N Summary"
```

`python <script> --help` prints the full flag surface for any of the four entry points. The two README files carry the exact command shapes used for stages 5 through 16.

There is no test suite and no configured linter or formatter (the `.gitignore` reserves `pytest`/`mypy`/`ruff` cache dirs, but none are set up). De facto verification is: run the smoke test, then run the relevant comparison matrix and read the generated `runs/*.md` summary.

## Conventions and gotchas

- **Keep the runner beside the model.** `cassandra_compare.py` imports `cassandra_tiny_transformer` as a sibling module; it only resolves because both files sit in the same directory (which Python puts on `sys.path`). Do not split them apart.
- **CUDA is the matrix default.** `cassandra_compare.py` defaults to `--device cuda`, and named configs inherit that device. `--device cpu` is an explicit diagnostic fallback, not the project default. If CUDA fails while the NVIDIA driver is present, the active Python likely has a CPU-only PyTorch build.
- **CPU and CUDA sampled rows are separate measurement families.** Same-seed
  sampled runs can choose different train and eval windows because the trainer
  uses `torch.Generator(device=device)`. Use `--eval-mode full` or a fresh CUDA
  rerun before treating an old CPU wall-clock or tiny sampled-NLL margin as a
  GPU-era decision.
- **`runs/` is gitignored**, as are `last_logits.pt` / `last_transformer.pt`. Run outputs are local artifacts; the durable record is the hand-written `RESULTS.md`. (This directory is not currently an initialized git repo despite the `.gitignore`.)
- **Determinism is load-bearing.** Seeds (default `7`, matrices use `7 11 19`) and corpus generator seeds are fixed so results replicate. Preserve this when adding experiments.
- **Every stage must beat an honest baseline** and be recorded with command shape, corpus/split, seed count, trainable-parameter count, val NLL plus bits/char, any task metric, and a short interpretation of what it does and does not prove. Failed and negative results are kept as evidence, not deleted.
- **Documentation home:** new findings go into the staged log in `RESULTS.md` and the relevant README, mirroring the existing stage format, rather than into a separate per-change doc.
- **Prose style:** AGENT.md and the project docs avoid em and en dashes, using a middle dot `·` or restructured sentences instead. Match that in any writing you add.
