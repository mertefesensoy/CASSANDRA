# Repository Guidelines

## Project Structure & Module Organization

Cassandra has two tracks. `AGENT.md` is a prose research artifact about accountable protective AI systems. The executable work lives in `experiments/tiny_language_lab/`, with model code in `cassandra_tiny_transformer.py`, the multi-seed runner in `cassandra_compare.py`, corpus generators named `make_*_corpus.py`, generated inputs in `corpus/`, and local run outputs in `runs/`. Durable experiment evidence belongs in `experiments/tiny_language_lab/RESULTS.md`. Planning and decisions live under `docs/hypotheses/` and `docs/decisions/`; outside-work notes live in `research/`.

## Build, Test, and Development Commands

Run commands from the repository root in PowerShell.

```powershell
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --device cuda --steps 20 --eval-interval 10 --max-new-tokens 80
```

Runs the standard CUDA smoke test.

```powershell
python .\experiments\tiny_language_lab\make_long_context_corpus.py --lines 512 --seed 20260617
```

Regenerates the deterministic long-context copy corpus.

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --device cuda --steps 50 --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_lora_r2 --out .\experiments\tiny_language_lab\runs\stageN.jsonl --summary .\experiments\tiny_language_lab\runs\stageN.md --title "Stage N Summary"
```

Runs a comparison matrix and writes JSONL plus Markdown artifacts.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation and type hints where local code already uses them. Keep experiment files beside the runner, since `cassandra_compare.py` imports the trainer as a sibling module. New configs should follow existing names such as `count_prior_lora_r2_copyw_floor`. Project prose should avoid em and en dashes.

## Testing Guidelines

There is no formal pytest suite. Verify Python edits with AST parsing:

```powershell
python -c "import ast, pathlib; ast.parse(pathlib.Path('experiments/tiny_language_lab/cassandra_compare.py').read_text())"
```

Before any full matrix, run a short smoke. Afterward, check JSONL row counts, seeds, configs, summary fields, and stale wording in docs.

## Commit & Pull Request Guidelines

The short history uses concise, stage-oriented messages such as `Hypothesis 9-12 tested`. Prefer a brief imperative or evidence-focused summary, for example `Record Stage 41 forced-choice probe`. PRs should include the hypothesis or ADR link, exact commands run, artifact paths, key metrics, and any known confounds. Include screenshots only for visual artifacts.

## Agent-Specific Instructions

Read `CODEX.md`, `CLAUDE.md`, and `GEMINI.md` before new experimental phases. Claude owns hypotheses and roadmap decisions, Gemini owns prior-art framing, and Codex owns implementation, runs, verification, and durable evidence.
