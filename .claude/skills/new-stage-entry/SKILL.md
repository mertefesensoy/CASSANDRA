---
name: new-stage-entry
description: Draft a RESULTS.md stage entry from finished run artifacts (jsonl, run summary, launcher log). Use when a comparison matrix or training run has finished and its results must enter the durable record.
---

# new-stage-entry · write the durable stage record

When to use: a run finished and `runs/*.jsonl` plus `runs/*.md` exist for
it, but `experiments/tiny_language_lab/RESULTS.md` has no entry yet (the
Stop-hook provenance reminder fires exactly on this condition).

## Exact steps, in order

1. Collect the artifacts: the run's `.jsonl` (one report dict per row),
   its generated `.md` summary, the launcher log (holds the exact command
   line), and the hypothesis or ADR the stage implements.
2. Extract, do not retype: command shape (from the launcher log), corpus
   and split (from the `[run]` log line: chars, train, val, vocab), seeds,
   trainable and frozen parameter counts, val NLL and bits/char per seed
   and mean, wall-clock seconds, peak CUDA MiB. Every number must come
   from an artifact; if a number is not in an artifact, it does not go in
   the entry.
3. Write the entry at the END of RESULTS.md as `## Stage N - <Title>`,
   with sections in this order: Date · Handoff (what the stage implements
   and why) · Code change (what was added or modified) · Verification or
   confirm-first audit · Primary command shape (fenced) · Artifacts
   (bulleted paths) · Results (a table) · Decision (which pass or fail
   line fired, in bold) · Interpretation (what it does AND does not
   prove).
4. State the decision against the hypothesis's pre-registered line, first
   match wins. Never invent a new verdict category.
5. Cross-check style: no em or en dashes anywhere (the PostToolUse hook
   flags them); backticked numbers; middle dot where needed.
6. If the stage resolves a hypothesis, note it; the ADR update itself is
   Claude's lane.

## Full example of a good final output (real entry: RESULTS.md Stage 53, abridged only inside bracketed ellipses)

```markdown
## Stage 53 - H020 Frozen Prior Free-Accelerator Test

Date: 2026-07-02

Handoff:

Stage 53 implements ADR 0013 D1 and Hypothesis 020. It tests the missing cell
from Stage 52: a 25.25M transformer with the frozen order-4 count n-gram prior
AND the full body trainable. The question was whether the prior is a free
accelerator under full training, a transient head start, or a late-training
handicap.

Code change:

Codex added the `count_prior_ng4_all` config in `cassandra_compare.py`:
`--residual-base count-ngram --prior-order 4 --train-scope all`, with no LoRA
keys. [...]

Confirm-first audit:

- The prior cache is consulted only when `--prior-cache-dir` is passed. Stage 53
  rows report `prior_cache_status=hit` [...]
- The 10-step CUDA smoke showed identical step-0 sampled validation NLL for
  `count_prior_ng4_lora_r2` and `count_prior_ng4_all`: `1.103458`.

Primary command shape:

    Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase4_visible.ps1`" -Mode stage53"

Artifacts:

- summary: `experiments/tiny_language_lab/runs/stage53_h020_free_accelerator_summary.md`
- main cells: `experiments/tiny_language_lab/runs/stage53_prior_all_25m_b{200,500,1000,2000}.jsonl` and `.md`
- interrupted partial evidence preserved as
  `stage53_prior_all_25m_b2000_interrupted_partial.jsonl` [...]

Stage 53 results:

| Budget | `count_prior_ng4_all` mean val NLL | Stage 52 `random_full` mean | Mean delta | Paired deltas by seed |
| ---: | ---: | ---: | ---: | --- |
| 200 | 1.101573 | 1.353240 | -0.251667 | `7:-0.254035`, `11:-0.244824`, `19:-0.256142` |
| 2000 | 1.022622 | 0.922147 | +0.100475 | `7:+0.101006`, `11:+0.106925`, `19:+0.093493` |

Decision:

Stage 53 is **KILL E-interfere** under H020. The frozen prior is a strong early
accelerator at 200 steps and still positive at 500 steps, but by 1000 and 2000
steps all three paired deltas are worse than `random_full` by more than `0.01`.
The required lower-LR rerun shrank the 2000-step gap but all three paired deltas
remained positive by about `0.05`.

Interpretation:

This result does not say the order-4 prior is useless. It says the current
full-body Muon recipe does not get the prior for free: the additive frozen base
helps early, then becomes a late-training handicap relative to random
initialization. [...] This stage remains a local NLL result, not a coherence or
sample-quality claim.
```

## Mistakes to avoid (each one actually happened here)

- **Do not read decisions off the sampled-16 eval.** The recorded 20k
  "seed spread" (0.579 to 0.609) collapsed to 0.0025 under the chunked
  8M-char eval; ADR 0014 D2 requires the chunked eval for closeout
  claims. Sampled numbers are monitoring only.
- **Do not treat per-invocation `seconds` as total wall clock** when a
  run resumed across legs; say so explicitly (Stage 55 closeout wording).
- **Do not claim coherence or sample quality from NLL.** Every entry ends
  by scoping what it does not prove; human review is a separate gate.
- **Never delete a failed row.** Interrupted or OOM evidence is preserved
  under a suffixed name and cited (Stage 52 OOM, Stage 53 interruption).
- **Em and en dashes are banned** in this repo's docs; use `·` or
  restructure (a hook enforces this).
- **Sleep-inflated timing rows poisoned a throughput estimate once**
  (Stage 52, two rows); flag anomalous timings instead of averaging over
  them.
