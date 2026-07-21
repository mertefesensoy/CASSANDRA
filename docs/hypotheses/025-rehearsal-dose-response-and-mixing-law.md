# Hypothesis 025 · A small rehearsal dose keeps the TinyStories register nearly free, or it does not: Stage 58's single 29.8 percent dose point, the replay literature's 5 to 10 percent practice, and a flat-toll alternative make competing predictions, and a proxy-fitted mixing law must publish its 85M prediction before the confirmation arms are paid for

- Status: OPEN. Specced for Codex as Stage 59 (README ladder rung 69).
  Audited 2026-07-21 (hypothesis-auditor pass; all eight required fixes
  applied: closed verdict partition with stated precedence, cross-budget
  anchor disclosure, registered escalation and retention baselines, command
  shapes, provisional Part 1 bound, pinned guard step, risk section).
  Pre-registered decision lines below; step counts inherit Recipe v2 and
  the measured throughput anchor, with a confirm-first smoke before any
  full arm.
- Date: 2026-07-21
- Author: Claude (hypothesis and roadmap role)
- Ladder rung: 69 (Codex stage number 59)
- Builds on: ADR 0016 (Phase 5 closeout; the reopen clauses this stage can
  fire), H024's Stage 58 evidence (the single measured mixture dose point:
  `+0.027977` text8 TEST cost and `2.730217` TinyStories retention gain versus
  COLD at the 29.8 percent dose,
  `runs/stage58_dev_mixture_85m_b42000_seed7_text8_test.json`,
  `runs/stage58_dev_mixture_retention.json`), the existing 20k COLD baselines
  (`runs/stage58_dev_cold_85m_b20000_seed11_text8_test.json` `1.410154`,
  `runs/stage58_dev_cold_85m_b20000_seed19_text8_test.json` `1.410779`),
  the H019 crossover scaling matrix (3.2M to 85M proxy lineage,
  `docs/hypotheses/019-crossover-scaling-law.md`), and Gemini's post-Phase-5
  notes (DoReMi and data mixing laws,
  `research/theme_3_training_dynamics_and_curriculums/17_data_mixing_laws_and_doremi.md`;
  replay buffers,
  `research/theme_3_training_dynamics_and_curriculums/18_catastrophic_forgetting_and_replay_buffers.md`).

## Why this, and why now

Phase 5 closed the ordering question: no acquisition order beats cold broad
training on the primary broad-text metric at fixed compute (ADR 0016, E-null).
But Stage 58 left behind exactly one measured point on a curve nobody has
swept: at a 29.8 percent TinyStories dose, interleaved rehearsal preserved the
narrow register almost perfectly (`0.826285` versus `3.556502` bits/char) for
a `+0.027977` broad-text toll. The continual-learning literature Gemini
surveyed says large labs run rehearsal at 5 to 10 percent, an order of
magnitude below our one measured dose. If the toll shrinks roughly in
proportion to dose while retention degrades slowly, there is a cheap operating
point where a model keeps a second register almost free. If the toll is flat
in dose, rehearsal is never free and Phase 6 should spend nothing more on it.

The second claim makes this stage more than a parameter sweep. The lab's
whole method is proxy scaling (H019 ran 3.2M to 85M). Ye et al.'s data mixing
laws claim small-proxy sweeps predict large-model mixture outcomes. Stage 59
tests that claim in the one place we can afford to check it: fit the law on
3.2M proxies, publish the predicted 85M cost BEFORE the confirmation runs
launch, then pay for exactly two 85M arms and compare.

## Design

Three parts, strictly ordered so the cheap evidence lands first.

### Part 0 · Behavior probe on the broad-trained COLD checkpoint (minutes)

Run the letters probe against the Stage 58 COLD seed-7 final checkpoint:

```powershell
python .\experiments\tiny_language_lab\eval_letters_copy_probe.py `
  --checkpoint C:\cassandra_runs\stage58_dev_cold_checkpoints\stage58_dev_cold_85m_b42000_seed7_random_full_seed7.pt `
  --device cuda `
  --out .\experiments\tiny_language_lab\runs\stage59_cold_letters_probe.json `
  --summary .\experiments\tiny_language_lab\runs\stage59_cold_letters_probe.md
```

(remaining flags at their Phase 5 probe defaults, recorded in the summary).
This executes ADR 0016 reopen clause 3 directly: constrained-choice accuracy
at or above `0.1625` (chance `0.0625` plus `0.10`, the registered threshold in
`runs/phase5_behavior_letters_probe.json`) reopens the behavior axis under a
new hypothesis about diverse-data circuit formation. Below the line, the axis
stays closed and this part is one recorded row. Part 0 has no bearing on
Parts 1 and 2 either way; it runs first only because it is nearly free.

### Part 1 · Proxy mixture sweep and mixing-law fit (the cheap curve)

Train the 3.2M-class Recipe v2 configuration (the H019 size lineage's
smallest rung; note this exact config, with the union 33-character
vocabulary, Muon, fp32, and cosine, has never been trained, so its true
parameter count and throughput belong in the smoke record, not here) on
mixture shards at TinyStories character-fraction `w` in
`{0, 0.05, 0.10, 0.20, 0.30, 0.50}`, seeds `{7, 11, 19}`.

Mixture shards come from the verified Stage 58 generator, one directory per
dose, with these weight pairs:

| `w` | `--tiny-weight` | `--broad-weight` |
| ---: | ---: | ---: |
| 0.05 | 1 | 19 |
| 0.10 | 1 | 9 |
| 0.20 | 1 | 4 |
| 0.30 | 3 | 7 |
| 0.50 | 1 | 1 |

```powershell
python .\experiments\tiny_language_lab\make_mixture_shards.py `
  --tiny-dir .\experiments\tiny_language_lab\corpus\tinystories_char_shards_500mb `
  --broad-dir <the Stage 58 broad shard dir> `
  --tiny-weight <pair> --broad-weight <pair> `
  --total-chars <set by the smoke> `
  --out-dir .\experiments\tiny_language_lab\corpus\stage59_mix_w<w> `
  --metadata-out .\experiments\tiny_language_lab\corpus\stage59_mix_w<w>.meta.json
```

The per-run step budget is set by a confirm-first throughput smoke (target:
the whole 18-run sweep completes inside roughly one GPU-evening; the smoke
sets the exact step count in the launcher log before the sweep starts,
mirroring the Stage 58 budget procedure). The sweep itself runs through
`cassandra_compare.py` with a new registered config branch for the 3.2M
Recipe v2 mixture family (added to `config_args` and the `--configs` choices
per the standing convention), seeds `7 11 19`, writing
`runs/stage59_proxy_sweep.jsonl` and `runs/stage59_proxy_sweep.md`. Every run
records sampled broad validation NLL and sampled TinyStories validation NLL
under the same eval conventions.

Fit the mixing-law functional form from Ye et al. (an exponential or power
law in `w`) to mean broad validation loss across seeds, with a new committed
script `make_mixing_law_fit.py` (new code, lives beside the trainer, writes
`runs/stage59_mixing_law_fit.json` and `.md`). From the fit, publish two
numbers in the fit summary BEFORE Part 2 launches:

1. The predicted 85M broad-text cost at `w = 0.10` relative to `w = 0`.
2. The predicted dose `w*` minimizing broad loss subject to a retention
   bound set at smoke time (provisional until the smoke: the bound is
   registered in the launcher log as the proxy `w = 0.30` arm's measured
   retention plus `0.5` bits/char; it is a Part 1 reporting device only and
   feeds no Part 2 verdict line).

Registering the prediction before the confirmation run is the point: it makes
the transfer claim falsifiable rather than retrofitted. A six-point fit is
under-determined for model selection between exponential and power forms;
the fit script must report both fits and their residuals, and the transfer
read uses whichever the fit summary registered as primary before Part 2.

### Part 2 · Two 85M confirmation arms against paired baselines

Train MIXTURE at `w = 0.10` (ratio 1:9, generated by `make_mixture_shards.py`
with the dose-control arithmetic Stage 58 already verified), 85.11M
parameters, Recipe v2, 20,000 steps, `--checkpoint-every 5000`, seeds 11
and 19, reusing the Stage 58 visible external launcher pattern with the
mixture shard directory swapped in. Two seeds instead of the standard
`7 11 19` is a REGISTERED deviation, justified because the paired text8
baselines exist only for seeds 11 and 19; the escalation rule below covers
the disagreement case with seed 7.

Score each final with the deterministic chunked conventions:

```powershell
python .\experiments\tiny_language_lab\eval_text8.py --split test --device cuda `
  --checkpoint <stage59 mixture final .pt>
python .\experiments\tiny_language_lab\eval_tinystories_retention.py --device cuda `
  --corpus .\experiments\tiny_language_lab\corpus\tinystories_char_shards_500mb\val.txt `
  --checkpoint <stage59 mixture final .pt> `
  --out-stem .\experiments\tiny_language_lab\runs\stage59_mixture_w10_retention
```

The text8 baselines are already on disk (Stage 58 COLD 20,000-step finals,
same seeds, same recipe: `1.410154` seed 11, `1.410779` seed 19). The two
retention baselines do NOT yet exist and are registered here as two cheap
evals of the existing COLD b20000 seed-11 and seed-19 finals under
`C:\cassandra_runs\stage58_dev_cold_checkpoints`, using the same
`eval_tinystories_retention.py` call. The comparison is paired by seed.

One registered indicative pre-step, mid-cosine caveat stated: score the
existing `stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step020000.pt`
on text8 TEST (about three minutes) to get the only obtainable 20k-step toll
reading at the 29.8 percent dose. It is indicative only (a step-20,000
checkpoint of a 42,000-step cosine run is not schedule-matched to a completed
20,000-step run) and feeds no decision line.

Estimated cost from the measured Recipe v2 throughput anchor (5,000 steps in
4,992.0457 seconds, `runs/stage58_dev_throughput_85m_b5000_seed7.jsonl`):
about 5.6 GPU-hours per arm, about 11.2 total, plus the Part 1 evening. All
launches follow the visible-launcher, keep-awake, checkpoint-preflight, and
disk-gate rules already in force (ADR 0012, Stage 58 closeout notes).

## Primary decision metric and pass or fail lines

Primary delta per seed: `d(seed) = MIXTURE_w10 minus COLD` on deterministic
text8 TEST bits/char at 20,000 steps. Retention gain per seed:
`r(seed) = COLD retention minus MIXTURE_w10 retention` on final TinyStories
validation bits/char (positive means the mixture remembers more).

Call a seed CONFIRM-side when `d` is at or below `+0.010` AND `r` is at or
above `1.0`; call a seed KILL-side when `d` is at or above `+0.020`. The
partition, in precedence order:

1. **INCONCLUSIVE (checked first)** = one seed CONFIRM-side and the other
   KILL-side. INCONCLUSIVE takes precedence over E-costly-rehearsal on a
   split. Escalation: run the seed-7 pair (one COLD 20,000-step arm, which
   does not yet exist as a completed schedule-matched run, plus one
   MIXTURE `w = 0.10` 20,000-step arm; about 11.2 additional GPU-hours,
   a registered contingency). Majority of the three seeds then decides,
   recorded as seed-sensitive either way.
2. **E-cheap-rehearsal (CONFIRM)** = both seeds CONFIRM-side. The register
   is kept nearly free: a third or less of the 29.8 percent dose's measured
   toll buys at least a full bit of retention.
3. **E-costly-rehearsal (KILL)** = both seeds KILL-side, or one KILL-side
   while the other is neither CONFIRM-side nor KILL-side. The toll does not
   shrink usefully with dose; rehearsal at this scale is never cheap, and
   Phase 6 spends nothing more on mixture composition for retention's sake.
4. **E-partial (GRADED)** = everything else: both seeds below `+0.020` but
   the CONFIRM conjunction fails on either seed (a `d` inside the
   `+0.010` to `+0.020` band, or an `r` below `1.0`). The curve bent but
   not enough to act on; worth one summary paragraph and no further stage.

This partition is exhaustive: every seed is CONFIRM-side, KILL-side, or
neither, and the four branches cover all nine pairings.

The `+0.010` line is set from measured quantities: it is about one third of
the `+0.027977` toll at the 29.8 percent dose, and three times the largest
same-recipe between-seed replica spread observed on this metric at 20k
(`0.003035`, H022 Stage 56 replicas), so it is neither noise-level nor
generous. The `1.0` bits/char retention line is deliberately far below the
`2.73` measured at the 29.8 percent dose and far above the `0.027` separation
between COLD and CURRICULUM finals, so it cleanly separates "kept a register"
from "kept nothing". One disclosed transfer: both anchors (`+0.027977` and
`2.730217`) are 42,000-step measurements, while Part 2 decides at 20,000
steps, where no toll at the 29.8 percent dose has ever been directly
measured. The transfer is justified because the toll's mechanism (broad-step
displacement by rehearsal characters) scales with the dose fraction, not the
step count, and the registered indicative pre-step in Part 2 gives the best
obtainable 20k reading before the lines are used; if that pre-step reads
wildly off the `+0.028` anchor, the discrepancy is recorded in the launcher
log before the arms start.

Secondary, descriptive (feeds the ADR, cannot fire a verdict): the mixing-law
transfer error, `predicted 85M d at w = 0.10` versus measured mean `d`. The
registered read: transfer is USEFUL if the prediction lands within a factor
of two of the measured mean and on the correct side of the E-costly line;
otherwise proxy mixing laws are recorded as not yet decision-grade at this
scale and DoReMi-style proxy optimization drops out of the Phase 6 toolkit.

## Expected signal, baseline, and honesty constraints

- Baseline: the paired same-seed, same-budget, same-recipe COLD finals named
  above. No new baseline compute is spent.
- Expected if the replay literature transfers: `d` in the `+0.003` to
  `+0.010` region at `w = 0.10` with `r` above `2.0` (retention curves in
  Stage 58 stayed low whenever TinyStories remained in the stream).
- Expected if Stage 58's toll is composition-independent overhead: `d` near
  `+0.028` regardless of dose, firing E-costly-rehearsal.
- Both 85M arms must pass the Stage 58 instability guard analog, with the
  guard step pinned here: with `--checkpoint-every 5000`, each arm's final
  sampled broad validation NLL must improve on its own step-10,000 value,
  else that arm is INCONCLUSIVE and reruns before any verdict is read.
- Failed and negative results are recorded in full, per the standing
  evidence standard.

## Risks

- **Thin seeds.** Two paired seeds decide; a split costs the 11.2 GPU-hour
  seed-7 escalation. Accepted because the paired baselines are free and the
  registered lines sit three replica-spreads above noise.
- **Cross-budget anchors.** The decision lines are derived from 42,000-step
  measurements and applied at 20,000 steps; the disclosed justification and
  the indicative pre-step above bound this, but a genuine budget interaction
  would surface as an E-partial read that is really a measurement artifact.
- **Under-determined fit.** Six dose points cannot separate exponential from
  power-law forms reliably; the fit script must publish both, and the
  transfer read is secondary precisely so a bad fit cannot corrupt the
  primary verdict.
- **Novel proxy config.** The 3.2M Recipe v2 union-vocab configuration has
  never been trained; the confirm-first smoke exists to catch a broken or
  mis-sized proxy before 18 runs are spent on it.

## What result would change the plan

- **E-cheap-rehearsal** hands Phase 6 a free-register recipe knob and makes
  multi-domain retention a live axis (and, per ADR 0016 reopen clause 1, a
  composition that ever BEATS cold broad training by `0.05` reopens the
  ordering axis itself; nothing in this design predicts that, but the sweep
  would surface it).
- **E-costly-rehearsal** closes the rehearsal branch at this scale the same
  way ADR 0016 closed ordering, and Phase 6 concentrates on the recipe and
  capacity frontier (the axis that measurably moved in Phase 5).
- The secondary mixing-law read decides whether proxy-predicted data
  decisions enter the lab's standard toolkit or wait for larger scales.
- Part 0 firing its `0.1625` line reopens the behavior axis under a new
  hypothesis; staying silent leaves it closed with one more honest data
  point.

## Links

- `docs/decisions/0016-phase-5-closeout-developmental-null-recipe-frontier.md`
- `docs/hypotheses/024-developmental-acquisition-order.md` (resolved; the
  design this stage inherits its protocols from)
- `experiments/tiny_language_lab/RESULTS.md` (Stage 56 through 58)
- `research/theme_3_training_dynamics_and_curriculums/17_data_mixing_laws_and_doremi.md`
- `research/theme_3_training_dynamics_and_curriculums/18_catastrophic_forgetting_and_replay_buffers.md`
- `research/theme_2_in_context_learning_and_rag/11_induction_heads_and_zero_shot_failure.md`
  (the axis Part 0 can reopen)
