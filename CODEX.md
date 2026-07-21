# CODEX.md

This file provides guidance to Codex when working in the Cassandra repository.

## Repository Overview

Cassandra is a research workspace with two connected tracks:

1. `AGENT.md` is a working reference for accountable protective AI systems.
2. `experiments/tiny_language_lab/` is the executable laptop-scale language
   model lab.

Most Codex work belongs to the second track. The tiny language lab asks whether
useful language-model behavior can be formed with less brute-force gradient
training by using analytic priors, frozen structure, small trainable surfaces,
verifier-guided correction data, retrieval, and staged curricula.

## Codex Role

Codex owns executable progress for Cassandra.

Codex turns hypotheses into runnable experiments, runs comparisons, verifies the
outputs, and documents every new stage. Codex is the lab engineer and evidence
keeper. A weak, partial, or negative result is still valuable if it narrows the
truth and is recorded honestly.

Codex must not treat implementation as the whole job. A stage is complete only
when the code, run artifacts, verification, and durable documentation all exist.

## The Three-Agent Workflow

Cassandra uses three model-facing role files. Codex must understand them before
moving to a new phase.

The files are not interchangeable. Codex should read them as a coordination
protocol:

- `CLAUDE.md` defines what question should be answered next.
- `GEMINI.md` defines how the question should be placed against outside work.
- `CODEX.md` defines how the question becomes code, runs, artifacts, and
  project evidence.

Codex should never silently replace Claude's hypothesis with a nearby easier
experiment. If local evidence suggests a better stage, Codex may propose or run
the smallest honest variant, but the documentation must say whether it came from
Claude, Gemini, or Codex's own measured result.

### Gemini

Gemini owns research awareness.

Gemini reads the outside world and compares Cassandra against papers, public
repositories, known methods, and technical baselines. Gemini's job is to prevent
Cassandra from experimenting in a vacuum or accidentally overstating novelty.

Gemini outputs:

- research briefs,
- prior-art comparisons,
- source-backed notes,
- suggested baselines,
- citation and vocabulary suggestions,
- warnings when a Cassandra idea already exists under another name.

Codex should treat Gemini's work as world context. If Gemini identifies a public
baseline, Codex should prefer converting it into a local, runnable comparison or
explicitly document why it cannot be tested in the tiny lab.

Gemini's output changes Codex's evidence standard. When Gemini provides a
source-backed note, Codex should look for three things before running or
recording a stage:

- a public baseline that can become a local comparison,
- a vocabulary correction that keeps Cassandra from renaming known work,
- a caveat that limits what a local metric can honestly claim.

If no stage-specific Gemini note exists, Codex may proceed, but must state that
the outside-world comparison is still pending and avoid novelty claims.

### Claude

Claude owns hypotheses, ADRs, and roadmap decisions.

Claude turns the user's broad research goal into crisp questions, staged plans,
and decision records. Claude decides what should be tested next and why, using
Codex results as measured evidence and Gemini notes as outside context.

Claude outputs:

- falsifiable hypotheses,
- ADRs,
- roadmaps,
- experiment priorities,
- expected signals,
- baselines and risks,
- decision points that would change the plan.

Codex should treat Claude's work as planning context. If Claude proposes a
hypothesis, Codex should implement the smallest honest experiment that can test
it, then report back with measured evidence.

Claude's output changes Codex's execution target. A Claude hypothesis is stronger
than a loose README queue because it defines a decision metric, pass line, kill
line, risks, and confounds. When a Claude hypothesis supersedes a previously
queued stage, Codex should follow the hypothesis and explicitly update the
durable docs so the queue does not drift away from the decision record.

### Codex

Codex owns executable evidence.

Codex implements experiments, runs smoke tests and comparison matrices, writes
raw JSONL and summary Markdown under `experiments/tiny_language_lab/runs`, then
updates the durable project documentation.

Codex outputs:

- implemented experiment code,
- reproducible commands,
- smoke-test evidence,
- multi-seed comparison results,
- raw run artifacts,
- concise stage interpretations,
- next-step suggestions grounded in measured behavior.

## How Codex Consumes Claude And Gemini Work

Codex has read `CLAUDE.md` and `GEMINI.md` for this phase. The working
understanding is:

- Claude is the research planner, not a second implementation agent. Claude's
  hypotheses and ADRs define the question, the expected signal, the baseline, the
  pass line, the kill line, and the roadmap consequence. Codex should preserve
  that decision surface when implementing a stage.
- Gemini is the outside-world researcher, not a second roadmap owner. Gemini
  identifies prior art, names public baselines, corrects vocabulary, and warns
  when Cassandra is rediscovering known methods. Codex should turn Gemini's notes
  into runnable comparisons when possible, or record why the comparison remains a
  caveat instead of a local test.
- Codex is the executable evidence layer. Codex decides how to make the smallest
  honest local run that tests the active question, then writes enough artifacts
  for Claude to update the roadmap and Gemini to compare the result with the
  world.

When the roles interact, Codex should use this merge rule:

1. Claude selects the local decision question.
2. Gemini constrains the claims and names outside baselines.
3. Codex implements, runs, verifies, and records the local evidence.
4. If Claude and Gemini point in different directions, Codex should run the
   smallest experiment that preserves Claude's decision metric while satisfying
   Gemini's strongest caveat, or explicitly document why the caveat remains
   pending.

The project should never let a good local number become a vague novelty claim.
After each stage, Codex should leave two handoff hooks:

- For Claude: what decision this result changes, supports, weakens, or kills.
- For Gemini: which public methods, baselines, or terms should be checked before
  the result is described outside this repo.

Current reading of the partner roles:

- Claude has closed the compact text-prefix memory branch through ADR 0001 unless
  a redesigned interface hypothesis appears. Codex should not drift back into
  more compact-memory variants just because they are easy to run.
- Claude's H005 and H006 reframed the bigram recipe as a bounded regime question,
  and ADR 0002 locked it as an early-compute accelerator rather than an
  asymptotic replacement for full training.
- Claude's H007 and H008 opened and then characterized the higher-order
  analytic-prior branch. Stage 28 answered H008 with a fair `V = 8` source-order
  by prior-order surface, and Claude accepted ADR 0003 as the current decision.
  The law is graded: exact source/prior order match is best, severe
  under-specification fails, one-step under-specification can still help, and
  over-specification is penalized by sparsity.
- Claude's active handoff is H009:
  `docs/hypotheses/009-natural-text-finite-order-prior-sweet-spot.md`. H009
  supersedes the older "maybe try natural text" note. It asks whether a frozen
  finite-order count prior plus rank-2 LoRA keeps a durable early-compute edge on
  natural text, where every finite prior is misspecified. The decision metric is
  advantage over `random_full` across prior orders, coverage, and budgets through
  500 steps. The pass line is a coverage-bounded sweet spot; the kill line is no
  durable finite-order advantage.
- Gemini's role is now stage-specific, not only standing context. Gemini note 05,
  `research/theme_1_architecture_and_priors/05_ngram_order_selection_and_bias_variance.md`,
  frames H009 as classical n-gram order selection and bias-variance, not as a
  novelty claim. Codex must report coverage by order, use strong multilevel
  backoff or state the smoother as a limitation, avoid claims of beating
  transformers, and leave a handoff for Gemini to compare any future neural plus
  n-gram residual wording against known interpolation and backoff hybrids.
- Codex's current execution constraint is therefore precise: do not add more
  pure-Markov cells unless testing a named confound, and do not jump to the
  non-gradient residual branch until H009's natural-text external-validity
  measurement has been implemented, run, verified, and documented.

## Required Intake Before a New Phase

Before beginning any new experimental phase, Codex must do a short intake pass:

1. Read or re-read the relevant current project state.
2. Check whether `CLAUDE.md` contains roadmap or hypothesis guidance relevant to
   the next stage.
3. Check whether `docs/hypotheses/` contains a newer open hypothesis that
   supersedes older README queue notes.
4. Check whether `GEMINI.md` changes the evidence standard or suggests outside
   baselines relevant to the next stage.
5. Inspect the latest `experiments/tiny_language_lab/RESULTS.md` section and the
   latest `runs/*.md` summaries.
6. State the exact hypothesis being tested before editing code or running a long
   comparison.
7. If a newer Claude hypothesis conflicts with an older Codex next-step note,
   follow Claude's falsifiable design and record the supersession.
8. If Gemini has no dedicated research note for the stage, keep the claim local:
   "this is a lab-scale result," not "this is novel."

If Claude or Gemini has not yet produced a dedicated note for the next phase,
Codex may continue from the latest measured result, but must say that the stage
is being derived from Codex's current evidence rather than from a new Claude or
Gemini handoff.

### Current Phase Intake: H009

On 2026-06-17 Codex re-read the partner files before moving to Stage 30:

- `CLAUDE.md`: Claude owns hypotheses, ADRs, roadmaps, expected signals, risks,
  and decision consequences. For this phase, Claude's concrete artifact is H009,
  which defines the natural-text sweet-spot claim, pass/partial/kill lines,
  corpus requirements, smoothing risk, coverage diagnostics, and the exact budget
  family Codex should measure.
- `GEMINI.md`: Gemini owns outside-world awareness and evidence standards. For
  this phase, Gemini's concrete artifact is note 05, which says the right public
  framing is n-gram model-order selection, Katz or Kneser-Ney style backoff, and
  bias-variance under sparsity. Gemini's note prevents Codex from treating a good
  local number as novelty.
- `docs/decisions/0003-graded-source-prior-order-law.md`: Claude has accepted
  the graded source/prior order law on controlled Markov sources and selected
  natural-text external validity as the next branch.
- `docs/hypotheses/009-natural-text-finite-order-prior-sweet-spot.md`: at intake,
  the active Codex task was Stage 30. Codex had to build or select a
  deterministic natural-text corpus near a moderate character vocabulary, record
  normalization and split, use multilevel backoff, run `random_full` against
  `count_prior_ng{1,2,3}_lora_r2` across budgets `10, 25, 50, 100, 200, 500` and
  seeds `7, 11, 19`, add order 4 only if memory allowed, then document coverage,
  bits per character, advantage curves, and the pass/partial/kill
  interpretation.

This intake is part of the experiment contract. If the code changes before Stage
30 completes, re-read these files and update this section if the handoff changes.

## Current Coordination State

As of 2026-06-16, Codex has measured Claude's memory-reliance gate, the LoRA
capacity sweep, a Codex-derived non-identity memory mapping probe, Claude
Hypothesis 004's held-out external-memory value test, the smaller rank-2
rehearsal follow-up allowed by ADR 0001, and Claude Hypothesis 005's
corpus-complexity regime sweep, Claude Hypothesis 006's time-budget surface, and
Claude Hypothesis 007's higher-order analytic-prior durability test. Codex has
re-read `CLAUDE.md`, `GEMINI.md`, and this file before moving on to each phase,
per the user's workflow instruction.

Stage 19, from Hypothesis 003, corrupted the train-split memory hint at probe
time. The full model dropped from `0.960526` copy accuracy with correct memory
to `0.824561` with corrupted memory. This shows partial memory reliance, but not
a clean collapse, because the model can still fall back on the local key.

Stage 20, from Hypothesis 002, swept LoRA ranks 4 and 8 for simultaneous versus
staged correction-retrieval curricula. The staged-minus-simultaneous gap moved
from `-0.065790` at rank 2 to `-0.013158` at rank 4, then back to `-0.061404`
at rank 8. Capacity alone is not enough to explain the Stage 17 split.

Gemini's standing research context is not a stage-specific brief yet. Gemini's
file sets the evidence standard: every outside comparison must be source-backed,
must distinguish direct relevance from analogy, and must warn when Cassandra is
rediscovering known methods such as retrieval-augmented generation, kNN language
models, RETRO-style retrieval, LoRA, distillation, verifier-guided training,
context-faithfulness probes, or continual-learning rehearsal under different
names.

Stage 21 built that non-identity memory mapping task. The mapping was
`a->h b->e c->g d->a e->c f->b g->d h->f`. Correct compact memory hurt the
full model relative to no hint: `0.450216` versus `0.722944` copy accuracy.
Corrupted memory reached `0.398268`, showing the memory prefix is behaviorally
active but not useful without interface training. The next memory-stage
candidate was initially non-identity retrieval-interface training with
`retrieval_mixed`.

Claude then wrote Hypothesis 004, "External memory can supply a mapping the
weights never learned," which superseded that simpler all-key
retrieval-interface run. Stage 22 implemented H004 with held-out keys `g h`,
seen keys `abcdef`, and the non-bijective mapping
`a->h b->e c->g d->a e->c f->b g->h h->e`. The train split contained zero
`key=g` or `key=h` examples, while `--copy-probe-memory-scope all` exposed all
eight mappings at probe time.

Stage 22 did not support H4 under the current compact interface. The full model
held-out accuracy was no hint `0.194444`, correct memory `0.166667`, and
corrupted memory `0.166667`. Seen-key accuracy was also only weakly sensitive:
no hint `0.502564`, correct memory `0.502564`, corrupted memory `0.492308`.
This is a local kill for the present `key=x answer=y` fact-substitution
mechanism, not a global proof against retrieval.

Gemini has no stage-specific note for H004 yet; the prior-art framing remains
pending. Codex should keep future H4 claims local until Gemini provides a
source-backed comparison for in-context-versus-parametric knowledge.

ADR 0001 retired the compact text-prefix external-memory interface and redirected
the project toward the core positive claim: characterize where the frozen count
prior plus tiny residual surface beats full training. Codex still measured the
lower-value Stage 20 follow-up as Stage 23. Rank-2 staged rehearsal reached
`0.245614` copy accuracy at rehearsal fraction `0.05`, only slightly above clean
staged `0.236842` and still below simultaneous `0.302632`; rehearsal `0.10`
fell to `0.228070`. This does not rescue the phase switch.

Stage 24 then implemented H005 by mixing the structured corpus generator and the
long-context corpus generator at long fractions `p = 0.00`, `0.25`, `0.50`,
`0.75`, and `1.00`, with `--block-size 96`, sampled evaluation, seeds `7 11 19`,
and the plain language-model configs `random_full`, `count_prior_head`,
`count_prior_lora_r1`, and `count_prior_lora_r2`.

Stage 24 measured the cheap-minus-full advantage as `random_full` mean validation
NLL minus `count_prior_lora_r2` mean validation NLL:

| Long fraction | Count-bigram bits | 50-step advantage | 100-step advantage |
| ---: | ---: | ---: | ---: |
| 0.00 | 2.905610 | 0.059159 | -0.156425 |
| 0.25 | 3.105543 | 0.098079 | -0.101866 |
| 0.50 | 3.030076 | 0.078996 | -0.230854 |
| 0.75 | 2.792105 | -0.066921 | -0.438576 |
| 1.00 | 2.483352 | -0.162385 | -0.642078 |

This supports the cheap recipe as a budget-dependent regime, not as a universal
replacement for full training. At 50 steps, the frozen count prior plus rank-2
LoRA wins through `p = 0.50` and loses at `p = 0.75` and `1.00`. At 100 steps,
the full random transformer wins at every measured point, including all
structured generated data. The count-bigram bits proxy is not enough by itself,
because the all-long corpus has the lowest count-bigram bits but the strongest
full-model advantage.

This led to the time-budget surface around Stage 24: run the same corpus axis at
smaller budgets such as 10 and 25 steps. A new memory attempt should wait for a
redesigned interface hypothesis from Claude or source-backed baseline guidance
from Gemini.

Stage 25 then implemented H006 by running the Stage 24 corpus axis at 10 and 25
steps. The cheap-minus-full advantage was:

| Long fraction | 10-step advantage | 25-step advantage | 50-step advantage | 100-step advantage |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.688454 | 0.247912 | 0.059159 | -0.156425 |
| 0.25 | 0.701242 | 0.278451 | 0.098079 | -0.101866 |
| 0.50 | 0.787087 | 0.299924 | 0.078996 | -0.230854 |
| 0.75 | 0.856700 | 0.247387 | -0.066922 | -0.438575 |
| 1.00 | 0.948164 | 0.241261 | -0.162385 | -0.642078 |

This confirms the surface-shape part of H006: for every measured long fraction,
advantage decreases as steps increase, and the crossover contour moves from
above the measured range at 10 and 25 steps to about `p = 0.635343` at 50 steps
and below the measured range by 100 steps. The mechanism clause is partial
rather than clean: after 10 steps the full random model is already far below
`ln(vocab_size)`, so it is learning quickly, but not quickly enough to catch the
frozen-prior surface.

Claude then accepted ADR 0002, which locked the bigram frozen-prior recipe as a
bounded early-compute inductive-bias accelerator rather than an asymptotic
replacement for full training. ADR 0002 opened the next method branch:
higher-order analytic priors.

Stage 26 implemented H007 with a pure Markov generator and a frozen trigram
residual base. On the pure order-2 source, the matched trigram prior stayed
strongly positive through 200 steps:

| Steps | Bigram advantage | Trigram advantage |
| ---: | ---: | ---: |
| 10 | 0.031391 | 0.661099 |
| 25 | 0.017248 | 0.644896 |
| 50 | 0.010246 | 0.644227 |
| 100 | 0.008005 | 0.637544 |
| 200 | -0.000367 | 0.637384 |

On the pure order-1 control, the matched bigram prior stayed positive at 50 and
100 steps, while the over-specified trigram prior was near tied or slightly
negative. H007 passes: analytic priors are not doomed to be only head starts when
the prior family matches the source. The next strongest executable branch is a
source-order versus prior-order match surface, not more budget sweeps of the old
bigram recipe.

Stage 27 completed that immediate closeout for source orders 1 and 2, prior
orders 1 and 2, and budgets 10, 25, 50, 100, and 200. This was a Codex-derived
completion of the Stage 26 grid at `V = 16`. The matched cells stayed positive
through 200 steps:

| Steps | Order-1 source, bigram prior | Order-2 source, trigram prior |
| ---: | ---: | ---: |
| 10 | 0.213916 | 0.661099 |
| 25 | 0.083939 | 0.644896 |
| 50 | 0.042389 | 0.644227 |
| 100 | 0.029240 | 0.637544 |
| 200 | 0.018835 | 0.637384 |

The mismatched cells did not have the same durability. The order-2 source with a
bigram prior decayed to tied by 200 steps (`-0.000367` advantage), and the
order-1 source with an over-specified trigram prior became negative by 100 and
200 steps (`-0.004205`, then `-0.015480`). The current local rule is therefore:
prior order should match source order; under-specification behaves like a head
start, while over-specification can hurt.

After Stage 27, Claude added H008. That hypothesis supersedes the lighter "add
order-0 or move to natural text" next-step note. The active next phase is the
H008 fair order surface: re-run the surface at `V = 8`, add source order 3 and
prior order 3, implement a general order-n frozen count prior, and decide whether
the matched diagonal remains durable when the high-order count table is
estimable.

Stage 28 implemented H008 with `--residual-base count-ngram`, `--prior-order`,
and `count_prior_ng{1,2,3}_lora_r2` configs. The full `V = 8` surface used source
orders 1, 2, and 3; budgets 10, 25, 50, 100, and 200; seeds 7, 11, and 19; and
the same rank-2 LoRA residual surface for every prior order. Aggregate artifacts:
`experiments/tiny_language_lab/runs/stage28_h008_summary.md` and `.jsonl`.

At 200 steps, the advantage matrix was:

| Source order \ Prior order | k = 1 | k = 2 | k = 3 |
| ---: | ---: | ---: | ---: |
| s = 1 | 0.006804 | 0.002280 | -0.021482 |
| s = 2 | -0.096815 | 0.436673 | 0.405249 |
| s = 3 | -0.000504 | 0.106932 | 0.630598 |

H008 passes on the sharpest cell: `A(3,3)` is strongly positive at 200 steps
with full highest-order coverage. The diagonal is positive and increasing. The
strict surface shape is partial rather than clean because `A(3,2)` remains
meaningfully positive, so one-step-under priors can still harvest useful
lower-order structure. The next Claude/ADR decision should phrase the law as
graded: matching is best, severe under-specification fails, one-step
under-specification can still help, and over-specification carries a sparsity
penalty.

Codex then prepared
`docs/decisions/0003-graded-source-prior-order-law.codex-draft.md` as a proposed
evidence draft for Claude. Claude accepted it with revisions as
`docs/decisions/0003-graded-source-prior-order-law.md`. The accepted ADR states
the graded source/prior order law for synthetic Markov sources and selects
natural-text external validity as the next branch.

Stage 29 ran that Codex-derived exploratory smoke test on
`experiments/tiny_language_lab/corpus/tiny_seed.txt`, a 1,129-character
project-prose corpus. It is not a natural-language claim, but it checks whether
the finite-order prior machinery immediately collapses outside pure Markov
sources. At 100 steps, `count_prior_ng3_lora_r2` beat `random_full` by
`0.399935` NLL, `count_prior_ng2_lora_r2` by `0.350333`, and
`count_prior_ng1_lora_r2` by `0.217754`. The corpus is far too small for a
roadmap decision; ADR 0003 and H009 treat it as a machinery check only.

Gemini then produced note 05 on n-gram order selection and bias-variance, and
Claude wrote H009. That made Stage 30 the next phase: natural-text finite-order
priors with coverage diagnostics and strong backoff. Codex must not describe
Stage 30 as a novelty test. It is a controlled measurement of whether a classical
finite-order statistical prior, frozen under a tiny neural residual, is still an
early-compute accelerator on natural text.

Stage 30 implemented H009 on normalized Tiny Shakespeare. The corpus has
`1,100,721` characters and `V = 33`; the matrix used recursive add-alpha
interpolation with `count_alpha=0.1`, `ngram_backoff=10`, budgets `10`, `25`,
`50`, `100`, `200`, and `500`, seeds `7`, `11`, and `19`, and configs
`random_full` plus `count_prior_ng{1,2,3}_lora_r2`. At 500 steps, ng2 remained
positive with `+0.110081` mean NLL advantage and ng3 remained strongly positive
with `+0.340641`; ng1 became negative at `-0.266069`. Validation hit coverage
was high for the useful priors: `0.999927` for order 2 and `0.995997` for order
3.

The Stage 30 decision is positive for natural-text transfer but incomplete for
the humped sweet-spot law. The measured order curve is monotone increasing
through order 3, so the descending limb is still unmeasured. At Stage 30
closeout, the workflow-safe options were order 4, a harsher corpus or
vocabulary, or a direct comparison against classical neural plus n-gram
interpolation before an ADR upgraded the natural-text claim.

Stage 31 ran H009's optional order-4 extension as a Codex-owned follow-up, not a
new Claude hypothesis. Codex generalized `count-ngram` to order 4 and added
`count_prior_ng4_lora_r2`. On the same normalized Tiny Shakespeare corpus, the
order-4 prior has `42,802,056` frozen logits and was feasible on CPU. It became
the best measured prior at every budget. At 500 steps, ng4 advantage was
`+0.463572`, above ng3 at `+0.340641`, ng2 at `+0.110081`, and ng1 at
`-0.266069`.

Stage 32 measured Claude H009b on a cross-domain validation suffix built from
normalized Cassandra project prose in the same `V = 33` alphabet. Hit coverage
fell but stayed high, with order 4 at `0.889803`, and the advantage curve stayed
monotone through order 4 at 100, 200, and 500 steps. At 500 steps, ng4 advantage
was `+0.291191`, above ng3 at `+0.199130` and ng2 at `+0.016767`; ng1 was
negative at `-0.203702`. This locally kills the predicted moderate-order hump on
the implemented split, while preserving the caveat that the validation source was
project prose rather than a second public-domain author.

Stage 33 measured Claude H010, the mixed prior-loss curriculum filter. The
uniform 200-step target was `2.040189` mean validation NLL. No filtered arm
reached that target earlier than uniform: `f=0.25` reached it only at 200 steps,
`f=0.50` did not reach it through 500 steps, and pure high-loss `f=1.00` was
worse at every budget. H010 is a local kill for this fixed frozen-prior-NLL
hard-example sampler.

Stage 34 measured Claude H011, the dynamic reducible-loss curriculum filter. A
fixed 4096-window pool was re-scored every 25 steps and high-current-loss windows
with positive smoothed deltas were mixed at `f=0.25` and `f=0.50`. Neither
dynamic arm reached the uniform 200-step target at any measured budget, and both
were slower than uniform because of re-scoring overhead. H011 kills the final
data-side selection attempt for the frozen-prior rank-2 residual.

Stage 35 measured Claude H012, the frozen recency base. Codex interpolated the
order-2 count prior with an analytic exponential-recency cache at `tau=96` and
`lambda=0.25`. The recency arm was worse than order-2 count-only at every budget
by `+0.062` to `+0.086` validation NLL, and slower. The order-3 count diagnostic
was far better than order 2 at every budget. H012 kills this default character
cache, but not every model-side frozen primitive. Gemini note 08 now frames this
as a character-level cache failure and points toward order-preserving frozen
kernels or n-gram caches. The remaining later branches are non-gradient residual
formation, a better-grounded model-side frozen primitive, retrieval-interface
redesign, and distilled-data experiments, unless Claude prioritizes one with a new
hypothesis.

## Experiment Responsibilities

Codex should:

- implement experiment code in `experiments/tiny_language_lab`,
- preserve existing deterministic seeds and corpus paths unless the experiment
  explicitly tests a new corpus or seed regime,
- run a small smoke test before any long matrix,
- run comparison matrices with fixed seeds, corpus paths, step counts, and
  evaluation settings,
- launch real experiment runs, smoke tests, and comparison matrices from a
  visible `powershell.exe` terminal so Mert can watch progress live. The Codex
  tool terminal is acceptable for file inspection, AST checks, and short
  non-experiment commands, but not for research runs that produce evidence,
- save raw JSONL and summary Markdown under `experiments/tiny_language_lab/runs`,
- update `README.md`, `experiments/tiny_language_lab/README.md`,
  `experiments/tiny_language_lab/RESULTS.md`, and
  `docs/LOW_HARDWARE_LM_RESEARCH.md`,
- keep failed experiments as evidence,
- keep PowerShell commands reproducible from the repository root.

## Evidence Standard

Every recorded stage must include:

- exact command shape,
- corpus path and split protocol,
- number of seeds,
- trainable parameter counts,
- validation NLL,
- validation bits per character,
- task-specific behavior metrics,
- wall time,
- comparison baseline,
- what the result proves,
- what the result does not prove,
- the next experiment suggested by the evidence.

Codex should not claim a method works because a sample looks nice. For the copy
task, copy accuracy and copy NLL are the behavior metrics. Ordinary validation
NLL is necessary but not sufficient.

## Current Tiny Language Lab Mental Model

Codex should keep this model in mind when extending the lab:

- `cassandra_tiny_lm.py` is the inspectable bigram surface where count,
  coordinate search, and gradient training form the same parameter matrix.
- `cassandra_tiny_transformer.py` is the single-run trainer and source of truth
  for model code, samplers, probes, LoRA/adapters, analytic priors, and reports.
- `cassandra_compare.py` is the multi-seed orchestrator. It imports `train` from
  `cassandra_tiny_transformer.py`, builds named configs, writes JSONL, and
  writes summary Markdown.
- `--init count-bigram` bakes a count prior into trainable weights.
- `--residual-base count-bigram` keeps the count prior frozen outside the
  trainable residual path. This is the stronger Cassandra direction.
- `--train-scope head`, `adapters`, and `lora` test small trainable surfaces on
  top of a frozen prior.
- The long-context corpus uses verified `key=... answer=...` examples to test
  whether behavior changes, not only next-character loss.
- Generated correction examples, retrieval-use examples, simultaneous curricula,
  and staged curricula are all attempts to move behavior formation from broad
  brute-force training into verifier-generated structure and external context.

## Working Style

Codex should prefer small, controlled experiments over dramatic jumps. A stage is
useful when it distinguishes between explanations.

Before moving to a new stage, Codex should ask:

- Which Claude hypothesis, Gemini finding, or prior result motivates this?
- What exact claim is being tested?
- What baseline would make the result honest?
- What metric decides whether the stage helped?
- What artifact will preserve the result for Gemini and Claude?

## Handoff Back To Claude And Gemini

After a stage, Codex should leave enough evidence for the other agents:

- Claude should be able to turn the result into a roadmap decision.
- Gemini should be able to compare the result with outside methods.

That means the final stage interpretation should be specific. Prefer statements
like "late staged retrieval improved full-model copy accuracy from 0.600877 to
0.881579 but still trailed Stage 14 at 0.960526" over vague claims like "staged
training helped."

## Conventions And Gotchas

- Keep `cassandra_compare.py` beside `cassandra_tiny_transformer.py`; the runner
  imports the trainer as a sibling module.
- Matrix configs are CUDA-first. `cassandra_compare.py` defaults to `--device cuda` so the laptop RTX 4070 is used for real matrices. Use `--device cpu` only for tiny diagnostics or when explicitly documenting a CUDA environment blocker.
- CPU and CUDA sampled runs are not bitwise interchangeable. The trainer uses
  `torch.Generator(device=device)`, so same-seed CPU and CUDA rows can sample
  different train/eval windows. Treat CUDA reruns as new measurements unless
  using full evaluation or a future device-independent RNG path.
- `runs/` artifacts are local evidence. The durable record is
  `experiments/tiny_language_lab/RESULTS.md`.
- Seeds and corpus generator seeds are part of the experiment contract.
- Add a config branch in `config_args` and register the config in `--configs`
  choices when adding a named comparison.
- Use AST parsing rather than `py_compile` for syntax checks on this Windows
  setup, because cache writes may fail under OneDrive.
- Project prose should avoid em and en dashes. Prefer commas, periods, or
  restructured sentences.
