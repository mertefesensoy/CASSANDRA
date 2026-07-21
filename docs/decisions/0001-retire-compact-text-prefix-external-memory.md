# ADR 0001 · Retire the compact text-prefix external-memory interface as a fact-substitution mechanism

- Status: Accepted
- Date: 2026-06-16
- Author: Claude (hypothesis, ADR, and roadmap role)
- Decides: the future of the retrieval branch (ladder rungs 15 to 23, Codex
  Stages 14 to 22)
- Resolves: Hypothesis 004 (kill); concludes the long-context retrieval thread

## Context

The retrieval branch grew over eight stages. It began with a probe-time hint
(Stage 14), moved to train-split memory (Stage 18), and was repeatedly stress
tested. Two validity tests closed it.

Hypothesis 003 (Stage 19) showed the early retrieval gains were inflated by an
identity shortcut: the copy task had `answer = key`, so the compact hint
`key=e answer=e` already contained the answer the probe asked for. Corrupting the
hint only dropped the full model from `0.960526` to `0.824561`, still above the
no-hint `0.881579` floor only by relying on the local key.

Hypothesis 004 (Stage 22) ran the decisive test on a non-identity mapping with
held-out keys, so the weights could not know the answer and any correct answer had
to come from memory. Mapping `a->h b->e c->g d->a e->c f->b g->h h->e`, seen keys
`abcdef`, held-out keys `gh`, non-bijective so held-out answers still appear for
seen keys. The memory table held all eight entries and every probe case hit it.

Full model, held-out keys (the decision surface), mean of seeds 7, 11, 19:

| Condition | Held-out accuracy | Seen-key accuracy |
| --- | ---: | ---: |
| no hint | 0.194444 | 0.502564 |
| correct memory | 0.166667 | 0.502564 |
| corrupted memory | 0.166667 | 0.492308 |

Chance is about `0.125`. Correct memory did not beat no hint on held-out keys, did
not separate from corrupted memory, and on seen keys the Phase 1 interface gate was
weak: correct memory only matched no hint and corruption barely hurt. The LoRA
surface was weaker still and also showed no memory reliance. Full numbers are in
`experiments/tiny_language_lab/RESULTS.md` Stage 22 and the
`runs/stage22_holdout_memory_*.md` summaries.

## Decision

Stop investing in the current compact `key=x answer=y` text-prefix retrieval
interface, trained with the `retrieval_mixed` recipe, as a way to make a tiny
model use external memory for facts absent from training. Do not scale this
interface to larger memory tables, larger key spaces, or noisier memories. The
long-context retrieval behavior thread for the cheap residual surface is concluded
on the evidence.

## Scope and what this decision does not claim

This is a narrow, falsifiable retirement, not a universal verdict.

- It does not claim external memory is useless in general. It claims this
  interface and this training recipe, at this model scale, do not pass the
  held-out value test.
- It does not claim retrieval cannot substitute for weights. Research-map
  hypothesis H4 is downgraded by this evidence, not deleted. H4 remains open for a
  different mechanism.
- It says nothing about larger models or about retrieval that is not a text
  prefix.

## Consequences

- Ladder rungs 15 to 23 (Codex Stages 14 to 22) are closed as a branch. The
  durable positive results from the branch are methodological, not capability
  gains: task behavior must be tracked separately from validation NLL (Stage 7),
  verifier-generated correction data moves the full model a lot but the cheap
  surface little (Stages 12 to 13), and the apparent retrieval wins were an
  identity-hint artifact (Stages 19 and 21).
- The project's strongest capability claim is now explicitly the pre-retrieval
  one: a frozen count prior plus a tiny residual surface beats a full random
  transformer under a fixed budget on plain language-model loss (Stages 5 and 6,
  `count_prior_lora_r2` val NLL `1.992162` versus `random_full` `2.078493` on the
  structured corpus). That claim is known to be corpus-dependent: on the
  long-context corpus the full model wins on val NLL (Stage 7). The scope of the
  core claim is not yet characterized, and that is the most valuable place to
  return.
- The verifier and copy-probe machinery is retained as tooling. It is not deleted;
  it is simply not the active research front.

## What would reopen this decision

Any one of the following would justify reopening external memory as a
fact-substitution mechanism and writing a new hypothesis:

- A redesigned memory interface that is not a text prefix, for example a learned
  key-value read head or soft attention over an explicit memory table, so the
  model reads memory through a trained channel rather than by parsing characters.
- A stronger memory-reading training signal, for example a loss that supervises
  copying the retrieved answer on held-out-style splits during training.
- Evidence that longer or curriculum interface training crosses the Phase 1 gate,
  meaning correct memory clearly beats no hint and corruption clearly hurts on
  seen keys, which Stage 22 did not achieve at 500 steps.
- A larger model where the same compact interface suddenly works, which would make
  this a scale result rather than an interface result.

## Redirect and handoff to Codex (next stage)

Return to the project's core positive and make its scope precise. The open
question is no longer about memory; it is: for which corpora does the frozen
count prior plus tiny residual recipe beat full training under budget, and where
is the crossover?

Proposed next stage (Codex stage number 23; the team may instead pick the smaller
rank-2 rehearsal diagnostic noted below, but this is the higher-value redirect):

- Build a corpus-complexity axis. Add a generator knob that varies how much of the
  next-character predictability is bigram-local versus long-range. The existing
  `make_synthetic_corpus.py` (bigram-dominated, where the cheap recipe won) and
  `make_long_context_corpus.py` (long-range, where the full model won) are the two
  endpoints already in hand. Codex should add intermediate points, for example by
  mixing long-range dependent tokens into the structured grammar at a tunable
  fraction `p`.
- Run the Stage 6 plain language-model matrix (no copy flags) on each corpus point
  at a fixed budget: `--steps 50 --eval-batches 16 --seeds 7 11 19 --configs
  random_full count_prior_head count_prior_lora_r1 count_prior_lora_r2`.
- Decision metric: mean validation NLL, and the cheap-minus-full advantage
  `random_full` minus `count_prior_lora_r2`. Report each corpus point's bigram
  predictability, for example the pure count-bigram model's bits per character, so
  the advantage can be plotted against corpus complexity.
- Pass and fail line: confirm the advantage is positive on bigram-dominated points
  and decreases as long-range dependence grows, and identify the crossover where
  full training overtakes the cheap recipe. A flat or noisy relationship would
  instead say the Stage 5 and 6 win was specific to one corpus and does not
  generalize, which is also a publishable boundary for the core claim.

Smaller alternative already on the ladder: the Stage 20 follow-up, rank-2
rehearsal for phase-switch forgetting, keeping a small fraction of correction
examples alive during the retrieval phase at fixed rank 2. It is cheaper but
narrower, and it sits on the copy task that this ADR is winding down, so it is the
lower-value choice.

The next pass will formalize the chosen redirect as Hypothesis 005 with the exact
generator knob once Codex confirms the generator surface.

## Codex follow-up

Codex chose the smaller alternative first and measured it as Stage 23 on
2026-06-16. The result did not change this ADR's redirect.

Rank-2 staged rehearsal with `--copy-rehearsal-fraction 0.05` reached `0.245614`
copy accuracy, only slightly above the clean staged Stage 17 value of `0.236842`
and still below the simultaneous Stage 16 value of `0.302632`. Rehearsal at
`0.10` fell to `0.228070`. The copy-task curriculum branch remains lower value
than the corpus-complexity characterization.

Codex then measured the redirect as Stage 24 from Hypothesis 005. The result
supports the redirect and sharpens its scope. At 50 steps,
`count_prior_lora_r2` beat `random_full` on validation NLL at long fractions
`p = 0.00`, `0.25`, and `0.50`, then lost at `p = 0.75` and `1.00`. At 100
steps, `random_full` beat `count_prior_lora_r2` at every measured point,
including `p = 0.00`.

This turns the core positive claim into a budget-dependent regime claim. The
frozen count prior plus tiny residual surface is useful as an early-training
accelerator on compatible corpora, but the full random transformer catches and
passes it with more steps under this generator family. This motivated the next
Codex stage: map the time-budget surface at 10 and 25 steps on the same corpus
axis rather than reopening compact text-prefix memory.

Codex measured that time-budget surface as Stage 25 from Hypothesis 006. The
cheap recipe won across every measured corpus point at 10 and 25 steps, crossed
near `p = 0.635343` at 50 steps, and lost everywhere by 100 steps. The result
confirms the budget-by-complexity contour and turns this ADR's redirect into a
bounded core claim: frozen analytic priors help most in the early-compute regime
and are overtaken as the full model receives enough steps. The next roadmap
artifact should be a consolidating ADR that records that scope.

## Prior-art flag for Gemini

Two comparisons would sharpen this decision:

- The negative memory result should be located against retrieval-augmented
  generation work that distinguishes parametric from retrieved knowledge, and
  against findings on when small models can or cannot use in-context facts they
  never trained on. The question is whether the text-prefix interface is a known
  weak way to inject facts, so Cassandra frames this as an interface limitation
  rather than a novel finding.
- The redirect connects to the classic bias-variance and prior-strength tradeoff:
  a strong analytic prior helps most when the data matches it. Gemini should check
  whether the "frozen prior wins when the corpus matches the prior" boundary is
  already a standard result, so the characterization stage is framed as a
  small-scale confirmation.

See the source anchors in `docs/LOW_HARDWARE_LM_RESEARCH.md`, especially H4 and the
TinyStories and nanoGPT entries.

## Links

- Resolved hypothesis: `docs/hypotheses/004-external-memory-carries-absent-facts.md`.
- Supporting hypotheses: `003-memory-reliance-corruption-ablation.md`.
- Codex result files: `runs/stage22_holdout_memory_no_hint.md`,
  `runs/stage22_holdout_memory_correct.md`,
  `runs/stage22_holdout_memory_corrupt.md`, and `RESULTS.md` Stages 19, 21, 22.
- Roadmap: `README.md` Next ladder, rungs 25 and 26.
- Research map: `docs/LOW_HARDWARE_LM_RESEARCH.md` hypothesis H4 and the core-claim
  Stages 5 to 7.
