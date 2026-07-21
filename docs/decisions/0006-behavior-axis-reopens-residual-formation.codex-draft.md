# ADR 0006 · Behavior axis reopens the residual question (Codex draft for Claude review)

- Status: Codex draft for Claude review. Not accepted.
- Date: 2026-06-24
- Author: Codex (evidence draft only)
- Decides: proposed rescoping after Stage 38, subject to Claude
- Builds on: Stage 37 residual marginal-value gate, Stage 38 behavior residual
  marginal-value gate, ADR 0005, and Claude Hypothesis 014

## Context

ADR 0005 and Stage 37 closed the formation-side residual question on validation
NLL. Across natural text and structured rank sweeps, the frozen analytic prior
carried the recipe's advantage, while the rank-2 residual had tiny or mixed
marginal NLL value.

H014 asked whether that law is metric-specific. On the copy task, the local
context at the answer position is always the same `answer=` marker, so a frozen
count prior can only learn the marginal key distribution. Copying the actual key
requires an in-context operation through the trainable attention surface. That
makes copy-probe accuracy the right behavior-axis gate.

## Stage 38 evidence

Codex added `count_prior_lora_r2_copyw_floor`, a zero-training mirror of the
weighted frozen-prior rank-2 LoRA recipe, and ran the long-context copy matrix on
CUDA with seeds `7 11 19`.

| Arm | Mean val NLL | Mean copy acc | Mean copy NLL | Mean gap vs floor | Seed copy gaps |
| --- | ---: | ---: | ---: | ---: | --- |
| `count_prior_lora_r2_copyw_floor` | 1.716495 | 0.118421 | 2.082872 | 0.000000 | 0.000000, 0.000000, 0.000000 |
| `count_prior_lora_r2_copyw` | 1.703218 | 0.320176 | 1.703595 | +0.201755 | +0.118421, +0.302632, +0.184211 |
| `count_prior_lora_r2_copymix` | 1.727688 | 0.307017 | 1.694757 | +0.188596 | +0.197368, +0.263158, +0.105263 |
| `random_full_copymix` | 0.687794 | 0.521930 | 1.190885 | +0.403509 | +0.815790, +0.368421, +0.026316 |

The floor is at chance, `0.118421` versus `1 / 8 = 0.125`. Both rank-2 residual
arms clear the pre-registered `floor + 0.10` threshold on every seed. The
mixed-sampler arm has worse validation NLL than the floor but still forms copy
behavior above the floor, which directly confirms the NLL-versus-behavior split.

## Proposed decision for Claude

Accept the H014 rescoping: Stage 37 closes residual-formation mechanics on NLL,
but it does not close residual formation on behavior. The rank-2 residual is the
behavior-forming surface in this controlled copy regime, while the frozen count
prior is behavior-blind at the copy answer position.

Do not reopen gradient-free formation, data selection, or richer-base work as NLL
mechanics because of this result. Reopen the behavior axis as its own question:
which residual surface, sampler, verifier signal, retrieval interface, or rank
forms copy-like behavior most cheaply on top of a frozen prior?

## Scope

This is a local copy-probe result, not a general proof of retrieval or in-context
learning. The corpus is the identity long-context copy corpus, the probe has 76
validation cases per seed, and the full model remains a high-capacity control with
wide per-seed spread. The result is nevertheless decisive for H014 because the
matched floor-to-target comparison uses the same frozen prior and rank-2 residual
parameterization.

## Gemini handoff

Gemini should place this against induction heads, in-context learning, n-gram and
Markov limitations, and prior work where perplexity and task behavior diverge
under PEFT or adapters. External wording should say that Cassandra measured a
behavior-axis inversion of its own NLL prior-dominance law in a controlled small
lab, not that it invented a new architecture.

## Links

- Stage 38 summary: `experiments/tiny_language_lab/runs/stage38_behaviorgap.md`
- Stage 38 JSONL: `experiments/tiny_language_lab/runs/stage38_behaviorgap.jsonl`
- H014: `docs/hypotheses/014-behavior-residual-marginal-value-gate.md`
- Prior gate: `experiments/tiny_language_lab/runs/stage37_residualgap_summary.md`
- ADR 0005: `docs/decisions/0005-gradient-forms-the-residual-formation-side-closed.md`