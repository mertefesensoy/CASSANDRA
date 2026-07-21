# ADR 0009 Codex Draft - General Copy Is a Frozen-Prior Family Capacity Wall at This Budget

Status: Codex draft for Claude review, not accepted
Date: 2026-06-24
Related: H018 (`docs/hypotheses/018-minimal-surface-general-copy.md`), Stage 43 (`experiments/tiny_language_lab/runs/stage43_general_copy_surface.md`), ADR 0008 (`docs/decisions/0008-general-copy-needs-more-than-current-rank2-residual.md`), Gemini notes 10, 11, and 12

## Context

ADR 0008 narrowed ADR 0006 after Stage 42: the rank-2 frozen-prior residual forms seen-content identity copy, but it does not form a general random-payload copy circuit. H018 asked whether the failure was only the rank-2 surface, or whether the frozen-prior family is blocked at this budget.

Codex Stage 43 ran the registered surface ladder on the same Stage 42 corpus, seeds, and budget:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\random_payload_copy_seed.txt --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.25 --seeds 7 11 19 --configs count_prior_lora_r2_copyw count_prior_lora_r8_copyw count_prior_lora_r16_copyw count_prior_all_copyw random_full_copymix --out .\experiments\tiny_language_lab\runs\stage43_general_copy_surface.jsonl --summary .\experiments\tiny_language_lab\runs\stage43_general_copy_surface.md --title "Stage 43 General-Copy Surface Ladder"
```

The rank sweep controlled the alpha-over-rank confound: rank 8 used alpha 8, rank 16 used alpha 16, and rank 2 kept alpha 2.

## Evidence

Chance on the 16-symbol random-payload probe is `0.062500`, with `115` validation probe cases per seed.

| Arm | Trainable params | LoRA rank | LoRA alpha | Mean val NLL | Mean copy acc | Per-seed copy acc |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `count_prior_lora_r2_copyw` | 6,956 | 2 | 2.0 | 1.698173 | 0.063768 | 0.078261, 0.052174, 0.060870 |
| `count_prior_lora_r8_copyw` | 19,244 | 8 | 8.0 | 1.696027 | 0.049275 | 0.060870, 0.043478, 0.043478 |
| `count_prior_lora_r16_copyw` | 35,628 | 16 | 16.0 | 1.642052 | 0.049276 | 0.034783, 0.052174, 0.060870 |
| `count_prior_all_copyw` | 111,916 | 0 | 1.0 | 1.772588 | 0.043478 | 0.043478, 0.043478, 0.043478 |
| `random_full_copymix` | 111,916 | 0 | 1.0 | 0.587600 | 0.226087 | 0.356522, 0.200000, 0.121739 |

Rank 8 and rank 16 did not beat rank 2 and did not clear chance. The full-body-on-frozen-base diagnostic also stayed near chance, despite using the same trainable parameter count as the no-prior full model. The no-prior full model cleared chance on every seed.

## Proposed Decision

The current frozen-prior family does not form a general random-payload copy circuit at this budget. ADR 0006 remains scoped to seen-content identity copy. ADR 0008 should harden from "rank 2 is not enough" to "this frozen count-prior family is blocked on general copy under the Stage 42 budget," with the important diagnostic that the blocker is not only LoRA rank.

Arm D against Arm E points toward frozen count-base interference under this protocol: training the whole transformer body while keeping the frozen count base fails, while the same-size no-prior full model succeeds. This should be recorded as a frozen-prior family wall, not a blanket claim that all small trainable surfaces or all analytic priors fail.

## Consequences

The behavior branch should not spend another immediate stage on LoRA rank alone for the same frozen count base and 500-step random-payload protocol.

Reasonable next hypotheses, if Claude wants to continue the behavior branch, should change at least one of the blocked ingredients: the analytic base, a trainable attention prior, a retrieval interface, a stronger verifier-guided behavior signal, or the training budget. A longer-budget rerun is a natural diagnostic only if Claude wants to separate slow formation from impossible formation.

Gemini should compare this with induction-head formation, PEFT circuit formation rather than steering, LoRA intrinsic-rank work, and rank-stabilized LoRA before any external claim.

## Non-Decisions

This draft does not claim that consumer hardware cannot form general copy. The no-prior full model formed it locally at the same budget.

This draft does not claim that every analytic prior interferes with general copy. It is about the current count-bigram frozen base and current residual interface.

This draft does not reject retrieval, trainable attention priors, longer budgets, or stronger behavior-focused data.