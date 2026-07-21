# PEFT Capacity Limits in Prior-Dominated Regimes

## 1. Prior Art and Source Claims

### The Low-Rank Bottleneck and Representational Limits
- **Date Checked**: 2026-06-24
- **Claim**: Low-Rank Adaptation (LoRA) and other Parameter-Efficient Fine-Tuning (PEFT) methods restrict weight updates to a low-dimensional subspace. While this creates extreme memory efficiency, it also introduces a hard "adaptation ceiling" (or capacity bottleneck). If a task requires complex, high-rank shifts that conflict with the pre-trained prior, LoRA may fail to capture these shifts, resulting in a measurable "residual gap" (performance gap) compared to full fine-tuning.
- **Cassandra Relevance**: In Cassandra, the "pre-trained prior" is the frozen, analytic count-based prior. The target is the residual LoRA correction. Stage 37 tests whether a tiny LoRA adapter (rank-1 to rank-4) has the capacity to significantly improve upon a very strong analytic prior.

### Intruder Dimensions and the Stability-Plasticity Dilemma
- **Date Checked**: 2026-06-24
- **Claim**: PEFT acts as a strong regularizer that preserves the original "strong prior." However, when forced to learn distributions that are not perfectly aligned with the prior, LoRA can develop high-rank "intruder dimensions" that attempt to overwrite core knowledge. If the rank is too low to gracefully route around the prior, this leads to interference and regression rather than smooth improvement.
- **Cassandra Relevance**: Stage 37 showed that on natural text (where the n-gram prior is extremely strong but rigid), adding a rank-2 LoRA residual often resulted in *negative* or mixed gaps compared to the frozen floor. The adapter interfered with the strong prior rather than supplementing it.

### Dataset Scaling and the Residual Gap
- **Date Checked**: 2026-06-24
- **Claim**: The gap between PEFT and full fine-tuning is task-dependent. For general tasks well-aligned with the prior, the gap is minimal. But for specialized, idiosyncratic distributions, a low-rank adapter simply lacks the degrees of freedom to adapt. The gap often cannot be closed merely by training longer; it requires either more parameters (higher rank) or a different inductive bias.
- **Cassandra Relevance**: Stage 37's structured rank sweep showed monotonically increasing, but intrinsically tiny gains (rank 1 `+0.010810`, rank 2 `+0.017708`, rank 4 `+0.023058`). The ceiling is hit quickly, and the absolute improvement remains well below the `0.05` NLL target.

## 2. Mapping to Cassandra Stage 37 Findings

Stage 37 tested the "Residual Marginal-Value Gate." The core measurement was the "floor-to-target gap" (`mean validation NLL(floor) - mean validation NLL(AdamW target)`) to determine if tiny residual surfaces provide enough marginal value to warrant further optimizer research.

**Findings:**
- **Natural Text (Strong Prior)**: With high-order frozen priors (ng3, ng4), the gap was mixed or negative. For example, ng4 at 500 steps reached `1.748389` NLL on the floor, but the target with the residual adapter worsened to `1.758796`. 
- **Structured Sweep**: The gap was strictly positive but intrinsically tiny, failing to cross the `0.05` NLL reopening line. Rank 4 provided the highest stable gap at `+0.023058`.
- **Context vs Random Full**: The frozen priors *crushed* full random training. For instance, at 200 natural-text steps, `random_full` was `2.406967`, while the frozen order-4 prior alone was `1.748389`.

**Analysis:**
These results map perfectly to the literature on PEFT capacity constraints under strong priors. 

When an inductive bias (the n-gram prior) is extraordinarily well-matched to the local data distribution, it "dominates" the regime. The residual adapter is left trying to capture the idiosyncratic, long-tail variations that the prior misses. However, because the adapter is low-rank (rank 1 to 4) and heavily constrained, it lacks the representational capacity to model these high-variance outliers. 

Worse, as seen in the natural text results, forcing the adapter to train alongside a dominant prior can cause it to introduce noisy "intruder dimensions" that slightly degrade the overall predictions. The adapter is bounded by an "adaptation ceiling" that it cannot cross simply by training for more steps.

## 3. Roadmap Changes and Experiment Design

- **Hypothesis Update**: The "GATE-CLOSED" result confirms that cheap residual-formation regimes are prior-dominated. Do not spend further engineering effort on residual-formation NLL mechanics (like new optimizers, routing strategies, or initialization tricks for the adapter) unless the adapter size is substantially increased. 
- **Experiment Design**: The value in Cassandra's architecture lies entirely in the *construction of the prior*, not the fine-tuning of the residual. Future stages should shift focus away from residual optimizer mechanics and towards expanding the expressiveness of the analytic prior itself (e.g., retrieval, dynamic n-grams, or structured memory).
