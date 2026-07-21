# 15. Capacity Bottlenecks in Data Selection

**Relevant Cassandra Stages:** Stage 34 (Hypothesis 011 Kill)

## Exhaustive Methodology Review

In Stage 34, Cassandra tested Hypothesis 011: a dynamic reducible-loss filter designed to overcome the failure of static loss proxies (Stage 33). The dynamic filter periodically re-scored a candidate pool using the live residual's loss trajectory, oversampling windows that were "hard but actively improving" (high NLL, positive delta). Despite matching the theoretical ideal of active learning and reducible-loss data pruning, the dynamic filter failed to beat uniform sampling and was slower due to re-scoring overhead.

### 1. The Breakdown of Data Pruning at Small Scales
- **The Finding in Stage 34:** True iterative data selection provided no convergence advantage over strict uniform random sampling for the rank-2 LoRA residual (~6.2K trainable parameters).
- **Prior Art Context:** While data pruning and dynamic curriculum learning are highly effective for large models (e.g., Sheared LLaMA, DoReMi), the literature indicates that these methods rely on the model having sufficient *capacity* to rapidly assimilate the selected high-value examples. If a model is severely bottlenecked by its parameter count, the benefits of data ordering vanish.

### 2. The LoRA Capacity Bottleneck
- **The Core Mechanic:** A rank-2 LoRA adapter enforces a very strict low-rank subspace on the weight updates. Because the representational capacity is so small, the optimizer essentially finds the single best global low-rank approximation of the corpus.
- **Why it matters:** When the trainable surface is this constrained, the model is "data-order insensitive." It cannot memorize specific hard cases or rapidly adapt to a focused curriculum batch because it lacks the degrees of freedom to do so without unlearning the rest of the corpus. The bottleneck is the architecture, not the data proxy.

### 3. Dynamic Rank Allocation vs. Fixed Rank
- **The Solution in Literature:** Recent research into parameter-efficient fine-tuning (PEFT) and data pruning (e.g., PRILoRA, LoPrune) shows that fixed-rank adapters often struggle with complex data distributions. Successful approaches either:
    1. Align the data pruning metric explicitly with the gradient subspace of the adapter (Trainable Subspace Alignment).
    2. Dynamically increase the rank of the adapter during training to provide capacity exactly when and where the curriculum demands it.

## Relevance to Cassandra's Tiny Language Lab

**Direct Link to Staged Experiments:**
- **Stage 34:** Conclusively proves that the failure of data-side selection for the rank-2 residual is a **capacity problem**, not just a proxy problem. Even the "correct" dynamic proxy could not accelerate training.
- **Stages 11, 12, 33, 34:** This four-stage sequence of failures provides a rigorous, documented proof that curriculum filtering and data pruning are ineffective for extremely low-rank, parameter-constrained residuals on natural text.

## Roadmap Impact & Experimental Imperatives

1. **Retire Data-Side Selection:** The data-side curriculum branch for the frozen-prior rank-2 residual must be retired. No further data-filtering hypotheses should be tested on this specific architecture.
2. **Pivot to Model-Side Architecture:** The roadmap must now turn to the structural, model-side branch (Dr. Akba's time-series/feature-selection priors) to provide a richer frozen base that handles context better than bigram counts.
3. **Future Capacity Tests:** If data selection is ever revisited, it must be combined with a capacity increase—either a much higher rank for the residual, or dynamic rank allocation techniques (like PRILoRA) that allow the model to absorb the curriculum.
