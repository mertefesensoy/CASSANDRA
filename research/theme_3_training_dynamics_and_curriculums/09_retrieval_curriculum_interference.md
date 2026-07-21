# 3. Difficulty-Aware Retrieval Curriculums

**Relevant Cassandra Stages:** Stages 15-17

## Exhaustive Methodology Review

Retrieval-Augmented Generation (RAG) integrates external knowledge into the generation process. However, a major unsolved problem in RAG systems is **parametric interference**: when a model is trained simultaneously on intrinsic reasoning tasks (relying on its internal weights) and retrieval tasks (relying on external context), the two objectives cannibalize each other. 

Models suffer from "retrieval reliance," where they lose the ability to perform basic reasoning without an external hint, or they suffer from "context ignorance," where they ignore the retrieval hint and hallucinate based on their parametric prior.

### UR²: Unify RAG and Reasoning
*UR² (arXiv:2508.06165)* proposes solving this interference through a "difficulty-aware curriculum" governed by Reinforcement Learning.
1. **Difficulty Probing:** For a given query, the system first attempts to answer it using only its internal parametric memory (zero-shot).
2. **Dynamic Gating:** The query's "difficulty" is measured by the model's confidence or correctness on the zero-shot attempt.
3. **Staged Exposure:** Easy queries are trained without retrieval. Hard queries trigger the retrieval module, and the model is trained on the retrieval-augmented trajectory.

This curriculum prevents the model from treating retrieval as a universal crutch. It forces the internal weights to maintain their baseline capabilities, only relying on the external memory interface when the intrinsic manifold fails.

## Relevance to Cassandra's Tiny Language Lab

**Direct Link to Staged Experiments:**
Cassandra observed severe parametric interference in **Stage 16**. When the model was trained simultaneously on correction examples (intrinsic reasoning) and retrieval examples (external memory), performance collapsed. The model became confused about whether to copy from the local key or the retrieval hint.

In **Stage 17**, Cassandra implemented a "Staged Curriculum," switching from pure correction training to pure retrieval training halfway through the budget (`--copy-curriculum-switch-fraction`). This dramatically improved performance for the full model, perfectly mirroring the literature's claim that separation of these objectives is required.

**The Divergence:**
Cassandra's Stage 17 curriculum is chronologically rigid. It uses a fixed epoch switch (e.g., at 50% or 80% of steps) rather than the dynamic, difficulty-aware gating proposed by UR². Furthermore, Cassandra's LoRA path actually *failed* under the staged curriculum, performing better under simultaneous mixing. This suggests that while large models benefit from staged isolation, extremely bottlenecked residual paths may suffer from catastrophic forgetting during the phase switch.

## Roadmap Impact & Experimental Imperatives

1. **Dynamic Difficulty Gating:** Claude must design an experiment that moves beyond the rigid chronological switch.
2. **Next Necessary Baseline:** Codex must implement an active curriculum sampler.
   - **Mechanism:** During the training loop, the sampler should evaluate the current model's logits on a batch of correction traces. Only if the model's loss exceeds a certain threshold (difficulty gating) should the sampler append a retrieval hint to the training example.
   - **Hypothesis:** A dynamic, error-triggered retrieval curriculum will preserve the intrinsic correction capabilities of both the full model and the LoRA adapter while still teaching the memory interface, outperforming the rigid Stage 17 phase switch.
