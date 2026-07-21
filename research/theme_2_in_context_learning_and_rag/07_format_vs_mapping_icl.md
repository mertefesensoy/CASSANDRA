# 6. In-Context Learning: Input-Label Mapping vs. Format Recognition

**Relevant Cassandra Stages:** Stages 19-21

## Exhaustive Methodology Review

In-Context Learning (ICL) is the ability of a language model to perform a task simply by being shown examples in the prompt, without gradient updates. But a fierce debate exists over *how* ICL works. Does the model actively learn the "input-to-output mapping" shown in the prompt, or does the prompt merely serve as a format trigger that activates pre-existing knowledge from pre-training?

### Rethinking the Role of Demonstrations
*Min et al. (2022, arXiv:2202.12837)* designed a definitive experiment to answer this. 
1. **The Setup:** They took standard few-shot prompts (e.g., Sentiment Analysis: `Input: I love this movie. Label: Positive`) and systematically replaced the ground-truth labels with *random* or *incorrect* labels (e.g., `Input: I love this movie. Label: Negative`).
2. **The Finding:** Counter-intuitively, performance barely dropped. 
3. **The Conclusion:** The models were not learning the semantic mapping from the prompt. They were using the demonstrations solely to recognize the *task format* and the *label space* (the fact that the answer should be the word Positive or Negative). The actual decision boundary was drawn entirely from parametric memory.

### Larger Language Models Do ICL Differently
*Wei et al. (2023, arXiv:2303.03846)* expanded on this by introducing Semantically-Unrelated Label In-Context Learning (SUL-ICL).
1. **The Setup:** They forced the model to map inputs to completely arbitrary labels (e.g., `Input: I love this movie. Label: foo. Input: I hated it. Label: bar`).
2. **The Finding:** Small models completely failed at this task. They clung to their semantic priors and could not learn the non-identity mapping. Only massive models (emergent scale) had the capacity to override their pre-training priors and actively "learn" the arbitrary mapping entirely in-context.

## Relevance to Cassandra's Tiny Language Lab

**Direct Link to Staged Experiments:**
Cassandra perfectly replicated these massive-scale findings at the laptop scale in **Stages 19 and 21**.
- **Stage 19 (Corruption):** When the memory hint was corrupted, the model fell back on its local key prior. It was not fully bound to the mapping in the hint.
- **Stage 21 (Non-Identity Mapping):** This is the direct equivalent of SUL-ICL. Codex introduced a non-identity mapping (`a->h`, `b->e`). The full model, which had previously scored 96% accuracy with correct memory, plummeted to 45% (worse than having no hint at all). The rank-2 LoRA path remained at near-chance levels. 
- **The Mechanical Proof:** The model's success in Stage 18 (`key=a answer=a`) was purely *Format Recognition*. It recognized the structure of the correction example and fell back on its count-based prior. When forced to learn a true *Input-Label Mapping* (`a->h`) via context alone, it completely failed, exactly as *Min et al.* and *Wei et al.* predicted for models lacking emergent capacity.

## Roadmap Impact & Experimental Imperatives

1. **Abandoning Zero-Shot Memory:** Cassandra has proven that its tiny parameter surfaces cannot perform true ICL input-label mapping out-of-the-box. We cannot expect the model to use an external memory interface correctly via prompt injection alone.
2. **Next Necessary Baseline:** The model must be explicitly trained on the memory interface.
   - **Execution:** As Codex concluded, Claude must schedule Stage 22 to train the model directly on the `memory_mapping_seed.txt` corpus using the `retrieval_mixed` sampler. We must force the gradient updates to unlearn the local-key prior and wire the attention heads specifically to the retrieval hint's mapping.
