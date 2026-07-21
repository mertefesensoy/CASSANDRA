# 7. RAG and Out-of-Distribution (OOD) Generalization

**Relevant Cassandra Stages:** Stage 22

## Exhaustive Methodology Review

A core promise of Retrieval-Augmented Generation (RAG) is that a model can answer questions about facts it has *never seen during training*, simply by reading them from the retrieved context window. However, recent literature reveals a severe limitation: models often fail to generalize their in-context learning (ICL) capabilities to Out-of-Distribution (OOD) or unseen mappings.

### Can In-context Learning Really Generalize to Out-of-distribution Tasks?
*Wang et al. (2024, arXiv:2410.09695)* investigated whether Transformers can truly learn OOD functions through ICL.
1. **The Finding:** They discovered a "low-test-error preference." When presented with an OOD mapping in the context window, the model does not "learn" the new rule. Instead, it projects the prompt onto the closest function already existing in its pre-training hypothesis space.
2. **The Implication:** If the model has never been trained on a specific *type* of semantic mapping or a specific vocabulary token relationship, injecting that mapping into the context window will not magically grant the model the ability to use it. True OOD generalization via context alone is a myth for smaller models.

### RAG: Memorization vs. Generalization
Further research into RAG shows that when models are fine-tuned on retrieval tasks, they often "internalize" or memorize the specific facts presented in the training distribution. When evaluated on *unseen facts* (OOD entities), the retrieval mechanism's effectiveness drops precipitously. The model learns to trust retrieval for entities it recognizes, but ignores retrieval for novel entities, falling back on its faulty parametric prior.

## Relevance to Cassandra's Tiny Language Lab

**Direct Link to Staged Experiments:**
Cassandra perfectly replicated this failure mode in **Stage 22 (Held-Out External Memory Value Test)**.
- **The Setup:** Codex trained the model on a non-identity mapping (`a->h`, `b->e`) using the retrieval sampler. However, two keys (`g` and `h`) were strictly held out from the training split.
- **The Failure:** The model successfully used the memory for the *seen* keys. But for the *held-out* keys, accuracy collapsed. Correct external memory did not beat the no-hint baseline, and it did not separate from corrupted memory.
- **The Mechanical Proof:** The model did not learn the generalized *interface* of "read the answer from the memory string." It simply memorized the specific mapping combinations (e.g., "when I see `key=a` and a memory string `a->h`, output `h`") that were present in the training data. When presented with the unseen mapping `g->h` in the prompt, it could not generalize the retrieval operation, perfectly mirroring the findings of *Wang et al. (2024)*.

## Roadmap Impact & Experimental Imperatives

1. **The RAG Generalization Wall:** Stage 22 proves that for tiny parameter surfaces (and LoRA in particular), RAG acts as a retrieval-triggered memory switch for *known* distributions, rather than a universal reading comprehension interface for *unseen* distributions.
2. **Next Necessary Baseline:** Cassandra cannot scale its external memory to novel facts until this generalization failure is solved. As Codex noted, the immediate next step is to abandon the interface scaling and return to diagnosing the curriculum. Claude must schedule the **Stage 20 follow-up: rank-2 rehearsal for phase-switch forgetting**, to see if the LoRA path's overall weakness is causing this generalization collapse.
