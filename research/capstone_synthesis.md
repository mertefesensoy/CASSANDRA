# Capstone Synthesis: Re-evaluating the Bitter Lesson in Low-Hardware Constraints

**Date:** June 17, 2026
**Scope:** Cassandra Stages 1 through 29

## Executive Summary
Since the inception of the Cassandra project, the core experimental methodology has centered on a single driving question: *How much language-model behavior can be formed when the training budget is treated as the scarce resource?*

Across 29 discrete stages, Cassandra transitioned from a basic evaluation of small parameter architectures (Stages 1-10) to a complex exploration of In-Context Learning and RAG (Stages 11-22), and finally to a deep geometric analysis of Inductive Bias, N-gram Backoff, and Compute Scaling (Stages 23-29). 

This document serves as the master synthesis of those 29 stages, directly correlating the local lab findings with external literature (as cataloged in the thematic research directories).

---

## Part 1: Architecture, Priors, and The Bitter Lesson

### The "Cheap Recipe" and the Inductive Bias Advantage
In **Stages 1 through 10**, Cassandra explored whether an analytic prior (a log-smoothed N-gram matrix) could serve as a frozen base upon which a tiny Low-Rank Adaptation (LoRA) residual was trained. 
- **The Finding:** This "cheap recipe" completely bypassed the early memorization phase of training, drastically outperforming randomly initialized full models at very early compute budgets (10 and 25 steps).
- **Correlation with Prior Art:** This aligns with literature on *Inductive Bias Alignment*. When a model's structural bias matches the data generation function, learning is nearly instantaneous. 

### The Crossover Point
However, **Stages 24 and 25** mapped out the exact time-budget surface across a corpus complexity axis.
- **The Finding:** The advantage is strictly bounded. By 50 steps, the analytic prior's advantage narrows. By 100 steps, full neural training overtakes the frozen prior entirely.
- **Correlation with Prior Art:** This is a perfect miniature replication of *Rich Sutton's "The Bitter Lesson"* and *Rectified Scaling Laws*. The analytic prior is a human heuristic. It provides a massive early-compute shortcut, but as compute scales, the general, high-dimensional neural representations discovered by raw backpropagation will inevitably overtake rigid mathematical priors.

### The Graded Law of Misspecified Priors
In **Stages 26 through 29**, the project asked what happens when the prior does *not* perfectly match the data.
- **The Finding (ADR 0003):** If the data is order-3 Markov, an order-3 prior wins completely. But crucially, an order-2 prior (which is misspecified/too simple) *still* provides a meaningful head start over random initialization. It acts as a safety net.
- **Correlation with Prior Art:** This maps perfectly to classic NLP *Kneser-Ney Smoothing* and *N-gram Backoff*. A lower-order prior acts as a hierarchical baseline. Even in natural text, providing a bigram backoff allows the neural capacity to focus exclusively on learning higher-order residuals.

---

## Part 2: In-Context Learning, RAG, and Memory

When the project shifted to testing how these small models retrieved information, the results challenged standard assumptions about RAG.

### Synthetic Correction Traces
In **Stages 11 through 14**, Cassandra attempted to correct the model's failures.
- **The Finding:** Replaying failed cases was useless. However, generating explicit, distilled "correction traces" (e.g., `key=e answer=e`) and placing them in the context window resulted in massive accuracy jumps (from ~42% to 96% copy accuracy).
- **Correlation with Prior Art:** This validates research into *Verifier-Guided Data Generation* and *Chain-of-Thought Distillation*. The model needed the specific deductive trace, not just repeated exposure to the failure.

### Format vs. Mapping in External Memory
In **Stages 19 through 22**, the project tested true external memory lookup using non-identity mappings (e.g., `key=a` maps to `answer=h`).
- **The Finding:** The small models could not generalize. Even when provided with the correct memory hint, if the mapping was unseen in training, the model ignored the memory or hallucinated.
- **Correlation with Prior Art:** This perfectly replicates the findings of *Min et al. (2022)* and *Wei et al. (2023)*. Small language models use In-Context Learning primarily to deduce the *format* of the task, not to learn novel *mappings* on the fly. Because the `a->h` mapping was out-of-distribution, RAG failed completely. The memory was functionally ignored.

---

## Part 3: Curriculums and Catastrophic Forgetting

### Simultaneous vs. Staged Curriculums
In **Stages 15 through 17**, Cassandra attempted to teach the model to use external memory while simultaneously teaching it general linguistic behaviors.
- **The Finding:** Simultaneous training created extreme parametric interference. The full model performed significantly better when the curriculum was strictly staged (Learn corrections first, *then* learn retrieval). 
- **The Exception:** The tiny rank-2 LoRA model performed *worse* when staged, because it lacked the parametric capacity to remember the first phase after switching to the second.
- **Correlation with Prior Art:** This mirrors research on *Catastrophic Forgetting* and *Geometric Limits of PEFT*. Low-rank updates constrain the model to a tiny subspace. When the task shifts, that entire subspace must rotate, destroying the previous behavior. Full models, with dense parameter spaces, can partition knowledge safely across phases.

---

## Final Conclusion
Cassandra's 29 stages prove that laptop-scale models do not just behave like "bad" large models—they exhibit distinct mathematical boundaries. 
1. **Analytic priors are backoff mechanisms, not asymptotic solutions.** They win early and lose late.
2. **Small models use RAG for format, not fact lookup.** If a mapping is completely unseen in training, providing it in context will fail.
3. **PEFT capacity dictates curriculum ordering.** Low-rank matrices cannot handle strictly staged curriculums due to catastrophic forgetting; they require simultaneous rehearsal. 

These mapped constraints are now firmly anchored in the literature, clearing the path for the next phase of the Cassandra program.
