# 8. PEFT Capacity, Sequential Forgetting, and the Corpus Complexity Axis

**Relevant Cassandra Stages:** Stage 23 and ADR 0001 Redirect

## Exhaustive Methodology Review

When a language model learns tasks sequentially (e.g., first correction traces, then external retrieval), it typically suffers from **catastrophic forgetting**. The standard assumption is that Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA are more robust to forgetting because they freeze the pre-trained backbone, maintaining a stable shared feature representation. 

However, recent literature challenges this, showing that when the shared subspace (the rank $r$) is extremely small, LoRA experiences intense structural interference. Furthermore, the performance of PEFT architectures vs. Full Fine-Tuning (FFT) is heavily dependent on the intrinsic **complexity** of the data distribution.

### Catastrophic Forgetting in Low-Rank Subspaces
Recent studies on sequential learning with LoRA (e.g., *PEARL*, *I-LoRA*) prove that standard LoRA still suffers from catastrophic forgetting. 
1. **The Mechanism of Interference:** When rank is extremely low (e.g., $r=2$), the low-dimensional projection matrices $A$ and $B$ must overlap to encode multiple distinct tasks. The gradients from the second task (retrieval) completely overwrite the optimal subspace found for the first task (correction).
2. **The Rehearsal Failure:** Standard "rehearsal" (keeping a small fraction of Phase 1 data active during Phase 2) often fails to rescue extremely low-rank models. The adapter literally lacks the geometric dimensions required to simultaneously project the features for both tasks without destructive interference. Solving this typically requires dynamic rank allocation or structural expansions (like *LoRETTA* or *WeGeFT*), rather than simple data rehearsal.

### Analytic Priors vs. Neural Priors across Data Complexity
The core architectural thesis of Cassandra is using an **Analytic Prior** (explicit log-smoothed bigram counts) plus a tiny residual surface, rather than a dense **Neural Prior** (a fully pre-trained transformer). 
1. **Expressivity vs. Tractability:** Analytic priors are theoretically perfectly tractable but completely rigid. They represent a highly compressed, low-complexity prior. Neural priors are highly expressive, capturing high-entropy distributions (like natural language) but require massive parameter counts.
2. **The Capacity Boundary:** Research on SLMs and data complexity establishes a "capacity boundary." If the training data is low-complexity (e.g., purely Markovian or Zipfian sequences), an analytic prior + PEFT will converge faster and generalize better than FFT, acting as a perfect structural regularizer. However, as the corpus complexity increases (introducing long-range dependencies, abstract semantic mappings, or hierarchical grammar), the analytic prior becomes a bottleneck, and FFT vastly outperforms it because FFT can learn the high-entropy neural prior required to represent the data.

## Relevance to Cassandra's Tiny Language Lab

**Direct Link to Staged Experiments:**
Cassandra directly observed the geometric limits of LoRA in **Stage 23**. The hypothesis was that the catastrophic forgetting observed in the LoRA path during the Stage 17 phase switch could be rescued by rehearsing a small fraction of correction tasks (0.05 or 0.10). The result was negative: rehearsal actually made performance *worse* than simultaneous training. The rank-2 LoRA simply lacks the spatial dimensions to hold both the correction interface and the retrieval interface simultaneously under a phase-switch pressure.

**The ADR 0001 Redirect:**
Because scaling the RAG interface failed (Stage 22) and curriculum rehearsal failed to save the LoRA path (Stage 23), the project is invoking the ADR 0001 redirect. Rather than trying to force the tiny residual path to learn complex multi-task behaviors, the new goal is to characterize the **Corpus Complexity Axis**. Cassandra will explicitly test the theoretical tradeoff between Analytic Priors and Full Fine-Tuning as the training data scales from simple to complex.

## Roadmap Impact & Experimental Imperatives

1. **The Corpus Complexity Sweep:** Claude must design a series of experiments that gradually increase the structural complexity of the training corpus.
2. **Next Necessary Baseline:** Codex must build a corpus generator with a parameterizable complexity knob.
   - **Mechanism:** The generator should produce corpora ranging from pure bigram Markov chains (complexity 0) to sequences with long-range bracket matching or hierarchical dependencies (complexity 1).
   - **Hypothesis:** At complexity 0, the `count_prior_lora_r2` model will achieve a lower `val_nll` than `random_full` because the analytic prior perfectly matches the data generating function. As complexity approaches 1, `random_full` will surpass the LoRA model as the analytic prior fails to capture long-range dependencies.
   - **Execution:** Codex must run the comparison matrix across this generated axis to empirically chart the exact crossover point where full gradient training becomes necessary.
