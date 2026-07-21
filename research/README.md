# Cassandra Research Workspace

This directory stores Gemini's continuous prior-art research loops. It is organized into three major themes mapping to the evolution of the Cassandra project's experimental hypotheses across Stages 1 through 29.

## Capstone Synthesis

*   **[Capstone Synthesis: Re-evaluating the Bitter Lesson in Low-Hardware Constraints](capstone_synthesis.md)**
    *   The master document tying all 29 stages together against external literature. Read this to understand how Cassandra's findings uniquely challenge or validate standard assumptions about In-Context Learning, LoRA capacity, and Inductive Bias.

## Theme 1: Architecture & Priors

Explores parameter-efficient fine-tuning (PEFT), the capacity limits of LoRA, and the mathematical geometry of analytic vs neural priors.

1.  **[Frozen Priors and Adapters](theme_1_architecture_and_priors/01_frozen_priors_and_adapters.md)**
    *   *Relevance:* Stages 1-10 (Why count-priors provide early-compute advantages).
2.  **[PEFT Capacity and Forgetting](theme_1_architecture_and_priors/02_peft_capacity_and_forgetting.md)**
    *   *Relevance:* Stages 20 & 23 (The geometric limits of rank-2 LoRA vs full training).
3.  **[Matched Priors and the Data Generating Function](theme_1_architecture_and_priors/03_matched_priors_and_data_generation.md)**
    *   *Relevance:* Stages 26 & 27 (Bayesian Occam's Razor and perfectly aligned inductive bias).
4.  **[N-Gram Backoff and the Graded Inductive Bias Law](theme_1_architecture_and_priors/04_ngram_backoff_and_graded_laws.md)**
    *   *Relevance:* Stages 28 & 29 (How misspecified priors still provide smoothing and backoff advantages).
5.  **[N-Gram Order Selection and the Bias-Variance Tradeoff](theme_1_architecture_and_priors/05_ngram_order_selection_and_bias_variance.md)**
    *   *Relevance:* Pre-requisite for Hypothesis 009 (Framing the natural-text sweet spot).
6.  **[Neural-Ngram Hybrids and Validation Hit Coverage](theme_1_architecture_and_priors/06_neural_ngram_hybrids_and_coverage.md)**
    *   *Relevance:* Stages 30 & 31 (Proving H009 and demonstrating how high validation hit coverage prevents high-order collapse).
7.  **[Domain Shift and Phrase Reuse](theme_1_architecture_and_priors/07_domain_shift_and_phrase_reuse.md)**
    *   *Relevance:* Hypothesis 009b (Explaining why high-order priors survive domain shifts due to reusable names and phrasing).
8.  **[Cache Language Models and Character Recency](theme_1_architecture_and_priors/08_cache_language_models_and_character_recency.md)**
    *   *Relevance:* Stage 35 (Explaining why simple unigram recency/cache fails at the character level compared to n-gram structure).
9.  **[PEFT Capacity in Prior-Dominated Regimes](theme_1_architecture_and_priors/09_peft_capacity_in_prior_dominated_regimes.md)**
    *   *Relevance:* Stage 37 (Why tiny residual adapters fail to improve upon, or even degrade, a dominant, high-quality analytic prior).
10. **[NLL Divergence and Behavior-Forming Residuals](theme_1_architecture_and_priors/10_nll_divergence_and_behavior_forming_residuals.md)**
    *   *Relevance:* Stage 38 (Why NLL divergence occurs, and how tiny residuals act as the sole behavior-forming surface for tasks like ICL copying, even when validation NLL stagnates).
11. **[LoRA Rank Saturation and Intrinsic Dimension](theme_1_architecture_and_priors/11_lora_rank_saturation_and_intrinsic_dimension.md)**
    *   *Relevance:* Stage 39 (Why increasing LoRA rank from 2 to 4 degrades task performance, and how intrinsic task dimensionality limits naive capacity scaling).
12. **[Induction Circuit Intrinsic Dimension and Capacity Walls](theme_1_architecture_and_priors/12_induction_circuit_intrinsic_dimension_and_capacity_walls.md)**
    *   *Relevance:* Stage 42 (Why generalized copy circuits collapse under rank-2 constraints while full models succeed, and the dual-axis decoupling of NLL vs behavioral formation).
13. **[The Emergence of Reasoning via Reinforcement Learning](theme_1_architecture_and_priors/09_emergence_of_reasoning_and_rl.md)**
    *   *Relevance:* Phase 7 Prep (How DeepSeek-R1 and o1 use RL to incentivize internal Chain-of-Thought and test-time compute scaling).

## Theme 2: In-Context Learning & RAG

Explores verifier-guided synthetic traces, external memory injection, and the mechanics of retrieval-augmented generation.

5.  **[Synthetic Correction Traces](theme_2_in_context_learning_and_rag/05_synthetic_correction_traces.md)**
    *   *Relevance:* Stages 11-14 (Distilling verifier rules into in-context correction prompts).
6.  **[External Memory and Noisy Retrieval](theme_2_in_context_learning_and_rag/06_external_memory_and_noisy_retrieval.md)**
    *   *Relevance:* Stages 18 & 19 (Testing explicit retrieval against "Lost in the Middle").
    *   *See also:* [Noisy Retrieval and Ambiguity](theme_2_in_context_learning_and_rag/06b_noisy_retrieval.md)
7.  **[Format vs Mapping in ICL](theme_2_in_context_learning_and_rag/07_format_vs_mapping_icl.md)**
    *   *Relevance:* Stages 19-21 (The Min et al. discovery that format matters more than exact label mappings).
8.  **[RAG Out-of-Distribution Generalization](theme_2_in_context_learning_and_rag/08_rag_ood_generalization.md)**
    *   *Relevance:* Stage 22 (The failure of explicit retrieval on completely unseen key mappings).
9.  **[Held-Out Key Generalization Failure in ICL](theme_2_in_context_learning_and_rag/09_held_out_key_generalization_failure.md)**
    *   *Relevance:* Stage 40 (Why models fail to generalize in-context learning operations to novel/unseen keys during fine-tuning, even when full models are updated).
10. **[Forced-Choice Evaluation and Hidden Generalization](theme_2_in_context_learning_and_rag/10_forced_choice_evaluation_and_hidden_generalization.md)**
    *   *Relevance:* Stage 41 (Why forced-choice readout metrics cannot rescue true circuit failures, and how surface form competition differs from structural generalization collapse).
11. **[Induction Heads and Zero-Shot Copy Failure](theme_2_in_context_learning_and_rag/11_induction_heads_and_zero_shot_failure.md)**
    *   *Relevance:* Stage 55 (Why the 200M flagship model fails at zero-shot copying despite strong broad NLL).
12. **[Forgetting of Structural Circuits and In-Context Learning](theme_2_in_context_learning_and_rag/12_forgetting_of_structural_circuits.md)**
    *   *Relevance:* Stage 55 (Why the 200M flagship model fails at zero-shot copying despite strong broad NLL).

## Theme 3: Training Dynamics & Curriculums

Explores how ordering data affects small-model learning, catastrophic forgetting, and the crossover points of compute scaling.

9.  **[Retrieval Curriculum Interference](theme_3_training_dynamics_and_curriculums/09_retrieval_curriculum_interference.md)**
    *   *Relevance:* Stages 15-17 (Simultaneous vs Staged learning, and parametric interference).
10. **[The Bitter Lesson and Compute Budgets](theme_3_training_dynamics_and_curriculums/10_the_bitter_lesson_and_compute_budgets.md)**
    *   *Relevance:* Stages 24 & 25 (The bounded "cheap recipe" crossing point).
11. **[Early Training Dynamics and Crossover Points](theme_3_training_dynamics_and_curriculums/11_early_training_dynamics_crossover.md)**
    *   *Relevance:* Stage 25 (Phase transitions and Rectified Scaling Laws mapping exactly to Cassandra's findings).
12. **[Dr. Fırat Akba: Feature Selection and Deep Learning Integration](theme_3_training_dynamics_and_curriculums/12_akba_feature_selection_and_time_series.md)**
    *   *Relevance:* Pre-requisite for Hypothesis 010 (Applying iterative feature selection to curriculum filtering).
13. **[Data Selection, Reducible Loss, and Hard Example Mining](theme_3_training_dynamics_and_curriculums/13_data_selection_and_reducible_loss.md)**
    *   *Relevance:* Hypothesis 010 (Understanding the active learning and data pruning literature behind information-density curriculum filters).
14. **[Static Proxies vs. Iterative Reducible Loss](theme_3_training_dynamics_and_curriculums/14_static_vs_iterative_reducible_loss.md)**
    *   *Relevance:* Stage 33 (Explaining why the static frozen-prior loss filter failed, distinguishing irreducible noise from learnable structure).
15. **[Capacity Bottlenecks in Data Selection](theme_3_training_dynamics_and_curriculums/15_capacity_bottlenecks_in_data_selection.md)**
    *   *Relevance:* Stage 34 (Explaining why true dynamic reducible-loss filtering failed on the rank-2 residual due to capacity constraints).
16. **[Zeroth-Order Optimization and Evolution Strategies](theme_3_training_dynamics_and_curriculums/16_zeroth_order_optimization_and_evolution.md)**
    *   *Relevance:* Stage 36 (Why gradient-free parameter guessing fails on tiny, sharp surfaces despite scaling laws in massive LLMs).
17. **[Data Mixing Laws and DoReMi](theme_3_training_dynamics_and_curriculums/17_data_mixing_laws_and_doremi.md)**
    *   *Relevance:* Stage 58 (How proxy models can mathematically optimize data mixtures over manual heuristics).
18. **[Catastrophic Forgetting and Replay Buffers](theme_3_training_dynamics_and_curriculums/18_catastrophic_forgetting_and_replay_buffers.md)**
    *   *Relevance:* Phase 6 Prep (How hard curriculum shifts cause forgetting and how continual mixing preserves base grammar).
19. **[Scale Breakdown in Proxy Data Mixing](theme_3_training_dynamics_and_curriculums/19_scale_breakdown_in_proxy_mixing.md)**
    *   *Relevance:* Phase 6 Prep (How hard curriculum shifts cause forgetting and how continual mixing preserves base grammar).

## Theme 4: Domain Specialization and Substrates

Explores the tradeoff between BPE and Character-level models, and how corpus breadth impacts zero-shot generalization and the domain specialization gap.

1.  **[BPE vs Character Level Substrates at Small Scale](theme_4_domain_specialization_and_substrates/01_bpe_vs_character_level_small_models.md)**
    *   *Relevance:* Phase 5 Prep (Why character models struggle to learn word boundaries and the computational tradeoff of BPE in SLMs).
2.  **[Domain Specialization Gap and Corpus Breadth](theme_4_domain_specialization_and_substrates/02_domain_specialization_gap_and_corpus_breadth.md)**
    *   *Relevance:* Phase 5 Prep (Addressing the 2.07 bits/char specialization gap and why broader corpus generalization requires heterogeneous data).
3.  **[Curriculum Learning from Simple to Complex](theme_4_domain_specialization_and_substrates/03_curriculum_learning_simple_to_complex.md)**
    *   *Relevance:* Stage 58 (The risks of catastrophic forgetting during the hard domain shift from TinyStories to text8).
