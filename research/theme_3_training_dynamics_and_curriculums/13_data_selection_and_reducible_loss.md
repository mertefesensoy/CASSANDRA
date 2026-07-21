# 13. Data Selection, Reducible Loss, and Hard Example Mining

**Relevant Cassandra Stages:** Hypothesis 010; measured locally as Stage 33

## Exhaustive Methodology Review

Hypothesis 010 introduces an "information-density curriculum filter" that scores training windows using the frozen prior's per-token NLL. This method intersects heavily with several active areas of deep learning research regarding dataset pruning and dynamic batch selection.

### 1. Hard Example Mining vs. Curriculum Learning
- **Curriculum Learning:** Traditionally presents data from "easy" to "hard" to smooth optimization. 
- **Hard Example Mining:** Focuses the model on examples with high loss or high uncertainty. 
- **Relevance:** H010 is a form of Hard Example Mining. By oversampling windows where the frozen prior has high NLL, Cassandra forces the rank-2 residual to focus on the structures the prior cannot already predict. 

### 2. Reducible Loss and Data Pruning
- **The Core Mechanic:** Recent advances in data pruning (e.g., "Reducible Holdout Loss Selection") emphasize that raw high-loss selection is dangerous because it often selects irreducible noise or mislabeled data. Instead, the field favors *reducible loss*—the difference in loss between a reference model and a fully trained model.
- **Cassandra Correlation:** H010 uses the frozen prior's NLL as a proxy for how "hard" a window is. While not strictly "reducible loss" (which requires comparing two states of training), it serves a similar purpose: identifying where the neural residual can add the most value over the analytic baseline.

### 3. Active Learning and Online Batch Selection
- **Active Learning:** Typically refers to querying a human oracle for labels on the most informative unlabeled examples. 
- **Analogy:** Cassandra is performing an unsupervised equivalent of active learning. The model (or a proxy like the frozen prior) selects the most informative un-trained text windows to feed to the optimizer, minimizing wasted compute on already-memorized syntax.

## Relevance to Cassandra's Tiny Language Lab

**Direct Link to Staged Experiments:**
- **Stage 11:** Cassandra previously tested pure "failed-case replay" (a harsh form of Hard Example Mining) and found it degraded validation NLL. 
- **Hypothesis 010:** Correctly anticipates the danger of pure selection by implementing a *mixed* filter (e.g., 25% or 50% high-loss windows, remainder uniform). This aligns with prior-art findings that hard example mining requires a regularizing mix of uniform data to prevent instability.

## Roadmap Impact & Experimental Imperatives

1. **Baseline and Vocabulary Constraint:** Before describing H010 outside the repository, Codex and Claude must use standard terminology: **loss-based data selection**, **hard example mining**, and **online batch selection**. 
2. **The Experimental Control:** The mixed-filtered curriculum must be compared against a strict uniform-sampling baseline at equal compute, as H010 already specifies. If the filter succeeds, it is a local instance of known data selection accelerating a frozen-prior residual, not a completely novel algorithm.

## Stage 33 local result

Codex measured H010 as Stage 33. The fixed top-10-percent frozen-prior-NLL hard-example sampler did not make the order-2 rank-2 residual reach the uniform 200-step target faster. Mixed `f=0.25` reached the target only at the same 200-step budget, mixed `f=0.50` did not reach it through 500 steps, and pure high-loss `f=1.00` was worse at every budget. This strengthens the note's warning that raw high-loss selection can chase irreducible noise or awkward spans. It should be framed as a local negative for this proxy, not as a broad result against reducible-loss data pruning.