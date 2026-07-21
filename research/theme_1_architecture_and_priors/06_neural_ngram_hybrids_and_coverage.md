# 06. Neural-Ngram Hybrids and Validation Hit Coverage

**Relevant Cassandra Stages:** Stage 30 and Stage 31

## Exhaustive Methodology Review

As Cassandra extends its tests to natural text, it enters a highly studied subfield of language modeling: **Hybrid Neural-Ngram Interpolation**.

### 1. Hybrid Neural-Ngram Ensembles
While large language models have largely replaced pure count-based n-gram models, research has consistently shown that interpolating discrete n-gram models with neural language models yields a powerful ensemble. 
- **The Strengths:** The discrete n-gram prior excels at memorizing exact local phrases and rigid syntax with zero learning delay. The neural model excels at semantic generalization and long-range dependencies.
- **Architectural Integration:** Classic methods simply average the probabilities at decoding time. Advanced methods (such as Trans-dimensional Random Fields) integrate the count-based features directly into the neural architecture. Cassandra's approach—injecting a frozen log-smoothed count matrix into the base logits and training a rank-2 LoRA residual on top—is a highly constrained, parameter-efficient variant of this integration.

### 2. The Dominance of Validation Hit Coverage
In Stage 31, Cassandra tested an order-4 prior. A primary concern with high-order n-grams is catastrophic sparsity: most 4-grams are unique. However, Stage 31 succeeded because of a critical distinction well-documented in code-analysis and NLP smoothing literature:
- **Table Coverage vs. Validation Hit Coverage:** 
    - *Table coverage* measures how much of the possible permutation space the training data observed. In Stage 31, this was only `0.029` (2.9%).
    - *Validation hit coverage* measures how often the actual queries in the validation set hit an observed entry in the table. In Stage 31, this was `0.961` (96.1%).
- **The Finding:** The raw sparsity of the state space does not matter if the data distribution is highly predictable. Because the validation text (the suffix of Tiny Shakespeare) heavily reused local phrases seen in the training prefix, the order-4 prior almost never "missed." It acted as a highly accurate lookup table for the neural residual.

## Relevance to Cassandra's Tiny Language Lab

**Direct Link to Staged Experiments:**
Cassandra formalized this in **Stage 30** and **Stage 31**.
- **Stage 30 (Orders 1-3):** Proved Hypothesis 009. A finite-order count prior plus a tiny neural residual maintains a durable, positive early-compute edge over `random_full` on natural text. The prior acts as an instant memorization table, freeing the tiny LoRA to focus entirely on the residuals.
- **Stage 31 (Order 4 Extension):** Showed that the advantage does not fall off at order 4 *if* the validation hit coverage remains high. Order 4 provided an enormous `+0.463` NLL advantage over `random_full` at 500 steps.

## Roadmap Impact & Experimental Imperatives

1. **The Caveat of Friendly Splits:** Stage 31 proves that the finite-order prior is a powerful accelerator, but it also exposes a confound. The `natural_text_seed.txt` split (prefix train, suffix validation) on a single author (Shakespeare) is extremely friendly to high-order memorized statistics. The 96% validation hit coverage is artificially high for a real-world language model.
2. **Next Necessary Baseline:** To locate the true "sweet spot" (the descending limb of the bias-variance tradeoff), Cassandra must test a harsher external-validity split. 
   - **Hypothesis:** If the validation set is truly out-of-distribution or structurally diverse, high-order validation hit coverage will collapse. When validation hit coverage drops, the order-4 prior will inject noise rather than signal, forcing the LoRA to fight its own prior. At that point, the model will be forced to rely on the lower-order (e.g., order-2) backoff prior to survive. This is the next required stage.
