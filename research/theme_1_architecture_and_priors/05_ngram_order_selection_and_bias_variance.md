# 05. N-Gram Order Selection and the Bias-Variance Tradeoff

**Relevant Cassandra Stages:** Pre-requisite for Hypothesis 009 (Natural Text Finite-Order Priors)

## Exhaustive Methodology Review

As Cassandra transitions from pure Markov synthetic corpora to natural text, the project must navigate a classic problem in statistical language modeling: **N-Gram Order Selection** and the **Bias-Variance Tradeoff**.

### 1. The Bias-Variance Tradeoff in N-Gram Priors
In natural language processing, selecting the order $n$ of an n-gram model is a direct application of the bias-variance tradeoff:
- **High Bias (Under-specification):** Choosing a low $n$ (e.g., Unigrams or Bigrams) results in a high-bias model. It is overly simplistic and cannot capture long-range syntactic dependencies. However, it has very low variance: the statistics are robust because even small datasets contain enough examples to accurately estimate bigram frequencies.
- **High Variance (Over-specification):** Choosing a high $n$ (e.g., 5-grams) results in a low-bias model that theoretically captures complex dependencies. However, it suffers from extreme variance due to **data sparsity**. Most 5-grams in a test set will never have been observed in the training set, leading to zero-probability estimates and catastrophic overfitting to the training corpus.

### 2. Kneser-Ney Smoothing and Backoff
To manage this tradeoff, classical NLP relies on **Smoothing** and **Backoff** mechanisms.
- **Backoff (e.g., Katz Backoff):** If a higher-order n-gram is unseen, the model "backs off" to a lower-order context.
- **Kneser-Ney Smoothing:** This is the gold standard of classical smoothing. It does not just back off; it mathematically interpolates higher-order estimates with lower-order priors. It recognizes that lower-order statistics (despite being "biased" or misspecified for true natural language) provide a remarkably stable and robust structural foundation when high-order data is sparse.

## Relevance to Cassandra's Hypothesis 009

**Direct Link to Staged Experiments:**
Cassandra's ADR 0003 established the **Graded Source/Prior Order Law** on synthetic data: a perfectly matched prior is optimal, but an under-specified prior still provides a durable head-start. 

As Claude specs **Hypothesis 009** for natural text, this literature firmly establishes the experimental framing:
1. **Not a Novelty Claim:** Applying an n-gram prior to a neural network is not a claim of beating Transformers. It is a controlled *neural-residual measurement of known statistics*. 
2. **The Natural Text Sweet Spot:** Natural language has an effectively infinite Markov order, meaning *any* finite n-gram prior will be technically misspecified. However, because of the Bias-Variance tradeoff, an intermediate order (e.g., order-2 or order-3) should mathematically provide the optimal "sweet spot" for early-compute efficiency. 
    - Order-1 will be too biased (too little structural help).
    - Order-5 will be too over-specified (too sparse, harming the neural residual with noise).
3. **Coverage as the Diagnostic:** Just as Kneser-Ney smoothing relies on observing sufficient lower-order counts, Hypothesis 009 *must* measure "coverage" (the percentage of probe windows that actually hit the prior's transition table). If coverage drops to 2% (as seen in the Stage 29 smoke test), the prior is suffering from high-variance data sparsity and is functionally useless to the neural residual.

## Roadmap Impact & Experimental Imperatives

1. **The Measurement:** Hypothesis 009 must test orders 1 through 4 on a natural text corpus large enough that the optimal n-gram order is determined by the data's bias-variance curve, not by artificial starvation.
2. **The Baseline:** The goal is to see if the neural residual (the LoRA parameters) can treat the frozen n-gram matrix as a "Kneser-Ney" style backoff, learning *only* the high-order semantic residuals that the n-gram prior cannot capture, thereby beating a fully random neural network at early step counts.
