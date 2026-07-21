# 07. Domain Shift, Phrase Reuse, and Neural N-gram Interpolation

**Relevant Cassandra Stages:** Stage 32 (Hypothesis 009b)

## Exhaustive Methodology Review

As Cassandra measures the durability of frozen count priors under external validity checks, the results map directly onto established findings in domain adaptation and hybrid language modeling. 

### 1. Domain Shift and N-gram Overlap
Stage 32 subjected the model to a cross-domain character split (training on Tiny Shakespeare, validating on Cassandra project prose). 
- **The Finding:** High-order validation hit coverage dropped (e.g., order 4 fell from `0.961` to `0.889`) but did not collapse. The advantage curve remained monotone through order 4, killing the expected moderate-order sweet spot for this specific split.
- **Prior Art Context:** It is a canonical result in classical language modeling (e.g., Katz backoff, Kneser-Ney) that n-gram models perform exceptionally well when test data shares phraseology with training data, and degrade severely under domain shift. The Stage 32 result shows that even across domains, low-level character sequences (orthography, basic syntax) maintain enough structural overlap ("phrase reuse") for the high-order prior to remain a net positive.

### 2. Neural N-gram Interpolation as a Safety Net
Cassandra uses a frozen count prior with a small trainable LoRA residual. This is a specific instance of **Neural-Ngram Interpolation**.
- **The Core Mechanic:** Modern interpolation methods often use n-grams to handle domain shift by injecting target-domain statistics into a general neural model. Cassandra flips this: it uses a source-domain frozen prior to accelerate a tiny neural residual.
- **Why it works:** The discrete n-gram prior handles local memorization and explicit phrase matching (phrase reuse) with zero learning delay. The neural residual handles semantic generalization and smoothing. Even when the domain shifts, the neural residual can learn to discount the prior where it is wrong and rely on it where it is right.

## Relevance to Cassandra's Tiny Language Lab

**Direct Link to Staged Experiments:**
- **Stage 32:** Proved that the high-order prior (order 4) remains the strongest accelerator even under a moderate domain shift, because validation hit coverage did not collapse entirely.

## Roadmap Impact & Experimental Imperatives

1. **The Source-Choice Caveat:** The lack of a humped sweet spot in Stage 32 is a *local* kill. The domain shift was not harsh enough to completely invalidate the order-4 prior. Future claims must caveat that the ideal prior order is highly sensitive to the exact domain distance.
2. **Terminology Update:** When describing Cassandra's architecture externally, we must adopt the terminology of **phrase reuse** and **neural-ngram interpolation**. We must avoid claiming novelty for the general concept of interpolating count statistics with neural representations, focusing instead on our specific budget-constrained, frozen-prior residual implementation.
