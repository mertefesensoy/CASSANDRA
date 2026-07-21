# 08. Cache Language Models and Character Recency

**Relevant Cassandra Stages:** Stage 35 (Hypothesis 012 Kill)

## Exhaustive Methodology Review

In Stage 35, Cassandra tested Hypothesis 012: enriching the order-2 count prior with a frozen, exponential-decay "recency base" to capture long-range phrase reuse. The expectation was that by tracking recently seen characters, the model could solve context limits. Instead, the recency interpolation severely degraded validation NLL compared to the count prior alone, and was massively outperformed by a simple order-3 count diagnostic.

### 1. The History of Cache Language Models
- **Origins:** Introduced in 1990 by Kuhn and De Mori, cache language models augment traditional n-gram models with a short-term memory store. They interpolate the static background probabilities (n-grams) with the dynamic probabilities of items recently observed in the text (the cache).
- **The Core Mechanic:** If a word appears in a document, its probability of appearing again soon is significantly higher than its background frequency (the "burstiness" of language). A linear interpolation of `P_ngram` and `P_cache` effectively captures this topic-specific repetition.

### 2. The Character-Level Failure Mode
- **The Finding in Stage 35:** Applying exponential smoothing and cache mechanisms at the *character level* degraded performance.
- **Why it failed:** Cache language models are predominantly effective at the **word or token level**. A word carries semantic and topical weight. Characters, however, are structural building blocks. The fact that the letter 'e' or 'a' appeared 10 characters ago provides almost zero predictive signal about whether it should be the *next* character. Next-character prediction relies on strict sequential context (e.g., 't' followed by 'h'), not a bag-of-recent-characters. 
- By interpolating the order-2 count prior with a simple character-recency decay, Stage 35 injected noise that disrupted the local syntax signal without providing a usable semantic signal.

### 3. Time-Series Matrices vs. Simple Recency
- Dr. Akba's proposed "time-series matrix" primitive is often interpreted as recency or exponential smoothing. However, Stage 35 proves that a 1D exponential decay over single characters is too weak a primitive.
- To succeed, a time-series or long-range base must track *sequences* or *states* (like State Space Models or Convolutional kernels), or operate over larger functional tokens. A simple character-cache interpolation is a known null result for structured modeling.

## Relevance to Cassandra's Tiny Language Lab

**Direct Link to Staged Experiments:**
- **Stage 35:** The failure confirms that raw character recency is not a substitute for structural context. The diagnostic jump from order-2 to order-3 count priors (-0.23 NLL) reinforces that precise, multi-step sequential conditioning (n-grams) is vastly superior to structure-blind recency.

## Roadmap Impact & Experimental Imperatives

1. **Abandon Simple Character Caches:** Unigram-level recency operators should not be pursued as a frozen base.
2. **Elevate the Primitive:** If the model-side branch continues, it must test primitives that preserve sequential ordering. A frozen State Space Model (SSM) kernel, a 1D convolution, or a mechanism that caches *n-grams* rather than unigrams would be the correct theoretical next step.
3. **Re-evaluate the Baseline:** The immense strength of the order-3 diagnostic suggests that for the `tiny_language_lab`, maximizing the count prior order (with appropriate smoothing) may simply be the theoretical ceiling for an analytic base.
