# 14. Static Proxies vs. Iterative Reducible Loss

**Relevant Cassandra Stages:** Stage 33 (Hypothesis 010 Kill)

## Exhaustive Methodology Review

In Stage 33, Cassandra tested Hypothesis 010: using the frozen order-2 count prior's per-token NLL to filter and oversample high-loss training windows for a rank-2 LoRA residual. The experiment failed to beat the uniform sampling baseline. This outcome perfectly reflects the frontier understanding of data pruning and curriculum learning in language models.

### 1. The Danger of Static High-Loss Proxies
- **The Finding in Stage 33:** Selecting the top 10% highest-loss windows under the frozen prior did not speed up convergence. At a pure `f=1.0` mix, it degraded performance significantly. Even the mixed `f=0.25` filter lagged slightly behind uniform sampling.
- **Prior Art Context:** A frozen prior’s loss is a *static proxy*. In the deep learning literature, data with high static loss often contains a high degree of **irreducible loss**—meaning the data is inherently unpredictable due to noise, mislabeled examples, rare character sequences, or chaotic formatting. When a model focuses on irreducible noise, it wastes capacity and destabilizes training.

### 2. Irreducible vs. Reducible Loss
- **The Core Mechanic:** Neural scaling laws decompose cross-entropy loss into irreducible loss (the inherent entropy of the data) and reducible loss (the gap between the model's distribution and the true distribution).
- **Why it matters:** Efficient data pruning methods must target *reducible loss*. If a window has high loss under the frozen prior but the neural residual also cannot learn it (e.g., a random hash string), the loss is irreducible. The rank-2 LoRA residual in Stage 33 was likely forced to memorize noise rather than learning generalizable syntax.

### 3. Iterative and Dynamic Scoring
- **The Solution in Literature:** To successfully implement loss-based data selection, researchers use *iterative* or *dynamic* scoring. This involves comparing the loss of a reference model to the loss of a trained model, or tracking the gradient norm/loss trajectory of individual examples over multiple training epochs. By observing how loss *changes*, the curriculum can select windows that the model is actively learning from, rather than windows that are permanently "hard."

## Relevance to Cassandra's Tiny Language Lab

**Direct Link to Staged Experiments:**
- **Stage 33:** Conclusively kills the idea that a cheap, static, single-pass loss filter (using the analytic prior) can effectively guide the neural residual.
- **Stage 11 & 12:** This result mirrors the earlier finding where replaying static failed copy cases performed worse than mixing in structured, synthesized correction traces.

## Roadmap Impact & Experimental Imperatives

1. **Pivot to Dynamic Scoring or Architecture:** Because static data selection failed, Cassandra has two valid paths forward:
   - **Data-side:** Implement true *iterative reducible loss* tracking (measuring the delta between the frozen prior and a live partially-trained residual) to identify learnable windows.
   - **Model-side:** Abandon data filtering and pursue Dr. Akba's time-series structural optimization (replacing the count prior with a lightweight time-series matrix) as originally scoped in Hypothesis 010's framing.
2. **Terminology Constraint:** The failure in Stage 33 must be documented strictly as a failure of "static loss proxies" and "irreducible noise oversampling," preventing claims that "data pruning does not work."
