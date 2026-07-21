# 1. Frozen Priors and Residual Adapters

**Relevant Cassandra Stages:** Stages 1-10

## Exhaustive Methodology Review

Parameter-Efficient Fine-Tuning (PEFT) has become the dominant paradigm for adapting large language models without updating the entire parameter matrix. The core assumption of PEFT is that the pre-trained weights constitute a stable, information-dense "prior" manifold. Any new task can be learned by finding a small displacement vector or low-rank subspace that shifts the model's behavior while remaining anchored to the prior.

### The Mathematics of Low-Rank Adaptation (LoRA)
In standard fine-tuning, a weight matrix $W_0 \in \mathbb{R}^{d \times k}$ receives a full-rank update $\Delta W$. LoRA constrains this update by representing $\Delta W$ as the product of two low-rank matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, where the rank $r \ll \min(d, k)$.
The forward pass is modified to:
$h = W_0 x + \frac{\alpha}{r} BA x$
Crucially, $A$ is initialized from a random Gaussian, while $B$ is initialized to zero. This zero-initialization guarantees that at step 0, $\Delta W = 0$. The model acts strictly as an identity function of its frozen prior.

### Typical Scale and Empirical Observations
- Standard models apply LoRA primarily to the query ($W_q$) and value ($W_v$) projection matrices in the self-attention blocks.
- Typical ranks used in production range from $r=8$ to $r=64$.
- The scaling factor $\alpha$ controls the magnitude of the residual update, preventing the adapter from collapsing under the massive scale of the frozen $W_0$.

## Relevance to Cassandra's Tiny Language Lab

**Direct Architectural Parallels:**
Cassandra explicitly mirrors the zero-initialized PEFT architecture. The lab uses `--residual-base count-bigram` as the $W_0$ and applies a small residual surface on top. At step 0, Cassandra's output perfectly matches the analytic bigram counts because of the `--zero-residual-head` configuration.

**The Divergence: Neural vs. Analytic Priors:**
In the papers (e.g., *Hu et al., 2021*), the "frozen prior" is a massively pre-trained neural network that already contains deep syntactic and semantic representations. The LoRA layers merely steer this vast capability. 

In Cassandra, the prior is an *analytic count table*. It is literally the log-smoothed occurrence counts of character bigrams from the synthetic corpus: $\log(\text{counts} + \alpha)$. This means Cassandra's frozen prior contains zero deep abstraction—it only contains literal surface frequencies. 

This creates a radically different learning environment for the LoRA adapter. In standard PEFT, LoRA can "reach" into the prior to extract deep features. In Cassandra, the rank-2 LoRA must build any abstract reasoning (like the copy task) entirely from scratch within its tiny parameter budget, while fighting the surface-level statistical pull of the analytic prior.

## Roadmap Impact & Experimental Imperatives

1. **The Upper Bound Problem:** Because Cassandra's prior is analytic, we do not know if the LoRA path's failures in later stages (e.g., Stage 16, Stage 21) are due to the small capacity of the adapter, or because the analytic prior is too rigid and devoid of latent features.
2. **Next Necessary Baseline:** Claude must schedule a formal ablation comparing the analytic prior against a "Tiny Neural Prior."
   - **Hypothesis:** A frozen, randomly-initialized but fully pre-trained 1M parameter transformer will provide a richer manifold for rank-2 LoRA than the analytic bigram prior, dramatically improving performance on non-identity memory tasks.
   - **Execution:** Codex must pre-train a base model on the long-context corpus, freeze it, and then attach the exact same rank-2 LoRA surface used in Stage 10. The `val_nll` and `copy_accuracy` deltas will definitively isolate the effect of the prior's composition.
