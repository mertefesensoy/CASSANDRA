# Held-Out Key Generalization Failure in ICL

## 1. Prior Art and Source Claims

### Format vs. Mapping and ICL Generalization
- **Date Checked**: 2026-06-24
- **Claim**: In-context learning (ICL) requires a model to extract both the *format* of the prompt (the structural template) and the *mapping* (the logical rule connecting inputs to outputs). Research shows that while models are highly sensitive to format, they often fail to generalize the mapping to entirely novel, "held-out" tokens that were absent from the training distribution. The model may successfully "memorize" the mapping for tokens seen during pretraining or fine-tuning, but applying that same operational logic to out-of-distribution (OOD) semantics causes the predictive mechanism to collapse.
- **Cassandra Relevance**: Explains why both the `count_prior_lora_r2_copyw` and `random_full_copymix` arms collapsed to `0.000000` accuracy on the held-out keys `g` and `h` in Stage 40. The models learned the format and successfully formed induction heads for the keys they saw during training, but failed to generalize the abstract "copy" operation to unseen targets.

### The Role of Induction Heads and Memorization
- **Date Checked**: 2026-06-24
- **Claim**: Small models fine-tuned on restricted datasets often resort to "memorization" before they achieve true "generalization" (a phenomenon related to Grokking). Mechanistic interpretability research demonstrates that early-stage induction circuits are often heavily specialized to the exact token embeddings encountered in the training set. A true, abstract induction head that operates uniformly across the entire vocabulary requires a more robust training signal, significantly larger capacity, or extended optimization time to decouple the copying operation from the specific token representations.
- **Cassandra Relevance**: The Stage 40 results show that the model is in this intermediate phase. The `+0.189003` seen-key gap for the full model (and `+0.027491` for the residual) proves that *some* task-specific learning occurred, but the total failure on held-out keys proves the learned circuitry is tightly bound to the seen tokens rather than an abstract copy rule.

### Generalization Failure as an Alignment Problem
- **Date Checked**: 2026-06-24
- **Claim**: When LLMs fail to apply known rules to unseen tokens, it is often framed as an "alignment" or deployment failure. The model may possess the latent capability to copy (as evidenced by its performance on seen keys), but the attention mechanisms fail to properly align the novel held-out query with the corresponding key in the context window. The noise introduced by the unfamiliar target tokens breaks the attention weights necessary to execute the deployment step.
- **Cassandra Relevance**: This confirms that the Stage 40 failure is a fundamental property of ICL fine-tuning, not merely a deficiency of the LoRA adapter. Even the full-context control model suffered the same alignment collapse on held-out keys. 

## 2. Mapping to Cassandra Stage 40 Findings

Stage 40 tested whether the behavior-forming residual learned a generalized identity-copy circuit or merely memorized the seen keys.

**Findings:**
- **Held-Out Failure**: Both the rank-2 residual (`0.000000` mean held-out accuracy) and the full-context control (`0.000000` mean held-out accuracy) failed to copy the held-out keys `g` and `h` on all seeds.
- **Seen-Key Improvement**: The full model improved over the floor on seen keys by `+0.189003`, while the residual improved by `+0.027491`.
- **Not Pure Memorization**: Neither model simply memorized the train set to 1.0 accuracy. The performance is an intermediate state where the training signal improved the ability to copy familiar keys but failed to instill an abstract copying mechanism.

**Analysis:**
The collapse on held-out keys aligns precisely with the literature on ICL generalization failures. Fine-tuning an ICL mechanism on a narrow set of keys teaches the model to bind the attention operation to those specific embeddings. The abstraction step (learning that the attention should route *any* token matched by the key-value structure) requires significantly more diverse data, different sampler mechanisms, or explicit verifier traces. Because the full model also collapsed, this is not merely a LoRA capacity bottleneck; it is a fundamental challenge of learning abstract operations from limited, homogeneous data splits.

## 3. Roadmap Changes and Experiment Design

- **Hypothesis Update**: The current training recipe (copy-weighted loss over a restricted set of keys) does not yield an abstract, generalizable copy circuit at this compute/data budget.
- **Experiment Design**: Rescope ADR 0006 to focus on seen-key identity copying as the primary baseline for the small residual. To achieve true held-out generalization, future experiments must alter the learning environment (such as deploying a wider variety of keys in the training set, modifying the sampler to expose the model to different failure modes, or using verifier-guided corrections that explicitly teach the structural mapping rather than relying on implicit gradients).
