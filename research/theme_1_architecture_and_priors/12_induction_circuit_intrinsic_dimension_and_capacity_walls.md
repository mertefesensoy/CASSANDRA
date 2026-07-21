# Induction Circuit Intrinsic Dimension and Capacity Walls

## 1. Prior Art and Source Claims

### Induction Heads and Intrinsic Dimension
- **Date Checked**: 2026-06-24
- **Claim**: In mechanistic interpretability, in-context learning (ICL) and copying behaviors are driven by "induction heads," which consist of a QK (Query-Key) circuit for pattern matching and an OV (Output-Value) circuit for information retrieval. The emergence of these circuits often corresponds to phase transitions in a model's intrinsic dimension. Because these operations require routing specific dimensional subspaces to compute correlations, they possess a theoretical minimum intrinsic dimension. If a parameter-efficient fine-tuning (PEFT) method restricts the update matrix to a rank lower than this intrinsic dimension, the model hits a hard capacity wall and physically cannot represent the generalized circuit, regardless of the optimizer or training time.
- **Cassandra Relevance**: Stage 42 demonstrated exactly this capacity wall. When the task was made "memorization-proof" (random payload keys, forcing a general copy operation rather than seen-token identity binding), the rank-2 residual adapter collapsed to chance accuracy (`0.0638` vs `0.0625` chance). Meanwhile, the full-parameter model successfully learned the circuit (`0.226`). This strongly implies that a generalized induction head requires a representational rank greater than 2.

### Dual-Axis Divergence: NLL vs. Behavior
- **Date Checked**: 2026-06-24
- **Claim**: Literature on "grokking" and the "mirage of emergent abilities" highlights a fundamental divergence between continuous metrics (NLL) and discrete metrics (behavioral accuracy). Models can decrease NLL by memorizing syntax or improving confidence on easy tokens, while discrete reasoning capabilities (like exact-match copying) remain flat. Conversely, structural reasoning circuits can form and drastically improve downstream accuracy without moving the global NLL average.
- **Cassandra Relevance**: Stage 42 provides the perfect mirror image to Stage 38. In Stage 38, the rank-2 residual formed copy behavior but *did not* improve validation NLL over the floor. In Stage 42, the rank-2 residual *improved* validation NLL by `0.062` over the floor, yet failed entirely to form the copy behavior. This dual-axis symmetry definitively proves that NLL changes and reasoning-circuit formation are decoupled. A PEFT surface can be useful for NLL without forming behavior, or useful for behavior without improving NLL.

### LoRA Capacity Walls and the Alpha/Rank Confound
- **Date Checked**: 2026-06-24
- **Claim**: When practitioners attempt to overcome LoRA capacity walls by increasing rank ($r$), they often inadvertently change the effective learning rate and update magnitude because of the scaling factor ($\alpha/r$). A common heuristic is $\alpha = 2r$, which masks whether performance changes are due to newly available geometric capacity or simply a louder training signal. Proper scaling experiments must decouple capacity from adaptation strength.
- **Cassandra Relevance**: Hypothesis 018 incorporates this prior art. To test whether the rank-2 failure in Stage 42 is genuinely a geometric capacity wall, the follow-up tests `r8` and `r16` while explicitly matching alpha to rank. This controls the alpha/rank confound, ensuring that any capability gain is due to the expanded intrinsic dimension rather than a shifted update magnitude.

## 2. Roadmap Changes and Experiment Design

- **Hypothesis Update**: The minimal trainable surface for a generalized copy circuit is greater than rank 2. The failure is not due to a lack of total compute or an impossible task, as the full-model control succeeded under the same constraints.
- **Experiment Design**: Advance to Hypothesis 018 to isolate the failure mode. Scale rank (`r8`, `r16`) with strictly controlled alpha scaling to test geometric capacity. Additionally, test a full-parameter train on top of the frozen base (`count_prior_all`) to rule out the possibility that the strong analytic prior itself actively interferes with the induction circuit formation.
