# Forced-Choice Evaluation and Hidden Generalization

## 1. Prior Art and Source Claims

### Surface Form Competition in Generative Metrics
- **Date Checked**: 2026-06-24
- **Claim**: When Large Language Models (LLMs) are evaluated on generative tasks (like next-token prediction), their performance is often hindered by "surface form competition." A generative model assigns probability mass to all possible strings. If multiple strings represent the same underlying concept or if the model expects a different format (e.g., an extra space, a synonym, or a newline), the probability of the strictly "correct" string drops. This can cause a model to fail generative evaluation even when its internal representations have correctly solved the task.
- **Cassandra Relevance**: This hypothesis drove Stage 41. The question was whether the rank-2 residual in Stage 40 actually "knew" the held-out keys but failed to generate them because the raw logit pathway was clouded by format artifacts or vocabulary competition.

### Forced-Choice Evaluation to Reveal Generalization
- **Date Checked**: 2026-06-24
- **Claim**: To mitigate surface form competition, researchers often use "forced-choice" evaluation (multiple choice). Instead of requiring the model to generate the correct string from the entire vocabulary, forced choice restricts the evaluation to a predefined set of candidate labels (e.g., options A, B, C, D). By calculating the argmax over only the valid candidates, forced-choice evaluation often reveals "hidden generalization" or capabilities that generative metrics underestimate.
- **Cassandra Relevance**: Stage 41 implemented this exactly by restricting the argmax prediction to the specific key candidates (`abcdefgh`). If the model had formed a latent copy circuit for the held-out keys but was just poorly calibrated, the forced-choice accuracy or Mean Reciprocal Rank (MRR) would have spiked above chance.

## 2. Mapping to Cassandra Stage 41 Findings

Stage 41 tested whether forced-choice evaluation could rescue the held-out generalization failure from Stage 40.

**Findings:**
- **No Hidden Generalization**: The `count_prior_lora_r2_copyw` arm stayed at `0.000000` held-out choice accuracy on all seeds.
- **Floor-Level MRR**: The mean held-out Mean Reciprocal Rank (MRR) was only `+0.002645` above the frozen floor, meaning the correct held-out keys were not even consistently ranked second or third.
- **Full Model Collapse**: The `random_full_copymix` control also scored `0.000000` on held-out forced choice, despite having a strong seen-key choice accuracy of `0.364261`.

**Analysis:**
The Stage 41 results definitively prove that the held-out generalization failure is a structural circuit failure, not a generative calibration artifact. If surface form competition were to blame, restricting the logit comparison to the 8 valid keys would have revealed the latent copy signal. Because the forced-choice MRR remained at the floor, we know the attention mechanism fundamentally failed to route the unseen keys. The model did not calculate the correct induction correlation at all. This aligns with the harsher interpretations of ICL fine-tuning: models easily overfit to the exact embeddings of seen tokens and fail to learn abstract relational operations unless forced by diverse data or explicit reasoning traces.

## 3. Roadmap Changes and Experiment Design

- **Hypothesis Update**: The failure to copy held-out keys is not an emission artifact. It is a genuine failure of the learned circuitry to transfer to out-of-distribution embeddings.
- **Experiment Design**: Do not pursue further readout or evaluation tricks (like calibration tuning or different probes) to rescue the Stage 38 copy recipe. The recipe itself is incapable of held-out transfer. Future experiments aiming for abstract generalization must change the training signal (e.g., verifier traces, altered samplers, or significantly wider token distributions) rather than the evaluation mechanism.
