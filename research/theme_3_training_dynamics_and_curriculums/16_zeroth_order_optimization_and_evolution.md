# Zeroth-Order Optimization and Evolution Strategies in Language Models

## 1. Prior Art and Source Claims

### MeZO: Fine-Tuning Language Models with Just Forward Passes (Malladi et al., 2023)
- **Date Checked**: 2026-06-23
- **Claim**: Zeroth-Order Stochastic Gradient Descent (ZO-SGD) can fine-tune Large Language Models (up to 66B parameters) with performance comparable to standard backpropagation (often within 1% absolute), while reducing memory usage by up to 12x. It estimates gradients in-place using two forward passes with perturbed parameters.
- **Cassandra Relevance**: Directly relevant to Stage 36's coordinate search and zeroth-order optimization hypotheses. MeZO proves that non-gradient optimization can theoretically reach parity with AdamW when applied to the full high-dimensional surface of pretrained models.
- **Limitations**: MeZO relies on "the blessings of scale"—the phenomenon where massive models have smoother, more predictable loss landscapes that allow random perturbations to reliably find useful directions. 

### Evolution Strategies as a Scalable Alternative to Reinforcement Learning (Salimans et al., 2017) & Recent ES LLM Alignment
- **Date Checked**: 2026-06-23
- **Claim**: Evolution Strategies (ES) can optimize neural networks without backpropagation by generating a population of perturbed parameter vectors and updating the base model toward the most successful mutations. Recent extensions show ES can align LLMs with greater stability and less run-to-run variance than Reinforcement Learning (e.g., PPO), particularly on long-horizon or sparse-reward tasks.
- **Cassandra Relevance**: Directly relevant to Stage 36's ES implementation for rank-2 LoRA.
- **Analogies**: ES acts as a stochastic approximation of the gradient, trading sample efficiency for architectural simplicity and gradient-free robustness. 

### Block Coordinate Descent (BCD) for Memory-Efficient Fine-Tuning
- **Date Checked**: 2026-06-23
- **Claim**: Modifying parameters block-by-block (or coordinate-by-coordinate) reduces optimizer state memory. While blindly applying BCD is inefficient due to wasteful backpropagation through inactive blocks, combining it with zeroth-order optimization or specific expansion matrices allows memory-efficient fine-tuning on consumer hardware.
- **Cassandra Relevance**: Stage 36's coordinate search iterates and changes one parameter block at a time, keeping strict improvements.

## 2. Mapping to Cassandra Stage 36 Findings

In Stage 36, Cassandra tested non-gradient residual formation (Evolution Strategies and Coordinate Search) against AdamW backpropagation on a frozen count prior. The trainable surface was a small LoRA adapter (4648 to 6696 parameters).

**Findings:**
- **AdamW (Rank-2 LoRA)**: Reached `2.000801` mean val NLL.
- **Coordinate Search (Rank-1 LoRA)**: Reached `2.011522` mean val NLL (barely beating the frozen floor of `2.018509` on one seed).
- **Evolution Strategies (Rank-2 LoRA)**: Worsened to `2.060750` mean val NLL, despite 803 formation forward passes.

**Analysis:**
The failure of ES and the weak performance of Coordinate Search in Stage 36 directly contrast with the success of MeZO and ES in frontier LLMs. The crucial difference lies in the "blessing of scale." MeZO and modern ES work on massive parameter models because large pretrained models possess highly redundant, smooth loss landscapes where random perturbations easily find descent directions. 

Cassandra's Stage 36 uses a tiny, constrained surface (a few thousand parameters) learning on a sharp, non-redundant loss landscape. In this highly constrained geometric setting, random perturbations (ES) mostly find destructive directions, and coordinate search is too slow to navigate the narrow valleys that AdamW traverses efficiently using exact gradients and momentum. 

## 3. Roadmap Changes and Experiment Design

- **Hypothesis Update**: Do not pursue zeroth-order optimization or unguided neuroevolution for tiny, constrained adapters. The geometry is too sharp and the parameters too few. Non-gradient parameter guessing does not scale *down*.
- **Experiment Design**: The results validate that AdamW remains the necessary control for optimizing tight residual surfaces. Future non-gradient experiments should focus on analytic, closed-form construction (like the count prior) rather than stochastic parameter guessing. Non-gradient optimization should only be revisited if the objective is strictly non-differentiable (e.g., exact string match rewards) and proxy reward gradients are unavailable.
