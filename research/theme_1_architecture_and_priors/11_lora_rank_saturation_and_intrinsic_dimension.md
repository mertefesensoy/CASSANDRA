# LoRA Rank Saturation and Intrinsic Dimension

## 1. Prior Art and Source Claims

### Intrinsic Rank and Over-Parameterization
- **Date Checked**: 2026-06-24
- **Claim**: The "intrinsic rank" hypothesis states that the weight updates required to adapt a language model to a new downstream task reside in a very low-dimensional subspace. Even if the base model has billions of parameters, the actual $\Delta W$ needed for the task can be captured by LoRA matrices with a rank ($r$) that is exponentially smaller. 
- **Cassandra Relevance**: In Cassandra Stage 39, the task is a specific long-context identity copy. The hypothesis implies that the required "induction circuit" to perform this copying has a very low intrinsic dimension, meaning it does not strictly need high-rank matrices to form.

### Rank Saturation and Non-Monotonic Scaling
- **Date Checked**: 2026-06-24
- **Claim**: Increasing LoRA rank does not guarantee monotonic improvements in downstream task performance. In fact, many tasks exhibit "rank saturation" where performance plateaus very early (e.g., $r=4$ to $r=16$). Furthermore, without careful re-tuning of hyperparameters (specifically the learning rate and the scaling factor $\alpha$), increasing the rank can actually *degrade* performance due to "stunted learning dynamics," optimization instability, or overfitting to small target distributions.
- **Cassandra Relevance**: This directly explains the Stage 39 finding where a rank-4 adapter performed *worse* on average than a rank-2 adapter. The Stage 39 sweep fixed `lora_alpha=2.0` and the learning rate across all arms. By increasing capacity without scaling the hyperparameters or expanding the training data, the rank-4 arm likely encountered optimization instability or overfit the training split, leading to poorer generalization on the 76 validation cases.

### The Role of Alpha and Rank-Stabilized LoRA (rsLoRA)
- **Date Checked**: 2026-06-24
- **Claim**: Standard LoRA implementations scale the adapter output by $\frac{\alpha}{r}$. If $r$ is increased while $\alpha$ is held constant, the magnitude of the adapter's influence changes, which can destabilize training. Recent literature (like rsLoRA) suggests scaling by $\frac{\alpha}{\sqrt{r}}$ to maintain stability at higher ranks.
- **Cassandra Relevance**: The failure of rank 4 in Stage 39 is a classic symptom of uncoupled hyperparameters. The drop from `0.320176` (rank 2) to `0.271930` (rank 4) indicates that the problem is not a lack of capacity, but an optimization or regularization failure at higher ranks.

## 2. Mapping to Cassandra Stage 39 Findings

Stage 39 tested the "Behavior Rank Sweep" to determine if the behavior-forming surface (the residual adapter) was bottlenecked by its rank-2 capacity on the copy task.

**Findings:**
- **Rank 1**: Reached `0.250000` mean copy accuracy.
- **Rank 2**: Reached `0.320176` mean copy accuracy.
- **Rank 4**: Reached `0.271930` mean copy accuracy.
- **Seed Spread**: Rank 4 did not consistently beat rank 1 (it lost on seed 19) and was worse than rank 2 on two out of three seeds. 

**Analysis:**
The naive assumption that "more trainable parameters = better behavior" is demonstrably false in this regime. This is a perfect mapping to the "rank saturation" phenomenon. The intrinsic dimensionality of the identity-copy task is clearly met (or nearly met) by rank 2. 

When the capacity was doubled to rank 4 without adjusting the optimization environment (alpha, learning rate, or batch size), the model did not discover a better circuit. Instead, the wider matrices likely introduced noise, destabilized the gradient descent path, or overfit to the `33,048` characters of training data, thereby generalizing worse to the validation set.

## 3. Roadmap Changes and Experiment Design

- **Hypothesis Update**: The behavioral capacity of the residual adapter saturates at or before rank 2 for the current copy task under the current hyperparameter regime. Higher ranks are not the bottleneck.
- **Experiment Design**: Do not pursue further rank sweeps as a primary method for improving behavior. As the Codex handoff suggested, the focus must shift to optimization stability, the sampler mechanism, or verifier-guided training. If larger ranks are ever revisited, they must be paired with Rank-Stabilized scaling ($\sqrt{r}$) and strict learning rate sweeps.
