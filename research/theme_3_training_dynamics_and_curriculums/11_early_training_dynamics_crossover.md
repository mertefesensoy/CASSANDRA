# 10. Early Training Dynamics and Inductive Bias Crossover Points

**Relevant Cassandra Stages:** Stage 25

## Exhaustive Methodology Review

The trajectory of a language model's loss curve during training is not merely a smooth optimization process; it is characterized by distinct **Phase Transitions**. During early iterations, the model's behavior is dominated by its architectural constraints and initialization (its **Inductive Bias**). As training progresses, the model undergoes a representational reorganization, transitioning from relying on its built-in symmetries to learning high-dimensional latent features from the data.

### Grokking and Phase Transitions
Recent literature studying "grokking" (the delayed emergence of generalization) maps these transitions dynamically:
1. **The Memorization Phase:** In early training, highly parameterized models often memorize the training set, achieving low training loss but high validation loss. 
2. **The Representation Phase (The Crossover):** At a critical threshold of steps, the model undergoes a sudden phase transition (often modeled via non-equilibrium statistical physics), where the weights realign into a generalizing circuit.
3. **Inductive Bias as a Shortcut:** Providing a model with a strong inductive bias (e.g., initializing it with the true task symmetries, or using an analytic prior) completely *bypasses* the early memorization phase. The model begins in a structured state.

### Rectified Scaling Laws
Standard scaling laws assume smooth continuous improvements. However, "Rectified Scaling Laws" account for these architectural biases. They predict that models with strong inductive biases will show a massive **early-compute advantage**, flattening out quickly. Conversely, general neural architectures will start worse but scale steeper, eventually creating a precise **crossover point** where the general method overtakes the biased method.

## Relevance to Cassandra's Tiny Language Lab

**Direct Link to Staged Experiments:**
Cassandra explicitly mapped this phase transition crossover point in **Stage 25**.
- **The Setup:** Codex ran the time-budget surface matrix at 10, 25, 50, and 100 steps across varying data complexities ($p=0.0$ to $p=1.0$).
- **The Early Regime (10 and 25 steps):** At the very beginning of training, the general method (`random_full`) is essentially outputting noise. The "cheap recipe" (`count_prior_lora_r2`) wins everywhere, regardless of data complexity. The analytic prior provides an immediate, perfect structural shortcut.
- **The Crossover (50 steps):** At 50 steps, the phase transition occurs depending on data complexity. For simple data ($p < 0.635$), the analytic prior still wins. For complex data ($p > 0.635$), the general model's feature learning overtakes the rigid prior.
- **The Asymptotic Regime (100 steps):** By 100 steps, the phase transition is complete. The full model has reorganized its weights and learned the task structure, beating the analytic prior across all complexities.

## Roadmap Impact & Experimental Imperatives

1. **Locking the Claim:** Stage 25 finalizes Cassandra's primary empirical claim for the bigram prior recipe. The analytic prior + LoRA is an **early-compute inductive-bias advantage**, not an asymptotic replacement for full training on the Stage 24 corpus family.
2. **ADR Completed:** Claude accepted ADR 0002, locking that exact empirical claim and boundary. Codex then measured the first new branch, H007, as Stage 26.

## Codex Local-Result Correction

The Stage 25 mechanism should be phrased carefully. The full random model was not
pure noise after 10 steps. It began near the uniform floor, but by 10 steps it had
already moved far below `ln(vocab_size)` on every measured corpus point. The
cheap recipe still won because the frozen prior gave it a much better low-step
surface while the full model was catching up.

Stage 26 also narrows the "not asymptotic" wording. The bigram prior recipe is
not asymptotic on the old corpus family, but an order-matched trigram prior on a
pure order-2 Markov source stayed strongly ahead through 200 steps. The next
research comparison should distinguish misspecified priors, which behave like
head starts, from well-specified analytic priors, which can be durable on a
matching data source.

Stage 27 completed the immediate source-order/prior-order surface for orders 1
and 2. Matched priors stayed positive through 200 steps, while the mismatched
order-2 plus bigram cell decayed to tied and the over-specified order-1 plus
trigram cell became negative. This makes the crossover story conditional on
model specification: a misspecified prior crosses over, while a well-specified
finite-order prior can remain ahead in the measured range.

Claude's H008 was the next required comparison for this framing because it added
order 3 at `V = 8`. The key research question was whether the crossover is about
misspecification alone, or whether high-order finite-count priors lose their edge
through sparsity even when they match the source.

Codex measured H008 as Stage 28. The result preserves the crossover framing but
makes it graded. The matched order-3 prior stayed strongly ahead at 200 steps
(`A(3,3)=0.630598`) with full coverage, while the order-2 prior on the order-3
source also stayed meaningfully positive (`A(3,2)=0.106932`). The comparison to
outside work should therefore include model-order selection and lower-order
backoff, not only matched versus mismatched priors.
