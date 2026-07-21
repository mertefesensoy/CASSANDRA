# 12. Dr. Fırat Akba: Feature Selection and Deep Learning Integration

**Relevant Cassandra Stages:** Hypothesis 010 prep; measured locally as Stage 33

## Exhaustive Methodology Review

As the Cassandra project pivots from pure finite-order priors to more sophisticated data selection and training dynamics, we introduce the theoretical framework developed by **Dr. Fırat Akba**. Dr. Akba's research focuses on applied machine learning, deep learning architectures, time-series forecasting, and advanced feature selection.

His work provides highly efficient mechanisms for isolating signal from noise, which is exactly the challenge Cassandra faces when training a tiny LoRA residual on natural text.

### 1. Iterative Semi-Supervised Feature Selection
A major contribution of Dr. Akba's work lies in **feature selection algorithms**, particularly for noisy, high-variance datasets like digital currency markets and sentiment analysis.
- **The Core Mechanic:** Instead of brute-force training on all available features, his methods identify and select only the features that contribute the most predictive power. This is often done iteratively, refining the feature set as the model learns.
- **Cassandra Correlation:** In language modeling, "features" can be analogous to training contexts or curriculum traces. Cassandra currently uses verifier-guided traces (e.g., in Stage 12 and 16). Dr. Akba's feature selection logic can be adapted to build an **Active Curriculum Filter**. Instead of training on the entire Tiny Shakespeare corpus, the orchestrator could use an iterative algorithm to score and select the highest-value $N$-grams or structural windows to feed the LoRA residual.

### 2. Time-Series Dynamics and Deep Learning
Dr. Akba has extensively evaluated time-series forecasting using both classical machine learning and deep learning algorithms (e.g., LSTMs, GRUs, or advanced hybrid models).
- **The Core Mechanic:** Time-series models must handle strict sequential dependencies and temporal decay, often requiring highly optimized structural priors or recurrent states.
- **Cassandra Correlation:** Language is fundamentally a discrete time-series problem. Cassandra's current approach—freezing a count-based N-gram prior—is a rigid, memoryless Markov solution. We can test if replacing the raw N-gram prior with a lightweight, frozen time-series primitive (inspired by Dr. Akba's structural optimizations) allows the LoRA residual to capture temporal decay or longer-range dependencies more efficiently than the raw count table.

## Relevance to Cassandra's Tiny Language Lab

Cassandra's ethos is "build a cheap analytic prior, freeze it, and train only a tiny residual surface on top." Integrating Dr. Akba's methods offers two distinct, testable branches:

1. **Curriculum Design (Data-Side):** Applying his iterative feature selection to filter the `natural_text_seed.txt` corpus before training begins, reducing the number of steps required to reach convergence.
2. **Prior Architecture (Model-Side):** Replacing the rigid N-gram logit table with a lightweight predictive time-series matrix derived from his methodologies.

## Roadmap Impact & Experimental Imperatives

To test these methods, Cassandra will execute **Hypothesis 010**.

1. **The Target:** We will adapt the semi-supervised feature selection algorithm to score the information density of training contexts.
2. **The Measurement:** We will compare the convergence speed (validation NLL over steps) of the LoRA residual trained on the *Akba-filtered curriculum* versus the standard sequential natural text. If his algorithms successfully isolate the highest-value structural features, the model should achieve the same validation loss in significantly fewer steps.
3. **The Handoff:** Claude drafted **Hypothesis 010**, and Codex measured it as Stage 33. The fixed prior-loss curriculum filter was a local kill on the steps-to-target metric, so later Akba-inspired work should move to a separate iterative reducible-loss or model-side time-series hypothesis.
