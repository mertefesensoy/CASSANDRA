# Data Mixing Laws and DoReMi

**Date Checked**: 2026-07-21
**Source 1**: "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining" (Xie et al., 2023) [arXiv:2305.10429]
**Source 2**: "Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance" (Ye et al., 2024) [arXiv:2403.16952]

## What the Source Claims
Recent work demonstrates that data mixing ratios in pretraining do not have to be manually tuned heuristics. 
- **DoReMi** uses a small proxy model to perform distributionally robust optimization across domains, determining weights that minimize the worst-case loss and drastically speeding up convergence of the full-scale model.
- **Data Mixing Laws** prove that the relationship between mixture proportions and final model loss follows predictable functional forms (often exponential/power laws). Small-scale training runs can perfectly predict the optimal mixture weights for massive models.

## Direct Relevance to Cassandra
In Stage 58, Cassandra relies on a manual heuristic mixture of 25% TinyStories (simple) and 75% text8 (broad) to balance the domain specialization gap without losing grammatical coherence. The mixture was adjusted dynamically only to account for prior Curriculum steps, rather than optimized mathematically.

## Analogies and Limits
Cassandra's Stage 58 operates on a laptop-scale compute budget, so training even smaller "proxy models" to optimize the 25/75 mix might consume a significant portion of the total budget. However, since the lab's methodology is entirely built on small proxy scaling (e.g., the H019 Crossover Scaling Matrix from 3M to 85M), Cassandra is perfectly positioned to apply Data Mixing Laws. 

## Roadmap & Experiment Change
**Hypothesis formulation for Phase 6**: A statically determined mixture (1:3 TinyStories to text8) is suboptimal compared to a proxy-derived data mixture.
- *Next Baseline*: Implement a mini-DoReMi loop. Train a 3.2M parameter model on varying ratios of TinyStories vs Text8 for 1000 steps. Plot the resulting validation NLLs to fit a Data Mixing Law, predicting the optimal mixture for the 85M or 200M model. Compare this predicted optimal mixture against the manual 25/75 baseline in Stage 58.
