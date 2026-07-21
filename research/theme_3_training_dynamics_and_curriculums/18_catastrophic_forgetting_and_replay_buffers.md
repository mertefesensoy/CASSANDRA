# Catastrophic Forgetting and Replay Buffers

**Date Checked**: 2026-07-21
**Source**: General literature on Continual Learning and Catastrophic Forgetting (e.g., Elastic Weight Consolidation, Replay Buffers).

## What the Source Claims
When a model is trained on a narrow, simple domain and then abruptly shifted to a broad, complex domain, it suffers from **catastrophic forgetting**. Gradient descent minimizes the loss on the *current* batch of data. If the new data requires different representations, the optimizer overwrites the weights that encoded the simple grammar.
Large labs avoid a "hard shift" curriculum (where dataset A stops entirely and dataset B begins). Instead, they use **continuous data mixing** or **replay buffers** (e.g., mixing 5-10% old data into new batches) so gradients continually penalize forgetting the foundational grammar.

## Direct Relevance to Cassandra
Stage 58 tests a hard domain shift (12.5k steps TinyStories -> 29.5k steps text8). The risk is catastrophic forgetting of the "simple" phase grammar when adapting to the complex phase.

## Roadmap & Experiment Change
**Hypothesis formulation for Phase 6**: 
Do not do hard curriculum shifts. Re-run Phase 2 of Stage 58 with an interleaved replay buffer (e.g., 10% TinyStories mixed into the text8 batches) to see if it preserves the 2.07 bits/char grammar advantage without losing broad text generalization.
