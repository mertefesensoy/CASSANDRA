# Curriculum Learning from Simple to Complex

**Date Checked**: 2026-07-21
**Source**: General literature on Curriculum Learning in LLMs (Bengio et al., 2009; various recent surveys).

## What the Source Claims
Curriculum Learning (CL) mimics human pedagogy by introducing easy, foundational examples (shorter sequences, lower perplexity, higher quality text) before complex, noisy ones. The goal is faster convergence and better generalization. However, the LLM literature is heavily mixed. Many studies find that fixed "simple to complex" curricula do not outperform random data shuffling for standard pre-training, often due to the difficulty of defining what constitutes "simple" for an LLM and the risk of catastrophic forgetting when moving phases.

## Direct Relevance to Cassandra
Cassandra's Stage 58 directly tests this: 12,500 steps of TinyStories (simple) followed by 29,500 steps of text8 (complex/broad) vs. a 42,000-step COLD text8 baseline vs. a MIXTURE. The Stage 56 result confirmed that the 2.07 bits/char specialization gap is purely a data-distribution effect (the character substrate *can* learn broad text directly). Curriculum Learning is being tested as a way to bootstrap grammar (TinyStories) before tackling broad knowledge (text8).

## Analogies and Limits
In large-scale LLMs, defining "simple" is abstract (e.g., based on model perplexity). In Cassandra, "simple" is structurally grounded in the TinyStories domain (a bounded vocabulary of 33 characters and child-like grammar). The transition from Phase 1 to Phase 2 in Stage 58 is a hard domain shift, risking catastrophic forgetting of the TinyStories grammar once the model enters the broad text8 phase.

## Roadmap & Experiment Change
**Hypothesis formulation for Phase 6**: A hard Simple -> Complex staged curriculum will result in catastrophic forgetting of the "simple" phase, making the final model indistinguishable from (or worse than) the MIXTURE baseline.
- *Next Baseline*: Analyze the retention metric for the Stage 58 Curriculum Phase 2 checkpoint. If retention drops dramatically (catastrophic forgetting), propose an interleaved curriculum (where simple data decays slowly rather than dropping to 0%) or an adaptive loss-based curriculum (Online Data Mixing) as the next iteration for Phase 6.
