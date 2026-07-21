# The Emergence of Reasoning via Reinforcement Learning

**Date Checked**: 2026-07-21
**Source**: OpenAI o1 System Card & DeepSeek-R1 Technical Report (2024-2025).

## What the Source Claims
Once a model can write grammatically correct sentences, scaling up next-word prediction (Supervised Fine-Tuning) hits a ceiling on logic and reasoning. To make models "think", companies shift from **imitation** to **search and reinforcement learning (RL)**.
- **Incentivizing Chain-of-Thought**: The model is given a prompt and a reward function based on the final answer correctness. It learns through trial and error that generating internal "thinking tokens" (scratchpad space) before answering increases its success rate.
- **Test-Time Compute**: This creates a new scaling law where scaling the time the model is allowed to "think" during inference yields better final answers.

## Direct Relevance to Cassandra
Cassandra's "Verifier-guided micro-reasoning" (Stage 9-12), where a verifier generates correction traces for the model to learn from, is a miniature, deterministic version of this RL reasoning loop. The model is shaped by an external correctness reward, not just next-character mimicking.

## Roadmap & Experiment Change
**Hypothesis formulation for Phase 7**: 
Bootstrapping Reasoning via RL. Once the base model stabilizes (via Phase 6 data mixing), introduce a pure RL-style task. Freeze the base model and use a reinforcement reward (not a next-token cross-entropy loss) on the generated output of a scratchpad. Test whether the small model can learn to allocate test-time compute to verify its own copies/math internally before producing the final answer.
