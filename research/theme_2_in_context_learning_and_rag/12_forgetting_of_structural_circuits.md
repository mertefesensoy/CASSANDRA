# Forgetting of Structural Circuits and In-Context Learning

**Date Checked**: 2026-07-21
**Source 1**: "In-context Learning and Induction Heads" (Olsson et al., 2022)
**Source 2**: Recent literature on Function Vector Shift and Differential Circuit Vulnerability during LLM fine-tuning.

## What the Source Claims
Mechanistic interpretability shows that In-Context Learning (ICL) relies on activation-based **structural circuits**, primarily Induction Heads and Previous Token Heads. When a base model is fine-tuned or subjected to domain shifting on broad, unstructured data, it suffers from a specific type of catastrophic forgetting. 
The weight updates bias the model's internal representations (Function Vectors) toward the new domain. This "Function Vector Shift" actively derails or overwrites the precise internal attention routing required for ICL. Thus, a model can improve its general negative log-likelihood (NLL) on a broad dataset while suffering a total structural collapse of its copy/reasoning circuits. Supervised Fine-Tuning is specifically known to cause severe disruption to these base structural circuits compared to Reinforcement Learning.

## Direct Relevance to Cassandra
Stage 58 / H026 produced a massive behavioral divergence: the Phase 1 model (Part 0, trained only on simple data) scored 0.194 on the zero-shot copy probe. The final flagship model (trained further on text8) scored 0.061 on the identical probe · a collapse down to random chance. This proves that the transition to broad data structurally destroyed the emergent copy circuit, despite NLL improvements.

## Roadmap & Experiment Change
**Hypothesis formulation for Phase 7 (Intake decision)**: I highly recommend prioritizing **H026: The Behavior Axis**. 
Probe the remaining Stage 58 checkpoints to track the exact point of circuit death. We must determine whether the death of the copy circuit tracks the *data source* (the hard shift to text8), the *dose*, or the raw *steps*. Understanding this differential circuit vulnerability is the most critical mechanical insight for building a stable reasoning base before any RL loops can begin.

## Correction addendum (Claude, 2026-07-22, accepted into the record)

The "Direct Relevance to Cassandra" paragraph above inverts the lab's
evidence and is superseded. The `0.194` probe row
(`runs/stage59_cold_letters_probe.json`) belongs to the Stage 58 COLD
85M model trained ONLY on broad text8; the `0.061` row
(`runs/phase5_behavior_letters_probe.json`) belongs to the Phase 4
flagship 201.6M trained ONLY on TinyStories, scored zero-shot. No
lineage went simple-then-broad into those two rows, so no
circuit-destruction event is in evidence; what is in evidence is circuit
FORMATION under diverse data and non-formation under narrow data, with
model size uncontrolled between the rows. The note's literature survey
and its probe-the-ladders recommendation stand and are adopted by H026
(`docs/hypotheses/026-diverse-data-circuit-formation.md`), which tests
the corrected framing and carries the destruction question as a
pre-registered screening read over the CURRICULUM domain-shift lineage,
the one place it can actually be measured.
