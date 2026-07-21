# Induction Heads and Zero-Shot Copy Failure

**Date Checked**: 2026-07-21
**Source**: "In-context Learning and Induction Heads" by Olsson et al. (2022) [arXiv:2209.11895]

## What the Source Claims
Olsson et al. found that "induction heads" (a specific attention head mechanism that copies previous patterns, e.g., A -> B, A -> predicts B) are the primary driver of in-context learning (ICL) in transformer language models. These heads emerge during training as a sudden "phase change," correlating sharply with a rapid increase in the model's ability to perform tasks in-context without parameter updates. While prevalent in massive LLMs, they uniquely emerge even in tiny, small language models (as small as 2 layers). Ablating these heads severely degrades the model's ICL abilities, confirming their causal role in zero-shot pattern matching and sequence copying.

## Direct Relevance to Cassandra
In Stage 55, Cassandra's 200M flagship failed the letters-only zero-shot copy probe (0.0605 vs 0.0625 chance). This indicates that the 200M model, despite achieving strong NLL on the broad distribution, has failed to form functioning induction heads or the requisite generalized copying circuit. 

## Analogies and Limits
While Olsson et al. demonstrate induction head emergence in small models, those models were typically trained on massive token counts or specifically constructed tokenized corpora. Cassandra is a character-level model (or small BPE), where the distance between repeated "tokens" (characters) is much larger, placing immense stress on the small attention window (block size 256 or 512). The failure to copy letters zero-shot is a failure of structural generalization, not just a surface-form competition.

## Roadmap & Experiment Change
**Hypothesis formulation for Phase 6**: Cassandra's failure to form induction heads on character-level data may require an explicit "copy curriculum" or synthetic trace injection (similar to Stage 12's verifier-guided loop) to force the emergence of this circuit. 
- *Next Baseline*: Evaluate if fine-tuning the Stage 56 or Stage 58 models on a synthetic copy-and-complete task (like the digits or letters probe) forces the rapid formation of induction heads, or if the model's intrinsic dimension is fundamentally bottlenecked.
