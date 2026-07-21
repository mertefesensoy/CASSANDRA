# 2. Verifier-Guided Synthetic Correction Traces

**Relevant Cassandra Stages:** Stages 11-14

## Exhaustive Methodology Review

Teaching small language models (SLMs) to recover from errors is a frontier challenge. When a model predicts the wrong token, the standard supervised fine-tuning approach is to simply penalize the wrong token and upweight the correct token. However, this "sparse" supervision fails for complex tasks because it does not teach the model the *reasoning path* required to detect the error and pivot to the correct answer.

### Project Aletheia: Distilling Backtracking
*Project Aletheia (arXiv:2601.14290)* introduces the concept of "Verifier-Guided Distillation of Backtracking." The core mechanism involves:
1. **Conflict Detection:** Training the model to explicitly output tokens that signal an error (e.g., `<wait>`, `<rethink>`).
2. **Synthetic Trajectory Generation:** Using a larger, stronger model (or a symbolic verifier) to generate a step-by-step reasoning trace that bridges the gap between the model's initial mistake and the correct answer.
3. **Trace Distillation:** Fine-tuning the small model on these synthetic trajectories. The model learns to generate the correction trace itself.

### Fission-GRPO: Group Relative Policy Optimization for Tool Use
*Fission-GRPO (arXiv:2601.15625)* applies this to tool-use and execution errors. Rather than using standard Proximal Policy Optimization (PPO), which relies on a separate value network, GRPO samples multiple trajectories, scores them using a verifier, and normalizes the rewards within the group. 
"Fissioning" refers to explicitly splitting failed trajectories. When an execution error occurs, the trajectory is halted, and a corrective supervision signal is injected directly into the training data. The model learns a specific policy for "what to do immediately after receiving an error signal."

## Relevance to Cassandra's Tiny Language Lab

**Direct Link to Staged Experiments:**
Cassandra explicitly maps to these findings across Stages 11-14:
- **Stage 11 (Failed Case Replay):** Cassandra attempted simple "adaptive error replay," showing the model cases it had failed. This yielded poor results. As predicted by the literature, merely exposing the model to the failure without a correction trace does not teach recovery.
- **Stages 12 & 13 (Synthetic Corrections):** Cassandra introduced `find_verified_copy_targets` (a deterministic verifier) to construct synthetic correction traces. The `prefix` template (`case 0436 key=d noise=c ... answer=`) explicitly bridges the gap, giving the model a structured path to the correct answer. This mirrors the distillation of backtracking, albeit at a tiny character level.

**The Divergence:**
The external literature focuses on massive Chain-of-Thought (CoT) reasoning paths, often spanning hundreds of tokens for math or code generation. Cassandra distills this concept down to a few characters. Furthermore, Cassandra does not use Reinforcement Learning (no PPO/GRPO); it achieves the correction behavior purely through specialized supervised samplers (`copy_correction_template`).

## Roadmap Impact & Experimental Imperatives

1. **Contrastive Correction Context:** Currently, Cassandra's correction templates (like `prefix` and `focus`) completely replace the failed generation. To truly replicate *Project Aletheia's* backtracking, the model must see its own mistake in the context window.
2. **Next Necessary Baseline:** Codex must implement a `contrastive` correction template.
   - **Template Shape:** `case 0435 key=d noise=c ... answer=a [ERROR] answer=d`
   - **Hypothesis:** Forcing the LoRA adapter to process the error token (`a`), the explicit rejection signal (`[ERROR]`), and the subsequent correction (`d`) will create a much stronger gradient signal for robust recovery than simply replacing the prompt. Claude must schedule this as a priority extension to the correction curriculum.
