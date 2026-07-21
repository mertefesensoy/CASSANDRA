# NLL Divergence and Behavior-Forming Residuals

## 1. Prior Art and Source Claims

### Perplexity vs. Downstream Accuracy Divergence
- **Date Checked**: 2026-06-24
- **Claim**: In Large Language Models, intrinsic metrics like Negative Log-Likelihood (NLL) or perplexity often diverge from downstream task accuracy. NLL is a global metric that averages across all tokens, meaning it is heavily influenced by easy-to-predict, frequent tokens (like syntax and stopwords). It can mask the model's performance on the rare, critical tokens necessary for complex reasoning, in-context learning, or instruction following. 
- **Cassandra Relevance**: Explains the apparent paradox between Stage 37 and Stage 38. In Stage 37, the residual adapter failed to improve global NLL because the strong count-prior already optimally predicted the frequent, easy tokens. In Stage 38, that same residual adapter proved essential for a specific, high-value behavior (in-context copying).

### The Role of PEFT in "Unlocking" Behavior
- **Date Checked**: 2026-06-24
- **Claim**: Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA often do not drastically alter the global probability distribution (and thus perplexity may stay stable or even regress slightly due to optimization instability). However, they function as a "steering mechanism" or "behavior unlocker." By modifying just a tiny fraction of parameters, PEFT can activate specific architectural pathways (like induction heads) that enable complex downstream behaviors.
- **Cassandra Relevance**: In Stage 38, the rank-2 LoRA residual acts as exactly this "behavior-forming surface." It lacks the capacity to lower the broad NLL of the text distribution, but it possesses exactly the capacity needed to wire up an attention-mediated copy mechanism.

### Mechanistic Views of In-Context Learning (ICL)
- **Date Checked**: 2026-06-24
- **Claim**: ICL relies on activating specific internal circuits (e.g., induction heads for copying). While a frozen statistical prior (like an n-gram model) can achieve excellent NLL by memorizing local token frequencies, it physically lacks the attention mechanism required to perform dynamic, long-distance ICL.
- **Cassandra Relevance**: The frozen count prior in Stage 38 copies at chance (`0.118421`) because it only sees the immediate `answer=` context. It has no mechanism to look back at the prompt key. The trainable residual is strictly required to implement the ICL copying circuit.

## 2. Mapping to Cassandra Stage 38 Findings

Stage 38 tested the "Behavior Residual Marginal-Value Gate" to see if the prior-dominance observed on validation NLL in Stage 37 held true for a specific, attention-mediated task: long-context copying.

**Findings:**
- **Frozen Floor (No Residual)**: Achieved a copy accuracy of `0.118421` (near the `0.125` chance line), despite having strong validation NLL.
- **Rank-2 Residual (AdamW)**: Both `copyw` and `copymix` training regimes cleared the `floor + 0.10` behavior threshold on every seed, reaching mean accuracies of `0.320176` and `0.307017` respectively.
- **NLL vs. Behavior Divergence**: The weighted residual (`copyw`) only improved mean validation NLL by a microscopic `0.013277` over the floor. The mixed residual (`copymix`) actually *worsened* validation NLL by `0.011193`. Yet, both formed functional copy behavior.

**Analysis:**
Stage 38 provides a textbook demonstration of NLL divergence. The Stage 37 conclusion that cheap residuals are "prior-dominated" is strictly an NLL phenomenon. The global loss landscape is dominated by the frozen analytic prior because it handles the bulk of standard text prediction. 

However, on the behavior axis, the relationship inverts. The trainable residual is the **exclusive behavior-forming surface**. The statistical prior cannot perform in-context copying; it requires the trainable attention weights in the residual LoRA to learn the induction circuit. The small rank-2 capacity is insufficient to move global NLL, but it is perfectly sufficient to learn a targeted routing mechanism. 

## 3. Roadmap Changes and Experiment Design

- **Hypothesis Update**: The divergence between NLL and behavior is confirmed. Optimization regimes cannot be judged solely by validation NLL if the goal is to form specific behaviors.
- **Experiment Design**: Future experiments must maintain dual-axis tracking (NLL and a targeted behavior probe). Since the residual is the behavior-forming surface, any attempt to replace AdamW with non-gradient methods (like the failed Stage 36 ES) must be evaluated on whether it can form these specific behavioral circuits, not just whether it minimizes next-token loss. The next steps should explore how to make this behavior-forming surface more robust without vastly increasing parameter count.
