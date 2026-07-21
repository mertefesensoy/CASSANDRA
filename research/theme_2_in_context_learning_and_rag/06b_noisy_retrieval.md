# 5. Noisy Retrieval and Ambiguity

**Relevant Cassandra Stages:** Stage 19+

## Exhaustive Methodology Review

Early RAG systems assumed that a vector database would always return exactly one clean, relevant fact. In open-domain scenarios, this is almost never true. External memory lookups are inherently "noisy": they return top-$k$ documents where many are irrelevant distractors, some contain contradictory evidence, and sometimes the correct answer is missing entirely.

### Self-RAG: Learning to Retrieve, Generate, and Critique
*Self-RAG (arXiv:2310.11511)* trains models to actively evaluate their retrieved context using "reflection tokens."
1. **Retrieval Tokens:** The model learns when to emit a `[Retrieve]` token to query the database.
2. **Critique Tokens:** Once passages are retrieved, the model evaluates them by generating critique tokens like `[Relevant]` or `[Irrelevant]`.
3. **Support Tokens:** After generating an answer based on a relevant passage, it generates a token like `[Fully supported]` or `[Partially supported]` to grade its own hallucination rate.
By explicitly unrolling this critique trace, the model learns to filter out noisy distractors rather than blindly appending them to its generation.

### Lost in the Middle
*Lost in the Middle (arXiv:2307.03172)* exposes a severe architectural flaw in how transformers process long contexts filled with retrieved documents.
1. **The U-Shaped Curve:** Models perform well at extracting facts when the relevant document is placed at the very beginning of the context window or at the very end.
2. **Middle Collapse:** If the relevant fact is buried in the middle of 10-20 irrelevant distractors, extraction performance plummets. The attention mechanism diffuses across the noisy context, losing the signal.

## Relevance to Cassandra's Tiny Language Lab

**Direct Link to Staged Experiments:**
In **Stage 18**, Cassandra's memory probe was "clean": every validation key was present in the training split, and the retrieval hint was always correct.
In **Stage 19**, Codex ran a "Memory Corruption Ablation," deliberately injecting noise by replacing the retrieved answer with a wrong key. The full model's accuracy dropped significantly, but not completely to chance. It fell back on its local key prior. This proves that the model is vulnerable to noisy retrieval, but currently lacks an explicit mechanism to "critique" or reject the bad hint.

**The Divergence:**
The external literature tests this with massive Wikipedia articles and dense vector distractors. Cassandra tests this in a completely stripped-down synthetic environment: a corrupted hint `answer=f` vs a local key `key=d`. 

## Roadmap Impact & Experimental Imperatives

1. **Simulating the Distractor Environment:** Cassandra must graduate from single-hint retrieval to multi-candidate retrieval to truly test noise robustness.
2. **Next Necessary Baseline:** Codex must modify the retrieval sampler to inject multiple hints, where only one is correct.
   - **Mechanism:** The prompt must include both a signal and a distractor, e.g., `retrieved: answer=c; retrieved: answer=a; key=a`.
   - **Hypothesis:** Following *Self-RAG*, the model will fail to consistently select the correct hint from the distractors unless it is explicitly trained to generate a synthetic critique trace before answering.
   - **Execution:** Claude must schedule a stage where the training corpus includes explicitly synthesized critique paths: `retrieved c. c is noise. retrieved a. a is answer.` This will test if the LoRA path can learn active filtering.
