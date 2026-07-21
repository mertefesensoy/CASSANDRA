# 4. External Memory and Indexed Lookups

**Relevant Cassandra Stages:** Stage 18

## Exhaustive Methodology Review

Standard language models encode all world knowledge implicitly within their feed-forward parameter weights. This is highly inefficient; storing rare facts requires massive parameter counts, and updating facts requires expensive retraining. External memory models decouple computation from storage by allowing a frozen or semi-frozen network to query an explicit datastore during inference.

### kNN-LM: Generalization through Memorization
*kNN-LM (arXiv:1911.00172)* introduces a non-parametric method to augment pre-trained language models.
1. **The Datastore:** An offline pass over a massive corpus constructs a key-value datastore. The keys are the high-dimensional hidden states from the last layer of the LM; the values are the target next tokens.
2. **Inference Interpolation:** At generation time, the model computes its standard parametric distribution over the vocabulary. Simultaneously, it uses its current hidden state to perform a $k$-nearest neighbors search over the datastore. The distances are converted into a probability distribution over the retrieved tokens.
3. **The Result:** The final prediction is a learned interpolation between the parametric distribution and the kNN distribution. This drastically improves perplexity on rare patterns without a single gradient update to the model weights.

### RETRO: Retrieving from Trillions of Tokens
*RETRO (arXiv:2112.04426)* integrates retrieval directly into the network architecture via cross-attention.
1. **Chunked Retrieval:** The input is split into chunks. Each chunk is encoded and used to query a massive chunked database.
2. **Chunked Cross-Attention:** The retrieved chunks are incorporated into the LM's forward pass using specialized Chunked Cross-Attention (CCA) layers interspersed with standard self-attention layers.
3. **The Result:** A 7 billion parameter RETRO model matches the performance of a 175 billion parameter dense model (like GPT-3) on knowledge-intensive tasks, proving that external memory is vastly more parameter-efficient than storing facts in weights.

## Relevance to Cassandra's Tiny Language Lab

**Direct Link to Staged Experiments:**
Cassandra's Stage 18 explicitly tests this core mechanic (`--copy-probe-retrieval-source memory`). Rather than the model relying on its training weights to copy the target, it must perform a lookup. The training split constructs a tiny explicit key-value table. At probe time, the local context key is used to "query" the table, and the resulting answer is formatted as a prefix hint.

**The Divergence:**
*RETRO* and *kNN-LM* use deep, high-dimensional vector embeddings for continuous retrieval. Cassandra uses discrete, exact-match string lookup at the python level before the forward pass. Furthermore, Cassandra's memory is vanishingly small (8 possible entries). The goal in Cassandra is not to scale the datastore, but to see if a rank-2 LoRA surface can learn the *interface* of reading from an external hint rather than its local prior.

## Roadmap Impact & Experimental Imperatives

1. **The Limits of Static Retrieval:** Stage 18 proved that the full model could use the static memory string perfectly, but the LoRA path failed. A static prefix hint is insufficient for a highly bottlenecked adapter.
2. **Next Necessary Baseline:** Claude must design an experiment that tests active, continuous memory interactions rather than static prompt injection.
   - **Hypothesis:** Implementing a kNN-LM style continuous caching mechanism—where the model can write arbitrary tokens to a buffer during early generation and read them during late generation via cross-attention—will allow the LoRA path to solve the memory task.
   - **Execution:** Codex must modify `TinyTransformer` to include an explicit external read/write buffer accessible via a custom attention mask.
