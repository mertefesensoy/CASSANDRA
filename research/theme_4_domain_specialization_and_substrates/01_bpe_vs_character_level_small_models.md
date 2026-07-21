# Subword (BPE) vs Character Level Tokenization Performance at Small Scale

*   **Date Checked:** 2026-07-07
*   **Topic:** Evaluating Byte Pair Encoding (BPE) against character-level/byte-level tokenization for parameter-constrained Small Language Models (SLMs).

## Core Prior Art: Charformer and ByT5

Two seminal papers address the trade-offs between character-level (or byte-level) and subword-level tokenization in modern transformers:
1.  **"ByT5: Towards a token-free future with pre-trained byte-to-byte models" (Xue et al., 2022)**
2.  **"Charformer: Fast Character Transformers via Gradient-based Subword Tokenization" (Tay et al., 2022)**

### 1. ByT5: Methodologies, Metrics, and Limitations

**Methodology:**
ByT5 completely abandons subword tokenizers (like SentencePiece or WordPiece) in favor of treating raw UTF-8 bytes as the atomic input tokens. It maps text to 256 possible byte values. The architecture uses a standard T5 sequence-to-sequence transformer, but shifts the parameter allocation heavily. Because the vocabulary matrix shrinks from typical sizes (e.g., 32,000 or 250,000 tokens) down to just 256, the embedding table becomes negligible. The authors repurpose those saved parameters into deeper transformer layers (more compute). 

**Metrics & Findings:**
ByT5 models demonstrate superior robustness to spelling errors, casing variations, and noise compared to standard T5 models. However, because byte sequences are significantly longer than subword sequences (typically 2.5x to 4x longer), the model requires much more compute (FLOPs) during both training and inference. To compensate for the "ballooning" sequence length, ByT5 unbalances the encoder-decoder depth, using a very deep encoder and a shallow decoder.

**Limitations:**
The primary limitation is the quadratic scaling of attention over long sequences. At small parameter scales (where compute and memory are heavily constrained), the model spends a vast majority of its capacity simply composing bytes into coherent words rather than learning high-level semantic reasoning.

### 2. Charformer: Methodologies, Metrics, and Limitations

**Methodology:**
Charformer attempts to hybridize the robustness of character models with the efficiency of subword models. It introduces the Gradient-based Subword Tokenization (GBST) module. Instead of using a fixed, deterministic tokenizer (like BPE), Charformer enumerates potential subword blocks (e.g., chunks of 1 to 4 characters) and uses a trainable block scoring network with a softmax function to score them. The latent subword representation is a weighted sum of these candidate blocks, trained end-to-end.

**Metrics & Findings:**
Charformer achieves speed improvements of 28% to 100% over standard character-level transformers by effectively downsampling the sequence length internally, recovering the speed and attention-efficiency of BPE while retaining the out-of-vocabulary (OOV) robustness of character models.

**Limitations:**
The GBST module adds architectural complexity and local parameter overhead. It is still computationally heavier than a pure pre-tokenized BPE model, and the downsampling approach can sometimes obscure fine-grained character-level tasks where exact position matters.

## Mapping to Cassandra's Specific Stages

### Cassandra Stage 49 & 50 (BPE Smoke and Matrix)
Cassandra explicitly tested the BPE-vs-Character tradeoff at the tiny scale. 
*   **Stage 49:** Introduced a tiny local BPE tokenizer (vocab size 256) on the TinyStories corpus, achieving an compression of `2.238` source characters per BPE token.
*   **Stage 50:** Evaluated a BPE-token bigram prior plus rank-2 LoRA against a full BPE model. The result was a stark failure for the frozen prior arm (`3.344` NLL) compared to full training (`2.404` NLL). More importantly, the full BPE model's performance roughly translated to `1.549` bits/source-char, which was drastically *worse* than the Stage 45 character-level baseline (which reached `1.052` NLL on characters).

**Synthesis:** 
Cassandra's Stage 50 reflects the exact limitation described in the literature for highly constrained models: when the vocabulary is tiny (256 BPE tokens), the embedding table is small, but the model still suffers from sequence length inefficiency without the semantic density of a true large-vocabulary BPE (e.g., 32k). 

### Phase 5 Roadmap Adjustments
The literature strongly suggests that for a small model (85M sizing control in Phase 5) to learn broad, zero-shot generalization (e.g., closing the text8 gap), it needs the semantic density of a true subword tokenizer. The pure character-level model (Stage 55, 201.6M parameters) spends too much capacity on word formation. 
*   **Next Experiment:** Phase 5 must implement a larger vocabulary BPE (e.g., 5k-10k) to drastically shorten the sequence length and increase semantic density per token, accepting the embedding table parameter cost in exchange for compute efficiency over a broader corpus.
