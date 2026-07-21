# Tiny Language Lab

This is Cassandra's first executable testbed for the question:

> Can a very small language model be formed with less than normal brute-force training?

The lab starts with a character-level bigram model and then adds a tiny causal
transformer. The bigram model is intentionally small: the whole parameter surface
is a `vocab_size x vocab_size` matrix, which makes it possible to compare three
ways of forming the same parameters.

## Bigram methods

- `count` constructs logits from smoothed bigram counts. This is the no-gradient baseline.
- `coordinate` changes one parameter at a time and keeps the change if validation loss improves.
- `gradient` trains the same logits with AdamW.

## Run the bigram lab

```powershell
python .\experiments\tiny_language_lab\cassandra_tiny_lm.py --method count --prompt "cassandra "
python .\experiments\tiny_language_lab\cassandra_tiny_lm.py --method coordinate --steps 300 --prompt "cassandra "
python .\experiments\tiny_language_lab\cassandra_tiny_lm.py --method gradient --steps 300 --prompt "cassandra "
```

Optional:

```powershell
python .\experiments\tiny_language_lab\cassandra_tiny_lm.py --method coordinate --init count --steps 300
python .\experiments\tiny_language_lab\cassandra_tiny_lm.py --method gradient --out experiments\tiny_language_lab\last_logits.pt
```

## Run the transformer baseline

```powershell
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --steps 200 --prompt "cassandra "
```

Structured initialization:

```powershell
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --init count-bigram --steps 50 --prompt "cassandra "
```

Frozen count prior with a learned residual path:

```powershell
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --residual-base count-bigram --residual-l2 1.0 --steps 50 --prompt "cassandra "
```

Small trainable residual surfaces:

```powershell
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --residual-base count-bigram --train-scope head --steps 50 --prompt "cassandra "
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --residual-base count-bigram --train-scope adapters --adapter-rank 4 --steps 50 --prompt "cassandra "
```

Generate the larger structured corpus and run the replication matrix:

```powershell
python .\experiments\tiny_language_lab\make_synthetic_corpus.py --lines 320 --seed 20260616
python .\experiments\tiny_language_lab\cassandra_compare.py --steps 50 --eval-batches 16 --seeds 7 11 19
```

Run the LoRA comparison matrix:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --steps 50 --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_head count_prior_adapter_r4 count_prior_lora_r1 count_prior_lora_r2 --out .\experiments\tiny_language_lab\runs\stage6_lora.jsonl --summary .\experiments\tiny_language_lab\runs\stage6_lora.md --title "Stage 6 LoRA Summary"
```

Run the long-context copy probe:

```powershell
python .\experiments\tiny_language_lab\make_long_context_corpus.py --lines 512 --seed 20260617
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 100 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --seeds 7 11 19 --configs random_full count_prior_head count_prior_adapter_r4 count_prior_lora_r2 --out .\experiments\tiny_language_lab\runs\stage7_long_context.jsonl --summary .\experiments\tiny_language_lab\runs\stage7_long_context.md --title "Stage 7 Long-Context Summary"
```

Run task-aware copy-position training:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --seeds 7 11 19 --configs random_full_copyw count_prior_lora_r2_copyw --out .\experiments\tiny_language_lab\runs\stage8_copy_weight200.jsonl --summary .\experiments\tiny_language_lab\runs\stage8_copy_weight200.md --title "Stage 8 Copy Weight 200 Summary"
```

Run verifier-guided mixed sampling:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-sample-fraction 0.25 --copy-loss-weight 200 --seeds 7 11 19 --configs random_full_copymix count_prior_lora_r2_copymix --out .\experiments\tiny_language_lab\runs\stage9_mixed_sampler_weight200.jsonl --summary .\experiments\tiny_language_lab\runs\stage9_mixed_sampler_weight200.md --title "Stage 9 Mixed Sampler Weight 200"
```

Run verifier choice-loss correction:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 50 --copy-choice-weight 0.25 --seeds 7 11 19 --configs random_full_copyw_choice count_prior_lora_r2_copyw_choice --out .\experiments\tiny_language_lab\runs\stage10_choice_loss_weight025.jsonl --summary .\experiments\tiny_language_lab\runs\stage10_choice_loss_weight025.md --title "Stage 10 Choice Loss Weight 0.25"
```

Run failed-case replay:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.1 --copy-mine-every 100 --seeds 7 11 19 --configs random_full_copyfailmix count_prior_lora_r2_copyfailmix --out .\experiments\tiny_language_lab\runs\stage11_failed_mixed_replay_weight200.jsonl --summary .\experiments\tiny_language_lab\runs\stage11_failed_mixed_replay_weight200.md --title "Stage 11 Failed Mixed Replay Weight 200"
```

Run generated correction examples:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage12_correction_examples_gentle_500.jsonl --summary .\experiments\tiny_language_lab\runs\stage12_correction_examples_gentle_500.md --title "Stage 12 Gentle Correction Examples 500"
```

Run correction-template comparison:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage13_prefix_corrections_500.jsonl --summary .\experiments\tiny_language_lab\runs\stage13_prefix_corrections_500.md --title "Stage 13 Prefix Corrections 500"
```

Run retrieval-style copy probe:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage14_retrieval_probe_compact.jsonl --summary .\experiments\tiny_language_lab\runs\stage14_retrieval_probe_compact.md --title "Stage 14 Retrieval Probe Compact"
```

Run retrieval-use training:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_retmix count_prior_lora_r2_retmix --out .\experiments\tiny_language_lab\runs\stage15_retrieval_use_training.jsonl --summary .\experiments\tiny_language_lab\runs\stage15_retrieval_use_training.md --title "Stage 15 Retrieval Use Training"
```

Run the correction-retrieval curriculum:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_corrretmix count_prior_lora_r2_corrretmix --out .\experiments\tiny_language_lab\runs\stage16_correction_retrieval_curriculum.jsonl --summary .\experiments\tiny_language_lab\runs\stage16_correction_retrieval_curriculum.md --title "Stage 16 Correction Retrieval Curriculum"
```

Run the staged correction-then-retrieval curriculum:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-curriculum-switch-fraction 0.5 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_corrthenret count_prior_lora_r2_corrthenret --out .\experiments\tiny_language_lab\runs\stage17_staged_correction_then_retrieval.jsonl --summary .\experiments\tiny_language_lab\runs\stage17_staged_correction_then_retrieval.md --title "Stage 17 Staged Correction Then Retrieval"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-curriculum-switch-fraction 0.8 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_corrthenret count_prior_lora_r2_corrthenret --out .\experiments\tiny_language_lab\runs\stage17_staged_correction_then_retrieval_late.jsonl --summary .\experiments\tiny_language_lab\runs\stage17_staged_correction_then_retrieval_late.md --title "Stage 17 Staged Correction Then Retrieval Late Switch"
```

Run the train-split memory retrieval probe:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage18_memory_retrieval_probe.jsonl --summary .\experiments\tiny_language_lab\runs\stage18_memory_retrieval_probe.md --title "Stage 18 Memory Retrieval Probe"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-train-marker "answer=" --copy-loss-weight 10 --seeds 7 11 19 --configs random_full_copyw count_prior_lora_r2_copyw --out .\experiments\tiny_language_lab\runs\stage18_memory_retrieval_baseline.jsonl --summary .\experiments\tiny_language_lab\runs\stage18_memory_retrieval_baseline.md --title "Stage 18 Memory Retrieval Baseline"
```

Run the memory-reliance corruption ablation:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-probe-retrieval-corrupt wrong-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage19_memory_corruption_ablation.jsonl --summary .\experiments\tiny_language_lab\runs\stage19_memory_corruption_ablation.md --title "Stage 19 Memory Corruption Ablation"
```

Run the LoRA capacity sweep for curriculum interference:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-curriculum-switch-fraction 0.8 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs count_prior_lora_r4_corrthenret count_prior_lora_r8_corrthenret --out .\experiments\tiny_language_lab\runs\stage20_rank_sweep_staged.jsonl --summary .\experiments\tiny_language_lab\runs\stage20_rank_sweep_staged.md --title "Stage 20 Rank Sweep Staged 0.8"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs count_prior_lora_r4_corrretmix count_prior_lora_r8_corrretmix --out .\experiments\tiny_language_lab\runs\stage20_rank_sweep_simultaneous.jsonl --summary .\experiments\tiny_language_lab\runs\stage20_rank_sweep_simultaneous.md --title "Stage 20 Rank Sweep Simultaneous"
```

Run the non-identity memory mapping probe:

```powershell
python .\experiments\tiny_language_lab\make_memory_mapping_corpus.py --lines 512 --seed 20260618 --out .\experiments\tiny_language_lab\corpus\memory_mapping_seed.txt
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage21_memory_mapping_no_hint.jsonl --summary .\experiments\tiny_language_lab\runs\stage21_memory_mapping_no_hint.md --title "Stage 21 Memory Mapping No Hint"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage21_memory_mapping_correct.jsonl --summary .\experiments\tiny_language_lab\runs\stage21_memory_mapping_correct.md --title "Stage 21 Memory Mapping Correct Memory"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-probe-retrieval-corrupt wrong-answer --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage21_memory_mapping_corrupt.jsonl --summary .\experiments\tiny_language_lab\runs\stage21_memory_mapping_corrupt.md --title "Stage 21 Memory Mapping Corrupted Memory"
```

Run the held-out external memory value test:

```powershell
python .\experiments\tiny_language_lab\make_memory_mapping_corpus.py --lines 512 --seed 20260618 --holdout-keys g h --out .\experiments\tiny_language_lab\corpus\memory_mapping_holdout_seed.txt
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_holdout_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template none --copy-probe-holdout-keys g h --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_retmix count_prior_lora_r2_retmix --out .\experiments\tiny_language_lab\runs\stage22_holdout_memory_no_hint.jsonl --summary .\experiments\tiny_language_lab\runs\stage22_holdout_memory_no_hint.md --title "Stage 22 Holdout Memory No Hint"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_holdout_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-probe-memory-scope all --copy-probe-holdout-keys g h --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_retmix count_prior_lora_r2_retmix --out .\experiments\tiny_language_lab\runs\stage22_holdout_memory_correct.jsonl --summary .\experiments\tiny_language_lab\runs\stage22_holdout_memory_correct.md --title "Stage 22 Holdout Memory Correct"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_holdout_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-probe-memory-scope all --copy-probe-retrieval-corrupt wrong-answer --copy-probe-holdout-keys g h --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_retmix count_prior_lora_r2_retmix --out .\experiments\tiny_language_lab\runs\stage22_holdout_memory_corrupt.jsonl --summary .\experiments\tiny_language_lab\runs\stage22_holdout_memory_corrupt.md --title "Stage 22 Holdout Memory Corrupt"
```

Run rank-2 staged rehearsal:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-rehearsal-fraction 0.05 --copy-curriculum-switch-fraction 0.8 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs count_prior_lora_r2_corrthenret_rehearsal --out .\experiments\tiny_language_lab\runs\stage23_rank2_rehearsal_frac005.jsonl --summary .\experiments\tiny_language_lab\runs\stage23_rank2_rehearsal_frac005.md --title "Stage 23 Rank 2 Rehearsal Fraction 0.05"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-rehearsal-fraction 0.10 --copy-curriculum-switch-fraction 0.8 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs count_prior_lora_r2_corrthenret_rehearsal --out .\experiments\tiny_language_lab\runs\stage23_rank2_rehearsal_frac010.jsonl --summary .\experiments\tiny_language_lab\runs\stage23_rank2_rehearsal_frac010.md --title "Stage 23 Rank 2 Rehearsal Fraction 0.10"
```

Run the corpus-complexity regime sweep:

```powershell
python .\experiments\tiny_language_lab\make_complexity_corpus.py --lines 512 --seed 20260619 --long-fraction 0.00 --out .\experiments\tiny_language_lab\corpus\complexity_p000_seed.txt
python .\experiments\tiny_language_lab\make_complexity_corpus.py --lines 512 --seed 20260619 --long-fraction 0.25 --out .\experiments\tiny_language_lab\corpus\complexity_p025_seed.txt
python .\experiments\tiny_language_lab\make_complexity_corpus.py --lines 512 --seed 20260619 --long-fraction 0.50 --out .\experiments\tiny_language_lab\corpus\complexity_p050_seed.txt
python .\experiments\tiny_language_lab\make_complexity_corpus.py --lines 512 --seed 20260619 --long-fraction 0.75 --out .\experiments\tiny_language_lab\corpus\complexity_p075_seed.txt
python .\experiments\tiny_language_lab\make_complexity_corpus.py --lines 512 --seed 20260619 --long-fraction 1.00 --out .\experiments\tiny_language_lab\corpus\complexity_p100_seed.txt
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\complexity_p050_seed.txt --steps 50 --block-size 96 --eval-mode sampled --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_head count_prior_lora_r1 count_prior_lora_r2 --out .\experiments\tiny_language_lab\runs\stage24_complexity_p050_s50.jsonl --summary .\experiments\tiny_language_lab\runs\stage24_complexity_p050_s50.md --title "Stage 24 Complexity p050 50 steps"
```

Run the matrix command above for each `complexity_p000`, `p025`, `p050`, `p075`,
and `p100` corpus at both 50 and 100 steps.

Run the low-step time-budget surface:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\complexity_p050_seed.txt --steps 10 --block-size 96 --eval-mode sampled --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_head count_prior_lora_r1 count_prior_lora_r2 --out .\experiments\tiny_language_lab\runs\stage25_timebudget_p050_s10.jsonl --summary .\experiments\tiny_language_lab\runs\stage25_timebudget_p050_s10.md --title "Stage 25 Time Budget p050 10 steps"
```

Run the matrix command above for each `complexity_p000`, `p025`, `p050`, `p075`,
and `p100` corpus at both 10 and 25 steps.

Run the order-matched analytic prior durability test:

```powershell
python .\experiments\tiny_language_lab\make_markov_corpus.py --order 2 --vocab 16 --lines 512 --seed 20260620 --out .\experiments\tiny_language_lab\corpus\markov_order2_seed.txt
python .\experiments\tiny_language_lab\make_markov_corpus.py --order 1 --vocab 16 --lines 512 --seed 20260620 --out .\experiments\tiny_language_lab\corpus\markov_order1_seed.txt
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\markov_order2_seed.txt --steps 100 --block-size 96 --eval-mode sampled --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_lora_r2 count_prior_tri_lora_r2 --out .\experiments\tiny_language_lab\runs\stage26_markov_order2_s100.jsonl --summary .\experiments\tiny_language_lab\runs\stage26_markov_order2_s100.md --title "Stage 26 Markov Order2 100 steps"
```

Run the order-2 matrix command above at steps 10, 25, 50, 100, and 200. Run the
same matrix on `markov_order1_seed.txt` at steps 50 and 100 for the order-1
control.

Complete the Stage 27 order-1/order-2 closeout:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\markov_order1_seed.txt --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_lora_r2 count_prior_tri_lora_r2 --out .\experiments\tiny_language_lab\runs\stage27_matchsurface_order1_s200.jsonl --summary .\experiments\tiny_language_lab\runs\stage27_matchsurface_order1_s200.md --title "Stage 27 Match Surface Order1 200 steps"
```

Run the command above at steps 10, 25, and 200. Stage 26 already supplied the
order-1 rows at 50 and 100 and all order-2 rows at 10, 25, 50, 100, and 200.
The aggregate file is
`experiments/tiny_language_lab/runs/stage27_matchsurface_summary.md`.
Claude's H008 defined the next, stronger surface: rerun at `V = 8`, add source
order 3 and prior order 3, and use a general order-n count prior with sparsity
diagnostics. Stage 28 implemented that surface.

Run the H008 `V = 8` order surface:

```powershell
python .\experiments\tiny_language_lab\make_markov_corpus.py --order 3 --vocab 8 --lines 512 --line-length 80 --seed 20260621 --out .\experiments\tiny_language_lab\corpus\markov_order3_v8_seed.txt
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\markov_order3_v8_seed.txt --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_ng1_lora_r2 count_prior_ng2_lora_r2 count_prior_ng3_lora_r2 --out .\experiments\tiny_language_lab\runs\stage28_h008_s3_b200.jsonl --summary .\experiments\tiny_language_lab\runs\stage28_h008_s3_b200.md --title "Stage 28 H008 source 3 budget 200"
```

Run the corpus generator for source orders 1, 2, and 3, then run the matrix
command above for budgets 10, 25, 50, 100, and 200. The aggregate file is
`experiments/tiny_language_lab/runs/stage28_h008_summary.md`.

Useful small GPU smoke test:

```powershell
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --device cuda --steps 20 --eval-interval 10 --max-new-tokens 80
```

Use `--device cpu` only for tiny diagnostics when CUDA is temporarily unavailable.

The transformer uses deterministic full-split evaluation by default. For larger
corpora, switch to sampled evaluation with `--eval-mode sampled --eval-batches 8`.

## What to look for

The important comparison is not sample beauty. It is:

- Does the method reduce validation loss?
- How many parameters were trainable or changed?
- How many seconds did it take?
- Does a structured seed help more than random or zero initialization?

Current verified numbers are recorded in `RESULTS.md`.

Stage 24's latest regime result: at 50 steps, the frozen count prior plus rank-2
LoRA beats the full random transformer through long fraction `p = 0.50` and loses
at `p = 0.75` and `1.00`. At 100 steps, full training wins on every measured
point, so the cheap recipe is currently best described as an early-budget
accelerator whose boundary must be mapped more finely.

Stage 25's latest time-budget result: the cheap recipe wins across the full
measured corpus axis at 10 and 25 steps. The crossover is above the measured
range at 10 and 25 steps, about `p = 0.635343` at 50 steps, and below the
measured range by 100 steps. This scopes the current positive claim as an
early-compute inductive-bias advantage.

Stage 26's latest higher-order result: on a pure order-2 Markov source, the
matched trigram prior stays strongly ahead through 200 steps, while the
mismatched bigram prior decays to tied. On a pure order-1 source, the matched
bigram prior stays positive and the over-specified trigram prior does not help.

Stage 27's latest closeout result: across source orders 1 and 2 and prior orders
1 and 2 at `V = 16`, matched priors stayed positive through 200 steps. The order-1
source with a bigram prior retained `0.018835` NLL advantage at 200 steps, and
the order-2 source with a trigram prior retained `0.637384`. The order-2 source
with a bigram prior decayed to tied, and the order-1 source with a trigram prior
turned negative by 100 and 200 steps.

Stage 28's H008 result: at `V = 8`, the matched diagonal stayed positive and
increased at 200 steps: `A(1,1)=0.006804`, `A(2,2)=0.436673`, and
`A(3,3)=0.630598`. The strict lower-triangle prediction is partial because
`A(3,2)=0.106932` remains meaningfully positive. The best current rule is graded:
match source order when possible, expect severe under-specification to fail,
allow one-step under-specification to retain some lower-order structure, and
treat over-specification as a sparsity/backoff tradeoff.

Stage 29's tiny-prose smoke result: on the 1,129-character `tiny_seed.txt`
project-prose corpus, higher-order frozen priors beat `random_full` at 10, 50,
and 100 steps. At 100 steps, ng3 advantage was `0.399935` NLL, ng2 was
`0.350333`, and ng1 was `0.217754`. This is exploratory evidence only, because
the corpus is too small for a natural-language claim.

Stage 30's H009 natural-text result: on normalized Tiny Shakespeare
(`1,100,721` chars, `V=33`), `count_prior_ng2_lora_r2` and
`count_prior_ng3_lora_r2` keep a positive advantage over `random_full` through
500 steps with recursive backoff smoothing (`count_alpha=0.1`,
`ngram_backoff=10`). At 500 steps, ng2 advantage is `+0.110081` NLL and ng3 is
`+0.340641`; ng1 falls to `-0.266069`. Validation hit coverage is high
(`0.999927` for order 2 and `0.995997` for order 3), so this is not the Stage 29
starvation artifact. The exact humped sweet-spot claim remains open because the
measured curve is monotone increasing through order 3.

Stage 31's order-4 extension: `count_prior_ng4_lora_r2` is feasible on the same
normalized corpus, with `42,802,056` frozen logits. It becomes the best measured
prior at every budget. At 500 steps, ng4 advantage is `+0.463572` NLL, above ng3
at `+0.340641`. The descending limb remains unmeasured because order-4 validation
hit coverage is still high (`0.961867`) despite sparse table coverage
(`0.029713`). Stage 32 tested one such harsher validation gate.

Stage 32's cross-domain validity gate: Codex kept the Stage 30 Shakespeare train
prefix but replaced the validation suffix with normalized Cassandra project prose
in the same `V=33` character set. The split lowered high-order validation-hit
coverage, but not enough to produce the predicted hump. Order 4 remained the best
measured prior at the decision budgets. At 500 steps, ng4 advantage was
`+0.291191` NLL, above ng3 at `+0.199130` and ng2 at `+0.016767`; ng1 was
negative at `-0.203702`. This locally kills H009b on the implemented split while
leaving the caveat that a more genre-matched public-domain validation source might
still be harsher. See `experiments/tiny_language_lab/runs/stage32_crossdomain_summary.md`.
Stage 33's H010 curriculum-filter result: the mixed prior-loss sampler did not
speed order-2 rank-2 residual convergence on normalized Tiny Shakespeare. The
uniform 200-step target was `2.040189` mean validation NLL. `f=0.25` reached it
only at 200 steps, `f=0.50` did not reach it through 500 steps, and pure
high-loss `f=1.00` was worse at every budget. At 500 steps, uniform was
`2.050701`, `f=0.25` was `2.051195`, `f=0.50` was `2.052930`, and `f=1.00` was
`2.065845`. This kills H010 for the fixed top-10-percent frozen-prior-NLL
sampler. See `experiments/tiny_language_lab/runs/stage33_filter_summary.md`.

Stage 34's H011 dynamic reducible-loss result: re-scoring a fixed 4096-window
pool every 25 steps did not make the rank-2 residual reach the uniform 200-step
target faster. The target was again `2.040189` mean validation NLL. Dynamic
`f=0.50` and dynamic `f=0.25` did not reach it at any measured budget, including
500 steps. At 500 steps, uniform was `2.050701`, dynamic `f=0.50` was
`2.060916`, and dynamic `f=0.25` was `2.053026`. Re-scoring also added
wall-clock overhead, so H011 kills the final data-side selection branch for this
ladder. See `experiments/tiny_language_lab/runs/stage34_dynfilter_summary.md`.

Stage 35's H012 frozen-recency result: adding an analytic exponential-recency
cache to the order-2 count prior did not improve natural-text validation NLL. At
500 steps, order-2 count-only was `2.050701`, order-2 plus recency was
`2.112697`, and the order-3 count diagnostic was `1.820142`. Recency was worse
than order-2 count-only at every measured budget and slower at equal steps. This
kills the default character-recency interpolation, but not every possible
model-side frozen primitive. Gemini note 08 frames the failure as character-level
cache noise and points toward order-preserving frozen kernels or n-gram caches.
See `experiments/tiny_language_lab/runs/stage35_recency_summary.md`.

Stage 36's H013 non-gradient residual result: Codex added a residual optimizer
switch and compared the frozen-prior floor, rank-2 AdamW, full rank-2 ES, and a
rank-1 coordinate feasibility arm on the structured corpus. The AdamW residual
improved the floor from `2.018509` to `2.000801` mean validation NLL. ES missed
the floor at `2.060750` despite `803` formation forward passes, while rank-1
coordinate reached `2.011522`, about `39.5%` mean gap recovery but with unstable
per-seed behavior. H013 does not pass; the current recipe still needs gradients
for the rank-2 residual. See `experiments/tiny_language_lab/runs/stage36_h013.md`.

Run a Stage 36 non-gradient residual matrix:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\structured_seed.txt --device cuda --steps 50 --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_lora_r2_floor count_prior_lora_r2 count_prior_lora_r2_es count_prior_lora_r1_coord --out .\experiments\tiny_language_lab\runs\stage36_h013.jsonl --summary .\experiments\tiny_language_lab\runs\stage36_h013.md --title "Stage 36 H013 Non-Gradient Residual Formation"
```

Stage 37's residual marginal-value gate: Codex measured the floor-to-target gap
using the Stage 36 `--residual-optim none` floor switch. Natural text at 200 and
500 steps, orders 2 to 4, showed mixed or negative residual gaps. The structured
rank sweep was stable and positive but small: rank 1 `+0.010810`, rank 2
`+0.017708`, and rank 4 `+0.023058` mean validation NLL. The gate closes because
no regime reached the `0.05` reopening line, and the largest stable gap stayed
below `0.03`. See `experiments/tiny_language_lab/runs/stage37_residualgap_summary.md`.

Run the Stage 37 natural-text residual-gap matrix:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\natural_text_seed.txt --device cuda --steps 500 --eval-batches 16 --seeds 7 11 19 --configs count_prior_ng2_lora_r2_floor count_prior_ng2_lora_r2 count_prior_ng3_lora_r2_floor count_prior_ng3_lora_r2 count_prior_ng4_lora_r2_floor count_prior_ng4_lora_r2 random_full --out .\experiments\tiny_language_lab\runs\stage37_residualgap_natural.jsonl --summary .\experiments\tiny_language_lab\runs\stage37_residualgap_natural.md --title "Stage 37 Residual Marginal-Value Gate (natural text)"
```

Run the Stage 37 structured rank sweep:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\structured_seed.txt --device cuda --steps 50 --eval-batches 16 --seeds 7 11 19 --configs count_prior_lora_r1_floor count_prior_lora_r1 count_prior_lora_r2_floor count_prior_lora_r2 count_prior_lora_r4_floor count_prior_lora_r4 random_full --out .\experiments\tiny_language_lab\runs\stage37_residualgap_rank.jsonl --summary .\experiments\tiny_language_lab\runs\stage37_residualgap_rank.md --title "Stage 37 Residual Marginal-Value Gate (rank sweep)"
```

Run the Stage 32 cross-domain corpus builder:

```powershell
python .\experiments\tiny_language_lab\make_cross_domain_corpus.py --train-source .\experiments\tiny_language_lab\corpus\natural_text_seed.txt --out .\experiments\tiny_language_lab\corpus\natural_text_crossdomain_seed.txt
```

Run a Stage 32 budget point:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\natural_text_crossdomain_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --count-alpha 0.1 --ngram-backoff 10 --seeds 7 11 19 --configs random_full count_prior_ng1_lora_r2 count_prior_ng2_lora_r2 count_prior_ng3_lora_r2 count_prior_ng4_lora_r2 --out .\experiments\tiny_language_lab\runs\stage32_crossdomain_b500.jsonl --summary .\experiments\tiny_language_lab\runs\stage32_crossdomain_b500.md --title Stage_32_Cross_Domain_500_steps
```

Run a Stage 33 curriculum-filter budget point:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\natural_text_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --count-alpha 0.1 --ngram-backoff 10 --seeds 7 11 19 --configs count_prior_ng2_lora_r2 count_prior_ng2_lora_r2_filter_f025 count_prior_ng2_lora_r2_filter_f050 count_prior_ng2_lora_r2_filter_f100 --out .\experiments\tiny_language_lab\runs\stage33_filter_b500.jsonl --summary .\experiments\tiny_language_lab\runs\stage33_filter_b500.md --title "Stage 33 Curriculum Filter 500 steps"
```

Run a Stage 34 dynamic-filter budget point:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\natural_text_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --count-alpha 0.1 --ngram-backoff 10 --curriculum-rescore-every 25 --curriculum-pool-size 4096 --seeds 7 11 19 --configs count_prior_ng2_lora_r2 count_prior_ng2_lora_r2_dynfilter_f050 count_prior_ng2_lora_r2_dynfilter_f025 --out .\experiments\tiny_language_lab\runs\stage34_dynfilter_b500.jsonl --summary .\experiments\tiny_language_lab\runs\stage34_dynfilter_b500.md --title "Stage 34 Dynamic Filter 500 steps"
```

Run a Stage 35 recency-base budget point:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\natural_text_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --count-alpha 0.1 --ngram-backoff 10 --recency-tau 96 --recency-lambda 0.25 --seeds 7 11 19 --configs count_prior_ng2_lora_r2 count_prior_ng2_recency_lora_r2 count_prior_ng3_lora_r2 --out .\experiments\tiny_language_lab\runs\stage35_recency_b500.jsonl --summary .\experiments\tiny_language_lab\runs\stage35_recency_b500.md --title "Stage 35 Frozen Recency Base 500 steps"
```

Run the Stage 30 corpus normalizer:

```powershell
python .\experiments\tiny_language_lab\make_natural_text_corpus.py --source .\experiments\tiny_language_lab\corpus\tiny_shakespeare_raw.txt --out .\experiments\tiny_language_lab\corpus\natural_text_seed.txt --block-size 96
```

Run a Stage 30 budget point:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\natural_text_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --count-alpha 0.1 --ngram-backoff 10 --seeds 7 11 19 --configs random_full count_prior_ng1_lora_r2 count_prior_ng2_lora_r2 count_prior_ng3_lora_r2 --out .\experiments\tiny_language_lab\runs\stage30_naturaltext_b500.jsonl --summary .\experiments\tiny_language_lab\runs\stage30_naturaltext_b500.md --title "Stage 30 Natural Text 500 steps"
```

Run a Stage 31 order-4 budget point:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\natural_text_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --count-alpha 0.1 --ngram-backoff 10 --seeds 7 11 19 --configs count_prior_ng4_lora_r2 --out .\experiments\tiny_language_lab\runs\stage31_order4_b500.jsonl --summary .\experiments\tiny_language_lab\runs\stage31_order4_b500.md --title "Stage 31 Order4 500 steps"
```

Stage 38's H014 behavior residual gate: Codex added
`count_prior_lora_r2_copyw_floor` and measured the copy-probe floor-to-target gap
on `long_context_seed.txt`. The frozen-prior floor copied at `0.118421`, near
`1 / 8` chance. The rank-2 residual arms reached `0.320176` for `copyw` and
`0.307017` for `copymix`, with all seed gaps above `0.10`. This confirms H014:
Stage 37's prior-dominance result is NLL-specific, while copy behavior is formed
by the trainable residual in this controlled probe. See
`experiments/tiny_language_lab/runs/stage38_behaviorgap.md`.

Run the Stage 38 behavior residual gate:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.25 --seeds 7 11 19 --configs count_prior_lora_r2_copyw_floor count_prior_lora_r2_copyw count_prior_lora_r2_copymix random_full_copymix --out .\experiments\tiny_language_lab\runs\stage38_behaviorgap.jsonl --summary .\experiments\tiny_language_lab\runs\stage38_behaviorgap.md --title "Stage 38 Behavior Residual Marginal-Value Gate"
```

Stage 39's behavior rank sweep: Codex added `count_prior_lora_r1_copyw` and
`count_prior_lora_r4_copyw`, then re-used the Stage 38 long-context copy protocol.
All trained ranks beat the `0.118421` floor, but the result is not monotone with
rank: rank 1 reaches `0.250000`, rank 2 reaches `0.320176`, and rank 4 reaches
`0.271930` mean copy accuracy. Rank 4 does not beat rank 1 with stable sign and
falls below rank 2 on two seeds, so this does not support a simple
capacity-limited explanation. See
`experiments/tiny_language_lab/runs/stage39_behavior_rank.md`.

Run the Stage 39 behavior rank sweep:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.25 --seeds 7 11 19 --configs count_prior_lora_r2_copyw_floor count_prior_lora_r1_copyw count_prior_lora_r2_copyw count_prior_lora_r4_copyw --out .\experiments\tiny_language_lab\runs\stage39_behavior_rank.jsonl --summary .\experiments\tiny_language_lab\runs\stage39_behavior_rank.md --title "Stage 39 Behavior Rank Sweep"
```

Stage 40's held-out-key copy test: Codex added `--holdout-keys` to the
identity-copy corpus generator and generated `long_context_holdout_seed.txt` with
`g h` held out of training answer rows while kept in the character vocabulary.
The split check found 97 seen and 18 held-out validation copy cases per seed, and
zero held-out key or answer rows in the train split. The rank-2 residual did not
clear the floor on held-out keys: `count_prior_lora_r2_copyw` and the frozen
floor both scored `0.000000` held-out accuracy on every seed. The clean
memorization kill also does not fire because the residual's mean seen-key gain is
only `+0.027491`, and the full control also collapsed to `0.000000` held-out
accuracy. See `experiments/tiny_language_lab/runs/stage40_heldout_copy.md`.

Run the Stage 40 held-out-key copy test:

```powershell
python .\experiments\tiny_language_lab\make_long_context_corpus.py --holdout-keys g h --lines 768 --seed 20260617 --out .\experiments\tiny_language_lab\corpus\long_context_holdout_seed.txt
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_holdout_seed.txt --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.25 --copy-probe-holdout-keys g h --seeds 7 11 19 --configs count_prior_lora_r2_copyw_floor count_prior_lora_r2_copyw random_full_copymix --out .\experiments\tiny_language_lab\runs\stage40_heldout_copy.jsonl --summary .\experiments\tiny_language_lab\runs\stage40_heldout_copy.md --title "Stage 40 Held-Out-Key Copy Generalization"
```

Stage 41's forced-choice held-out copy test: Codex added forced-choice readout
fields to the copy probe without changing training. The probe restricts the
answer-position logits to the validation key alphabet, `abcdefgh`, and reports
seen and held-out choice accuracy plus correct-key MRR. Reusing the Stage 40
corpus and arms, forced choice did not lift held-out transfer: the rank-2 residual
and the full control both stayed at `0.000000` held-out choice accuracy on every
seed. Arm B's seen choice accuracy was only `0.202749`, below the clean
`+0.10` seen-power clause, while the full control reached `0.364261` mean seen
choice accuracy and still held-out collapsed. See
`experiments/tiny_language_lab/runs/stage41_forcedchoice_heldout.md`.

Run the Stage 41 forced-choice held-out copy test:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_holdout_seed.txt --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.25 --copy-probe-holdout-keys g h --seeds 7 11 19 --configs count_prior_lora_r2_copyw_floor count_prior_lora_r2_copyw random_full_copymix --out .\experiments\tiny_language_lab\runs\stage41_forcedchoice_heldout.jsonl --summary .\experiments\tiny_language_lab\runs\stage41_forcedchoice_heldout.md --title "Stage 41 Forced-Choice Held-Out Copy Circuit"
```
Stage 42's memorization-proof copy probe: Codex added `--random-payload` and
`--payload-alphabet-size` to `make_long_context_corpus.py`. The generated corpus
keeps the `key=X ... answer=X` format but draws `X` uniformly per line from a
seeded 16-symbol payload alphabet, `abcdefghijklmnop`, so there is no fixed key or
line-index mapping to memorize. The split audit found `58,405` total characters,
`49,644` train characters, `8,761` validation characters, `652` complete train
key/answer pairs, and `115` validation probe cases. Both train and validation
contain all 16 payload symbols, and all complete key/answer pairs match.

Result: H017 does not confirm. `count_prior_lora_r2_copyw` reaches `0.063768`
mean copy accuracy, only `+0.001268` over chance (`1 / 16 = 0.062500`) and
`+0.020290` over the frozen floor. Every cheap-residual seed stays inside the
registered `0.05` reversal band. `random_full_copymix` clears chance on all seeds
and reaches `0.226087` mean copy accuracy, so the stage is a local reversal kill
for the current cheap rank-2 residual recipe rather than an inconclusive task
failure. See `experiments/tiny_language_lab/runs/stage42_random_payload_copy.md`.

Run the Stage 42 corpus builder and matrix:

```powershell
python .\experiments\tiny_language_lab\make_long_context_corpus.py --random-payload --payload-alphabet-size 16 --lines 768 --seed 20260617 --out .\experiments\tiny_language_lab\corpus\random_payload_copy_seed.txt
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\random_payload_copy_seed.txt --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.25 --seeds 7 11 19 --configs count_prior_lora_r2_copyw_floor count_prior_lora_r2_copyw random_full_copymix --out .\experiments\tiny_language_lab\runs\stage42_random_payload_copy.jsonl --summary .\experiments\tiny_language_lab\runs\stage42_random_payload_copy.md --title "Stage 42 Memorization-Proof Copy Probe"
```

Roadmap impact: Stage 38's behavior-axis result remains valid for seen-key
identity copy, but it should not be upgraded to a general in-context copy circuit.
Codex prepared `docs/decisions/0008-general-copy-needs-more-than-current-rank2-residual.codex-draft.md`
for Claude review; Gemini owns the random-token copying and induction-head
prior-art pass.
Stage 43's minimal-surface general-copy ladder: Codex added
`count_prior_lora_r8_copyw`, `count_prior_lora_r16_copyw`, and
`count_prior_all_copyw`, then reused the Stage 42 random-payload corpus and CUDA
protocol. The new rank arms controlled the alpha-over-rank confound by setting
rank 8 to alpha 8 and rank 16 to alpha 16.

Result: H018 does not confirm, and the registered KILL line fires. Rank 8 reaches
`0.049275` mean copy accuracy and rank 16 reaches `0.049276`, both below chance
(`1 / 16 = 0.062500`) and below the rank-2 baseline (`0.063768`). The
full-body-on-frozen-base diagnostic also stays near chance at `0.043478`, while
`random_full_copymix` clears chance on every seed and reaches `0.226087` mean copy
accuracy. The rank trend is absent, and Arm D against Arm E points to frozen-base
interference under this protocol rather than LoRA rank alone. See
`experiments/tiny_language_lab/runs/stage43_general_copy_surface.md`.

Run the Stage 43 minimal-surface ladder:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\random_payload_copy_seed.txt --device cuda --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.25 --seeds 7 11 19 --configs count_prior_lora_r2_copyw count_prior_lora_r8_copyw count_prior_lora_r16_copyw count_prior_all_copyw random_full_copymix --out .\experiments\tiny_language_lab\runs\stage43_general_copy_surface.jsonl --summary .\experiments\tiny_language_lab\runs\stage43_general_copy_surface.md --title "Stage 43 General-Copy Surface Ladder"
```

Roadmap impact: the current frozen-prior family does not form general random-payload
copy at the Stage 42 budget, even with rank 16 LoRA or the full body trained under
the frozen count base. Claude owns whether to accept the Codex draft ADR or open a
longer-budget, different-prior, retrieval, or trainable-attention follow-up. Gemini
owns the rank and induction-circuit prior-art comparison.
The copy-aware training knobs are `--copy-train-marker`, `--copy-loss-weight`,
`--copy-sampler`, `--copy-sample-fraction`, and `--copy-choice-weight`.
`--copy-verify-mode identity` keeps the original copy-task verifier, where the
answer must equal the key. `--copy-verify-mode key-answer` accepts generated
rows where both `key=` and `answer=` are present, allowing non-identity mapping
tasks.
`--copy-mine-every` controls failed-case mining for `--copy-sampler failed` and
`--copy-sampler failed_mixed`; the generated correction samplers are
`--copy-sampler correction` and `--copy-sampler correction_mixed`.
`--copy-correction-template` chooses `compact`, `focus`, `prefix`, or `full`.
The retrieval-training samplers are `--copy-sampler retrieval` and
`--copy-sampler retrieval_mixed`; `--copy-train-retrieval-template` chooses
`compact`, `focus`, or `prefix` retrieval examples during training.
`--copy-sampler correction_retrieval_mixed` mixes ordinary corpus windows,
generated correction examples, and retrieval-use examples in one batch.
`--copy-sampler correction_then_retrieval_mixed` runs correction-mixed training
first, then retrieval-mixed training after `--copy-curriculum-switch-fraction`.
`--copy-sampler correction_then_retrieval_rehearsal_mixed` uses the same staged
switch, then keeps a correction rehearsal stream alive during the retrieval
phase. `--copy-rehearsal-fraction` controls that post-switch correction stream.
`--copy-probe-retrieval-template` adds `compact`, `focus`, or `prefix`
retrieval context only during the copy probe. `--copy-probe-retrieval-source`
chooses whether that probe hint is derived directly from the held-out target
(`target`, the historical default) or looked up from a train-split copy memory
table (`memory`). The memory source reports table entries, observations,
conflicts, hits, and misses in the run JSONL.
`--copy-probe-memory-scope all` lets the memory table include the full corpus,
which is useful for explicit external-memory tests where held-out mappings are
absent from training but present in the memory table. `--copy-probe-holdout-keys`
splits probe accuracy and NLL into seen-key and held-out-key subsets.
`--copy-probe-retrieval-corrupt wrong-answer` keeps the memory hit well formed
but swaps the retrieved answer for a deterministic different valid key
character. The run JSONL reports `copy_probe_retrieval_corrupted`.
Validation NLL remains ordinary next-character loss; the copy probe is the
behavior metric for whether the model learned the key-answer rule.

## Phase 2 TinyStories bridge baseline

ADR 0010 moves the lab toward a from-scratch TinyStories-scale build. The first
bridge run stays character-level so the existing frozen n-gram prior machinery
ports unchanged. Put locally downloaded TinyStories `.txt`, `.jsonl`, or `.json`
files under `experiments\tiny_language_lab\corpus\tinystories_raw`, then prepare
the normalized corpus:

```powershell
python .\experiments\tiny_language_lab\make_tinystories_corpus.py --source-dir .\experiments\tiny_language_lab\corpus\tinystories_raw --out .\experiments\tiny_language_lab\corpus\tinystories_char_seed.txt --metadata-out .\experiments\tiny_language_lab\corpus\tinystories_char_seed.meta.json --block-size 256 --min-chars 1000000 --shard-dir .\experiments\tiny_language_lab\corpus\tinystories_char_shards --shard-chars 5000000
```

Run a visible CUDA smoke before any long training:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode smoke-fast -KeepOpen"
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode smoke-prior -KeepOpen"
```

After the smoke passes and the user confirms, run the first bridge comparison:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode bridge100 -KeepOpen"
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode bridge500 -KeepOpen"
```

Stage 44 ran this bridge path on a bounded official TinyStories train slice. At
500 steps, `count_prior_ng4_lora_r2` reached `1.139715` mean validation NLL
versus `2.352297` for `random_full`, while training `41,249` parameters instead
of `3,209,249`. See
`experiments/tiny_language_lab/runs/phase2_tinystories_bridge_b500.md`.

Run the modern Phase 2 baseline smoke and matrix:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode modern-smoke -KeepOpen"
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode modern500 -KeepOpen"
```

Stage 45 ran the modern path with RoPE, Muon, gradient accumulation, and
activation checkpointing. At 500 steps, `random_full` reached `1.144942` mean
validation NLL, and `count_prior_ng4_lora_r2` reached `1.102748`. The prior arm
kept a `0.042194` NLL lead while training `41,249` parameters instead of
`3,176,481`. See
`experiments/tiny_language_lab/runs/phase2_tinystories_modern_b500.md`.

Score the saved generation samples:

```powershell
python .\experiments\tiny_language_lab\score_generation_samples.py --runs .\experiments\tiny_language_lab\runs\phase2_tinystories_modern_b500.jsonl --out .\experiments\tiny_language_lab\runs\phase2_tinystories_modern_b500_generation_quality.md --title "Stage 45 TinyStories Modern Generation Quality"
```

Stage 46 ran this score sheet on the Stage 45 modern b500 samples. The
deterministic proxy total was `5.667/6` for `count_prior_ng4_lora_r2` and
`3.000/6` for `random_full`. The score is for local trend tracking, not a public
generation-quality claim.

Smoke-test shard consumption for plain LM training:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode stream-smoke -KeepOpen"
```

Stage 47 ran this mode with RoPE, Muon, activation checkpointing, and gradient
accumulation. The run consumed the five `train_*.txt` TinyStories shards,
reported `train_chars = 8,500,000` and `train_eval_chars = 200,000`, and
completed at `2.216325` validation NLL. This shard mode is currently scoped to
plain LM batches; frozen-prior, copy-aware, and curriculum modes still use the
full train tensor.

Run the 1000-step modern crossover matrix:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode modern1000 -KeepOpen"
```

Stage 48 ran this mode across seeds `7 11 19`. At 1000 steps, `random_full`
reached `1.052559` mean validation NLL, while `count_prior_ng4_lora_r2` reached
`1.123161`. This reverses the Stage 45 500-step ordering, placing the modern
character-level crossover between 500 and 1000 steps. See
`experiments/tiny_language_lab/runs/phase2_tinystories_modern_b1000.md`.

Build and smoke-test the first BPE-token corpus:

```powershell
python .\experiments\tiny_language_lab\make_bpe_corpus.py --source .\experiments\tiny_language_lab\corpus\tinystories_char_seed.txt --out .\experiments\tiny_language_lab\corpus\tinystories_bpe_v256_seed.txt --metadata-out .\experiments\tiny_language_lab\corpus\tinystories_bpe_v256_seed.meta.json --vocab-size 256 --train-chars 500000 --max-chars 1000000
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode bpe-smoke -KeepOpen"
python .\experiments\tiny_language_lab\decode_bpe_samples.py --metadata .\experiments\tiny_language_lab\corpus\tinystories_bpe_v256_seed.meta.json --run .\experiments\tiny_language_lab\runs\phase2_tinystories_bpe_smoke.jsonl --out .\experiments\tiny_language_lab\runs\phase2_tinystories_bpe_smoke_decoded_samples.md --title "Stage 49 TinyStories BPE Smoke Decoded Samples"
```

Stage 49 encoded `1,000,000` TinyStories characters into `446,694` BPE tokens.
The visible smoke compared `random_full` with `count_prior_lora_r2`, a frozen
BPE-token bigram prior plus rank-2 LoRA. At 20 steps, the prior arm reached
`3.405228` validation NLL versus `3.847587` for `random_full`. This is a BPE
feasibility smoke, not the final BPE-vs-character result.

Run the BPE 500-step multi-seed matrix:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase2_visible.ps1`" -Mode bpe500 -KeepOpen"
python .\experiments\tiny_language_lab\decode_bpe_samples.py --metadata .\experiments\tiny_language_lab\corpus\tinystories_bpe_v256_seed.meta.json --run .\experiments\tiny_language_lab\runs\phase2_tinystories_bpe_b500.jsonl --out .\experiments\tiny_language_lab\runs\phase2_tinystories_bpe_b500_decoded_samples.md --title "Stage 50 TinyStories BPE 500 Decoded Samples"
```

Stage 50 ran this matrix across seeds `7 11 19`. At 500 steps, `random_full`
reached `2.404960` mean validation NLL, while the BPE-token bigram prior plus
rank-2 LoRA reached `3.344760`. Approximate source-normalized bits were
`1.549860` for full BPE and `2.155508` for the prior arm. The current Phase 2
default remains character-level TinyStories; this v256 BPE branch is secondary.

Build the Phase 3 coherence checkpoint and score its saved samples:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase3_visible.ps1`" -Mode stage51"
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase3_visible.ps1`" -Mode score51"
```

Stage 51 uses the rescaled `494,094,421` character TinyStories corpus and the
Stage 47 shard-streaming path. The 25.25M `random_full` checkpoint family ran
`5000` steps across seeds `7 11 19` and reached `0.818608` mean validation NLL,
`1.181001` mean bits/char, and `5.667/6` mean deterministic generation proxy
score. Human sample review remains pending before any genuine-coherence claim.

Run the Phase 3 H019 crossover-scaling matrix:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase3_visible.ps1`" -Mode stage52-matrix"
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase3_visible.ps1`" -Mode stage52-prior-sharded"
```

Stage 52 compares `random_full` with `count_prior_ng4_lora_r2` across sizes
`3m`, `10m`, `25m`, and `85m`, budgets `200 500 1000 2000`, and seeds
`7 11 19`, always with `--eval-mode sampled`. The first prior pass OOMed and is
preserved as failed evidence; the corrected shard-native prior cells completed
with `_sharded` artifact names. Crossovers were `1000`, `1000`, `1000`, and
`500` steps, so H019 is `GRADED`.

Run the Phase 4 H020 free-accelerator test:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase4_visible.ps1`" -Mode stage53"
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase4_visible.ps1`" -Mode stage53-cell -Budget 2000 -MuonLr 0.005"
```

Stage 53 adds `count_prior_ng4_all`, which keeps the order-4 count prior frozen
while training the full 25.25M body. The arm beats Stage 52 `random_full` at
200 and 500 steps, but loses on all paired seeds at 1000 and 2000 steps. The
required lower-LR sensitivity cell improves the 2000-step mean but still trails
`random_full`, so H020 is KILL E-interfere. See
`experiments/tiny_language_lab/runs/stage53_h020_free_accelerator_summary.md`.

Run the Phase 4 H021 prior-order floor-scaling test:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase4_visible.ps1`" -Mode stage54-smoke -KeepOpen"
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase4_visible.ps1`" -Mode stage54-gateA -KeepOpen"
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase4_visible.ps1`" -Mode stage54-phaseB-cell -Budget 1000 -KeepOpen"
```

Repeat `stage54-phaseB-cell` for budgets `200`, `500`, `1000`, and `2000` when
the gate passes. Stage 54 adds `count_prior_ng5_lora_r2` and
`count_prior_ng5_lora_r2_floor`, backed by a sparse shard-native order-5 prior
that falls back to the dense order-4 prior. The gate passed by `-0.117649` mean
paired NLL versus order 4, and the 25.25M crossover moved from 1000 to 2000
steps. Verdict: H021 CONFIRM for frozen-prior tiny-surface runs. See
`experiments/tiny_language_lab/runs/stage54_h021_prior_order_floor_scaling_summary.md`.

Run the Phase 4 Stage 55 flagship package:

```powershell
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase4_visible.ps1`" -Mode stage55-flagship-cell -Budget 50000 -Seed 7 -FlagshipCheckpointDir C:\cassandra_runs\stage55_flagship_checkpoints"
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase4_visible.ps1`" -Mode stage55-flagship-cell -Budget 20000 -Seed 11 -FlagshipCheckpointDir C:\cassandra_runs\stage55_flagship_checkpoints"
Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PWD\experiments\tiny_language_lab\run_phase4_visible.ps1`" -Mode stage55-flagship-cell -Budget 20000 -Seed 19 -FlagshipCheckpointDir C:\cassandra_runs\stage55_flagship_checkpoints"
```

Stage 55 completed the ADR 0013 flagship build with `random_full`, `201.61M`
trainable parameters, block 256, RoPE, activation checkpointing, Muon, sampled
eval, and the shard-streamed TinyStories corpus. Seed 7 ran 50,000 steps and
reached `0.556410` sampled report validation NLL (`0.802730` bits/char);
seeds 11 and 19 ran reduced 20,000-step replicas. Final checkpoints were
secured outside OneDrive and `%TEMP%`, then later pruned during the Stage 56
disk emergency after durable JSONL, Markdown, and export evidence had been
recorded. The seed-7 repo-local artifact checkpoint remains under
`experiments/tiny_language_lab/artifacts/phase4/checkpoints/` and is the
checkpoint used by later eval-only probes. The checked Nsight DL Designer export is
`experiments/tiny_language_lab/artifacts/phase4/nsight_dld/stage55_seed7_final_success_b1_s256.onnx`.
Proxy generation scoring is deterministic only; human review remains pending.

Run the Phase 5 Stage 56 H022 broad-corpus package:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase5_visible.ps1 -Mode stage56-prep
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase5_visible.ps1 -Mode stage56-cell -Budget 50000 -Seed 7 -CheckpointEvery 5000
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase5_visible.ps1 -Mode stage56-cell -Budget 20000 -Seed 11 -CheckpointEvery 20000
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase5_visible.ps1 -Mode stage56-cell -Budget 20000 -Seed 19 -CheckpointEvery 20000
```

Stage 56 CONFIRMED H022. The unchanged `85.10M` character recipe trained on
text8 train/valid data scored `1.485740` TEST bits/char for seed 7 at 50,000
steps. The 20,000-step replicas scored `1.532627` and `1.529591`, a `0.003035`
bits/char spread, so the Stage 55 broad-text gap reads as a data-distribution
effect under the registered test. Canonical checkpoints live in
`C:\cassandra_runs\stage56_broadchar_checkpoints`, and durable evidence is in
`experiments/tiny_language_lab/RESULTS.md`.

Stage 57 locked Recipe v2:

```powershell
--precision fp32 `
--lr-schedule cosine --lr-final-frac 0.1 `
--checkpoint-keep 1 `
--vocab-chars "<phase-specific union alphabet when cross-corpus scoring is required>"
```

bf16 was rejected on throughput, cosine LR was adopted, checkpoint retention
and fp16 model-only archive mode passed smoke tests, and the vocab-union smoke
proved a text8-trained 33-char checkpoint can score TinyStories validation text
without out-of-vocab errors. Block 512 is recorded only as a timing row.

Run the Phase 5 letters-only behavior probe:

```powershell
python .\experiments\tiny_language_lab\make_letters_copy_probe.py --lines 1024 --seed 20260709
python .\experiments\tiny_language_lab\eval_letters_copy_probe.py --device auto --max-cases 1024
```

This is eval-only and does no training. The generated probe uses payloads
`a` through `p`, letter-spelled case ids, `key ` / `answer ` markers, and no
digits or equals signs so it fits the Stage 55 33-character vocabulary. The
Stage 55 seed-7 flagship scored `0.060547` constrained-choice copy accuracy
against chance `0.062500`, below the `0.162500` reopen threshold. The behavior
axis stays closed; Stage 58 proceeds under Claude's H024. See
`runs/phase5_behavior_letters_probe.md`.

Phase 5 D2 release prep status:

- Corpus payloads were removed from the Git index while local files were left
  in place.
- Generated corpus payloads are ignored; `.meta.json` provenance files remain
  trackable.
- `.githooks/pre-commit` blocks staged additions over 50 MiB.
- Licensing notes and a flagship model-card draft live at
  `docs/phase5-licensing-notes.md` and `docs/phase5-model-card-draft.md`.
- fp16 model-only exports of the three Stage 56 final checkpoints live under
  `C:\cassandra_runs\phase5_model_only_exports`.
- History surgery, license choice, Hub upload, and public push remain
  user-gated. Stage 58 completed on 2026-07-21 and H024 resolved E-NULL,
  seed-robust in sign; see `docs/phase5-final-report.md` and ADR 0016 for
  the closeout evidence.

After all three H024 arms have deterministic text8 TEST and TinyStories
retention reports, regenerate the Phase 5 comparison figures and data with:

```powershell
python .\experiments\tiny_language_lab\make_phase5_figures.py
```
