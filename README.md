# CASSANDRA

Cassandra is a research workspace with two connected tracks:

1. `AGENT.md` · a working reference for accountable protective AI systems.
2. `docs/LOW_HARDWARE_LM_RESEARCH.md` · a laptop-scale language-model research program.

The second track asks a deliberately narrow version of a large dream:

> Can useful language-model behavior be formed on consumer hardware by reducing
> brute-force gradient training through structure, reuse, compression,
> retrieval, search, or analytic initialization?

This is not yet a claim that a laptop can build a frontier LLM. It is a lab for
testing smaller claims honestly, then keeping only the ones that survive.

## Collaboration workflow

Cassandra now uses three model-facing role files:

- `CODEX.md`: executable experiments, tests, runs, and stage documentation.
- `GEMINI.md`: outside research, prior art, and world comparison.
- `CLAUDE.md`: hypotheses, ADRs, and roadmap decisions.

The intended loop is: Gemini researches the world, Claude turns that context
into hypotheses and decisions, and Codex turns those hypotheses into measured
local experiments.

## Current executable experiments

The first runnable artifact is a character-level bigram lab:

```powershell
python .\experiments\tiny_language_lab\cassandra_tiny_lm.py --method count --prompt "cassandra "
python .\experiments\tiny_language_lab\cassandra_tiny_lm.py --method coordinate --steps 300 --prompt "cassandra "
python .\experiments\tiny_language_lab\cassandra_tiny_lm.py --method gradient --steps 300 --prompt "cassandra "
```

The three methods share the same parameter surface:

- `count` constructs parameters directly from corpus statistics, with no gradient training.
- `coordinate` changes one parameter at a time and keeps changes that reduce loss.
- `gradient` uses normal backpropagation as the control condition.

The point is not that bigrams are enough. The point is to make Cassandra's core
question measurable before moving to transformers.

A tiny causal transformer baseline is also available:

```powershell
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --steps 200 --prompt "cassandra "
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --init count-bigram --steps 50 --prompt "cassandra "
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --residual-base count-bigram --residual-l2 1.0 --steps 50 --prompt "cassandra "
python .\experiments\tiny_language_lab\cassandra_tiny_transformer.py --residual-base count-bigram --train-scope adapters --adapter-rank 4 --steps 50 --prompt "cassandra "
python .\experiments\tiny_language_lab\make_synthetic_corpus.py --lines 320 --seed 20260616
python .\experiments\tiny_language_lab\cassandra_compare.py --steps 50 --eval-batches 16 --seeds 7 11 19
python .\experiments\tiny_language_lab\cassandra_compare.py --steps 50 --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_head count_prior_adapter_r4 count_prior_lora_r1 count_prior_lora_r2 --out .\experiments\tiny_language_lab\runs\stage6_lora.jsonl --summary .\experiments\tiny_language_lab\runs\stage6_lora.md --title "Stage 6 LoRA Summary"
python .\experiments\tiny_language_lab\make_long_context_corpus.py --lines 512 --seed 20260617
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 100 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --seeds 7 11 19 --configs random_full count_prior_head count_prior_adapter_r4 count_prior_lora_r2 --out .\experiments\tiny_language_lab\runs\stage7_long_context.jsonl --summary .\experiments\tiny_language_lab\runs\stage7_long_context.md --title "Stage 7 Long-Context Summary"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --seeds 7 11 19 --configs random_full_copyw count_prior_lora_r2_copyw --out .\experiments\tiny_language_lab\runs\stage8_copy_weight200.jsonl --summary .\experiments\tiny_language_lab\runs\stage8_copy_weight200.md --title "Stage 8 Copy Weight 200 Summary"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-sample-fraction 0.25 --copy-loss-weight 200 --seeds 7 11 19 --configs random_full_copymix count_prior_lora_r2_copymix --out .\experiments\tiny_language_lab\runs\stage9_mixed_sampler_weight200.jsonl --summary .\experiments\tiny_language_lab\runs\stage9_mixed_sampler_weight200.md --title "Stage 9 Mixed Sampler Weight 200"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 50 --copy-choice-weight 0.25 --seeds 7 11 19 --configs random_full_copyw_choice count_prior_lora_r2_copyw_choice --out .\experiments\tiny_language_lab\runs\stage10_choice_loss_weight025.jsonl --summary .\experiments\tiny_language_lab\runs\stage10_choice_loss_weight025.md --title "Stage 10 Choice Loss Weight 0.25"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 200 --copy-sample-fraction 0.1 --copy-mine-every 100 --seeds 7 11 19 --configs random_full_copyfailmix count_prior_lora_r2_copyfailmix --out .\experiments\tiny_language_lab\runs\stage11_failed_mixed_replay_weight200.jsonl --summary .\experiments\tiny_language_lab\runs\stage11_failed_mixed_replay_weight200.md --title "Stage 11 Failed Mixed Replay Weight 200"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage12_correction_examples_gentle_500.jsonl --summary .\experiments\tiny_language_lab\runs\stage12_correction_examples_gentle_500.md --title "Stage 12 Gentle Correction Examples 500"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage13_prefix_corrections_500.jsonl --summary .\experiments\tiny_language_lab\runs\stage13_prefix_corrections_500.md --title "Stage 13 Prefix Corrections 500"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage14_retrieval_probe_compact.jsonl --summary .\experiments\tiny_language_lab\runs\stage14_retrieval_probe_compact.md --title "Stage 14 Retrieval Probe Compact"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_retmix count_prior_lora_r2_retmix --out .\experiments\tiny_language_lab\runs\stage15_retrieval_use_training.jsonl --summary .\experiments\tiny_language_lab\runs\stage15_retrieval_use_training.md --title "Stage 15 Retrieval Use Training"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_corrretmix count_prior_lora_r2_corrretmix --out .\experiments\tiny_language_lab\runs\stage16_correction_retrieval_curriculum.jsonl --summary .\experiments\tiny_language_lab\runs\stage16_correction_retrieval_curriculum.md --title "Stage 16 Correction Retrieval Curriculum"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-curriculum-switch-fraction 0.5 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_corrthenret count_prior_lora_r2_corrthenret --out .\experiments\tiny_language_lab\runs\stage17_staged_correction_then_retrieval.jsonl --summary .\experiments\tiny_language_lab\runs\stage17_staged_correction_then_retrieval.md --title "Stage 17 Staged Correction Then Retrieval"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-curriculum-switch-fraction 0.8 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_corrthenret count_prior_lora_r2_corrthenret --out .\experiments\tiny_language_lab\runs\stage17_staged_correction_then_retrieval_late.jsonl --summary .\experiments\tiny_language_lab\runs\stage17_staged_correction_then_retrieval_late.md --title "Stage 17 Staged Correction Then Retrieval Late Switch"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage18_memory_retrieval_probe.jsonl --summary .\experiments\tiny_language_lab\runs\stage18_memory_retrieval_probe.md --title "Stage 18 Memory Retrieval Probe"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-train-marker "answer=" --copy-loss-weight 10 --seeds 7 11 19 --configs random_full_copyw count_prior_lora_r2_copyw --out .\experiments\tiny_language_lab\runs\stage18_memory_retrieval_baseline.jsonl --summary .\experiments\tiny_language_lab\runs\stage18_memory_retrieval_baseline.md --title "Stage 18 Memory Retrieval Baseline"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-probe-retrieval-corrupt wrong-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage19_memory_corruption_ablation.jsonl --summary .\experiments\tiny_language_lab\runs\stage19_memory_corruption_ablation.md --title "Stage 19 Memory Corruption Ablation"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-curriculum-switch-fraction 0.8 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs count_prior_lora_r4_corrthenret count_prior_lora_r8_corrthenret --out .\experiments\tiny_language_lab\runs\stage20_rank_sweep_staged.jsonl --summary .\experiments\tiny_language_lab\runs\stage20_rank_sweep_staged.md --title "Stage 20 Rank Sweep Staged 0.8"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs count_prior_lora_r4_corrretmix count_prior_lora_r8_corrretmix --out .\experiments\tiny_language_lab\runs\stage20_rank_sweep_simultaneous.jsonl --summary .\experiments\tiny_language_lab\runs\stage20_rank_sweep_simultaneous.md --title "Stage 20 Rank Sweep Simultaneous"
python .\experiments\tiny_language_lab\make_memory_mapping_corpus.py --lines 512 --seed 20260618 --out .\experiments\tiny_language_lab\corpus\memory_mapping_seed.txt
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage21_memory_mapping_no_hint.jsonl --summary .\experiments\tiny_language_lab\runs\stage21_memory_mapping_no_hint.md --title "Stage 21 Memory Mapping No Hint"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage21_memory_mapping_correct.jsonl --summary .\experiments\tiny_language_lab\runs\stage21_memory_mapping_correct.md --title "Stage 21 Memory Mapping Correct Memory"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-probe-retrieval-corrupt wrong-answer --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-mine-every 100 --copy-correction-template prefix --seeds 7 11 19 --configs random_full_copycorrmix count_prior_lora_r2_copycorrmix --out .\experiments\tiny_language_lab\runs\stage21_memory_mapping_corrupt.jsonl --summary .\experiments\tiny_language_lab\runs\stage21_memory_mapping_corrupt.md --title "Stage 21 Memory Mapping Corrupted Memory"
python .\experiments\tiny_language_lab\make_memory_mapping_corpus.py --lines 512 --seed 20260618 --holdout-keys g h --out .\experiments\tiny_language_lab\corpus\memory_mapping_holdout_seed.txt
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_holdout_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template none --copy-probe-holdout-keys g h --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_retmix count_prior_lora_r2_retmix --out .\experiments\tiny_language_lab\runs\stage22_holdout_memory_no_hint.jsonl --summary .\experiments\tiny_language_lab\runs\stage22_holdout_memory_no_hint.md --title "Stage 22 Holdout Memory No Hint"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_holdout_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-probe-memory-scope all --copy-probe-holdout-keys g h --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_retmix count_prior_lora_r2_retmix --out .\experiments\tiny_language_lab\runs\stage22_holdout_memory_correct.jsonl --summary .\experiments\tiny_language_lab\runs\stage22_holdout_memory_correct.md --title "Stage 22 Holdout Memory Correct"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\memory_mapping_holdout_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-probe-retrieval-source memory --copy-probe-memory-scope all --copy-probe-retrieval-corrupt wrong-answer --copy-probe-holdout-keys g h --copy-verify-mode key-answer --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.05 --copy-train-retrieval-template compact --seeds 7 11 19 --configs random_full_retmix count_prior_lora_r2_retmix --out .\experiments\tiny_language_lab\runs\stage22_holdout_memory_corrupt.jsonl --summary .\experiments\tiny_language_lab\runs\stage22_holdout_memory_corrupt.md --title "Stage 22 Holdout Memory Corrupt"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-rehearsal-fraction 0.05 --copy-curriculum-switch-fraction 0.8 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs count_prior_lora_r2_corrthenret_rehearsal --out .\experiments\tiny_language_lab\runs\stage23_rank2_rehearsal_frac005.jsonl --summary .\experiments\tiny_language_lab\runs\stage23_rank2_rehearsal_frac005.md --title "Stage 23 Rank 2 Rehearsal Fraction 0.05"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\long_context_seed.txt --steps 500 --block-size 96 --eval-mode sampled --eval-batches 16 --copy-probe-marker "answer=" --copy-probe-retrieval-template compact --copy-train-marker "answer=" --copy-loss-weight 10 --copy-sample-fraction 0.1 --copy-rehearsal-fraction 0.10 --copy-curriculum-switch-fraction 0.8 --copy-mine-every 100 --copy-correction-template prefix --copy-train-retrieval-template compact --seeds 7 11 19 --configs count_prior_lora_r2_corrthenret_rehearsal --out .\experiments\tiny_language_lab\runs\stage23_rank2_rehearsal_frac010.jsonl --summary .\experiments\tiny_language_lab\runs\stage23_rank2_rehearsal_frac010.md --title "Stage 23 Rank 2 Rehearsal Fraction 0.10"
python .\experiments\tiny_language_lab\make_complexity_corpus.py --lines 512 --seed 20260619 --long-fraction 0.00 --out .\experiments\tiny_language_lab\corpus\complexity_p000_seed.txt
python .\experiments\tiny_language_lab\make_complexity_corpus.py --lines 512 --seed 20260619 --long-fraction 0.25 --out .\experiments\tiny_language_lab\corpus\complexity_p025_seed.txt
python .\experiments\tiny_language_lab\make_complexity_corpus.py --lines 512 --seed 20260619 --long-fraction 0.50 --out .\experiments\tiny_language_lab\corpus\complexity_p050_seed.txt
python .\experiments\tiny_language_lab\make_complexity_corpus.py --lines 512 --seed 20260619 --long-fraction 0.75 --out .\experiments\tiny_language_lab\corpus\complexity_p075_seed.txt
python .\experiments\tiny_language_lab\make_complexity_corpus.py --lines 512 --seed 20260619 --long-fraction 1.00 --out .\experiments\tiny_language_lab\corpus\complexity_p100_seed.txt
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\complexity_p050_seed.txt --steps 50 --block-size 96 --eval-mode sampled --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_head count_prior_lora_r1 count_prior_lora_r2 --out .\experiments\tiny_language_lab\runs\stage24_complexity_p050_s50.jsonl --summary .\experiments\tiny_language_lab\runs\stage24_complexity_p050_s50.md --title "Stage 24 Complexity p050 50 steps"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\complexity_p050_seed.txt --steps 10 --block-size 96 --eval-mode sampled --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_head count_prior_lora_r1 count_prior_lora_r2 --out .\experiments\tiny_language_lab\runs\stage25_timebudget_p050_s10.jsonl --summary .\experiments\tiny_language_lab\runs\stage25_timebudget_p050_s10.md --title "Stage 25 Time Budget p050 10 steps"
python .\experiments\tiny_language_lab\make_markov_corpus.py --order 2 --vocab 16 --lines 512 --seed 20260620 --out .\experiments\tiny_language_lab\corpus\markov_order2_seed.txt
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\markov_order2_seed.txt --steps 100 --block-size 96 --eval-mode sampled --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_lora_r2 count_prior_tri_lora_r2 --out .\experiments\tiny_language_lab\runs\stage26_markov_order2_s100.jsonl --summary .\experiments\tiny_language_lab\runs\stage26_markov_order2_s100.md --title "Stage 26 Markov Order2 100 steps"
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\markov_order1_seed.txt --steps 200 --block-size 96 --eval-mode sampled --eval-batches 16 --seeds 7 11 19 --configs random_full count_prior_lora_r2 count_prior_tri_lora_r2 --out .\experiments\tiny_language_lab\runs\stage27_matchsurface_order1_s200.jsonl --summary .\experiments\tiny_language_lab\runs\stage27_matchsurface_order1_s200.md --title "Stage 27 Match Surface Order1 200 steps"
```

This is the normal gradient-trained control model. Future Cassandra methods
should be compared against it under the same corpus, step budget, and metrics.

Latest local numbers are recorded in
`experiments/tiny_language_lab/RESULTS.md`.

## Next ladder

1. Bigram parameter lab · implemented.
2. Tiny transformer baseline · implemented.
3. Count-seeded transformer initialization · implemented.
4. Count-prior residual transformer · implemented.
5. Head-only and adapter-only residual updates · implemented.
6. Larger corpus and multi-seed replication · implemented.
7. LoRA-style experiments · implemented.
8. Long-context synthetic corpus and copy probe · implemented.
9. Task-aware copy-position training · implemented.
10. Verifier-guided mixed sampling · implemented.
11. Verifier choice-loss correction · implemented.
12. Failed-case replay correction loop · implemented.
13. Generated correction examples · implemented.
14. Structurally faithful correction templates · implemented.
15. Retrieval-style probe context · implemented.
16. Retrieval-use training · implemented.
17. Correction-retrieval curriculum · implemented.
18. Staged correction-retrieval curriculum · implemented. Result was a
    capacity-dependent split: ordering helped the full model (0.60 to 0.88 copy
    accuracy) but hurt the rank-2 LoRA surface (0.30 to 0.24). See
    `docs/hypotheses/001-staged-correction-retrieval-curriculum.md`.
19. Train-split memory retrieval probe · implemented. Stage 14's strongest
    full-model result survived when the compact probe hint came from a
    train-split memory table instead of the held-out target: copy accuracy stayed
    at 0.960526, while the same-weight baseline stayed weak at 0.403509.
20. Memory-reliance corruption ablation · implemented. A wrong but well-formed
    memory hint dropped full-model copy accuracy from 0.960526 to 0.824561,
    showing partial reliance on the retrieved answer, but not a full collapse.
    See
    `docs/hypotheses/003-memory-reliance-corruption-ablation.md`.
21. LoRA capacity sweep for curriculum interference · implemented. Rank 4
    narrowed the staged-minus-simultaneous gap from -0.065790 to -0.013158, but
    rank 8 fell back to -0.061404, so capacity alone is not enough. See
    `docs/hypotheses/002-lora-capacity-curriculum-interference.md`.
22. Non-identity memory generalization · implemented. A fixed mapping corpus
    removes the identity shortcut, but compact memory without interface training
    hurts the full model: no hint `0.722944`, correct memory `0.450216`,
    corrupted memory `0.398268`.
23. External-memory value test on held-out mappings · implemented. Stage 22
    trained `retrieval_mixed` on seen keys only, then probed held-out keys `g h`
    with no hint, correct full-corpus memory, and corrupted memory. The full
    model did not use correct memory to answer absent mappings: held-out
    accuracy was no hint `0.194444`, correct memory `0.166667`, corrupted memory
    `0.166667`. H4 is not supported by this compact interface. See
    `docs/hypotheses/004-external-memory-carries-absent-facts.md`. Branch retired
    by `docs/decisions/0001-retire-compact-text-prefix-external-memory.md`.
24. Rank-2 rehearsal for phase-switch forgetting · implemented. Codex chose the
    smaller alternative allowed by ADR 0001 before the higher-value redirect.
    Rehearsal at `0.05` only nudged rank-2 staged accuracy from `0.236842` to
    `0.245614`, still below simultaneous `0.302632`; rehearsal at `0.10`
    dropped to `0.228070`. This does not rescue the phase switch.
25. Characterize the cheap-recipe corpus regime · implemented. Stage 24 measured
    `complexity_p000` to `p100` at 50 and 100 steps. At 50 steps,
    `count_prior_lora_r2` beat `random_full` through `p = 0.50` and lost at
    `p = 0.75` and `1.00`. At 100 steps, full training won everywhere. See
    `docs/hypotheses/005-cheap-recipe-corpus-regime.md`.
26. Time-budget surface for the cheap recipe · implemented. Stage 25 measured
    10 and 25 steps across the Stage 24 corpus axis. The cheap recipe won across
    every measured point at 10 and 25 steps, crossed near `p = 0.635343` at
    50 steps, and lost everywhere by 100 steps. H006 is supported on the surface
    shape and partial on the simple mechanism clause. See
    `docs/hypotheses/006-time-budget-inductive-bias-boundary.md`.
27. Consolidating ADR for the core claim · implemented. ADR 0002 locks the
    frozen-prior recipe as a bounded early-compute inductive-bias accelerator, a
    head start that decays by about 100 steps, not an asymptotic replacement. The
    full model trains fast even at 10 steps; the prior wins early by giving a
    better low-step surface. See
    `docs/decisions/0002-frozen-prior-is-bounded-early-compute-accelerator.md`.
28. Order-matched analytic prior durability · implemented. Stage 26 added a
    pure Markov generator and a frozen trigram residual base. On the pure order-2
    corpus, the matched trigram prior stayed strongly positive at 100 and 200
    steps, while the mismatched bigram prior decayed to tied by 200 steps. On the
    order-1 control, the matched bigram prior stayed positive. See
    `docs/hypotheses/007-higher-order-analytic-prior-durability.md`.
29. Source-order versus prior-order closeout · implemented. Stage 27 completed
    the order-1/order-2 `V = 16` surface at budgets 10, 25, 50, 100, and 200. Matched
    priors stayed positive through 200 steps: order-1 plus bigram retained
    `0.018835` NLL advantage, and order-2 plus trigram retained `0.637384`.
    Mismatched priors decayed or turned negative. This was a Codex closeout of
    Stage 26's open cells, not the full H008 order-3 test. See
    `experiments/tiny_language_lab/runs/stage27_matchsurface_summary.md`.
30. H008 full source-order versus prior-order surface · implemented. Stage 28
    added `--residual-base count-ngram`, `--prior-order`, and the
    `count_prior_ng{1,2,3}_lora_r2` configs, then ran the `V = 8` three-by-three
    surface. The diagonal was positive and increasing at 200 steps:
    `A(1,1)=0.006804`, `A(2,2)=0.436673`, `A(3,3)=0.630598`. The strict lower
    triangle was partial because `A(3,2)=0.106932` stayed meaningfully positive.
    See `experiments/tiny_language_lab/runs/stage28_h008_summary.md` and
    `docs/hypotheses/008-source-order-prior-order-surface.md`.
31. Consolidating ADR for the graded source/prior order law · accepted. Claude
    accepted Codex's evidence draft with revisions as ADR 0003: the law is graded,
    matching is best and the matched advantage grows with source order, severe
    under-specification fails while one-step under-specification still helps, and
    over-specification carries a coverage-dependent sparsity penalty. The advantage
    is durable through 200 steps where the matched table has full coverage; it is a
    race between the prior's specification and the full model's learning speed. See
    `docs/decisions/0003-graded-source-prior-order-law.md`.
32. Tiny-prose finite-order prior smoke · implemented as Codex exploratory
    evidence. Stage 29 used the 1,129-character `tiny_seed.txt` project-prose
    corpus. Higher-order priors beat `random_full` at 10, 50, and 100 steps, with
    ng3 advantage `0.399935` NLL at 100 steps, but the corpus is much too small
    for a natural-language claim. See
    `experiments/tiny_language_lab/runs/stage29_tinyprose_ngram_summary.md`.
33. Natural-text external validity of finite-order priors · implemented as Stage
    30. On normalized Tiny Shakespeare (`1,100,721` chars, `V=33`), recursive
    backoff count priors plus rank-2 LoRA beat `random_full` through 500 steps for
    orders 2 and 3. At 500 steps, ng2 advantage was `+0.110081` NLL and ng3 was
    `+0.340641`; ng1 flipped negative at `-0.266069`. The core transfer result is
    positive, but the humped sweet-spot law is still open because the measured
    curve is monotone increasing through order 3. See
    `experiments/tiny_language_lab/runs/stage30_naturaltext_summary.md` and
    `docs/hypotheses/009-natural-text-finite-order-prior-sweet-spot.md`.
34. Order-4 natural-text extension · implemented as Stage 31. Codex added
    `count_prior_ng4_lora_r2` under H009's optional order-4 clause. Order 4 was
    feasible on the normalized Stage 30 corpus (`42,802,056` frozen logits) and
    became the best measured prior at every budget. At 500 steps, ng4 advantage
    was `+0.463572` NLL, stronger than ng3 at `+0.340641`. The descending limb is
    still not located because validation hit coverage remains high (`0.961867`)
    despite sparse table coverage (`0.029713`). See
    `experiments/tiny_language_lab/runs/stage31_order4_extension_summary.md`.
35. Harsher natural-text validity gate · implemented as Stage 32. Codex kept
    the Stage 30 Shakespeare train prefix but replaced the validation suffix with
    normalized Cassandra project prose in the same `V=33` alphabet. High-order
    validation-hit coverage dropped but stayed high (`0.889803` for order 4), and
    the advantage curve remained monotone through order 4 at 100, 200, and 500
    steps. At 500 steps, ng4 advantage was `+0.291191` NLL, above ng3 at
    `+0.199130`. This locally kills the predicted moderate-order hump on the
    implemented split, but leaves a caveat for a harsher or more genre-matched
    public-domain validation source. See
    `experiments/tiny_language_lab/runs/stage32_crossdomain_summary.md` and
    `docs/hypotheses/009b-ood-validation-true-sweet-spot.md`.
36. Mixed loss-based data-selection filter (Hypothesis 010) · implemented as
    Stage 33. The fixed top-10-percent frozen-prior-NLL filter did not make the
    order-2 rank-2 residual reach the uniform 200-step target earlier. At 500
    steps, both mixed filters were slightly worse than uniform, and pure
    high-loss selection was clearly worse. H010 is a local kill for this sampler,
    consistent with the Stage 11 hard-selection warning. See
    `experiments/tiny_language_lab/runs/stage33_filter_summary.md` and
    `docs/hypotheses/010-akba-curriculum-filter.md`.
37. Dynamic reducible-loss filter (Hypothesis 011) · implemented as Stage 34.
    Codex re-scored a fixed 4096-window pool every 25 steps and oversampled
    high-current-loss windows with positive smoothed loss deltas. Neither dynamic
    arm reached the uniform 200-step target faster, or at all within the measured
    budgets, and both were slower at equal steps because of scoring overhead. H011
    kills the final data-side selection attempt for the frozen-prior rank-2
    residual and retires this branch on a capacity and data-order-insensitivity
    explanation. Branch retired by ADR 0004
    (`docs/decisions/0004-retire-data-side-selection-rank2-residual.md`), which
    consolidates Stages 11, 12, 33, 34. See
    `experiments/tiny_language_lab/runs/stage34_dynfilter_summary.md` and
    `docs/hypotheses/011-dynamic-reducible-loss-filter.md`.
38. Frozen recency base (Hypothesis 012) · implemented as Stage 35. Codex added
    an analytic exponential-recency cache to the order-2 count prior with `tau=96`
    and `lambda=0.25`. The recency arm was worse than the order-2 count-only
    baseline at every measured budget by `+0.062` to `+0.086` validation NLL, and
    slower because the base is computed per block position. The order-3 count
    diagnostic was far better than order 2 at every budget, so this simple cache
    does not beat raising the count order. H012 kills the default frozen
    character-recency interpolation, without rejecting every model-side frozen
    primitive. Gemini note 08 explains this as the known character-level cache
    failure mode and points toward order-preserving frozen kernels or n-gram
    caches. See `experiments/tiny_language_lab/runs/stage35_recency_summary.md`
    and `docs/hypotheses/012-frozen-recency-base.md`.
39. Non-gradient residual formation (Hypothesis 013) · implemented as
    Stage 36. Codex added `--residual-optim {adamw,es,coord,none}` and compared
    the frozen-prior floor, the rank-2 AdamW target, full rank-2 ES, and a rank-1
    coordinate feasibility arm on the structured corpus. The frozen-prior-to-AdamW
    gap was `0.017708` NLL. ES missed the floor by `+0.042242` NLL on the mean
    despite `803` formation forward passes, so the full rank-2 no-backprop claim
    is not passed. Rank-1 coordinate recovered about `39.5%` of the gap on the
    mean, but only beat the floor on one of three seeds. See
    `experiments/tiny_language_lab/runs/stage36_h013.md` and
    `docs/hypotheses/013-non-gradient-residual-formation.md`.
40. Consolidating ADR for the formation-side trilogy · accepted. ADR 0005 retires
    unguided gradient-free formation of the rank-2 residual at this scale: full ES
    was worse than the frozen floor and the noisy estimator overfit a fixed batch,
    while non-gradient methods need the blessing of scale they do not have here
    (Gemini note 16). With ADR 0004 (data-side) and Stage 35 (analytic-base ceiling),
    the three cheap "form the residual more cleverly" levers are now bounded. The
    decision records the decomposition behind it: on the structured corpus the frozen
    prior carries about `83%` of the recipe's edge over `random_full` and the
    trainable residual only about `17%` (`0.017708` of `0.106704` NLL). See
    `docs/decisions/0005-gradient-forms-the-residual-formation-side-closed.md`.
41. Residual marginal-value gate · implemented as Stage 37. Codex measured
    the floor-to-target gap across natural-text Tiny Shakespeare at 200 and 500
    steps for orders 2 to 4, plus a rank 1/2/4 sweep on the structured corpus.
    The gate closed: no regime reached the `0.05` NLL reopening line with stable
    positive sign. Natural-text gaps were mixed or negative, and the largest stable
    structured gap was rank 4 at only `+0.023058` NLL. This confirms the tested
    cheap regimes are prior-dominated on NLL. Claude later opened H014 on the
    behavior axis. See
    `experiments/tiny_language_lab/runs/stage37_residualgap_summary.md`.
42. Lower-priority and now gate-closed · the model-side richer frozen base,
    retrieval-interface redesign, and distilled-data experiments. Stage 37 says
    further residual-formation NLL mechanics have no large local target in the
    tested cheap regimes. Claude has now chosen the pivot: turn to the behavior
    axis (rung 43) before the prior-ceiling axis, because the north star is
    behavior and Stage 37's closure was measured only on NLL.
43. Behavior residual marginal-value gate (Hypothesis 014) · implemented as
    Stage 38. Codex re-ran the Stage 37 floor-to-target gate on copy-probe
    accuracy instead of NLL, using the long-context copy corpus. The frozen-prior
    floor copied at `0.118421`, within `0.006579` of `1 / 8` chance, while the
    rank-2 residual arms reached `0.320176` (`copyw`) and `0.307017` (`copymix`)
    mean copy accuracy. Both residual arms beat the floor by more than `0.10` on
    every seed, so H014 confirms the inversion: Stage 37's prior-dominance law is
    NLL-specific, and behavior formation lives in the trainable residual. Codex
    prepared `docs/decisions/0006-behavior-axis-reopens-residual-formation.codex-draft.md`
    for Claude review; Gemini filed note 10 on the induction-head and
    in-context-learning prior-art placement. See
    `experiments/tiny_language_lab/runs/stage38_behaviorgap.md` and
    `docs/hypotheses/014-behavior-residual-marginal-value-gate.md`.
44. ADR 0006 for the behavior-axis reopening · accepted. Claude accepted the
    Stage 38 rescoping with revisions
    (`docs/decisions/0006-behavior-axis-reopens-residual-formation.md`): the
    formation-side closure is NLL-specific, the rank-2 residual is the
    behavior-forming surface, and the frozen count prior is behavior-blind at the
    copy answer position. The ADR adds a standing dual-axis tracking rule (every
    behavior-branch stage reports both NLL and a behavior probe), keeps the
    NLL-side levers closed, and scopes the claim tightly: identity copy, seen
    keys, 76 cases per seed, about `0.31` accuracy is formation not mastery, and
    the full model is a high-variance control (seed 19 random_full `0.144737`,
    below both cheap residual arms). Grounded by Gemini note 10, no novelty
    claimed. Reverses only if cheap surfaces plateau near the floor on a harder or
    generalizing probe while only full-model capacity forms the behavior.
45. Behavior rank sweep · implemented as Stage 39 from ADR 0006's provisional
    handoff. Codex added `count_prior_lora_r1_copyw` and
    `count_prior_lora_r4_copyw`, keeping the Stage 38 corpus and copy-weighted
    protocol fixed. The sweep does not support a simple capacity-limited rank law:
    rank 2 is best on mean copy accuracy (`0.320176`), rank 1 reaches `0.250000`,
    and rank 4 reaches `0.271930`. Rank 4 does not beat rank 1 with stable sign
    (`+0.078948`, `+0.026316`, `-0.039474`) and is below rank 2 on two seeds.
    Every trained rank stays above the `0.118421` floor, so behavior formation is
    real, but the next behavior question should not simply be "more rank". Claude
    owns the next hypothesis, likely around sampler signal, verifier signal,
    generalization, or optimization stability. See
    `experiments/tiny_language_lab/runs/stage39_behavior_rank.md` and
    `docs/decisions/0006-behavior-axis-reopens-residual-formation.md`.
46. Held-out-key copy generalization (Hypothesis 015) · implemented as Codex
    Stage 40. Codex added `--holdout-keys` to the identity-copy long-context
    generator, produced `long_context_holdout_seed.txt` with `g h` held out of
    training `key=...answer=` rows, verified zero held-out train rows and in-vocab
    `g h`, then reran the Stage 38 behavior arms with
    `--copy-probe-holdout-keys g h`. The rank-2 residual did not generalize:
    `count_prior_lora_r2_copyw` tied the frozen floor at `0.000000` held-out
    accuracy on all seeds. This does not cleanly fire the memorization reversal,
    because the residual's seen-key gain over the floor was only `+0.027491` on
    the mean, far below the `+0.10` seen-formation clause, and the full control
    also collapsed to `0.000000` held-out accuracy. Stage 40 scopes ADR 0006 to
    seen-key identity copy at this budget and hands the next behavior decision to
    Claude. See `experiments/tiny_language_lab/runs/stage40_heldout_copy.md` and
    `docs/hypotheses/015-held-out-key-copy-generalization.md`.
47. Forced-choice held-out copy circuit (Hypothesis 016) · implemented as Codex
    Stage 41. Codex added forced-choice copy metrics to the probe, using the
    validation key alphabet `abcdefgh` as the candidate set, and reran the exact
    Stage 40 corpus, arms, seeds, and training protocol. The readout did not
    rescue held-out transfer: `count_prior_lora_r2_copyw` stayed at `0.000000`
    held-out choice accuracy on every seed, with held-out MRR only `+0.002645`
    above the floor on the mean. This does not cleanly fire the memorization
    reversal either, because Arm B's seen choice accuracy was only `0.202749`,
    `+0.077749` above chance and below the `+0.10` seen-power clause. The full
    control is the stronger caveat: `random_full_copymix` reached `0.364261` mean
    seen choice accuracy and `0.670103` on seed `19`, but also scored `0.000000`
    held-out choice accuracy with floor-level held-out MRR. Stage 41 rejects the
    simple free-vocab emission-artifact rescue, but it still points to a task or
    budget failure under this protocol rather than a cheap-surface-only reversal.
    Claude owns the next branch decision; Gemini owns the forced-choice and
    logit-pathway prior-art pass. See
    `experiments/tiny_language_lab/runs/stage41_forcedchoice_heldout.md` and
    `docs/hypotheses/016-forced-choice-heldout-copy-circuit.md`.
48. Consolidating ADR for the held-out generalization sub-branch · accepted. ADR
    0007 reads Stages 40 and 41 together: every arm, including the full control,
    scored `0.000000` held-out under both free-vocabulary and forced-choice
    readouts, and the held-out choice MRR (about `0.134`) ranks held-out keys near
    last among eight, at or below uniform-random. The diagnosis is structural: in
    identity copy a held-out key token never appears at the answer position, so the
    output suppresses it regardless of any copy circuit, and the probe cannot
    express a held-out copy. Decision: scope ADR 0006 to seen-key identity copy
    (the reversal clause does not fire, because the full model fails identically),
    retire the held-out-key-token probe as a generalization instrument, and keep
    the rank and NLL levers closed. The question reopens only with a valid-readout
    probe carrying a full-model ceiling that actually generalizes. See
    `docs/decisions/0007-heldout-token-copy-probe-cannot-measure-generalization.md`.
49. Memorization-proof copy probe (Hypothesis 017) - implemented as Codex
    Stage 42. Codex added `--random-payload` to the long-context generator and
    built `random_payload_copy_seed.txt` with a seeded per-line random payload
    alphabet `abcdefghijklmnop` (`V = 16`, chance `0.062500`). The cheap rank-2
    residual did not form a general copy circuit: `count_prior_lora_r2_copyw`
    reached `0.063768` mean copy accuracy, only `+0.001268` above chance and
    `+0.020290` above the floor, with every seed within the registered `0.05`
    reversal band. The full control cleared chance on every seed and reached
    `0.226087` mean copy accuracy, so H017 fires the local reversal kill for the
    current cheap recipe. See
    `experiments/tiny_language_lab/runs/stage42_random_payload_copy.md` and
    `docs/hypotheses/017-memorization-proof-copy-probe.md`.
50. Consolidating ADR for the general-copy reversal · accepted. Claude accepted
    Codex's Stage 42 evidence draft with revisions as ADR 0008
    (`docs/decisions/0008-general-copy-needs-more-than-current-rank2-residual.md`).
    Stage 42 fires ADR 0006's reversal clause: the rank-2 residual forms seen-content
    copy behavior (Stage 38) but not a general in-context copy circuit, so ADR 0006 is
    narrowed to seen-content, not reversed wholesale. Crucially this is not a
    needs-more-than-this-hardware result: the full model formed the circuit at the same
    budget (`0.226087`), so the capacity and trainable-surface question reopens on the
    general task (Stage 39 closed rank only on the memorizable task). It is also a
    mirror-image dual-axis split: the cheap residual gained `0.062257` validation NLL
    over the floor yet formed zero copy behavior. Reverses toward the stronger claim
    only if a small surface forms the circuit (Hypothesis 018); hardens into a capacity
    wall if nothing short of the full body does.
51. Minimal surface for general copy (Hypothesis 018) · implemented as Codex
    Stage 43. Codex added `count_prior_lora_r8_copyw`,
    `count_prior_lora_r16_copyw`, and `count_prior_all_copyw`, then reran the
    Stage 42 memorization-proof corpus with alpha matched to rank for the new
    higher-rank LoRA arms. The registered KILL line fired: rank 8 (`0.049275`)
    and rank 16 (`0.049276`) stayed at chance and below rank 2 (`0.063768`), and
    the full-body-on-frozen-base diagnostic stayed at `0.043478` on every seed.
    Only `random_full_copymix` cleared chance on all seeds (`0.226087` mean). Arm
    D against Arm E points to frozen-base interference under this protocol, not
    merely low LoRA rank. H018 is tested; Codex prepared
    `docs/decisions/0009-general-copy-frozen-prior-capacity-wall.codex-draft.md`
    for Claude review. See
    `experiments/tiny_language_lab/runs/stage43_general_copy_surface.md` and
    `docs/hypotheses/018-minimal-surface-general-copy.md`.
52. Phase 2 TinyStories bridge · implemented as Codex Stage 44 from ADR 0010.
    Codex added retryable TinyStories download, character-level TinyStories
    corpus prep, visible PowerShell launch scripts, matrix progress logging, and
    a vectorized n-gram prior builder. The prepared bridge corpus has
    `10,000,001` normalized characters and `V = 33`. Visible CUDA smokes passed,
    then the b500 bridge matrix ran seeds `7 11 19` for `random_full`,
    `count_prior_ng3_lora_r2`, and `count_prior_ng4_lora_r2`. At 500 steps,
    order-4 frozen prior plus rank-2 LoRA reached `1.139715` mean validation NLL
    versus `2.352297` for the full random baseline, with `41,249` trainable
    parameters versus `3,209,249`. Generation from `once upon a time ` is rough
    but story-like for the frozen-prior arms and still noisy for the random full
    baseline. This is a successful Phase 2 bridge result, not yet the full
    modded-nanoGPT baseline. See
    `experiments/tiny_language_lab/runs/phase2_tinystories_bridge_b500.md`,
    `experiments/tiny_language_lab/RESULTS.md`, and
    `docs/decisions/0010-confirmation-report.md`.
53. Phase 2 modern TinyStories baseline · implemented as Codex Stage 45 from ADR
    0010. Codex added RoPE, gradient accumulation, activation checkpointing, and
    a single-device Muon optimizer path, then exposed those knobs through
    `cassandra_compare.py` and the visible PowerShell runner. The visible
    `modern-smoke` passed, then `modern500` ran seeds `7 11 19` for
    `random_full` and `count_prior_ng4_lora_r2` with `--pos-encoding rope`,
    `--optimizer muon`, `--grad-accum-steps 2`, and `--activation-checkpoint`.
    At 500 steps, the modern `random_full` baseline reached `1.144942` mean
    validation NLL, improving sharply over Stage 44's `2.352297` bridge baseline.
    The modern order-4 frozen prior reached `1.102748`, retaining a smaller
    `0.042194` NLL lead while training `41,249` parameters instead of
    `3,176,481`. See
    `experiments/tiny_language_lab/runs/phase2_tinystories_modern_b500.md`,
    `experiments/tiny_language_lab/RESULTS.md`, and
    `docs/decisions/0010-confirmation-report.md`.
54. Phase 2 generation-quality scoring · implemented as Codex Stage 46 from ADR
    0010. Codex added `score_generation_samples.py`, a deterministic local
    proxy score sheet over saved prompt completions. It scores coherence,
    grammar, and prompt relevance from `0` to `2` each, using prompt adherence,
    story cues, corpus-word ratio, punctuation, repetition, and bad-marker
    checks. On the Stage 45 modern b500 samples, `count_prior_ng4_lora_r2`
    scored `5.667/6` mean total versus `3.000/6` for `random_full`. See
    `experiments/tiny_language_lab/runs/phase2_tinystories_modern_b500_generation_quality.md`.
55. Phase 2 shard-consumption smoke · implemented as Codex Stage 47 from ADR
    0010. Codex added `--train-shard-dir` and `--stream-train-eval-chars` to the
    trainer and compare harness, plus a visible `stream-smoke` launcher mode.
    The smoke consumed the five TinyStories train shard files for plain LM
    batches, reported `train_chars = 8,500,000` and `train_eval_chars =
    200,000`, and completed a 20-step RoPE/Muon CUDA run at `2.216325`
    validation NLL. See
    `experiments/tiny_language_lab/runs/phase2_tinystories_stream_smoke.md`.
56. Phase 2 modern crossover · implemented as Codex Stage 48 from ADR 0010.
    Codex added a visible `modern1000` launcher mode and ran the modern
    TinyStories character baseline across seeds `7 11 19`. The NLL crossover
    fired: `random_full` reached `1.052559` mean validation NLL at 1000 steps,
    while `count_prior_ng4_lora_r2` reached `1.123161`, reversing the prior
    arm's 500-step lead. The prior-minus-full gap was positive on every seed:
    `+0.071768`, `+0.079357`, and `+0.060680`. The generation proxy still favored
    the prior arm because two random-full samples emitted `endoftext` artifacts.
    See `experiments/tiny_language_lab/runs/phase2_tinystories_modern_b1000.md`.
57. Phase 2 BPE smoke · implemented as Codex Stage 49 from ADR 0010. Codex added
    `make_bpe_corpus.py` and `decode_bpe_samples.py`, trained a local 256-token
    BPE tokenizer on a bounded TinyStories slice, encoded `1,000,000` characters
    into `446,694` BPE tokens, and ran a visible BPE smoke. The BPE-token bigram
    prior plus rank-2 LoRA reached `3.405228` validation NLL versus `3.847587`
    for `random_full` at 20 steps. This proves the BPE-token path and n-gram
    prior over BPE tokens run locally; it is not yet the durable BPE-vs-character
    decision. See
    `experiments/tiny_language_lab/runs/phase2_tinystories_bpe_smoke.md`.
58. BPE 500-step multi-seed matrix · implemented as Codex Stage 50 from ADR
    0010. The visible `bpe500` run tested the v256 BPE corpus across seeds
    `7 11 19`. At 500 steps, full BPE training won decisively:
    `random_full` reached `2.404960` mean validation NLL, while the BPE-token
    bigram prior plus rank-2 LoRA reached `3.344760`. Approximate
    source-normalized bits were `1.549860` vs `2.155508`. Decision: keep the
    character-level TinyStories baseline as the current Phase 2 default, and keep
    BPE as a secondary branch for larger-vocab or higher-order-prior follow-up.
    See `experiments/tiny_language_lab/runs/phase2_tinystories_bpe_b500.md`.
59. Phase 3 coherence checkpoint · implemented as Codex Stage 51 from ADR
    0011. Codex rebuilt the TinyStories character corpus from a `500,000,000`
    byte raw slice, yielding `494,094,421` normalized characters,
    `419,980,257` train characters, `74,114,164` validation characters, and
    `V = 33`, then trained the 25.25M `random_full` checkpoint family for
    `5000` steps across seeds `7 11 19`. Mean validation NLL reached
    `0.818608`, with mean bits/char `1.181001` and deterministic generation
    proxy score `5.667/6`. Human review of saved samples is still pending
    before any genuine-coherence claim. See
    `experiments/tiny_language_lab/runs/stage51_coherence_25m_b5000.md`.
60. H019 crossover-scaling matrix · implemented as Codex Stage 52 from ADR
    0011. The 85M confirmation smoke cleared the 8 GB laptop, so the matrix ran
    sizes `3m`, `10m`, `25m`, and `85m`, budgets `200 500 1000 2000`, both
    `random_full` and `count_prior_ng4_lora_r2`, and seeds `7 11 19` with
    sampled evaluation. The first prior pass OOMed through full-tensor prior
    construction and is preserved as failed evidence; the corrected shard-native
    prior pass completed. Crossovers were `1000`, `1000`, `1000`, and `500`
    steps respectively, so H019 is `GRADED`, not clean E1, E2, or E3. See
    `experiments/tiny_language_lab/runs/stage52_h019_crossover_scaling_summary.md`
    and `docs/hypotheses/019-crossover-scaling-law.md`.
61. Free-accelerator test · implemented as Codex Stage 53 from ADR 0013
    (H020). Codex added `count_prior_ng4_all`, a 25.25M model with the frozen
    order-4 prior and the full body trainable. The arm beat Stage 52
    `random_full` at 200 and 500 steps, then lost on all three paired seeds at
    1000 and 2000 steps. The required `--muon-lr 0.005` sensitivity rerun
    improved the 2000-step mean to `0.976702` but still trailed
    `random_full` by `+0.054555` mean NLL. Verdict: H020 KILL E-interfere.
    See
    `experiments/tiny_language_lab/runs/stage53_h020_free_accelerator_summary.md`
    and `docs/hypotheses/020-frozen-prior-free-accelerator.md`.
62. Prior-order floor scaling · implemented as Codex Stage 54 from ADR 0013
    (H021). Codex added a shard-native sparse order-5 backoff prior, gated it
    against the paired order-4 floor, then reran the 25.25M crossover column.
    The order-5 floor beat order 4 by `-0.117649` mean paired NLL, with all
    three seed deltas negative. The 25.25M crossover moved from 1000 steps to
    2000 steps: order 5 beat Stage 52 `random_full` by mean at 200, 500, and
    1000 steps, then lost at 2000. Verdict: H021 CONFIRM for the low-budget
    tiny-surface recipe. See
    `experiments/tiny_language_lab/runs/stage54_h021_prior_order_floor_scaling_summary.md`
    and `docs/hypotheses/021-prior-order-floor-scaling.md`.
63. Phase 4 flagship build · completed as Codex Stage 55 from ADR 0013. The
    laptop flagship uses random initialization, `201.61M` trainable parameters,
    block 256, RoPE, activation checkpointing, Muon, sampled eval, and the
    494M-character TinyStories corpus. Seed 7 ran 50,000 steps and reached
    `0.556410` sampled report validation NLL (`0.802730` bits/char); seeds 11
    and 19 ran reduced 20,000-step replicas. The mixed-budget generation proxy
    mean is `4.333/6`, with human review still pending. Final `.pt`
    checkpoints were originally secured in
    `C:\cassandra_runs\stage55_flagship_checkpoints`; the repo-local seed-7
    artifact checkpoint remains under
    `experiments/tiny_language_lab/artifacts/phase4/checkpoints/` after the
    Stage 56 disk cleanup. The verified DL Designer ONNX closeout lives under
    `experiments/tiny_language_lab/artifacts/phase4/nsight_dld/`.
64. Broad-corpus specialization-gap test (Hypothesis 022) · completed as
    Codex Stage 56 from ADR 0015. ADR 0014 measured the flagship's
    domain-specialization gap (`0.8126` bits/char in-domain vs `2.8817`
    zero-shot text8). Stage 56 trained the unchanged 85.10M character recipe
    on text8 train/valid data and scored the untouched text8 TEST split.
    Seed 7 at 50,000 steps reached `1.485740` TEST bits/char, below the
    `1.70` CONFIRM line; the 20,000-step replicas landed at `1.532627`
    (seed 11) and `1.529591` (seed 19), a `0.003035` bits/char spread.
    Verdict: H022 CONFIRM, so the gap is a data-distribution effect under the
    registered test and the Phase 5 developmental experiment stays on the char
    substrate unless a later ADR changes course. See
    `experiments/tiny_language_lab/RESULTS.md` and
    `docs/hypotheses/022-broad-corpus-specialization-gap.md`.
65. Recipe v2 gates · completed as Codex Stage 57 from ADR 0015. bf16
    matched fp32 NLL at 200 steps but was slower (`0.9058x` fp32 throughput),
    so fp32 stays. Cosine LR warmdown won the 25.25M 5000-step gate
    (`1.071579` vs `1.139756` final sampled val NLL), so Recipe v2 adopts
    `--lr-schedule cosine --lr-final-frac 0.1`. `--checkpoint-keep`,
    fp16 model-only archive mode, and `--vocab-chars` passed smoke tests.
    Block 512 produced an 85M timing row (`306.5595s` for 200 steps,
    `1,645.0039` peak CUDA MiB) but does not become the default. Phase 5's
    centerpiece (H024, the developmental experiment: COLD vs CURRICULUM vs
    MIXTURE at one fixed budget) remains pre-registered in ADR 0015 D1 and
    must be written by Claude before Codex starts Stage 58. See
    `experiments/tiny_language_lab/RESULTS.md` and
    `docs/decisions/0015-phase-5-developmental-training-and-open-source-posture.md`.
66. Phase 5 letters-only behavior probe · completed as an eval-only D4 check
    from ADR 0015. Codex added a 1,024-case letters-only copy probe because the
    Stage 55 flagship vocabulary has no digits or equals sign. The Stage 55
    seed-7 flagship scored `0.060547` constrained-choice copy accuracy against
    chance `0.062500`, below the reopen threshold `0.162500`, so the behavior
    axis stays closed. Stage 58 then ran under Claude's H024 (rung 68). See
    `experiments/tiny_language_lab/runs/phase5_behavior_letters_probe.md`.
67. Phase 5 D2 release prep · partially completed up to the user-sign-off
    boundary. Corpus payloads were removed from the Git index, generated corpus
    payloads are ignored while `.meta.json` stays trackable, a 50 MiB
    pre-commit guard was added, licensing notes and a flagship model-card draft
    were written, and fp16 model-only exports of the three Stage 56 finals were
    created under `C:\cassandra_runs\phase5_model_only_exports`. History
    surgery, license choice, Hub upload, public push, and Stage 58 remain gated.
    See `docs/phase5-d2-prep-status.md`.
68. Phase 5 developmental experiment (H024) · completed by Codex as Stage 58
    and resolved E-NULL, seed-robust in sign (2026-07-21, ADR 0016). Three
    equal-compute 85.11M arms decided the smartest acquisition order for
    breadth: COLD (full budget on broad text), CURRICULUM (a TinyStories
    childhood, then broad), and MIXTURE (interleaved at the dose-matched
    25:59 ratio). On deterministic text8 TEST bits/char, CURRICULUM minus
    COLD read `+0.005096` (seed 7, 42,000 steps), `+0.009845` (seed 11, 20k
    replica), and `+0.007791` (seed 19, 20k replica): every sign favors
    COLD, every magnitude sits inside the `+/-0.05` E-null band, and the
    escalation rule did not fire. MIXTURE scored `+0.027977` versus COLD on
    text8 while retaining TinyStories at `0.826285` bits/char against
    `3.55`/`3.53` for the other arms: rehearsal preserves the narrow
    register cheaply, but no ordering beats cold broad training on the
    primary metric. The cross-stage lesson: Recipe v2 improved matched-seed,
    matched-budget text8 TEST by about `0.12` bits/char over Stage 56's
    Recipe v1, roughly 24 times the largest ordering effect. See
    `docs/hypotheses/024-developmental-acquisition-order.md`,
    `docs/phase5-final-report.md`, `docs/figures/phase5/`, and
    `docs/decisions/0016-phase-5-closeout-developmental-null-recipe-frontier.md`.
69. Phase 6 rehearsal dose-response (H025) · specced by Claude and audited,
    awaiting Codex as Stage 59. Part 0 runs the letters copy probe on the
    Stage 58 broad-trained COLD final (reopen line `0.1625`). Part 1 sweeps
    TinyStories dose `w` in `{0, 0.05, 0.10, 0.20, 0.30, 0.50}` on 3.2M-class
    Recipe v2 proxies (seeds 7 11 19) and fits a data mixing law whose 85M
    prediction must be published before Part 2 launches. Part 2 trains two
    85M MIXTURE `w = 0.10` arms at 20,000 steps (seeds 11 and 19) against the
    existing paired Stage 58 COLD baselines; CONFIRM needs both seeds at
    `d <= +0.010` broad toll with `r >= 1.0` retention gain. See
    `docs/hypotheses/025-rehearsal-dose-response-and-mixing-law.md` and
    `docs/phase6-codex-goal-prompt.md`.
70. Phase 6.1 dual-register flagship (Stage 60) · GATED on rung 69 reading
    E-cheap-rehearsal (ADR 0017 D2); does not launch otherwise. A 200M-class
    Recipe v2 model in the Stage 55 configuration lineage trains on one
    continuous mixture at the validated dose, budget set by a sustained
    throughput measure inside a 30 to 40 GPU-hour envelope (50,000-step
    ceiling). Publish-worthiness bars are pre-registered in ADR 0017 D4
    (beat `1.357318` text8 TEST, retention at least `1.0` bits/char better
    than the `3.556502` no-rehearsal anchor, letters probe recorded, sample
    review); Hugging Face packaging is prepared by Codex, every public
    button is the user's. See
    `docs/decisions/0017-phase-6-rehearsal-dose-flagship-gate.md`.
