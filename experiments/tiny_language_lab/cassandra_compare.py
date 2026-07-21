from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from cassandra_tiny_transformer import train


DEFAULT_CORPUS = Path(__file__).with_name("corpus") / "structured_seed.txt"
DEFAULT_OUT = Path(__file__).with_name("runs") / "stage5_replication.jsonl"
DEFAULT_SUMMARY = Path(__file__).with_name("runs") / "stage5_replication.md"
STAGE59_UNION_VOCAB = Path(__file__).with_name("corpus") / "phase5_union_vocab.txt"


def stage59_proxy_recipe_args(steps: int) -> dict[str, object]:
    return {
        "block_size": 256,
        "batch_size": 8,
        "grad_accum_steps": 2,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "pos_encoding": "rope",
        "activation_checkpoint": True,
        "precision": "fp32",
        "optimizer": "muon",
        "muon_lr": 0.01,
        "lr_schedule": "cosine",
        "lr_final_frac": 0.1,
        "lr_total_steps": steps,
        "vocab_chars_file": STAGE59_UNION_VOCAB,
        "val_fraction": 0.05263157894736842,
        "eval_mode": "sampled",
        "eval_batches": 16,
        "copy_train_marker": "",
    }


def base_args(
    corpus: Path,
    device: str,
    seed: int,
    steps: int,
    eval_batches: int,
    block_size: int,
    eval_mode: str,
    count_alpha: float,
    ngram_backoff: float,
    recency_tau: float,
    curriculum_filter: str,
    curriculum_fraction: float,
    curriculum_rescore_every: int,
    curriculum_pool_size: int,
    search_batches: int,
    search_population: int,
    search_sigma: float,
    search_lr: float,
    coord_step_size: float,
    copy_probe_marker: str,
    copy_probe_retrieval_template: str,
    copy_probe_retrieval_source: str,
    copy_probe_retrieval_corrupt: str,
    copy_probe_memory_scope: str,
    copy_probe_holdout_keys: str | list[str],
    copy_verify_mode: str,
) -> dict[str, object]:
    return {
        "corpus": corpus,
        "device": device,
        "init": "random",
        "residual_base": "none",
        "count_alpha": count_alpha,
        "trigram_backoff": 1.0,
        "prior_order": 1,
        "ngram_backoff": ngram_backoff,
        "prior5_min_count": 10,
        "recency_tau": recency_tau,
        "recency_lambda": 0.0,
        "curriculum_filter": "off",
        "curriculum_fraction": curriculum_fraction,
        "curriculum_rescore_every": curriculum_rescore_every,
        "curriculum_pool_size": curriculum_pool_size,
        "count_embedding_scale": 1.0,
        "residual_scale": 1.0,
        "residual_l2": 0.0,
        "zero_residual_head": True,
        "steps": steps,
        "residual_optim": "adamw",
        "search_batches": search_batches,
        "search_population": search_population,
        "search_sigma": search_sigma,
        "search_lr": search_lr,
        "coord_step_size": coord_step_size,
        "train_scope": "all",
        "adapter_rank": 0,
        "lora_rank": 0,
        "lora_alpha": 1.0,
        "lora_dropout": 0.0,
        "batch_size": 16,
        "grad_accum_steps": 1,
        "train_shard_dir": None,
        "stream_train_eval_chars": 200_000,
        "retention_eval_corpus": None,
        "prior_cache_dir": None,
        "checkpoint_dir": None,
        "checkpoint_every": 0,
        "checkpoint_keep": 0,
        "resume_from": None,
        "resume_steps_additional": False,
        "model_only_out": None,
        "model_only_dtype": "fp16",
        "vocab_chars": "",
        "vocab_chars_file": None,
        "block_size": block_size,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 64,
        "pos_encoding": "learned",
        "activation_checkpoint": False,
        "dropout": 0.0,
        "precision": "fp32",
        "optimizer": "adamw",
        "lr": 0.003,
        "lr_schedule": "constant",
        "lr_final_frac": 0.1,
        "lr_total_steps": 0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_eps": 1e-10,
        "muon_lr": 0.02,
        "muon_momentum": 0.95,
        "muon_ns_steps": 5,
        "no_muon_nesterov": False,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "val_fraction": 0.15,
        "eval_mode": eval_mode,
        "eval_interval": steps,
        "eval_batches": eval_batches,
        "log_every": 0,
        "prompt": "cassandra ",
        "max_new_tokens": 80,
        "temperature": 0.9,
        "copy_train_marker": "",
        "copy_sampler": "random",
        "copy_sample_fraction": 0.5,
        "copy_rehearsal_fraction": 0.05,
        "copy_curriculum_switch_fraction": 0.5,
        "copy_loss_weight": 1.0,
        "copy_choice_weight": 0.0,
        "copy_verify_mode": copy_verify_mode,
        "copy_mine_every": 50,
        "copy_correction_template": "compact",
        "copy_train_retrieval_template": "compact",
        "copy_probe_marker": copy_probe_marker,
        "copy_probe_key_marker": "key=",
        "copy_probe_max_cases": 200,
        "copy_probe_retrieval_template": copy_probe_retrieval_template,
        "copy_probe_retrieval_source": copy_probe_retrieval_source,
        "copy_probe_retrieval_corrupt": copy_probe_retrieval_corrupt,
        "copy_probe_memory_scope": copy_probe_memory_scope,
        "copy_probe_holdout_keys": copy_probe_holdout_keys,
        "seed": seed,
        "out": None,
    }


def config_args(
    name: str,
    corpus: Path,
    device: str,
    seed: int,
    steps: int,
    eval_batches: int,
    block_size: int,
    eval_mode: str,
    count_alpha: float,
    ngram_backoff: float,
    recency_tau: float,
    recency_lambda: float,
    curriculum_filter: str,
    curriculum_fraction: float,
    curriculum_rescore_every: int,
    curriculum_pool_size: int,
    search_batches: int,
    search_population: int,
    search_sigma: float,
    search_lr: float,
    coord_step_size: float,
    copy_probe_marker: str,
    copy_probe_retrieval_template: str,
    copy_probe_retrieval_source: str,
    copy_probe_retrieval_corrupt: str,
    copy_probe_memory_scope: str,
    copy_probe_holdout_keys: str | list[str],
    copy_train_marker: str,
    copy_sampler: str,
    copy_sample_fraction: float,
    copy_rehearsal_fraction: float,
    copy_curriculum_switch_fraction: float,
    copy_loss_weight: float,
    copy_choice_weight: float,
    copy_verify_mode: str,
    copy_mine_every: int,
    copy_correction_template: str,
    copy_train_retrieval_template: str,
) -> SimpleNamespace:
    args = base_args(
        corpus,
        device,
        seed,
        steps,
        eval_batches,
        block_size,
        eval_mode,
        count_alpha,
        ngram_backoff,
        recency_tau,
        curriculum_filter,
        curriculum_fraction,
        curriculum_rescore_every,
        curriculum_pool_size,
        search_batches,
        search_population,
        search_sigma,
        search_lr,
        coord_step_size,
        copy_probe_marker,
        copy_probe_retrieval_template,
        copy_probe_retrieval_source,
        copy_probe_retrieval_corrupt,
        copy_probe_memory_scope,
        copy_probe_holdout_keys,
        copy_verify_mode,
    )
    args["copy_rehearsal_fraction"] = copy_rehearsal_fraction
    if name == "random_full":
        pass
    elif name == "stage59_proxy_random_full":
        args.update(stage59_proxy_recipe_args(steps))
    elif name == "random_full_copyw":
        args.update(
            {
                "copy_train_marker": copy_train_marker,
                "copy_sampler": copy_sampler,
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
            }
        )
    elif name == "random_full_copyw_choice":
        args.update(
            {
                "copy_train_marker": copy_train_marker,
                "copy_sampler": copy_sampler,
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
            }
        )
    elif name == "random_full_copyfail":
        args.update(
            {
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "failed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
                "copy_mine_every": copy_mine_every,
                "copy_correction_template": copy_correction_template,
            }
        )
    elif name == "random_full_copyfailmix":
        args.update(
            {
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "failed_mixed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
                "copy_mine_every": copy_mine_every,
                "copy_correction_template": copy_correction_template,
            }
        )
    elif name == "random_full_copycorr":
        args.update(
            {
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "correction",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
                "copy_mine_every": copy_mine_every,
                "copy_correction_template": copy_correction_template,
            }
        )
    elif name == "random_full_copycorrmix":
        args.update(
            {
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "correction_mixed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
                "copy_mine_every": copy_mine_every,
                "copy_correction_template": copy_correction_template,
            }
        )
    elif name == "random_full_retmix":
        args.update(
            {
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "retrieval_mixed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_train_retrieval_template": copy_train_retrieval_template,
            }
        )
    elif name == "random_full_corrretmix":
        args.update(
            {
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "correction_retrieval_mixed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
                "copy_mine_every": copy_mine_every,
                "copy_correction_template": copy_correction_template,
                "copy_train_retrieval_template": copy_train_retrieval_template,
            }
        )
    elif name == "random_full_corrthenret":
        args.update(
            {
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "correction_then_retrieval_mixed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_curriculum_switch_fraction": copy_curriculum_switch_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
                "copy_mine_every": copy_mine_every,
                "copy_correction_template": copy_correction_template,
                "copy_train_retrieval_template": copy_train_retrieval_template,
            }
        )
    elif name == "random_full_copys":
        args.update(
            {
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "answer",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
            }
        )
    elif name == "random_full_copymix_choice":
        args.update(
            {
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "mixed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
            }
        )
    elif name == "count_prior_lora_r2_copyfail":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "failed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
                "copy_mine_every": copy_mine_every,
                "copy_correction_template": copy_correction_template,
            }
        )
    elif name == "count_prior_lora_r2_copyfailmix":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "failed_mixed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
                "copy_mine_every": copy_mine_every,
                "copy_correction_template": copy_correction_template,
            }
        )
    elif name == "count_prior_lora_r2_copycorr":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "correction",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
                "copy_mine_every": copy_mine_every,
                "copy_correction_template": copy_correction_template,
            }
        )
    elif name == "count_prior_lora_r2_copycorrmix":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "correction_mixed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
                "copy_mine_every": copy_mine_every,
                "copy_correction_template": copy_correction_template,
            }
        )
    elif name == "count_prior_lora_r2_retmix":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "retrieval_mixed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_train_retrieval_template": copy_train_retrieval_template,
            }
        )
    elif name == "count_prior_lora_r2_corrretmix":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "correction_retrieval_mixed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
                "copy_mine_every": copy_mine_every,
                "copy_correction_template": copy_correction_template,
                "copy_train_retrieval_template": copy_train_retrieval_template,
            }
        )
    elif name == "count_prior_lora_r2_corrthenret":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "correction_then_retrieval_mixed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_curriculum_switch_fraction": copy_curriculum_switch_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
                "copy_mine_every": copy_mine_every,
                "copy_correction_template": copy_correction_template,
                "copy_train_retrieval_template": copy_train_retrieval_template,
            }
        )
    elif name == "count_prior_lora_r2_corrthenret_rehearsal":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "correction_then_retrieval_rehearsal_mixed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_rehearsal_fraction": copy_rehearsal_fraction,
                "copy_curriculum_switch_fraction": copy_curriculum_switch_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
                "copy_mine_every": copy_mine_every,
                "copy_correction_template": copy_correction_template,
                "copy_train_retrieval_template": copy_train_retrieval_template,
            }
        )
    elif name in {
        "count_prior_lora_r4_corrretmix",
        "count_prior_lora_r8_corrretmix",
        "count_prior_lora_r4_corrthenret",
        "count_prior_lora_r8_corrthenret",
    }:
        rank = 4 if "_r4_" in name else 8
        sampler = (
            "correction_then_retrieval_mixed"
            if name.endswith("_corrthenret")
            else "correction_retrieval_mixed"
        )
        update = {
            "residual_base": "count-bigram",
            "train_scope": "lora",
            "lora_rank": rank,
            "lora_alpha": float(rank),
            "copy_train_marker": copy_train_marker,
            "copy_sampler": sampler,
            "copy_sample_fraction": copy_sample_fraction,
            "copy_loss_weight": copy_loss_weight,
            "copy_choice_weight": copy_choice_weight,
            "copy_mine_every": copy_mine_every,
            "copy_correction_template": copy_correction_template,
            "copy_train_retrieval_template": copy_train_retrieval_template,
        }
        if sampler == "correction_then_retrieval_mixed":
            update["copy_curriculum_switch_fraction"] = copy_curriculum_switch_fraction
        args.update(update)
    elif name == "random_full_copymix":
        args.update(
            {
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "mixed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
            }
        )
    elif name == "count_prior_lora_r2_copyw_choice":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "copy_train_marker": copy_train_marker,
                "copy_sampler": copy_sampler,
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
            }
        )
    elif name == "count_prior_adapter_r4":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "adapters",
                "adapter_rank": 4,
            }
        )
    elif name == "count_prior_head":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "head",
            }
        )
    elif name in {
        "count_prior_lora_r1",
        "count_prior_lora_r2",
        "count_prior_lora_r4",
        "count_prior_lora_r1_floor",
        "count_prior_lora_r2_floor",
        "count_prior_lora_r4_floor",
    }:
        rank = int(name.split("_r", 1)[1].split("_", 1)[0])
        update = {
            "residual_base": "count-bigram",
            "train_scope": "lora",
            "lora_rank": rank,
            "lora_alpha": float(rank),
        }
        if name.endswith("_floor"):
            update["residual_optim"] = "none"
        args.update(update)
    elif name == "count_prior_lora_r2_es":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "residual_optim": "es",
            }
        )
    elif name == "count_prior_lora_r1_coord":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "lora",
                "lora_rank": 1,
                "lora_alpha": 1.0,
                "residual_optim": "coord",
            }
        )
    elif name == "count_prior_tri_lora_r2":
        args.update(
            {
                "residual_base": "count-trigram",
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
            }
        )
    elif name == "count_prior_ng1_lora_r2":
        args.update(
            {
                "residual_base": "count-ngram",
                "prior_order": 1,
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
            }
        )
    elif name in {"count_prior_ng2_lora_r2", "count_prior_ng2_lora_r2_floor"}:
        update = {
            "residual_base": "count-ngram",
            "prior_order": 2,
            "train_scope": "lora",
            "lora_rank": 2,
            "lora_alpha": 2.0,
        }
        if name.endswith("_floor"):
            update["residual_optim"] = "none"
        args.update(update)
    elif name == "count_prior_ng2_recency_lora_r2":
        args.update(
            {
                "residual_base": "count-ngram",
                "prior_order": 2,
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "recency_tau": recency_tau,
                "recency_lambda": recency_lambda,
            }
        )
    elif name == "count_prior_ng2_lora_r2_filter":
        args.update(
            {
                "residual_base": "count-ngram",
                "prior_order": 2,
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "curriculum_filter": "prior-loss" if curriculum_filter == "off" else curriculum_filter,
            }
        )
    elif name in {
        "count_prior_ng2_lora_r2_filter_f025",
        "count_prior_ng2_lora_r2_filter_f050",
        "count_prior_ng2_lora_r2_filter_f100",
    }:
        fixed_fraction = {
            "count_prior_ng2_lora_r2_filter_f025": 0.25,
            "count_prior_ng2_lora_r2_filter_f050": 0.50,
            "count_prior_ng2_lora_r2_filter_f100": 1.00,
        }[name]
        args.update(
            {
                "residual_base": "count-ngram",
                "prior_order": 2,
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "curriculum_filter": "prior-loss",
                "curriculum_fraction": fixed_fraction,
            }
        )
    elif name in {
        "count_prior_ng2_lora_r2_dynfilter_f025",
        "count_prior_ng2_lora_r2_dynfilter_f050",
    }:
        fixed_fraction = {
            "count_prior_ng2_lora_r2_dynfilter_f025": 0.25,
            "count_prior_ng2_lora_r2_dynfilter_f050": 0.50,
        }[name]
        args.update(
            {
                "residual_base": "count-ngram",
                "prior_order": 2,
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "curriculum_filter": "dynamic-reducible",
                "curriculum_fraction": fixed_fraction,
            }
        )
    elif name == "count_prior_ng4_all":
        args.update(
            {
                "residual_base": "count-ngram",
                "prior_order": 4,
                "train_scope": "all",
            }
        )
    elif name in {
        "count_prior_ng3_lora_r2",
        "count_prior_ng4_lora_r2",
        "count_prior_ng5_lora_r2",
        "count_prior_ng3_lora_r2_floor",
        "count_prior_ng4_lora_r2_floor",
        "count_prior_ng5_lora_r2_floor",
    }:
        order = int(name.split("_ng", 1)[1].split("_", 1)[0])
        update = {
            "residual_base": "count-ngram",
            "prior_order": order,
            "train_scope": "lora",
            "lora_rank": 2,
            "lora_alpha": 2.0,
        }
        if name.endswith("_floor"):
            update["residual_optim"] = "none"
        args.update(update)
    elif name in {
        "count_prior_lora_r1_copyw",
        "count_prior_lora_r2_copyw",
        "count_prior_lora_r4_copyw",
        "count_prior_lora_r8_copyw",
        "count_prior_lora_r16_copyw",
        "count_prior_lora_r2_copyw_floor",
    }:
        rank = 2 if name.endswith("_floor") else int(name.split("_r", 1)[1].split("_", 1)[0])
        alpha = float(rank) if name in {"count_prior_lora_r8_copyw", "count_prior_lora_r16_copyw"} else 2.0
        update = {
            "residual_base": "count-bigram",
            "train_scope": "lora",
            "lora_rank": rank,
            "lora_alpha": alpha,
            "copy_train_marker": copy_train_marker,
            "copy_sampler": copy_sampler,
            "copy_sample_fraction": copy_sample_fraction,
            "copy_loss_weight": copy_loss_weight,
        }
        if name.endswith("_floor"):
            update["residual_optim"] = "none"
        args.update(update)
    elif name == "count_prior_all_copyw":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "all",
                "copy_train_marker": copy_train_marker,
                "copy_sampler": copy_sampler,
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
            }
        )
    elif name == "count_prior_lora_r2_copymix_choice":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "mixed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
                "copy_choice_weight": copy_choice_weight,
            }
        )
    elif name == "count_prior_lora_r2_copys":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "answer",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
            }
        )
    elif name == "count_prior_lora_r2_copymix":
        args.update(
            {
                "residual_base": "count-bigram",
                "train_scope": "lora",
                "lora_rank": 2,
                "lora_alpha": 2.0,
                "copy_train_marker": copy_train_marker,
                "copy_sampler": "mixed",
                "copy_sample_fraction": copy_sample_fraction,
                "copy_loss_weight": copy_loss_weight,
            }
        )
    else:
        raise ValueError(f"Unknown config: {name}")
    return SimpleNamespace(**args)


def summarize(rows: list[dict[str, object]], title: str) -> str:
    error_rows = [row for row in rows if row.get("status") == "error"]
    rows = [row for row in rows if row.get("status") != "error"]
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        group_name = str(row["comparison_name"])
        if row.get("mixture_dose") is not None:
            group_name += f" w={float(row['mixture_dose']):.2f}"
        grouped.setdefault(group_name, []).append(row)

    lines = [
        f"# {title}",
        "",
    ]
    has_probe = any(row.get("copy_probe_accuracy") is not None for row in rows)
    has_split_probe = any(
        int(row.get("copy_probe_heldout_cases") or 0) > 0
        or bool(row.get("copy_probe_holdout_keys"))
        for row in rows
    )
    has_choice_probe = any(
        row.get("copy_probe_seen_choice_accuracy") is not None
        or row.get("copy_probe_heldout_choice_accuracy") is not None
        for row in rows
    )
    has_retention_eval = any(row.get("retention_val_nll") is not None for row in rows)

    if has_probe:
        if has_split_probe:
            if has_choice_probe:
                lines.extend(
                    [
                        "| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Min copy accuracy | Max copy accuracy | Mean seen accuracy | Min seen accuracy | Max seen accuracy | Mean heldout accuracy | Min heldout accuracy | Max heldout accuracy | Mean seen choice accuracy | Mean heldout choice accuracy | Mean seen choice MRR | Mean heldout choice MRR | Mean copy NLL | Mean seconds |",
                        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                    ]
                )
            else:
                lines.extend(
                    [
                        "| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Min copy accuracy | Max copy accuracy | Mean seen accuracy | Min seen accuracy | Max seen accuracy | Mean heldout accuracy | Min heldout accuracy | Max heldout accuracy | Mean copy NLL | Mean seconds |",
                        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                    ]
                )
        else:
            if has_choice_probe:
                lines.extend(
                    [
                        "| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Min copy accuracy | Max copy accuracy | Mean choice accuracy | Mean choice MRR | Mean copy NLL | Mean seconds |",
                        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                    ]
                )
            else:
                lines.extend(
                    [
                        "| Config | Seeds | Trainable params | Mean val NLL | Mean bits/char | Mean copy accuracy | Min copy accuracy | Max copy accuracy | Mean copy NLL | Mean seconds |",
                        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                    ]
                )
    else:
        if has_retention_eval:
            lines.extend(
                [
                    "| Config | Seeds | Optimizer | Residual route | Trainable params | Formation params | Mean broad val NLL | Min broad val NLL | Max broad val NLL | Mean broad bits/char | Mean retention val NLL | Mean retention bits/char | Mean seconds | Mean formation forward passes |",
                    "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
        else:
            lines.extend(
                [
                    "| Config | Seeds | Optimizer | Residual route | Trainable params | Formation params | Mean val NLL | Min val NLL | Max val NLL | Mean bits/char | Mean seconds | Mean formation forward passes |",
                    "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )

    for name in sorted(grouped):
        group = grouped[name]
        val = [float(row["val_nll"]) for row in group]
        bits = [float(row["val_bits_per_char"]) for row in group]
        seconds = [float(row["seconds"]) for row in group]
        trainable = int(group[0]["trainable_parameters"])
        optimizer = str(group[0].get("optimizer", ""))
        residual_optim = str(group[0].get("residual_optim", ""))
        formation_parameters = int(group[0].get("formation_parameters", trainable))
        formation_forward_passes = [float(row.get("formation_forward_passes", 0)) for row in group]
        if has_probe:
            probe_acc = [
                float(row["copy_probe_accuracy"])
                for row in group
                if row.get("copy_probe_accuracy") is not None
            ]
            probe_nll = [
                float(row["copy_probe_nll"])
                for row in group
                if row.get("copy_probe_nll") is not None
            ]
            if has_split_probe:
                seen_acc = [
                    float(row["copy_probe_seen_accuracy"])
                    for row in group
                    if row.get("copy_probe_seen_accuracy") is not None
                ]
                heldout_acc = [
                    float(row["copy_probe_heldout_accuracy"])
                    for row in group
                    if row.get("copy_probe_heldout_accuracy") is not None
                ]
                seen_text = f"{statistics.mean(seen_acc):.6f}" if seen_acc else "n/a"
                seen_min_text = f"{min(seen_acc):.6f}" if seen_acc else "n/a"
                seen_max_text = f"{max(seen_acc):.6f}" if seen_acc else "n/a"
                heldout_text = f"{statistics.mean(heldout_acc):.6f}" if heldout_acc else "n/a"
                heldout_min_text = f"{min(heldout_acc):.6f}" if heldout_acc else "n/a"
                heldout_max_text = f"{max(heldout_acc):.6f}" if heldout_acc else "n/a"
                row_text = (
                    "| "
                    f"{name} | {len(group)} | {trainable} | "
                    f"{statistics.mean(val):.6f} | {statistics.mean(bits):.6f} | "
                    f"{statistics.mean(probe_acc):.6f} | {min(probe_acc):.6f} | {max(probe_acc):.6f} | "
                    f"{seen_text} | {seen_min_text} | {seen_max_text} | "
                    f"{heldout_text} | {heldout_min_text} | {heldout_max_text} | "
                )
                if has_choice_probe:
                    seen_choice_acc = [
                        float(row["copy_probe_seen_choice_accuracy"])
                        for row in group
                        if row.get("copy_probe_seen_choice_accuracy") is not None
                    ]
                    heldout_choice_acc = [
                        float(row["copy_probe_heldout_choice_accuracy"])
                        for row in group
                        if row.get("copy_probe_heldout_choice_accuracy") is not None
                    ]
                    seen_choice_mrr = [
                        float(row["copy_probe_seen_choice_mrr"])
                        for row in group
                        if row.get("copy_probe_seen_choice_mrr") is not None
                    ]
                    heldout_choice_mrr = [
                        float(row["copy_probe_heldout_choice_mrr"])
                        for row in group
                        if row.get("copy_probe_heldout_choice_mrr") is not None
                    ]
                    row_text += f"{statistics.mean(seen_choice_acc):.6f} | " if seen_choice_acc else "n/a | "
                    row_text += f"{statistics.mean(heldout_choice_acc):.6f} | " if heldout_choice_acc else "n/a | "
                    row_text += f"{statistics.mean(seen_choice_mrr):.6f} | " if seen_choice_mrr else "n/a | "
                    row_text += f"{statistics.mean(heldout_choice_mrr):.6f} | " if heldout_choice_mrr else "n/a | "
                row_text += (
                    f"{statistics.mean(probe_nll):.6f} | "
                    f"{statistics.mean(seconds):.4f} |"
                )
                lines.append(row_text)
            else:
                if has_choice_probe:
                    choice_acc = [
                        float(row["copy_probe_seen_choice_accuracy"])
                        for row in group
                        if row.get("copy_probe_seen_choice_accuracy") is not None
                    ]
                    choice_mrr = [
                        float(row["copy_probe_seen_choice_mrr"])
                        for row in group
                        if row.get("copy_probe_seen_choice_mrr") is not None
                    ]
                    choice_acc_text = f"{statistics.mean(choice_acc):.6f}" if choice_acc else "n/a"
                    choice_mrr_text = f"{statistics.mean(choice_mrr):.6f}" if choice_mrr else "n/a"
                    lines.append(
                        "| "
                        f"{name} | {len(group)} | {trainable} | "
                        f"{statistics.mean(val):.6f} | {statistics.mean(bits):.6f} | "
                        f"{statistics.mean(probe_acc):.6f} | {min(probe_acc):.6f} | {max(probe_acc):.6f} | "
                        f"{choice_acc_text} | {choice_mrr_text} | "
                        f"{statistics.mean(probe_nll):.6f} | "
                        f"{statistics.mean(seconds):.4f} |"
                    )
                else:
                    lines.append(
                        "| "
                        f"{name} | {len(group)} | {trainable} | "
                        f"{statistics.mean(val):.6f} | {statistics.mean(bits):.6f} | "
                        f"{statistics.mean(probe_acc):.6f} | {min(probe_acc):.6f} | {max(probe_acc):.6f} | "
                        f"{statistics.mean(probe_nll):.6f} | "
                        f"{statistics.mean(seconds):.4f} |"
                    )
        else:
            row_text = (
                "| "
                f"{name} | {len(group)} | {optimizer} | {residual_optim} | "
                f"{trainable} | {formation_parameters} | "
                f"{statistics.mean(val):.6f} | {min(val):.6f} | {max(val):.6f} | "
                f"{statistics.mean(bits):.6f} | "
            )
            if has_retention_eval:
                retention_nll = [
                    float(row["retention_val_nll"])
                    for row in group
                    if row.get("retention_val_nll") is not None
                ]
                retention_bits = [
                    float(row["retention_val_bits_per_char"])
                    for row in group
                    if row.get("retention_val_bits_per_char") is not None
                ]
                row_text += (
                    f"{statistics.mean(retention_nll):.6f} | " if retention_nll else "n/a | "
                )
                row_text += (
                    f"{statistics.mean(retention_bits):.6f} | " if retention_bits else "n/a | "
                )
            row_text += (
                f"{statistics.mean(seconds):.4f} | "
                f"{statistics.mean(formation_forward_passes):.1f} |"
            )
            lines.append(row_text)

    prior5_rows = [row for row in rows if row.get("prior5_kept_contexts") is not None]
    if prior5_rows:
        lines.extend(
            [
                "",
                "## Sparse Order-5 Diagnostics",
                "",
                "| Config | Seed | Kept contexts | Total bytes | Build seconds | Validation backoff rate | Peak CUDA MiB | Cache status |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in prior5_rows:
            lines.append(
                "| "
                f"{row['comparison_name']} | {row['seed']} | "
                f"{int(row.get('prior5_kept_contexts') or 0)} | "
                f"{int(row.get('prior5_total_bytes') or 0)} | "
                f"{float(row.get('prior5_build_seconds') or 0.0):.4f} | "
                f"{float(row.get('prior5_validation_backoff_rate') or 0.0):.6f} | "
                f"{float(row.get('peak_cuda_memory_mib') or 0.0):.4f} | "
                f"{row.get('prior_cache_status', 'n/a')} |"
            )

    lines.extend(["", "## Raw Runs", ""])
    for row in rows:
        lines.append(
            "- "
            f"{row['comparison_name']} seed={row['seed']} "
            f"optimizer={row.get('optimizer')} "
            f"residual_route={row.get('residual_optim')} "
            f"pos={row.get('pos_encoding')} "
            f"accum={row.get('grad_accum_steps')} "
            f"trainable={row['trainable_parameters']} "
            f"formation_params={row.get('formation_parameters')} "
            f"formation_fwds={row.get('formation_forward_passes')} "
            f"val_nll={row['val_nll']} "
            f"bits={row['val_bits_per_char']} "
            f"mixture_dose={row.get('mixture_dose')} "
            f"retention_val_nll={row.get('retention_val_nll')} "
            f"retention_bits={row.get('retention_val_bits_per_char')} "
            f"copy_acc={row.get('copy_probe_accuracy')} "
            f"seen_acc={row.get('copy_probe_seen_accuracy')} "
            f"heldout_acc={row.get('copy_probe_heldout_accuracy')} "
            f"seen_choice_acc={row.get('copy_probe_seen_choice_accuracy')} "
            f"heldout_choice_acc={row.get('copy_probe_heldout_choice_accuracy')} "
            f"seen_choice_mrr={row.get('copy_probe_seen_choice_mrr')} "
            f"heldout_choice_mrr={row.get('copy_probe_heldout_choice_mrr')} "
            f"peak_cuda_mib={row.get('peak_cuda_memory_mib')} "
            f"seconds={row['seconds']}"
        )
    if error_rows:
        lines.extend(
            [
                "",
                "## Errors",
                "",
                "| Config | Seed | Error type | Seconds | Message |",
                "| --- | ---: | --- | ---: | --- |",
            ]
        )
        for row in error_rows:
            message = str(row.get("error", "")).replace("\n", " ").replace("|", "\\|")
            lines.append(
                "| "
                f"{row.get('comparison_name')} | {row.get('seed')} | "
                f"{row.get('error_type')} | {float(row.get('seconds', 0.0)):.4f} | "
                f"{message} |"
            )
    lines.append("")
    return "\n".join(lines)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cassandra tiny-lab comparison matrix")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--mixture-dose", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--resume-steps-additional", action="store_true")
    parser.add_argument("--title", type=str, default="Stage 5 Replication Summary")
    parser.add_argument("--device", choices=["cuda", "auto", "cpu"], default="cuda")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--train-shard-dir", type=Path, default=None)
    parser.add_argument("--stream-train-eval-chars", type=int, default=200_000)
    parser.add_argument("--retention-eval-corpus", type=Path, default=None)
    parser.add_argument("--prior-cache-dir", type=Path, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--checkpoint-keep", type=int, default=0)
    parser.add_argument("--model-only-out", type=Path, default=None)
    parser.add_argument("--model-only-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--vocab-chars", type=str, default="")
    parser.add_argument("--vocab-chars-file", type=Path, default=None)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--pos-encoding", choices=["learned", "rope"], default="learned")
    parser.add_argument("--activation-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--lr-final-frac", type=float, default=0.1)
    parser.add_argument("--lr-total-steps", type=int, default=0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-10)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--no-muon-nesterov", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-mode", choices=["full", "sampled"], default="sampled")
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="cassandra ")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--count-alpha", type=float, default=0.1)
    parser.add_argument("--ngram-backoff", type=float, default=1.0)
    parser.add_argument("--prior5-min-count", type=int, default=10)
    parser.add_argument("--recency-tau", type=float, default=96.0)
    parser.add_argument("--recency-lambda", type=float, default=0.25)
    parser.add_argument("--curriculum-filter", choices=["off", "prior-loss", "dynamic-reducible"], default="off")
    parser.add_argument("--curriculum-fraction", type=float, default=0.5)
    parser.add_argument("--curriculum-rescore-every", type=int, default=25)
    parser.add_argument("--curriculum-pool-size", type=int, default=4096)
    parser.add_argument("--search-batches", type=int, default=1)
    parser.add_argument("--search-population", type=int, default=8)
    parser.add_argument("--search-sigma", type=float, default=0.02)
    parser.add_argument("--search-lr", type=float, default=0.05)
    parser.add_argument("--coord-step-size", type=float, default=0.02)
    parser.add_argument("--copy-train-marker", type=str, default="answer=")
    parser.add_argument("--no-copy-train-marker", action="store_true")
    parser.add_argument(
        "--copy-sampler",
        choices=[
            "random",
            "answer",
            "mixed",
            "failed",
            "failed_mixed",
            "correction",
            "correction_mixed",
            "retrieval",
            "retrieval_mixed",
            "correction_retrieval_mixed",
            "correction_then_retrieval_mixed",
            "correction_then_retrieval_rehearsal_mixed",
        ],
        default="random",
    )
    parser.add_argument("--copy-sample-fraction", type=float, default=0.5)
    parser.add_argument("--copy-rehearsal-fraction", type=float, default=0.05)
    parser.add_argument("--copy-curriculum-switch-fraction", type=float, default=0.5)
    parser.add_argument("--copy-loss-weight", type=float, default=50.0)
    parser.add_argument("--copy-choice-weight", type=float, default=1.0)
    parser.add_argument("--copy-verify-mode", choices=["identity", "key-answer"], default="identity")
    parser.add_argument("--copy-mine-every", type=int, default=50)
    parser.add_argument(
        "--copy-correction-template",
        choices=["compact", "focus", "prefix", "full"],
        default="compact",
    )
    parser.add_argument(
        "--copy-train-retrieval-template",
        choices=["compact", "focus", "prefix"],
        default="compact",
    )
    parser.add_argument("--copy-probe-marker", type=str, default="")
    parser.add_argument("--copy-probe-key-marker", type=str, default="key=")
    parser.add_argument(
        "--copy-probe-retrieval-template",
        choices=["none", "compact", "focus", "prefix"],
        default="none",
    )
    parser.add_argument(
        "--copy-probe-retrieval-source",
        choices=["target", "memory"],
        default="target",
    )
    parser.add_argument(
        "--copy-probe-retrieval-corrupt",
        choices=["none", "wrong-answer"],
        default="none",
    )
    parser.add_argument(
        "--copy-probe-memory-scope",
        choices=["train", "all"],
        default="train",
    )
    parser.add_argument("--copy-probe-holdout-keys", nargs="*", default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["random_full", "count_prior_adapter_r4", "count_prior_head"],
        choices=[
            "random_full",
            "stage59_proxy_random_full",
            "random_full_copyw",
            "random_full_copyw_choice",
            "random_full_copyfail",
            "random_full_copyfailmix",
            "random_full_copycorr",
            "random_full_copycorrmix",
            "random_full_retmix",
            "random_full_corrretmix",
            "random_full_corrthenret",
            "random_full_copys",
            "random_full_copymix",
            "random_full_copymix_choice",
            "count_prior_adapter_r4",
            "count_prior_head",
            "count_prior_lora_r1",
            "count_prior_lora_r1_floor",
            "count_prior_lora_r2",
            "count_prior_lora_r2_floor",
            "count_prior_lora_r4",
            "count_prior_lora_r4_floor",
            "count_prior_lora_r2_es",
            "count_prior_lora_r1_coord",
            "count_prior_tri_lora_r2",
            "count_prior_ng1_lora_r2",
            "count_prior_ng2_lora_r2",
            "count_prior_ng2_lora_r2_floor",
            "count_prior_ng2_recency_lora_r2",
            "count_prior_ng2_lora_r2_filter",
            "count_prior_ng2_lora_r2_filter_f025",
            "count_prior_ng2_lora_r2_filter_f050",
            "count_prior_ng2_lora_r2_filter_f100",
            "count_prior_ng2_lora_r2_dynfilter_f025",
            "count_prior_ng2_lora_r2_dynfilter_f050",
            "count_prior_ng3_lora_r2",
            "count_prior_ng3_lora_r2_floor",
            "count_prior_ng4_lora_r2",
            "count_prior_ng4_all",
            "count_prior_ng4_lora_r2_floor",
            "count_prior_ng5_lora_r2",
            "count_prior_ng5_lora_r2_floor",
            "count_prior_lora_r1_copyw",
            "count_prior_lora_r2_copyw",
            "count_prior_lora_r2_copyw_floor",
            "count_prior_lora_r4_copyw",
            "count_prior_lora_r8_copyw",
            "count_prior_lora_r16_copyw",
            "count_prior_all_copyw",
            "count_prior_lora_r2_copyw_choice",
            "count_prior_lora_r2_copyfail",
            "count_prior_lora_r2_copyfailmix",
            "count_prior_lora_r2_copycorr",
            "count_prior_lora_r2_copycorrmix",
            "count_prior_lora_r2_retmix",
            "count_prior_lora_r2_corrretmix",
            "count_prior_lora_r2_corrthenret",
            "count_prior_lora_r2_corrthenret_rehearsal",
            "count_prior_lora_r4_corrretmix",
            "count_prior_lora_r8_corrretmix",
            "count_prior_lora_r4_corrthenret",
            "count_prior_lora_r8_corrthenret",
            "count_prior_lora_r2_copys",
            "count_prior_lora_r2_copymix",
            "count_prior_lora_r2_copymix_choice",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_interval < 0:
        raise ValueError("--eval-interval must be non-negative")
    if args.mixture_dose is not None and not 0.0 <= args.mixture_dose <= 1.0:
        raise ValueError("--mixture-dose must be in [0, 1]")
    if "stage59_proxy_random_full" in args.configs:
        if args.configs != ["stage59_proxy_random_full"]:
            raise ValueError("stage59_proxy_random_full must run as the only named config")
        for key, value in stage59_proxy_recipe_args(args.steps).items():
            setattr(args, key, value)
    if args.no_copy_train_marker:
        args.copy_train_marker = ""
    total_runs = len(args.seeds) * len(args.configs)
    if args.resume_from is not None and total_runs != 1:
        raise ValueError("--resume-from is only supported for a single config and seed")
    if args.model_only_out is not None and total_runs != 1:
        raise ValueError("--model-only-out is only supported for a single config and seed")
    print(
        "[matrix] "
        f"runs={total_runs} corpus={args.corpus} device={args.device} "
        f"steps={args.steps} block={args.block_size} batch={args.batch_size} "
        f"accum={args.grad_accum_steps} model=L{args.n_layer} H{args.n_head} D{args.n_embd} "
        f"pos={args.pos_encoding} checkpoint={args.activation_checkpoint} "
        f"precision={args.precision} optimizer={args.optimizer} lr_schedule={args.lr_schedule} "
        f"train_shards={args.train_shard_dir or 'off'} "
        f"retention_eval={args.retention_eval_corpus or 'off'} "
        f"mixture_dose={args.mixture_dose} append={args.append} "
        f"checkpoint_dir={args.checkpoint_dir or 'off'} "
        f"configs={','.join(args.configs)} seeds={','.join(str(seed) for seed in args.seeds)}",
        flush=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir is not None:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    if args.append and args.out.exists():
        for line_number, line in enumerate(args.out.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid existing JSONL row {line_number} in {args.out}: {exc}") from exc
    run_index = 0
    for seed in args.seeds:
        for config_name in args.configs:
            run_index += 1
            run_args = config_args(
                config_name,
                args.corpus,
                args.device,
                seed,
                args.steps,
                args.eval_batches,
                args.block_size,
                args.eval_mode,
                args.count_alpha,
                args.ngram_backoff,
                args.recency_tau,
                args.recency_lambda,
                args.curriculum_filter,
                args.curriculum_fraction,
                args.curriculum_rescore_every,
                args.curriculum_pool_size,
                args.search_batches,
                args.search_population,
                args.search_sigma,
                args.search_lr,
                args.coord_step_size,
                args.copy_probe_marker,
                args.copy_probe_retrieval_template,
                args.copy_probe_retrieval_source,
                args.copy_probe_retrieval_corrupt,
                args.copy_probe_memory_scope,
                args.copy_probe_holdout_keys,
                args.copy_train_marker,
                args.copy_sampler,
                args.copy_sample_fraction,
                args.copy_rehearsal_fraction,
                args.copy_curriculum_switch_fraction,
                args.copy_loss_weight,
                args.copy_choice_weight,
                args.copy_verify_mode,
                args.copy_mine_every,
                args.copy_correction_template,
                args.copy_train_retrieval_template,
            )
            run_args.batch_size = args.batch_size
            run_args.grad_accum_steps = args.grad_accum_steps
            run_args.train_shard_dir = args.train_shard_dir
            run_args.stream_train_eval_chars = args.stream_train_eval_chars
            run_args.retention_eval_corpus = args.retention_eval_corpus
            run_args.prior_cache_dir = args.prior_cache_dir
            run_args.val_fraction = args.val_fraction
            run_args.prior5_min_count = args.prior5_min_count
            run_args.checkpoint_dir = args.checkpoint_dir
            run_args.checkpoint_every = args.checkpoint_every
            run_args.checkpoint_keep = args.checkpoint_keep
            run_args.resume_from = args.resume_from
            run_args.resume_steps_additional = args.resume_steps_additional
            run_args.model_only_out = args.model_only_out
            run_args.model_only_dtype = args.model_only_dtype
            run_args.vocab_chars = args.vocab_chars
            run_args.vocab_chars_file = args.vocab_chars_file
            run_args.copy_probe_key_marker = args.copy_probe_key_marker
            run_args.n_layer = args.n_layer
            run_args.n_head = args.n_head
            run_args.n_embd = args.n_embd
            run_args.pos_encoding = args.pos_encoding
            run_args.activation_checkpoint = args.activation_checkpoint
            run_args.dropout = args.dropout
            run_args.precision = args.precision
            run_args.optimizer = args.optimizer
            run_args.lr = args.lr
            run_args.lr_schedule = args.lr_schedule
            run_args.lr_final_frac = args.lr_final_frac
            run_args.lr_total_steps = args.lr_total_steps
            run_args.adam_beta1 = args.adam_beta1
            run_args.adam_beta2 = args.adam_beta2
            run_args.adam_eps = args.adam_eps
            run_args.muon_lr = args.muon_lr
            run_args.muon_momentum = args.muon_momentum
            run_args.muon_ns_steps = args.muon_ns_steps
            run_args.no_muon_nesterov = args.no_muon_nesterov
            run_args.weight_decay = args.weight_decay
            run_args.eval_interval = args.eval_interval if args.eval_interval > 0 else args.steps
            run_args.log_every = args.log_every
            run_args.prompt = args.prompt
            run_args.max_new_tokens = args.max_new_tokens
            run_args.temperature = args.temperature
            if args.checkpoint_dir is not None:
                run_args.out = args.checkpoint_dir / f"{args.out.stem}_{config_name}_seed{seed}.pt"
            run_start = time.perf_counter()
            print(
                "[matrix] "
                f"start {run_index}/{total_runs} config={config_name} seed={seed}",
                flush=True,
            )
            try:
                report = train(run_args)
                report["comparison_name"] = config_name
                report["seed"] = seed
                report["mixture_dose"] = args.mixture_dose
                rows.append(report)
                print(
                    "[matrix] "
                    f"done {run_index}/{total_runs} config={config_name} seed={seed} "
                    f"val_nll={report['val_nll']} "
                    f"bits={report['val_bits_per_char']} "
                    f"trainable={report['trainable_parameters']} "
                    f"seconds={time.perf_counter() - run_start:.1f}",
                    flush=True,
                )
            except Exception as exc:
                error_report = {
                    "status": "error",
                    "comparison_name": config_name,
                    "seed": seed,
                    "mixture_dose": args.mixture_dose,
                    "corpus": str(args.corpus),
                    "device": args.device,
                    "steps": args.steps,
                    "block_size": args.block_size,
                    "batch_size": args.batch_size,
                    "grad_accum_steps": args.grad_accum_steps,
                    "n_layer": args.n_layer,
                    "n_head": args.n_head,
                    "n_embd": args.n_embd,
                    "pos_encoding": args.pos_encoding,
                    "activation_checkpoint": args.activation_checkpoint,
                    "precision": args.precision,
                    "optimizer": args.optimizer,
                    "lr_schedule": args.lr_schedule,
                    "eval_mode": args.eval_mode,
                    "eval_batches": args.eval_batches,
                    "seconds": round(time.perf_counter() - run_start, 4),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                rows.append(error_report)
                print(
                    "[matrix] "
                    f"error {run_index}/{total_runs} config={config_name} seed={seed} "
                    f"type={type(exc).__name__} message={exc}",
                    flush=True,
                )
            args.out.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            args.summary.write_text(summarize(rows, args.title), encoding="utf-8")
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    args.summary.write_text(summarize(rows, args.title), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
