# GPU Transition Validity Audit

Date: 2026-06-23

Status: CUDA is now the default measurement device for real comparison matrices.
Historical CPU results remain evidence, but CPU wall-clock is no longer the
planning baseline for future work.

## Environment

- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- PyTorch: `2.12.1+cu126`
- PyTorch CUDA runtime: `12.6`
- Driver-reported CUDA capability from `nvidia-smi`: `13.1`
- Runner default after the switch: `cassandra_compare.py --device cuda`

Package caveat: the CUDA PyTorch reinstall reported an `opencv-python` /
`numpy` version mismatch because `numpy` is now `2.4.4`. The tiny language lab
does not use OpenCV, so this does not affect the language-model evidence. It can
matter later for image or video tooling.

## Audit Checks

CUDA device probe:

```powershell
python .\experiments\tiny_language_lab\cassandra_compare.py --corpus .\experiments\tiny_language_lab\corpus\tiny_seed.txt --steps 1 --eval-mode sampled --eval-batches 1 --seeds 7 --configs random_full --out .\experiments\tiny_language_lab\runs\gpu_device_probe.jsonl --summary .\experiments\tiny_language_lab\runs\gpu_device_probe.md --title "GPU Device Probe"
```

Result: `device=cuda`, `device_name=NVIDIA GeForce RTX 4070 Laptop GPU`,
`random_full` val NLL `3.214120`.

CPU versus CUDA RNG diagnostic:

```text
cpu  [15, 92, 21, 86, 83, 47, 87, 79, 88, 61, 58, 31]
cuda [37, 28, 45, 93, 25, 93, 16, 52, 46, 31, 38, 53]
```

The same seed does not produce the same sample stream across CPU and CUDA when
using `torch.Generator(device=device)`. This affects sampled training windows,
sampled validation windows, dynamic-curriculum pools, and generated text.

Zero-step full-eval sanity check, no sampled windows:

| Device | `random_full` val NLL | `count_prior_lora_r2` val NLL |
| --- | ---: | ---: |
| CPU | 3.377009 | 2.502358 |
| CUDA | 3.377010 | 2.502358 |

With sampled windows removed, CPU and CUDA agree to displayed precision on the
tiny probe. The device switch does not by itself change the model definition or
the exact full-evaluation loss in a meaningful way.

Five-step sampled smoke:

| Device | `random_full` val NLL | `count_prior_lora_r2` val NLL |
| --- | ---: | ---: |
| CPU | 3.154839 | 2.507561 |
| CUDA | 2.912663 | 2.516801 |

The ordering survives on this tiny diagnostic, but the values differ because
sample streams differ. Treat same-seed CPU rows and CUDA rows as separate
measurement families unless the sampler is changed to use device-independent
CPU RNG for start indices.

## What The Switch Changes

Wall-clock conclusions are the main casualty. Any old statement that a method is
faster or slower in seconds is a CPU-era statement unless it has been rerun under
CUDA.

Exact numeric sampled losses are not bitwise portable across CPU and CUDA. The
old rows remain valid within their original stage because arms shared one device,
one corpus, one budget, and the same seed policy. They should not be merged with
new CUDA rows as if they were one continuous table.

Step-budget conclusions are much less affected. If a result was decided by a
large validation-NLL gap, copy-probe accuracy gap, coverage diagnostic, or
steps-to-target comparison inside one matched run family, the GPU switch does
not erase the result. It only says a CUDA rerun is a new measurement, not a
literal continuation of the CPU measurement.

## Killed Hypotheses And Results

| Result | Primary old evidence | GPU impact | Audit call |
| --- | --- | --- | --- |
| Stage 32 / H009b cross-domain sweet spot | Order 4 stayed best at 100, 200, and 500 steps on the implemented split. | Low. This is a corpus, coverage, and step-NLL result. | Still locally killed for that split. GPU mainly enables harsher or larger follow-up gates. |
| Stage 33 / H010 fixed prior-loss filter | No filtered arm reached the uniform 200-step target earlier; pure high-loss was clearly worse. | Low to medium. The `f=0.25` NLL differences are tiny, but the steps-to-target kill does not rely on wall-clock. | Still killed as local CPU evidence. Rerun on CUDA only if reopening data selection. |
| Stage 34 / H011 dynamic reducible filter | Dynamic arms did not reach target; `f=0.50` was clearly worse, `f=0.25` was slightly worse; CPU wall-clock overhead was recorded. | Medium. NLL direction likely stands, but the wall-clock clause is stale. | Keep the step/NLL kill. Do not use the old seconds table as GPU-era evidence. Rerun on CUDA before making a final GPU wall-clock claim. |
| ADR 0004 data-side selection retirement | Consolidates Stages 11, 12, 33, and 34. | Medium. It should lean on behavior and NLL failures, not CPU seconds. | Direction still justified. A CUDA H011 rerun is the cleanest check if the branch is challenged. |
| Stage 35 / H012 frozen recency base | Recency was worse than order-2 count-only by about `+0.062` to `+0.086` NLL; order 3 was far better. | Low. The margin is large and the failure is model-side, not CPU-specific. | Still killed. Only wall-clock optimization claims need CUDA remeasurement. |
| Stages 11 and 12 hard replay/correction sampling | Data-selection variants did not fix the copy/retrieval failures. | Low. These are behavior and selection results. | Still local negatives. GPU can rerun faster, but the old kills are not invalidated. |
| Stage 22 / H4 compact external memory | Correct memory hint failed to carry absent held-out mappings. | Low. This is an interface and behavior failure. | Still killed for the compact text-prefix interface. |
| Stages 24 and 25 / H005 and H006 cheap-recipe surfaces | Count-prior LoRA wins early in optimizer steps and loses later as full training catches up. | Medium if interpreted as seconds, low if interpreted as steps. | Preserve as early-step evidence. Avoid CPU-second claims unless rerun on CUDA. |
| ADR 0002 bounded early-compute accelerator | The frozen prior is an early-step head start, not an asymptotic replacement. | Low to medium. | Still valid as a step-budget law. GPU invites larger time-to-target reruns. |
| Stage 28 and ADR 0003 source/prior order law | Matching source order and prior order helps, with coverage-dependent sparsity penalties. | Low. | Still valid. GPU expands feasible order/capacity sweeps. |

## Policy Going Forward

Use CUDA for real matrices and keep the `device` and `device_name` fields in
every durable result. Use CPU only for diagnostics, fallbacks, or explicit
cross-device checks.

Do not overwrite historical CPU rows with CUDA reruns. Record CUDA reruns as new
stage rows, appendices, or audit checks with their own file names.

Before a past negative result is used as a hard roadmap stop in the GPU era,
rerun it on CUDA if either condition holds:

- the decision margin was smaller than about `0.003` validation NLL,
- the claim depends on wall-clock seconds rather than optimizer steps or task
  behavior.

When cross-device comparability matters, prefer `--eval-mode full` or add a
device-independent RNG path for sampled start indices. The current sampler is
deterministic within a device, but not identical across CPU and CUDA.

## Bottom Line

The GPU switch does not resurrect the killed hypotheses. It changes the
measurement regime. Strong NLL and behavior kills stay killed. CPU wall-clock
claims become historical notes. Borderline sampled comparisons, especially H011
`f=0.25` and any time-to-target wording, should be rerun on CUDA before they are
used as firm GPU-era decisions.
