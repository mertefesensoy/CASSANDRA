# Nsight DL Designer Workflow

Date: 2026-07-03

This workflow prepares Cassandra checkpoints for NVIDIA Nsight Deep Learning
Designer without changing the training path. DL Designer is for inference graph
inspection, ONNX or XDL editing, TensorRT or ONNX Runtime profiling, and
performance review. It is not the tool that resumes Cassandra training.

## What To Use It For

- Open a fixed inference copy of a Cassandra checkpoint.
- Inspect the transformer graph and tensor shapes.
- Run ONNX Runtime or TensorRT-oriented profiling from the DL Designer GUI.
- Compare inference graph changes separately from training evidence.

Do not use the exported ONNX file as training evidence. The training checkpoint
remains the `.pt` file with model, optimizer, generator, and loss-curve state.

## Export The Stage 55 Checkpoint

From the repository root:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\export_stage55_nsight_dld.ps1 `
  -Checkpoint C:\cassandra_runs\stage55_flagship_checkpoints\stage55_flagship_200m_b50000_seed7_random_full_seed7_step050000.pt `
  -OutDir .\experiments\tiny_language_lab\artifacts\phase4\nsight_dld `
  -Name stage55_seed7_final_success_b1_s256 `
  -SeqLen 256 `
  -BatchSize 1 `
  -Device cpu `
  -OpenFolder
```

The exporter writes:

- `experiments/tiny_language_lab/artifacts/phase4/nsight_dld/<name>.onnx`
- `experiments/tiny_language_lab/artifacts/phase4/nsight_dld/<name>.manifest.json`

ONNX exports and checkpoints are large local artifacts and are ignored by git.

## Phase 4 Success Artifact

At the end of a successful Phase 4 flagship run, create the DL Designer export
again from the final successful checkpoint and place it in the Phase 4 artifact
folder:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\export_stage55_nsight_dld.ps1 `
  -Checkpoint C:\cassandra_runs\stage55_flagship_checkpoints\stage55_flagship_200m_b50000_seed7_random_full_seed7_step050000.pt `
  -OutDir .\experiments\tiny_language_lab\artifacts\phase4\nsight_dld `
  -Name stage55_seed7_final_success_b1_s256 `
  -SeqLen 256 `
  -BatchSize 1 `
  -Device cpu
```

Record both generated paths in `experiments/tiny_language_lab/RESULTS.md`.
Create this only after Phase 4 success. For failed or partial runs, do not
promote a DL Designer export into `artifacts/`.

## Open In DL Designer

1. Launch NVIDIA Nsight Deep Learning Designer.
2. Open the exported `.onnx` file.
3. Confirm the input is `input_ids` with shape `[1, 256]`.
4. Confirm the output is `logits` with shape `[1, 256, 33]`.
5. Use the DL Designer profiling tools only as inference diagnostics.

For Stage 55, this ONNX graph is the random-init 200M character model at the
exported checkpoint step. It does not include optimizer state.

## Optional Checks

If the Python environment has `onnx` installed, add `-CheckOnnx`:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\export_stage55_nsight_dld.ps1 `
  -CheckOnnx
```

Use `-Device cuda` only if CPU export is too slow and the GPU is idle. Do not run
export on the GPU while a Cassandra training run is active.

## Training Continuation Remains Separate

Resume Cassandra training from the `.pt` checkpoint, not from ONNX:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\experiments\tiny_language_lab\run_phase4_visible.ps1 `
  -Mode stage55-flagship-cell `
  -Budget 20000 `
  -Seed 7 `
  -ResumeFrom C:\cassandra_runs\stage55_flagship_checkpoints\stage55_flagship_200m_b50000_seed7_random_full_seed7_step050000.pt
```

This keeps the Stage 55 evidence clean because the optimizer and generator state
come from the training checkpoint.
