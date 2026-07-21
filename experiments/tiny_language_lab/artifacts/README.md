# Tiny Language Lab Artifacts

This folder is for promoted local artifacts that are useful after a stage
succeeds but are too large or too tool-specific for the durable Markdown record.

Phase 4 closeout should place the NVIDIA Nsight Deep Learning Designer ONNX
export under:

```text
experiments/tiny_language_lab/artifacts/phase4/nsight_dld/
```

Keep training evidence in `runs/` and durable interpretation in `RESULTS.md`.
Large generated files in this tree (`.pt`, `.sha256`, `.onnx`, `.onnx.data`,
and `.manifest.json`) are ignored by git. Track only small README or manifest
notes when they add useful provenance.
