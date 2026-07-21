# Stage 55 Flagship Checkpoint Export

Artifact: `stage55_flagship_200m_b50000_seed7_random_full_seed7_step050000.pt`

Current durable source checkpoint:

`C:\cassandra_runs\stage55_flagship_checkpoints\stage55_flagship_200m_b50000_seed7_random_full_seed7_step050000.pt`

Original recovery source checkpoint:

`C:\Users\senso\AppData\Local\Temp\cassandra_runs\stage55_flagship_checkpoints\stage55_flagship_200m_b50000_seed7_random_full_seed7_step050000.pt`

Promoted checkpoint:

`experiments\tiny_language_lab\artifacts\phase4\checkpoints\stage55_flagship_200m_b50000_seed7\stage55_flagship_200m_b50000_seed7_random_full_seed7_step050000.pt`

Recorded: 2026-07-06

Bytes: `1619478113`

SHA256:

`AB4B9CE99462241F00728D65111C423A6A8ED328D8B7ABA8E60B36D164E0B8D7`

Verification:

- Source and promoted checkpoint byte counts match.
- Source and promoted checkpoint SHA256 hashes match.
- CPU `torch.load(..., map_location="cpu", weights_only=False)` succeeded from the promoted copy.
- Checkpoint `step` is `50000`.
- Final JSONL reports `formation_steps=50000`.
- Checkpoint `formation_forward_passes` is `100000`.
- Checkpoint args report `optimizer=muon` and `seed=7`.
- `model_state` is present.
- `optimizer_state` is present.
- RNG/generator state is present.
- `loss_curve` contains 9 points.

This is the resumable training checkpoint. The separate Nsight Deep Learning
Designer ONNX export is inference-only and does not preserve optimizer state.
