# Publishing the Stage 61 flagship to Hugging Face

Everything that can be prepared without your account is prepared. The
upload-ready package is staged at `C:\cassandra_runs\stage61_release\`. This
guide is the remaining sequence; the steps marked **(you)** are yours because
they require your account and token, which I never handle.

## What is already prepared

Built and verified by `experiments/tiny_language_lab/make_stage61_release.py`
(round-trip: the fp16 weights load and generate correctly). Staged in
`C:\cassandra_runs\stage61_release\`:

- `stage61_pure_broad_200m_text8_fp16.pt` (~406 MB) and its `.sha256`
- `config.json` (architecture, training provenance, recommended inference)
- `codec.json` (the 33-character alphabet, index = token id)
- `release_manifest.json` (SHA-256s, round-trip record)
- `README.md` (the model card), `inference_example.py`, `LICENSE`, `NOTICE`

Publish-worthiness (Cassandra ADR 0018 D5): text8 TEST `1.336059` beats the
`1.357318` bar (PASS), instrumentation complete (PASS), and your sample review
is given, with top-p 0.9 chosen as the recommended default (PASS).

## Steps to publish (you)

1. **(you)** Create a Hugging Face account at <https://huggingface.co> if you do
   not have one.
2. **(you)** Create a WRITE access token at
   <https://huggingface.co/settings/tokens>.
3. Install the client (one time):
   ```powershell
   pip install -U huggingface_hub
   ```
4. **(you)** Log in with your token (it is entered by you, stored locally, and
   never seen by me):
   ```powershell
   huggingface-cli login
   ```
5. **(you)** Create the public model repo (or do it in the web UI):
   ```powershell
   huggingface-cli repo create cassandra-200m-text8 --type model
   ```
6. **(you)** Upload the staged folder to your repo (replace `<username>`):
   ```powershell
   huggingface-cli upload <username>/cassandra-200m-text8 `
     C:\cassandra_runs\stage61_release . --repo-type model
   ```
   (Newer CLI: `hf upload <username>/cassandra-200m-text8 C:\cassandra_runs\stage61_release .`)
7. **(you)** Open the model page, confirm the card renders and the files are
   present, then download the weights once and run `inference_example.py` to
   confirm the public copy works end to end.

## Optional: private fp32 training archive (disk relief)

The public weights are fp16 (inference). If you also want the full fp32
resume-capable checkpoint archived (so the local 1.6 GB copy can later be
deleted), export and upload it to a PRIVATE repo:

```powershell
python .\experiments\tiny_language_lab\export_model_only_checkpoint.py `
  --checkpoint C:\cassandra_runs\stage61_pure_broad_200m_checkpoints\stage61_pure_broad_200m_seed7_random_full_seed7.pt `
  --out C:\cassandra_runs\stage61_release_fp32\stage61_pure_broad_200m_text8_fp32.pt --dtype fp32
huggingface-cli repo create cassandra-200m-text8-archive --type model --private
huggingface-cli upload <username>/cassandra-200m-text8-archive C:\cassandra_runs\stage61_release_fp32 . --repo-type model
```

Verify the upload (checksums) BEFORE deleting any local checkpoint.

## Two informed choices for you

- **Weights format.** The package ships a PyTorch `.pt` (pickle). It loads with
  the shipped code and is fully functional, but Hugging Face flags pickle files
  and prefers `safetensors`. If you want the cleaner, warning-free format, say
  so and I will add a `safetensors` weights file plus a tiny loader before you
  upload. The `.pt` is publishable as-is.
- **Repo name.** `cassandra-200m-text8` is a suggestion; rename freely. The
  model card links back to the GitHub code repo, so keep that link accurate if
  you fork or move it.

Nothing here is irreversible until step 6. I do not run the upload; that is your
button to press.
