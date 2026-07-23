# Publishing the Stage 61 flagship to Hugging Face

Everything that can be prepared without your account is prepared. The
upload-ready package is staged at `C:\cassandra_runs\stage61_release\`. This
guide is the remaining sequence; the steps marked **(you)** are yours because
they require your account and token, which I never handle.

## What is already prepared

Built and verified by `experiments/tiny_language_lab/make_stage61_release.py`
(round-trip: the fp16 weights load pickle-free and generate correctly). Staged
in `C:\cassandra_runs\stage61_release\`:

- `stage61_pure_broad_200m_text8_fp16.safetensors` (~406 MB) and its `.sha256`
  (safetensors, no pickle: Hugging Face loads it without a security warning)
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
   The command is `hf` (huggingface_hub 1.x; the old `huggingface-cli` name was
   removed). pip installs `hf.exe` into a Scripts folder that may not be on
   PATH, so add it to the current PowerShell session first:
   ```powershell
   $env:PATH = "$env:PATH;C:\Users\senso\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts"
   hf version   # should print version=1.24.0 or newer
   ```
   (If a fresh terminal ever cannot find `hf`, run that `$env:PATH` line again,
   or call the full path `& "C:\Users\senso\AppData\Local\...\Python313\Scripts\hf.exe"`.)
4. **(you)** Log in with your token (you enter it; it is stored locally and
   never seen by me):
   ```powershell
   hf auth login
   ```
5. **(you)** Upload the staged folder to your repo (replace `<username>`). This
   creates the public repo automatically if it does not exist:
   ```powershell
   hf upload <username>/cassandra-200m-text8 C:\cassandra_runs\stage61_release . --repo-type model
   ```
   (To create it explicitly first, or make it private: `hf repo create cassandra-200m-text8 --repo-type model`.)
6. **(you)** Open the model page, confirm the card renders and the files are
   present, then download the weights once and run `inference_example.py` to
   confirm the public copy works end to end.

## Optional: private fp32 training archive (disk relief)

The public weights are fp16 (inference-only). If you also want the full fp32
**resume-capable** checkpoint archived (so the local ~1.6 GB copy can later be
deleted without losing the ability to continue training), upload the original
checkpoint itself, which still carries optimizer and generator state, to a
PRIVATE repo:

```powershell
hf repo create cassandra-200m-text8-archive --repo-type model --private
hf upload <username>/cassandra-200m-text8-archive `
  C:\cassandra_runs\stage61_pure_broad_200m_checkpoints\stage61_pure_broad_200m_seed7_random_full_seed7.pt `
  stage61_pure_broad_200m_seed7_fp32_resume.pt --repo-type model
```

Its SHA-256 is `4e5c0c0540b7b019f7fb6a53636a8963cffae145e6182e2e41aa463b2f8bacd5`.
Verify the upload (checksums) BEFORE deleting any local checkpoint.

## One choice for you

- **Repo name.** `cassandra-200m-text8` is a suggestion; rename freely. The
  model card links back to the GitHub code repo, so keep that link accurate if
  you fork or move it.

Weights ship as `safetensors` (no pickle), so Hugging Face loads them without a
security warning and the file cannot execute code on load.

Nothing here is irreversible until step 6. I do not run the upload; that is your
button to press.
