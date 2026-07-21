param(
    [ValidateSet("smoke-fast", "smoke-prior", "bridge100", "bridge500", "modern-smoke", "modern500", "modern1000", "stream-smoke", "bpe-smoke", "bpe500")]
    [string]$Mode = "smoke-fast",
    [switch]$KeepOpen
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$RunDir = Join-Path $ScriptDir "runs"
$Corpus = Join-Path $ScriptDir "corpus\tinystories_char_seed.txt"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $RunDir ("phase2_{0}_{1}.log" -f $Mode, $Timestamp)

if (-not (Test-Path -LiteralPath $Corpus)) {
    throw "Missing TinyStories corpus at $Corpus"
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
Set-Location $RepoRoot

$Common = @(
    ".\experiments\tiny_language_lab\cassandra_compare.py",
    "--corpus", ".\experiments\tiny_language_lab\corpus\tinystories_char_seed.txt",
    "--device", "cuda",
    "--block-size", "128",
    "--batch-size", "8",
    "--n-layer", "4",
    "--n-head", "4",
    "--n-embd", "256",
    "--eval-mode", "sampled",
    "--prompt", "once upon a time ",
    "--max-new-tokens", "240"
)

switch ($Mode) {
    "smoke-fast" {
        $RunArgs = $Common + @(
            "--steps", "5",
            "--eval-batches", "2",
            "--eval-interval", "1",
            "--log-every", "1",
            "--seeds", "7",
            "--configs", "random_full",
            "--out", ".\experiments\tiny_language_lab\runs\phase2_tinystories_smoke_fast.jsonl",
            "--summary", ".\experiments\tiny_language_lab\runs\phase2_tinystories_smoke_fast.md",
            "--title", "Phase 2 TinyStories Fast Smoke"
        )
    }
    "smoke-prior" {
        $RunArgs = $Common + @(
            "--steps", "5",
            "--eval-batches", "2",
            "--eval-interval", "1",
            "--log-every", "1",
            "--seeds", "7",
            "--configs", "count_prior_ng3_lora_r2",
            "--out", ".\experiments\tiny_language_lab\runs\phase2_tinystories_smoke_prior.jsonl",
            "--summary", ".\experiments\tiny_language_lab\runs\phase2_tinystories_smoke_prior.md",
            "--title", "Phase 2 TinyStories Prior Smoke"
        )
    }
    "bridge100" {
        $RunArgs = $Common + @(
            "--steps", "100",
            "--eval-batches", "16",
            "--eval-interval", "25",
            "--log-every", "25",
            "--seeds", "7", "11", "19",
            "--configs", "random_full", "count_prior_ng3_lora_r2", "count_prior_ng4_lora_r2",
            "--out", ".\experiments\tiny_language_lab\runs\phase2_tinystories_bridge_b100.jsonl",
            "--summary", ".\experiments\tiny_language_lab\runs\phase2_tinystories_bridge_b100.md",
            "--title", "Phase 2 TinyStories Bridge 100 steps"
        )
    }
    "bridge500" {
        $RunArgs = $Common + @(
            "--steps", "500",
            "--eval-batches", "16",
            "--eval-interval", "100",
            "--log-every", "100",
            "--seeds", "7", "11", "19",
            "--configs", "random_full", "count_prior_ng3_lora_r2", "count_prior_ng4_lora_r2",
            "--out", ".\experiments\tiny_language_lab\runs\phase2_tinystories_bridge_b500.jsonl",
            "--summary", ".\experiments\tiny_language_lab\runs\phase2_tinystories_bridge_b500.md",
            "--title", "Phase 2 TinyStories Bridge 500 steps"
        )
    }
    "modern-smoke" {
        $RunArgs = $Common + @(
            "--steps", "20",
            "--eval-batches", "2",
            "--eval-interval", "5",
            "--log-every", "5",
            "--grad-accum-steps", "2",
            "--pos-encoding", "rope",
            "--activation-checkpoint",
            "--optimizer", "muon",
            "--muon-lr", "0.01",
            "--seeds", "7",
            "--configs", "random_full", "count_prior_ng4_lora_r2",
            "--out", ".\experiments\tiny_language_lab\runs\phase2_tinystories_modern_smoke.jsonl",
            "--summary", ".\experiments\tiny_language_lab\runs\phase2_tinystories_modern_smoke.md",
            "--title", "Phase 2 TinyStories Modern Smoke"
        )
    }
    "modern500" {
        $RunArgs = $Common + @(
            "--steps", "500",
            "--eval-batches", "16",
            "--eval-interval", "100",
            "--log-every", "100",
            "--grad-accum-steps", "2",
            "--pos-encoding", "rope",
            "--activation-checkpoint",
            "--optimizer", "muon",
            "--muon-lr", "0.01",
            "--seeds", "7", "11", "19",
            "--configs", "random_full", "count_prior_ng4_lora_r2",
            "--out", ".\experiments\tiny_language_lab\runs\phase2_tinystories_modern_b500.jsonl",
            "--summary", ".\experiments\tiny_language_lab\runs\phase2_tinystories_modern_b500.md",
            "--title", "Phase 2 TinyStories Modern Bridge 500 steps"
        )
    }
    "modern1000" {
        $RunArgs = $Common + @(
            "--steps", "1000",
            "--eval-batches", "16",
            "--eval-interval", "250",
            "--log-every", "250",
            "--grad-accum-steps", "2",
            "--pos-encoding", "rope",
            "--activation-checkpoint",
            "--optimizer", "muon",
            "--muon-lr", "0.01",
            "--seeds", "7", "11", "19",
            "--configs", "random_full", "count_prior_ng4_lora_r2",
            "--out", ".\experiments\tiny_language_lab\runs\phase2_tinystories_modern_b1000.jsonl",
            "--summary", ".\experiments\tiny_language_lab\runs\phase2_tinystories_modern_b1000.md",
            "--title", "Phase 2 TinyStories Modern Crossover 1000 steps"
        )
    }
    "stream-smoke" {
        $RunArgs = $Common + @(
            "--steps", "20",
            "--eval-batches", "2",
            "--eval-interval", "5",
            "--log-every", "5",
            "--grad-accum-steps", "2",
            "--train-shard-dir", ".\experiments\tiny_language_lab\corpus\tinystories_char_shards",
            "--stream-train-eval-chars", "200000",
            "--pos-encoding", "rope",
            "--activation-checkpoint",
            "--optimizer", "muon",
            "--muon-lr", "0.01",
            "--seeds", "7",
            "--configs", "random_full",
            "--out", ".\experiments\tiny_language_lab\runs\phase2_tinystories_stream_smoke.jsonl",
            "--summary", ".\experiments\tiny_language_lab\runs\phase2_tinystories_stream_smoke.md",
            "--title", "Phase 2 TinyStories Stream Smoke"
        )
    }
    "bpe-smoke" {
        $RunArgs = @(
            ".\experiments\tiny_language_lab\cassandra_compare.py",
            "--corpus", ".\experiments\tiny_language_lab\corpus\tinystories_bpe_v256_seed.txt",
            "--device", "cuda",
            "--block-size", "128",
            "--batch-size", "8",
            "--n-layer", "4",
            "--n-head", "4",
            "--n-embd", "256",
            "--eval-mode", "sampled",
            "--max-new-tokens", "80",
            "--steps", "20",
            "--eval-batches", "2",
            "--eval-interval", "5",
            "--log-every", "5",
            "--grad-accum-steps", "2",
            "--pos-encoding", "rope",
            "--activation-checkpoint",
            "--optimizer", "muon",
            "--muon-lr", "0.01",
            "--seeds", "7",
            "--configs", "random_full", "count_prior_lora_r2",
            "--out", ".\experiments\tiny_language_lab\runs\phase2_tinystories_bpe_smoke.jsonl",
            "--summary", ".\experiments\tiny_language_lab\runs\phase2_tinystories_bpe_smoke.md",
            "--title", "Phase 2 TinyStories BPE Smoke"
        )
    }
    "bpe500" {
        $RunArgs = @(
            ".\experiments\tiny_language_lab\cassandra_compare.py",
            "--corpus", ".\experiments\tiny_language_lab\corpus\tinystories_bpe_v256_seed.txt",
            "--device", "cuda",
            "--block-size", "128",
            "--batch-size", "8",
            "--n-layer", "4",
            "--n-head", "4",
            "--n-embd", "256",
            "--eval-mode", "sampled",
            "--max-new-tokens", "160",
            "--steps", "500",
            "--eval-batches", "16",
            "--eval-interval", "100",
            "--log-every", "100",
            "--grad-accum-steps", "2",
            "--pos-encoding", "rope",
            "--activation-checkpoint",
            "--optimizer", "muon",
            "--muon-lr", "0.01",
            "--seeds", "7", "11", "19",
            "--configs", "random_full", "count_prior_lora_r2",
            "--out", ".\experiments\tiny_language_lab\runs\phase2_tinystories_bpe_b500.jsonl",
            "--summary", ".\experiments\tiny_language_lab\runs\phase2_tinystories_bpe_b500.md",
            "--title", "Phase 2 TinyStories BPE 500 steps"
        )
    }
}

Write-Host "[visible] repo=$RepoRoot"
Write-Host "[visible] mode=$Mode"
Write-Host "[visible] log=$Log"
Write-Host "[visible] command=python $($RunArgs -join ' ')"
Write-Host ""

python @RunArgs 2>&1 | Tee-Object -FilePath $Log
$ExitCode = $LASTEXITCODE
Write-Host ""
Write-Host "[visible] exit_code=$ExitCode"
Write-Host "[visible] log=$Log"
if ($KeepOpen) {
    Read-Host "Run finished. Press Enter to close this window"
}
exit $ExitCode
