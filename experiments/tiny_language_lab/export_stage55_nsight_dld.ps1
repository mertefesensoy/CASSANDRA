param(
    [string]$Checkpoint = "C:\cassandra_runs\stage55_flagship_checkpoints\stage55_flagship_200m_b50000_seed7_random_full_seed7_step050000.pt",
    [string]$OutDir = ".\experiments\tiny_language_lab\artifacts\phase4\nsight_dld",
    [string]$Name = "stage55_seed7_final_success_b1_s256",
    [int]$SeqLen = 256,
    [int]$BatchSize = 1,
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",
    [switch]$DynamicBatch,
    [switch]$CheckOnnx,
    [switch]$OpenFolder
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $RepoRoot

$RunArgs = @(
    ".\experiments\tiny_language_lab\export_nsight_dld.py",
    "--checkpoint", $Checkpoint,
    "--out-dir", $OutDir,
    "--seq-len", [string]$SeqLen,
    "--batch-size", [string]$BatchSize,
    "--device", $Device
)

if ($Name -ne "") {
    $RunArgs += @("--name", $Name)
}
if ($DynamicBatch) {
    $RunArgs += "--dynamic-batch"
}
if ($CheckOnnx) {
    $RunArgs += "--check-onnx"
}

Write-Host "[nsight-dld] python $($RunArgs -join ' ')"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
python @RunArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($OpenFolder) {
    Invoke-Item $OutDir
}
