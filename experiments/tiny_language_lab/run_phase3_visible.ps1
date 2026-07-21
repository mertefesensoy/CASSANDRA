param(
    [ValidateSet("stage51", "score51", "stage52", "stage52-matrix", "stage52-prior-sharded")]
    [string]$Mode = "stage51",
    [ValidateSet("3m", "10m", "25m", "85m")]
    [string]$Size = "25m",
    [int]$Budget = 1000,
    [ValidateSet("random_full", "count_prior_ng4_lora_r2")]
    [string]$Config = "random_full",
    [switch]$DryRun,
    [switch]$KeepOpen
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$RunDir = Join-Path $ScriptDir "runs"
$Corpus = Join-Path $ScriptDir "corpus\tinystories_char_seed.txt"
$ShardDir = Join-Path $ScriptDir "corpus\tinystories_char_shards_500mb"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $RunDir ("phase3_{0}_{1}.log" -f $Mode, $Timestamp)
$LauncherLog = Join-Path $RunDir ("phase3_{0}_{1}_launcher.log" -f $Mode, $Timestamp)

if (-not (Test-Path -LiteralPath $Corpus)) {
    throw "Missing TinyStories corpus at $Corpus"
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
Set-Location $RepoRoot
"[visible] launcher_start=$(Get-Date -Format o)" | Set-Content -LiteralPath $LauncherLog

$SizeMap = @{
    "3m" = @{ Layers = "4"; Heads = "4"; Embd = "256" }
    "10m" = @{ Layers = "6"; Heads = "6"; Embd = "384" }
    "25m" = @{ Layers = "8"; Heads = "8"; Embd = "512" }
    "85m" = @{ Layers = "12"; Heads = "12"; Embd = "768" }
}

$Common = @(
    ".\experiments\tiny_language_lab\cassandra_compare.py",
    "--corpus", ".\experiments\tiny_language_lab\corpus\tinystories_char_seed.txt",
    "--device", "cuda",
    "--block-size", "128",
    "--batch-size", "8",
    "--grad-accum-steps", "2",
    "--pos-encoding", "rope",
    "--activation-checkpoint",
    "--optimizer", "muon",
    "--muon-lr", "0.01",
    "--eval-mode", "sampled",
    "--eval-batches", "16",
    "--no-copy-train-marker",
    "--prompt", "once upon a time "
)

function New-Stage52RunArgs {
    param(
        [string]$CellSize,
        [int]$CellBudget,
        [string]$CellConfig,
        [string]$OutSuffix = ""
    )
    $Shape = $SizeMap[$CellSize]
    $OutStem = "stage52_crossover_{0}_b{1}_{2}{3}" -f $CellSize, $CellBudget, $CellConfig, $OutSuffix
    $Args = $Common + @(
        "--steps", [string]$CellBudget,
        "--eval-interval", [string]$CellBudget,
        "--log-every", "500",
        "--n-layer", $Shape.Layers,
        "--n-head", $Shape.Heads,
        "--n-embd", $Shape.Embd,
        "--max-new-tokens", "120",
        "--seeds", "7", "11", "19",
        "--configs", $CellConfig,
        "--out", ".\experiments\tiny_language_lab\runs\$OutStem.jsonl",
        "--summary", ".\experiments\tiny_language_lab\runs\$OutStem.md",
        "--title", "Stage 52 Crossover $CellSize $CellBudget-step $CellConfig$OutSuffix",
        "--train-shard-dir", ".\experiments\tiny_language_lab\corpus\tinystories_char_shards_500mb",
        "--stream-train-eval-chars", "200000",
        "--prior-cache-dir", ".\experiments\tiny_language_lab\runs\stage52_prior_cache"
    )
    return $Args
}

function Test-JsonlComplete {
    param(
        [string]$Path,
        [int]$ExpectedRows = 3
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $Rows = @(Get-Content -LiteralPath $Path | Where-Object { $_.Trim().Length -gt 0 })
    return $Rows.Count -ge $ExpectedRows
}

switch ($Mode) {
    "stage51" {
        $RunArgs = $Common + @(
            "--steps", "5000",
            "--eval-interval", "5000",
            "--log-every", "500",
            "--n-layer", "8",
            "--n-head", "8",
            "--n-embd", "512",
            "--train-shard-dir", ".\experiments\tiny_language_lab\corpus\tinystories_char_shards_500mb",
            "--stream-train-eval-chars", "200000",
            "--max-new-tokens", "240",
            "--seeds", "7", "11", "19",
            "--configs", "random_full",
            "--checkpoint-dir", ".\experiments\tiny_language_lab\runs\stage51_checkpoints",
            "--out", ".\experiments\tiny_language_lab\runs\stage51_coherence_25m_b5000.jsonl",
            "--summary", ".\experiments\tiny_language_lab\runs\stage51_coherence_25m_b5000.md",
            "--title", "Stage 51 Coherence Checkpoint 25M 5000-step"
        )
    }
    "score51" {
        $RunArgs = @(
            ".\experiments\tiny_language_lab\score_generation_samples.py",
            "--runs", ".\experiments\tiny_language_lab\runs\stage51_coherence_25m_b5000.jsonl",
            "--corpus", ".\experiments\tiny_language_lab\corpus\tinystories_char_seed.txt",
            "--out", ".\experiments\tiny_language_lab\runs\stage51_coherence_25m_b5000_generation_quality.md",
            "--title", "Stage 51 Coherence Checkpoint Generation Quality"
        )
    }
    "stage52" {
        $RunArgs = New-Stage52RunArgs -CellSize $Size -CellBudget $Budget -CellConfig $Config
    }
    "stage52-matrix" {
    }
    "stage52-prior-sharded" {
    }
}

if ($Mode -eq "stage52-matrix" -or $Mode -eq "stage52-prior-sharded") {
    $MatrixSizes = @("3m", "10m", "25m", "85m")
    $MatrixBudgets = @(200, 500, 1000, 2000)
    $MatrixConfigs = @("random_full", "count_prior_ng4_lora_r2")
    $OutSuffix = ""
    if ($Mode -eq "stage52-prior-sharded") {
        $MatrixConfigs = @("count_prior_ng4_lora_r2")
        $OutSuffix = "_sharded"
    }
    Write-Host "[visible] repo=$RepoRoot"
    Write-Host "[visible] mode=$Mode"
    Write-Host "[visible] log=$Log"
    Write-Host "[visible] cells=$($MatrixSizes.Count * $MatrixBudgets.Count * $MatrixConfigs.Count)"
    Add-Content -LiteralPath $LauncherLog -Value "[visible] repo=$RepoRoot"
    Add-Content -LiteralPath $LauncherLog -Value "[visible] mode=$Mode"
    Add-Content -LiteralPath $LauncherLog -Value "[visible] log=$Log"
    Add-Content -LiteralPath $LauncherLog -Value "[visible] cells=$($MatrixSizes.Count * $MatrixBudgets.Count * $MatrixConfigs.Count)"
    if ($DryRun) {
        foreach ($CellSize in $MatrixSizes) {
            foreach ($CellBudget in $MatrixBudgets) {
                foreach ($CellConfig in $MatrixConfigs) {
                    $OutStem = "stage52_crossover_{0}_b{1}_{2}{3}" -f $CellSize, $CellBudget, $CellConfig, $OutSuffix
                    Write-Host "[visible] dry_cell=$OutStem"
                    Add-Content -LiteralPath $LauncherLog -Value "[visible] dry_cell=$OutStem"
                }
            }
        }
        Write-Host "[visible] dry_run=true"
        Add-Content -LiteralPath $LauncherLog -Value "[visible] dry_run=true"
        exit 0
    }

    $ProcessFailures = 0
    foreach ($CellSize in $MatrixSizes) {
        foreach ($CellBudget in $MatrixBudgets) {
            foreach ($CellConfig in $MatrixConfigs) {
                $OutStem = "stage52_crossover_{0}_b{1}_{2}{3}" -f $CellSize, $CellBudget, $CellConfig, $OutSuffix
                $OutPath = Join-Path $RunDir "$OutStem.jsonl"
                if (Test-JsonlComplete -Path $OutPath -ExpectedRows 3) {
                    "[matrix-visible] skip complete $OutStem" | Tee-Object -FilePath $Log -Append
                    continue
                }
                $CellArgs = New-Stage52RunArgs -CellSize $CellSize -CellBudget $CellBudget -CellConfig $CellConfig -OutSuffix $OutSuffix
                "[matrix-visible] start $OutStem" | Tee-Object -FilePath $Log -Append
                python @CellArgs 2>&1 | Tee-Object -FilePath $Log -Append
                $CellExit = $LASTEXITCODE
                "[matrix-visible] exit $OutStem code=$CellExit" | Tee-Object -FilePath $Log -Append
                if ($CellExit -ne 0) {
                    $ProcessFailures += 1
                    $Rows = foreach ($Seed in @(7, 11, 19)) {
                        [ordered]@{
                            status = "error"
                            comparison_name = $CellConfig
                            seed = $Seed
                            size = $CellSize
                            steps = $CellBudget
                            error_type = "ProcessExit"
                            error = "python exited with code $CellExit before cassandra_compare.py completed"
                        } | ConvertTo-Json -Compress
                    }
                    $Rows | Set-Content -LiteralPath $OutPath
                    "[matrix-visible] wrote process-error rows $OutStem" | Tee-Object -FilePath $Log -Append
                }
            }
        }
    }
    Write-Host "[visible] process_failures=$ProcessFailures"
    Add-Content -LiteralPath $LauncherLog -Value "[visible] process_failures=$ProcessFailures"
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit 0
}

Write-Host "[visible] repo=$RepoRoot"
Write-Host "[visible] mode=$Mode"
if ($Mode -eq "stage52") {
    Write-Host "[visible] size=$Size budget=$Budget config=$Config"
}
Write-Host "[visible] log=$Log"
Write-Host "[visible] command=python $($RunArgs -join ' ')"
Write-Host ""
Add-Content -LiteralPath $LauncherLog -Value "[visible] repo=$RepoRoot"
Add-Content -LiteralPath $LauncherLog -Value "[visible] mode=$Mode"
if ($Mode -eq "stage52") {
    Add-Content -LiteralPath $LauncherLog -Value "[visible] size=$Size budget=$Budget config=$Config"
}
Add-Content -LiteralPath $LauncherLog -Value "[visible] log=$Log"
Add-Content -LiteralPath $LauncherLog -Value "[visible] command=python $($RunArgs -join ' ')"

if ($DryRun) {
    Add-Content -LiteralPath $LauncherLog -Value "[visible] dry_run=true"
    Write-Host "[visible] dry_run=true"
    exit 0
}

try {
    python @RunArgs 2>&1 | Tee-Object -FilePath $Log
    $ExitCode = $LASTEXITCODE
} catch {
    Add-Content -LiteralPath $LauncherLog -Value "[visible] exception=$($_.Exception.Message)"
    throw
}
Write-Host ""
Write-Host "[visible] exit_code=$ExitCode"
Write-Host "[visible] log=$Log"
if ($KeepOpen) {
    Read-Host "Run finished. Press Enter to close this window"
}
exit $ExitCode
