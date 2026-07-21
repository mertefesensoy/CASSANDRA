param(
    [ValidateSet(
        "stage53-smoke",
        "stage53-cell",
        "stage53",
        "stage54-smoke",
        "stage54-gateA",
        "stage54-phaseB-cell",
        "stage55-size-85m-b128",
        "stage55-size-85m-b256",
        "stage55-size-200m-b128",
        "stage55-size-200m-b256",
        "stage55-resume-unbroken",
        "stage55-resume-interrupted",
        "stage55-resume-resumed",
        "stage55-flagship-cell"
    )]
    [string]$Mode = "stage53-smoke",
    [int]$Budget = 500,
    [int]$Seed = 7,
    [string]$ResumeFrom = "",
    [string]$FlagshipCheckpointDir = "",
    [string]$MuonLr = "0.01",
    [switch]$DryRun,
    [switch]$KeepOpen
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$RunDir = Join-Path $ScriptDir "runs"
$Corpus = Join-Path $ScriptDir "corpus\tinystories_char_seed.txt"
$ShardDir = Join-Path $ScriptDir "corpus\tinystories_char_shards_500mb"
$PriorCacheDir = Join-Path $RunDir "stage52_prior_cache"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $RunDir ("phase4_{0}_{1}.log" -f $Mode, $Timestamp)
$LauncherLog = Join-Path $RunDir ("phase4_{0}_{1}_launcher.log" -f $Mode, $Timestamp)

if (-not (Test-Path -LiteralPath $Corpus)) {
    throw "Missing TinyStories corpus at $Corpus"
}
if (-not (Test-Path -LiteralPath $ShardDir)) {
    throw "Missing TinyStories shard directory at $ShardDir"
}
if (-not (Test-Path -LiteralPath $PriorCacheDir)) {
    throw "Missing Stage 52 prior cache directory at $PriorCacheDir"
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
Set-Location $RepoRoot
"[visible] launcher_start=$(Get-Date -Format o)" | Set-Content -LiteralPath $LauncherLog

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
    "--muon-lr", $MuonLr,
    "--eval-mode", "sampled",
    "--eval-batches", "16",
    "--no-copy-train-marker",
    "--prompt", "once upon a time ",
    "--max-new-tokens", "120",
    "--n-layer", "8",
    "--n-head", "8",
    "--n-embd", "512",
    "--train-shard-dir", ".\experiments\tiny_language_lab\corpus\tinystories_char_shards_500mb",
    "--stream-train-eval-chars", "200000",
    "--prior-cache-dir", ".\experiments\tiny_language_lab\runs\stage52_prior_cache"
)

function New-Stage53CellArgs {
    param([int]$CellBudget)
    $LrSuffix = ""
    if ($MuonLr -ne "0.01") {
        $LrToken = $MuonLr -replace "^0\.", "" -replace "\.", "p"
        $LrSuffix = "_muonlr$LrToken"
    }
    $OutStem = "stage53_prior_all_25m_b{0}{1}" -f $CellBudget, $LrSuffix
    return $Common + @(
        "--steps", [string]$CellBudget,
        "--eval-interval", [string]$CellBudget,
        "--log-every", "500",
        "--seeds", "7", "11", "19",
        "--configs", "count_prior_ng4_all",
        "--out", ".\experiments\tiny_language_lab\runs\$OutStem.jsonl",
        "--summary", ".\experiments\tiny_language_lab\runs\$OutStem.md",
        "--title", "Stage 53 Prior-All 25M $CellBudget-step$LrSuffix"
    )
}

function New-Stage53SmokeArgs {
    $OutStem = "stage53_smoke_prior_all_10step"
    return $Common + @(
        "--steps", "10",
        "--eval-interval", "10",
        "--log-every", "10",
        "--seeds", "7",
        "--configs", "count_prior_ng4_lora_r2", "count_prior_ng4_all",
        "--out", ".\experiments\tiny_language_lab\runs\$OutStem.jsonl",
        "--summary", ".\experiments\tiny_language_lab\runs\$OutStem.md",
        "--title", "Stage 53 Prior-All 10-step Smoke"
    )
}

function New-Stage54SmokeArgs {
    $OutStem = "stage54_smoke_ng5_20step"
    return $Common + @(
        "--steps", "20",
        "--eval-interval", "20",
        "--log-every", "10",
        "--seeds", "7",
        "--configs", "count_prior_ng5_lora_r2",
        "--prior5-min-count", "10",
        "--out", ".\experiments\tiny_language_lab\runs\$OutStem.jsonl",
        "--summary", ".\experiments\tiny_language_lab\runs\$OutStem.md",
        "--title", "Stage 54 Order-5 20-step Smoke"
    )
}

function New-Stage54GateArgs {
    $OutStem = "stage54_gateA_floor_pair"
    return $Common + @(
        "--steps", "200",
        "--eval-interval", "200",
        "--log-every", "500",
        "--seeds", "7", "11", "19",
        "--configs", "count_prior_ng4_lora_r2_floor", "count_prior_ng5_lora_r2_floor",
        "--prior5-min-count", "10",
        "--out", ".\experiments\tiny_language_lab\runs\$OutStem.jsonl",
        "--summary", ".\experiments\tiny_language_lab\runs\$OutStem.md",
        "--title", "Stage 54 Gate A Order-5 vs Order-4 Floor"
    )
}

function New-Stage54PhaseBCellArgs {
    param([int]$CellBudget)
    $OutStem = "stage54_ng5_25m_b{0}" -f $CellBudget
    return $Common + @(
        "--steps", [string]$CellBudget,
        "--eval-interval", [string]$CellBudget,
        "--log-every", "500",
        "--seeds", "7", "11", "19",
        "--configs", "count_prior_ng5_lora_r2",
        "--prior5-min-count", "10",
        "--out", ".\experiments\tiny_language_lab\runs\$OutStem.jsonl",
        "--summary", ".\experiments\tiny_language_lab\runs\$OutStem.md",
        "--title", "Stage 54 Order-5 Crossover 25M $CellBudget-step"
    )
}

function New-Stage55SizeArgs {
    param(
        [string]$Label,
        [string]$BlockSize,
        [string]$Layers,
        [string]$Heads,
        [string]$Embd
    )
    $OutStem = "stage55_size_{0}" -f $Label
    return @(
        ".\experiments\tiny_language_lab\cassandra_compare.py",
        "--corpus", ".\experiments\tiny_language_lab\corpus\tinystories_char_seed.txt",
        "--device", "cuda",
        "--steps", "200",
        "--eval-interval", "200",
        "--log-every", "50",
        "--seeds", "7",
        "--configs", "random_full",
        "--block-size", $BlockSize,
        "--batch-size", "8",
        "--grad-accum-steps", "2",
        "--pos-encoding", "rope",
        "--activation-checkpoint",
        "--optimizer", "muon",
        "--muon-lr", $MuonLr,
        "--eval-mode", "sampled",
        "--eval-batches", "16",
        "--no-copy-train-marker",
        "--prompt", "once upon a time ",
        "--max-new-tokens", "120",
        "--n-layer", $Layers,
        "--n-head", $Heads,
        "--n-embd", $Embd,
        "--train-shard-dir", ".\experiments\tiny_language_lab\corpus\tinystories_char_shards_500mb",
        "--stream-train-eval-chars", "200000",
        "--out", ".\experiments\tiny_language_lab\runs\$OutStem.jsonl",
        "--summary", ".\experiments\tiny_language_lab\runs\$OutStem.md",
        "--title", "Stage 55 Size Gate $Label"
    )
}

function New-Stage55ResumeArgs {
    param(
        [string]$Label,
        [string]$ResumeFrom = ""
    )
    $CheckpointDir = ".\experiments\tiny_language_lab\runs\stage55_resume_checkpoints"
    $RunArgs = @(
        ".\experiments\tiny_language_lab\cassandra_compare.py",
        "--corpus", ".\experiments\tiny_language_lab\corpus\tinystories_char_seed.txt",
        "--device", "cuda",
        "--steps", "400",
        "--eval-interval", "200",
        "--log-every", "100",
        "--seeds", "7",
        "--configs", "random_full",
        "--block-size", "256",
        "--batch-size", "8",
        "--grad-accum-steps", "2",
        "--pos-encoding", "rope",
        "--activation-checkpoint",
        "--optimizer", "muon",
        "--muon-lr", $MuonLr,
        "--eval-mode", "sampled",
        "--eval-batches", "16",
        "--no-copy-train-marker",
        "--prompt", "once upon a time ",
        "--max-new-tokens", "120",
        "--n-layer", "16",
        "--n-head", "16",
        "--n-embd", "1024",
        "--train-shard-dir", ".\experiments\tiny_language_lab\corpus\tinystories_char_shards_500mb",
        "--stream-train-eval-chars", "200000",
        "--checkpoint-dir", $CheckpointDir,
        "--checkpoint-every", "200",
        "--out", ".\experiments\tiny_language_lab\runs\$Label.jsonl",
        "--summary", ".\experiments\tiny_language_lab\runs\$Label.md",
        "--title", "Stage 55 Resume Proof $Label"
    )
    if ($ResumeFrom -ne "") {
        $RunArgs += @("--resume-from", $ResumeFrom)
    }
    return $RunArgs
}

function New-Stage55FlagshipArgs {
    param(
        [int]$CellBudget,
        [int]$CellSeed,
        [string]$CellResumeFrom = ""
    )
    $CheckpointDir = "C:\cassandra_runs\stage55_flagship_checkpoints"
    if ($FlagshipCheckpointDir -ne "") {
        $CheckpointDir = $FlagshipCheckpointDir
    }
    $OutStem = "stage55_flagship_200m_b{0}_seed{1}" -f $CellBudget, $CellSeed
    $RunArgs = @(
        ".\experiments\tiny_language_lab\cassandra_compare.py",
        "--corpus", ".\experiments\tiny_language_lab\corpus\tinystories_char_seed.txt",
        "--device", "cuda",
        "--steps", [string]$CellBudget,
        "--eval-interval", "5000",
        "--log-every", "1000",
        "--seeds", [string]$CellSeed,
        "--configs", "random_full",
        "--block-size", "256",
        "--batch-size", "8",
        "--grad-accum-steps", "2",
        "--pos-encoding", "rope",
        "--activation-checkpoint",
        "--optimizer", "muon",
        "--muon-lr", $MuonLr,
        "--eval-mode", "sampled",
        "--eval-batches", "16",
        "--no-copy-train-marker",
        "--prompt", "once upon a time ",
        "--max-new-tokens", "240",
        "--n-layer", "16",
        "--n-head", "16",
        "--n-embd", "1024",
        "--train-shard-dir", ".\experiments\tiny_language_lab\corpus\tinystories_char_shards_500mb",
        "--stream-train-eval-chars", "200000",
        "--checkpoint-dir", $CheckpointDir,
        "--checkpoint-every", "5000",
        "--out", ".\experiments\tiny_language_lab\runs\$OutStem.jsonl",
        "--summary", ".\experiments\tiny_language_lab\runs\$OutStem.md",
        "--title", "Stage 55 Flagship 200M $CellBudget-step seed $CellSeed"
    )
    if ($CellResumeFrom -ne "") {
        $RunArgs += @("--resume-from", $CellResumeFrom)
    }
    return $RunArgs
}

function Test-JsonlComplete {
    param(
        [string]$Path,
        [int]$ExpectedRows
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $Rows = @(Get-Content -LiteralPath $Path | Where-Object { $_.Trim().Length -gt 0 })
    return $Rows.Count -ge $ExpectedRows
}

function Invoke-VisiblePython {
    param(
        [string]$Label,
        [string[]]$RunArgs
    )
    Write-Host "[visible] command=python $($RunArgs -join ' ')"
    Add-Content -LiteralPath $LauncherLog -Value "[visible] command=python $($RunArgs -join ' ')"
    if ($DryRun) {
        "[visible] dry_run $Label" | Tee-Object -FilePath $Log -Append
        return 0
    }
    "[matrix-visible] start $Label" | Tee-Object -FilePath $Log -Append
    python @RunArgs 2>&1 | Tee-Object -FilePath $Log -Append
    $CellExit = $LASTEXITCODE
    "[matrix-visible] exit $Label code=$CellExit" | Tee-Object -FilePath $Log -Append
    return $CellExit
}

Write-Host "[visible] repo=$RepoRoot"
Write-Host "[visible] mode=$Mode"
Write-Host "[visible] muon_lr=$MuonLr"
Write-Host "[visible] seed=$Seed"
Write-Host "[visible] resume_from=$ResumeFrom"
Write-Host "[visible] flagship_checkpoint_dir=$FlagshipCheckpointDir"
Write-Host "[visible] log=$Log"
Add-Content -LiteralPath $LauncherLog -Value "[visible] repo=$RepoRoot"
Add-Content -LiteralPath $LauncherLog -Value "[visible] mode=$Mode"
Add-Content -LiteralPath $LauncherLog -Value "[visible] muon_lr=$MuonLr"
Add-Content -LiteralPath $LauncherLog -Value "[visible] seed=$Seed"
Add-Content -LiteralPath $LauncherLog -Value "[visible] resume_from=$ResumeFrom"
Add-Content -LiteralPath $LauncherLog -Value "[visible] flagship_checkpoint_dir=$FlagshipCheckpointDir"
Add-Content -LiteralPath $LauncherLog -Value "[visible] log=$Log"

if ($Mode -eq "stage53-smoke") {
    $RunArgs = New-Stage53SmokeArgs
    $ExitCode = Invoke-VisiblePython -Label "stage53_smoke_prior_all_10step" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage53-cell") {
    $RunArgs = New-Stage53CellArgs -CellBudget $Budget
    $ExitCode = Invoke-VisiblePython -Label "stage53_prior_all_25m_b$Budget" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage54-smoke") {
    $RunArgs = New-Stage54SmokeArgs
    $ExitCode = Invoke-VisiblePython -Label "stage54_smoke_ng5_20step" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage54-gateA") {
    $RunArgs = New-Stage54GateArgs
    $ExitCode = Invoke-VisiblePython -Label "stage54_gateA_floor_pair" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage54-phaseB-cell") {
    $RunArgs = New-Stage54PhaseBCellArgs -CellBudget $Budget
    $ExitCode = Invoke-VisiblePython -Label "stage54_ng5_25m_b$Budget" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage55-size-85m-b128") {
    $RunArgs = New-Stage55SizeArgs -Label "85m_b128" -BlockSize "128" -Layers "12" -Heads "12" -Embd "768"
    $ExitCode = Invoke-VisiblePython -Label "stage55_size_85m_b128" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage55-size-85m-b256") {
    $RunArgs = New-Stage55SizeArgs -Label "85m_b256" -BlockSize "256" -Layers "12" -Heads "12" -Embd "768"
    $ExitCode = Invoke-VisiblePython -Label "stage55_size_85m_b256" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage55-size-200m-b128") {
    $RunArgs = New-Stage55SizeArgs -Label "200m_b128" -BlockSize "128" -Layers "16" -Heads "16" -Embd "1024"
    $ExitCode = Invoke-VisiblePython -Label "stage55_size_200m_b128" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage55-size-200m-b256") {
    $RunArgs = New-Stage55SizeArgs -Label "200m_b256" -BlockSize "256" -Layers "16" -Heads "16" -Embd "1024"
    $ExitCode = Invoke-VisiblePython -Label "stage55_size_200m_b256" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage55-resume-unbroken") {
    $RunArgs = New-Stage55ResumeArgs -Label "stage55_resume_unbroken_400"
    $ExitCode = Invoke-VisiblePython -Label "stage55_resume_unbroken_400" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage55-resume-interrupted") {
    $RunArgs = New-Stage55ResumeArgs -Label "stage55_resume_interrupted_400"
    $ExitCode = Invoke-VisiblePython -Label "stage55_resume_interrupted_400" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage55-resume-resumed") {
    $ResumePath = ".\experiments\tiny_language_lab\runs\stage55_resume_checkpoints\stage55_resume_interrupted_400_random_full_seed7_step000200.pt"
    if (-not (Test-Path -LiteralPath $ResumePath)) {
        throw "Missing resume checkpoint at $ResumePath"
    }
    $RunArgs = New-Stage55ResumeArgs -Label "stage55_resume_resumed_400" -ResumeFrom $ResumePath
    $ExitCode = Invoke-VisiblePython -Label "stage55_resume_resumed_400" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage55-flagship-cell") {
    $RunArgs = New-Stage55FlagshipArgs -CellBudget $Budget -CellSeed $Seed -CellResumeFrom $ResumeFrom
    $ExitCode = Invoke-VisiblePython -Label "stage55_flagship_200m_b${Budget}_seed${Seed}" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

$Budgets = @(200, 500, 1000, 2000)
$ProcessFailures = 0
foreach ($CellBudget in $Budgets) {
    $OutStem = "stage53_prior_all_25m_b{0}" -f $CellBudget
    $OutPath = Join-Path $RunDir "$OutStem.jsonl"
    if (Test-JsonlComplete -Path $OutPath -ExpectedRows 3) {
        "[matrix-visible] skip complete $OutStem" | Tee-Object -FilePath $Log -Append
        continue
    }
    $RunArgs = New-Stage53CellArgs -CellBudget $CellBudget
    $CellExit = Invoke-VisiblePython -Label $OutStem -RunArgs $RunArgs
    if ($CellExit -ne 0) {
        $ProcessFailures += 1
    }
}

Write-Host "[visible] process_failures=$ProcessFailures"
Add-Content -LiteralPath $LauncherLog -Value "[visible] process_failures=$ProcessFailures"
if ($KeepOpen) {
    Read-Host "Run finished. Press Enter to close this window"
}
exit $ProcessFailures
