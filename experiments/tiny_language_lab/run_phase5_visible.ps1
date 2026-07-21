param(
    [ValidateSet(
        "stage56-prep",
        "stage56-smoke",
        "stage56-cell",
        "stage58-prep",
        "stage58-throughput",
        "stage58-cold",
        "stage58-curriculum-phase1",
        "stage58-curriculum-phase2",
        "stage58-mixture"
    )]
    [string]$Mode = "stage56-smoke",
    [int]$Budget = 50000,
    [int]$Seed = 7,
    [string]$ResumeFrom = "",
    [string]$Stage56CheckpointDir = "",
    [string]$Stage58CheckpointRoot = "C:\cassandra_runs",
    [string]$Stage58TinyCorpus = "",
    [string]$Stage58TinyShardDir = "",
    [string]$Stage58BroadShardDir = "",
    [string]$Stage58MixtureShardDir = "",
    [int]$Phase1Steps = 12500,
    [int]$LrTotalSteps = 0,
    [int]$MixtureChars = 0,
    [int]$CheckpointKeep = 0,
    [int]$CheckpointEvery = 5000,
    [string]$MuonLr = "0.01",
    [switch]$DryRun,
    [switch]$KeepOpen
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$RunDir = Join-Path $ScriptDir "runs"
$Text8Raw = Join-Path $ScriptDir "corpus\text8\text8"
$Corpus = Join-Path $ScriptDir "corpus\text8_char_seed.txt"
$ShardDir = Join-Path $ScriptDir "corpus\text8_char_shards"
$UnionVocab = Join-Path $ScriptDir "corpus\phase5_union_vocab.txt"
$TinyCorpus = if ($Stage58TinyCorpus -ne "") { $Stage58TinyCorpus } else { Join-Path $ScriptDir "corpus\tinystories_char_seed.txt" }
$TinyShardDir = if ($Stage58TinyShardDir -ne "") { $Stage58TinyShardDir } else { Join-Path $ScriptDir "corpus\tinystories_char_shards_500mb" }
$BroadShardDir = if ($Stage58BroadShardDir -ne "") { $Stage58BroadShardDir } else { $ShardDir }
$MixtureShardDir = if ($Stage58MixtureShardDir -ne "") { $Stage58MixtureShardDir } else { Join-Path $ScriptDir "corpus\mixture_char_shards" }
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $RunDir ("phase5_{0}_{1}.log" -f $Mode, $Timestamp)
$LauncherLog = Join-Path $RunDir ("phase5_{0}_{1}_launcher.log" -f $Mode, $Timestamp)

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
Set-Location $RepoRoot
"[visible] launcher_start=$(Get-Date -Format o)" | Set-Content -LiteralPath $LauncherLog

function Assert-Text8Prepared {
    if ($DryRun) {
        return
    }
    if (-not (Test-Path -LiteralPath $Corpus)) {
        throw "Missing Stage 56 seed corpus at $Corpus. Run -Mode stage56-prep first."
    }
    if (-not (Test-Path -LiteralPath $ShardDir)) {
        throw "Missing Stage 56 shard directory at $ShardDir. Run -Mode stage56-prep first."
    }
}

function Assert-FreeSpaceForRun {
    if ($DryRun) {
        return
    }
    $FreeBytes = (Get-PSDrive -Name C).Free
    $FreeGiB = $FreeBytes / 1GB
    if ($FreeGiB -lt 15) {
        throw ("C: free space is {0:N2} GiB, below the 15 GiB Stage 58 launch gate." -f $FreeGiB)
    }
    Write-Host ("[visible] c_free_gib={0:N2}" -f $FreeGiB)
    Add-Content -LiteralPath $LauncherLog -Value ("[visible] c_free_gib={0:N2}" -f $FreeGiB)
}

function Assert-Stage58CommonPrepared {
    if ($DryRun) {
        return
    }
    Assert-Text8Prepared
    if (-not (Test-Path -LiteralPath $UnionVocab)) {
        throw "Missing Stage 58 union vocab at $UnionVocab."
    }
    if (-not (Test-Path -LiteralPath $BroadShardDir)) {
        throw "Missing broad shard directory at $BroadShardDir."
    }
}

function Assert-Stage58TinyPrepared {
    if ($DryRun) {
        return
    }
    Assert-Stage58CommonPrepared
    if (-not (Test-Path -LiteralPath $TinyCorpus)) {
        throw "Missing TinyStories corpus at $TinyCorpus. Regenerate the full TinyStories corpus before Stage 58 curriculum or mixture."
    }
    if (-not (Test-Path -LiteralPath $TinyShardDir)) {
        throw "Missing TinyStories shard directory at $TinyShardDir. Expected the full Stage 58 shard set."
    }
}

function Assert-Stage58CheckpointWriteReady {
    param(
        [string]$CheckpointDirName
    )
    if ($DryRun) {
        return
    }
    $CheckpointDir = Join-Path $Stage58CheckpointRoot $CheckpointDirName
    $ProbeStem = "phase5_checkpoint_probe_{0}" -f ([guid]::NewGuid().ToString("N"))
    $ProbeTemp = Join-Path $CheckpointDir "$ProbeStem.pt.tmp"
    $ProbeFinal = Join-Path $CheckpointDir "$ProbeStem.pt"
    if ((Test-Path -LiteralPath $ProbeTemp) -or (Test-Path -LiteralPath $ProbeFinal)) {
        throw "Refusing checkpoint probe because its generated path already exists."
    }
    $ProbePathsOwned = $true
    try {
        New-Item -ItemType Directory -Force -Path $CheckpointDir | Out-Null
        $ProbeCode = @"
import os
from pathlib import Path
import torch
tmp = Path(r"$ProbeTemp")
final = Path(r"$ProbeFinal")
torch.save({"probe": torch.tensor([1])}, tmp)
os.replace(tmp, final)
if not final.exists() or final.stat().st_size <= 0:
    raise RuntimeError("checkpoint probe did not produce a durable file")
"@
        python -c $ProbeCode
        if ($LASTEXITCODE -ne 0) {
            throw "PyTorch checkpoint probe exited with code $LASTEXITCODE"
        }
        Write-Host "[visible] checkpoint_write_probe=passed path=$CheckpointDir"
        Add-Content -LiteralPath $LauncherLog -Value "[visible] checkpoint_write_probe=passed path=$CheckpointDir"
    }
    catch {
        $Details = $_.Exception.Message
        throw "Stage 58 checkpoint-write probe failed for $CheckpointDir. $Details"
    }
    finally {
        if ($ProbePathsOwned -and (Test-Path -LiteralPath $ProbeTemp)) {
            Remove-Item -LiteralPath $ProbeTemp -Force -ErrorAction SilentlyContinue
        }
        if ($ProbePathsOwned -and (Test-Path -LiteralPath $ProbeFinal)) {
            Remove-Item -LiteralPath $ProbeFinal -Force -ErrorAction SilentlyContinue
        }
    }
}
function New-Stage56PrepArgs {
    if (-not (Test-Path -LiteralPath $Text8Raw)) {
        throw "Missing text8 source at $Text8Raw"
    }
    return @(
        ".\experiments\tiny_language_lab\make_text8_shards.py",
        "--text8", ".\experiments\tiny_language_lab\corpus\text8\text8",
        "--out-dir", ".\experiments\tiny_language_lab\corpus\text8_char_shards",
        "--seed-out", ".\experiments\tiny_language_lab\corpus\text8_char_seed.txt",
        "--metadata-out", ".\experiments\tiny_language_lab\corpus\text8_char_shards.meta.json",
        "--shard-chars", "10000000"
    )
}

function New-Stage58PrepArgs {
    $TargetChars = $MixtureChars
    if ($TargetChars -le 0) {
        $TargetChars = $Budget * 8 * 2 * 256
    }
    $BroadSteps = $Budget - $Phase1Steps
    if ($Phase1Steps -le 0 -or $BroadSteps -le 0) {
        throw "Stage 58 mixture needs positive TinyStories and broad-text step counts."
    }
    $TinyWeight = $Phase1Steps
    $BroadWeight = $BroadSteps
    while ($BroadWeight -ne 0) {
        $Remainder = $TinyWeight % $BroadWeight
        $TinyWeight = $BroadWeight
        $BroadWeight = $Remainder
    }
    $WeightDivisor = $TinyWeight
    $TinyWeight = [int]($Phase1Steps / $WeightDivisor)
    $BroadWeight = [int]($BroadSteps / $WeightDivisor)
    return @(
        ".\experiments\tiny_language_lab\make_mixture_shards.py",
        "--tiny-dir", $TinyShardDir,
        "--broad-dir", $BroadShardDir,
        "--out-dir", $MixtureShardDir,
        "--metadata-out", ".\experiments\tiny_language_lab\corpus\mixture_char_shards.meta.json",
        "--tiny-weight", [string]$TinyWeight,
        "--broad-weight", [string]$BroadWeight,
        "--total-chars", [string]$TargetChars,
        "--shard-chars", "10000000"
    )
}

function New-Stage58CellArgs {
    param(
        [string]$CellLabel,
        [string]$CellCorpus,
        [string]$CellShardDir,
        [string]$CheckpointDirName,
        [int]$CellBudget,
        [int]$CellSeed,
        [string]$CellBudgetLabel = "",
        [string]$CellResumeFrom = "",
        [switch]$AdditionalResumeSteps,
        [switch]$NoCheckpointEvery
    )
    $CellCheckpointDir = Join-Path $Stage58CheckpointRoot $CheckpointDirName
    $CellCheckpointEvery = if ($NoCheckpointEvery) { "0" } else { [string]$CheckpointEvery }
    $ScheduleSteps = if ($LrTotalSteps -gt 0) { $LrTotalSteps } else { $Budget }
    $OutBudget = if ($CellBudgetLabel -ne "") { $CellBudgetLabel } else { [string]$CellBudget }
    $OutStem = "stage58_dev_{0}_85m_b{1}_seed{2}" -f $CellLabel, $OutBudget, $CellSeed
    $Title = "Stage 58 Developmental $CellLabel 85M $OutBudget-step seed $CellSeed"
    $RunArgs = @(
        ".\experiments\tiny_language_lab\cassandra_compare.py",
        "--corpus", $CellCorpus,
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
        "--precision", "fp32",
        "--lr-schedule", "cosine",
        "--lr-final-frac", "0.1",
        "--lr-total-steps", [string]$ScheduleSteps,
        "--vocab-chars-file", ".\experiments\tiny_language_lab\corpus\phase5_union_vocab.txt",
        "--eval-mode", "sampled",
        "--eval-batches", "16",
        "--no-copy-train-marker",
        "--prompt", "the history of ",
        "--max-new-tokens", "240",
        "--n-layer", "12",
        "--n-head", "12",
        "--n-embd", "768",
        "--train-shard-dir", $CellShardDir,
        "--stream-train-eval-chars", "200000",
        "--val-fraction", "0.05263157894736842",
        "--checkpoint-dir", $CellCheckpointDir,
        "--checkpoint-every", $CellCheckpointEvery,
        "--checkpoint-keep", [string]$CheckpointKeep,
        "--out", ".\experiments\tiny_language_lab\runs\$OutStem.jsonl",
        "--summary", ".\experiments\tiny_language_lab\runs\$OutStem.md",
        "--title", $Title
    )
    if ($CellResumeFrom -ne "") {
        $RunArgs += @("--resume-from", $CellResumeFrom)
    }
    if ($AdditionalResumeSteps) {
        $RunArgs += @("--resume-steps-additional")
    }
    return $RunArgs
}

function New-Stage56CellArgs {
    param(
        [int]$CellBudget,
        [int]$CellSeed,
        [string]$CellResumeFrom = "",
        [switch]$Smoke
    )
    $CheckpointDir = "C:\cassandra_runs\stage56_broadchar_checkpoints"
    if ($Stage56CheckpointDir -ne "") {
        $CheckpointDir = $Stage56CheckpointDir
    }
    $CellCheckpointEvery = [string]$CheckpointEvery
    $EvalInterval = "5000"
    $LogEvery = "1000"
    $OutStem = "stage56_broadchar_85m_b{0}_seed{1}" -f $CellBudget, $CellSeed
    $Title = "Stage 56 Broad-Corpus Char 85M $CellBudget-step seed $CellSeed"
    if ($Smoke) {
        $CellCheckpointEvery = "10"
        $EvalInterval = "10"
        $LogEvery = "10"
        $OutStem = "stage56_smoke_broadchar_85m_b{0}_seed{1}" -f $CellBudget, $CellSeed
        $Title = "Stage 56 Broad-Corpus Char 85M $CellBudget-step smoke seed $CellSeed"
        if ($Stage56CheckpointDir -eq "") {
            $CheckpointDir = "C:\cassandra_runs\stage56_broadchar_smoke_checkpoints"
        }
    }
    $RunArgs = @(
        ".\experiments\tiny_language_lab\cassandra_compare.py",
        "--corpus", ".\experiments\tiny_language_lab\corpus\text8_char_seed.txt",
        "--device", "cuda",
        "--steps", [string]$CellBudget,
        "--eval-interval", $EvalInterval,
        "--log-every", $LogEvery,
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
        "--prompt", "the history of ",
        "--max-new-tokens", "240",
        "--n-layer", "12",
        "--n-head", "12",
        "--n-embd", "768",
        "--train-shard-dir", ".\experiments\tiny_language_lab\corpus\text8_char_shards",
        "--stream-train-eval-chars", "200000",
        "--val-fraction", "0.05263157894736842",
        "--checkpoint-dir", $CheckpointDir,
        "--checkpoint-every", $CellCheckpointEvery,
        "--out", ".\experiments\tiny_language_lab\runs\$OutStem.jsonl",
        "--summary", ".\experiments\tiny_language_lab\runs\$OutStem.md",
        "--title", $Title
    )
    if ($CellResumeFrom -ne "") {
        $RunArgs += @("--resume-from", $CellResumeFrom)
    }
    return $RunArgs
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
    "[phase5-visible] start $Label" | Tee-Object -FilePath $Log -Append
    python @RunArgs 2>&1 | Tee-Object -FilePath $Log -Append
    $CellExit = $LASTEXITCODE
    "[phase5-visible] exit $Label code=$CellExit" | Tee-Object -FilePath $Log -Append
    return $CellExit
}

Write-Host "[visible] repo=$RepoRoot"
Write-Host "[visible] mode=$Mode"
Write-Host "[visible] muon_lr=$MuonLr"
Write-Host "[visible] seed=$Seed"
Write-Host "[visible] resume_from=$ResumeFrom"
Write-Host "[visible] stage56_checkpoint_dir=$Stage56CheckpointDir"
Write-Host "[visible] stage58_checkpoint_root=$Stage58CheckpointRoot"
Write-Host "[visible] stage58_tiny_corpus=$TinyCorpus"
Write-Host "[visible] stage58_tiny_shard_dir=$TinyShardDir"
Write-Host "[visible] stage58_broad_shard_dir=$BroadShardDir"
Write-Host "[visible] stage58_mixture_shard_dir=$MixtureShardDir"
Write-Host "[visible] log=$Log"
Add-Content -LiteralPath $LauncherLog -Value "[visible] repo=$RepoRoot"
Add-Content -LiteralPath $LauncherLog -Value "[visible] mode=$Mode"
Add-Content -LiteralPath $LauncherLog -Value "[visible] muon_lr=$MuonLr"
Add-Content -LiteralPath $LauncherLog -Value "[visible] seed=$Seed"
Add-Content -LiteralPath $LauncherLog -Value "[visible] resume_from=$ResumeFrom"
Add-Content -LiteralPath $LauncherLog -Value "[visible] stage56_checkpoint_dir=$Stage56CheckpointDir"
Add-Content -LiteralPath $LauncherLog -Value "[visible] stage58_checkpoint_root=$Stage58CheckpointRoot"
Add-Content -LiteralPath $LauncherLog -Value "[visible] stage58_tiny_corpus=$TinyCorpus"
Add-Content -LiteralPath $LauncherLog -Value "[visible] stage58_tiny_shard_dir=$TinyShardDir"
Add-Content -LiteralPath $LauncherLog -Value "[visible] stage58_broad_shard_dir=$BroadShardDir"
Add-Content -LiteralPath $LauncherLog -Value "[visible] stage58_mixture_shard_dir=$MixtureShardDir"
Add-Content -LiteralPath $LauncherLog -Value "[visible] log=$Log"

if ($Mode -eq "stage56-prep") {
    $RunArgs = New-Stage56PrepArgs
    $ExitCode = Invoke-VisiblePython -Label "stage56_prep_text8_shards" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

Assert-Text8Prepared

if ($Mode -eq "stage56-smoke") {
    $RunArgs = New-Stage56CellArgs -CellBudget 20 -CellSeed $Seed -CellResumeFrom $ResumeFrom -Smoke
    $ExitCode = Invoke-VisiblePython -Label "stage56_smoke_broadchar_85m_b20_seed$Seed" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage56-cell") {
    $RunArgs = New-Stage56CellArgs -CellBudget $Budget -CellSeed $Seed -CellResumeFrom $ResumeFrom
    $ExitCode = Invoke-VisiblePython -Label "stage56_broadchar_85m_b${Budget}_seed${Seed}" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage58-prep") {
    Assert-Stage58TinyPrepared
    $RunArgs = New-Stage58PrepArgs
    $ExitCode = Invoke-VisiblePython -Label "stage58_prep_mixture_shards" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage58-throughput") {
    Assert-Stage58CommonPrepared
    Assert-FreeSpaceForRun
    Assert-Stage58CheckpointWriteReady -CheckpointDirName "stage58_dev_throughput_checkpoints"
    $RunArgs = New-Stage58CellArgs -CellLabel "throughput" -CellCorpus ".\experiments\tiny_language_lab\corpus\text8_char_seed.txt" -CellShardDir $BroadShardDir -CheckpointDirName "stage58_dev_throughput_checkpoints" -CellBudget $Budget -CellSeed $Seed -NoCheckpointEvery
    $ExitCode = Invoke-VisiblePython -Label "stage58_throughput_85m_b${Budget}_seed${Seed}" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage58-cold") {
    Assert-Stage58CommonPrepared
    Assert-FreeSpaceForRun
    Assert-Stage58CheckpointWriteReady -CheckpointDirName "stage58_dev_cold_checkpoints"
    $RunArgs = New-Stage58CellArgs -CellLabel "cold" -CellCorpus ".\experiments\tiny_language_lab\corpus\text8_char_seed.txt" -CellShardDir $BroadShardDir -CheckpointDirName "stage58_dev_cold_checkpoints" -CellBudget $Budget -CellSeed $Seed -CellResumeFrom $ResumeFrom
    $ExitCode = Invoke-VisiblePython -Label "stage58_cold_85m_b${Budget}_seed${Seed}" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage58-curriculum-phase1") {
    Assert-Stage58TinyPrepared
    Assert-FreeSpaceForRun
    Assert-Stage58CheckpointWriteReady -CheckpointDirName "stage58_dev_curriculum_checkpoints"
    $RunArgs = New-Stage58CellArgs -CellLabel "curriculum_phase1" -CellCorpus $TinyCorpus -CellShardDir $TinyShardDir -CheckpointDirName "stage58_dev_curriculum_checkpoints" -CellBudget $Phase1Steps -CellSeed $Seed
    $ExitCode = Invoke-VisiblePython -Label "stage58_curriculum_phase1_85m_b${Phase1Steps}_seed${Seed}" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage58-curriculum-phase2") {
    Assert-Stage58CommonPrepared
    Assert-FreeSpaceForRun
    Assert-Stage58CheckpointWriteReady -CheckpointDirName "stage58_dev_curriculum_checkpoints"
    if ($ResumeFrom -eq "") {
        throw "stage58-curriculum-phase2 requires -ResumeFrom pointing to the phase-1 checkpoint."
    }
    $ResumeName = Split-Path -Leaf $ResumeFrom
    $ResumeFromPhase1 = $ResumeName -like "*curriculum_phase1*"
    if ($ResumeFromPhase1) {
        $Phase2Steps = $Budget - $Phase1Steps
        if ($Phase2Steps -le 0) {
            throw "Budget must be larger than Phase1Steps for curriculum phase 2."
        }
        $RunArgs = New-Stage58CellArgs -CellLabel "curriculum_phase2" -CellCorpus ".\experiments\tiny_language_lab\corpus\text8_char_seed.txt" -CellShardDir $BroadShardDir -CheckpointDirName "stage58_dev_curriculum_checkpoints" -CellBudget $Phase2Steps -CellSeed $Seed -CellBudgetLabel ([string]$Budget) -CellResumeFrom $ResumeFrom -AdditionalResumeSteps
    } else {
        $RunArgs = New-Stage58CellArgs -CellLabel "curriculum_phase2" -CellCorpus ".\experiments\tiny_language_lab\corpus\text8_char_seed.txt" -CellShardDir $BroadShardDir -CheckpointDirName "stage58_dev_curriculum_checkpoints" -CellBudget $Budget -CellSeed $Seed -CellBudgetLabel ([string]$Budget) -CellResumeFrom $ResumeFrom
    }
    $ExitCode = Invoke-VisiblePython -Label "stage58_curriculum_phase2_85m_b${Budget}_seed${Seed}" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}

if ($Mode -eq "stage58-mixture") {
    Assert-Stage58TinyPrepared
    if (-not $DryRun -and -not (Test-Path -LiteralPath $MixtureShardDir)) {
        throw "Missing mixture shard directory at $MixtureShardDir. Run -Mode stage58-prep first."
    }
    Assert-FreeSpaceForRun
    Assert-Stage58CheckpointWriteReady -CheckpointDirName "stage58_dev_mixture_checkpoints"
    $RunArgs = New-Stage58CellArgs -CellLabel "mixture" -CellCorpus ".\experiments\tiny_language_lab\corpus\text8_char_seed.txt" -CellShardDir $MixtureShardDir -CheckpointDirName "stage58_dev_mixture_checkpoints" -CellBudget $Budget -CellSeed $Seed
    $ExitCode = Invoke-VisiblePython -Label "stage58_mixture_85m_b${Budget}_seed${Seed}" -RunArgs $RunArgs
    if ($KeepOpen) {
        Read-Host "Run finished. Press Enter to close this window"
    }
    exit $ExitCode
}
