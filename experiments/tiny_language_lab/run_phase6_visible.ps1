param(
    [ValidateSet(
        "stage59-smoke-prep",
        "stage59-proxy-smoke",
        "stage59-build-mixtures",
        "stage59-proxy-sweep",
        "stage59-fit",
        "stage59-build-w10-85m",
        "stage59-indicative-eval",
        "stage59-retention-baselines",
        "stage59-part2-arm",
        "stage59-part2-eval",
        "stage59-verdict",
        "stage59-seed7-cold-arm",
        "stage59-seed7-cold-eval",
        "stage59-seed7-majority-verdict"
    )]
    [string]$Mode = "stage59-proxy-smoke",
    [int]$Budget = 200,
    [int]$Seed = 11,
    [int64]$TotalChars = 0,
    [int]$SmokeTotalChars = 2000000,
    [string]$ResumeFrom = "",
    [switch]$AllowSeed7Escalation,
    [switch]$KeepOpen
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$RunDir = Join-Path $ScriptDir "runs"
$CorpusDir = Join-Path $ScriptDir "corpus"
$Text8Corpus = Join-Path $CorpusDir "text8_char_seed.txt"
$BroadShardDir = Join-Path $CorpusDir "text8_char_shards"
$TinyShardDir = Join-Path $CorpusDir "tinystories_char_shards_500mb"
$RetentionCorpus = Join-Path $TinyShardDir "val.txt"
$SmokeMixDir = Join-Path $CorpusDir "stage59_mix_smoke_w010"
$Part2MixDir = Join-Path $CorpusDir "stage59_mix_w010_85m"
$UnionVocab = Join-Path $CorpusDir "phase5_union_vocab.txt"
$Part2CheckpointDir = "C:\cassandra_runs\stage59_mixture_w10_checkpoints"
$Seed7ColdCheckpointDir = "C:\cassandra_runs\stage59_seed7_cold_checkpoints"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunLog = Join-Path $RunDir ("phase6_{0}_{1}.log" -f $Mode, $Timestamp)
$LauncherLog = Join-Path $RunDir ("phase6_{0}_{1}_launcher.log" -f $Mode, $Timestamp)
$KeepAwakeLog = Join-Path $RunDir ("keep_awake_{0}_{1}.log" -f $Mode, $Timestamp)

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
Set-Location $RepoRoot

function Write-LauncherLog {
    param([string]$Message)
    Write-Host $Message
    Add-Content -LiteralPath $LauncherLog -Value $Message
}

function Assert-FreeSpaceForRun {
    $FreeGiB = (Get-PSDrive -Name C).Free / 1GB
    if ($FreeGiB -lt 15) {
        throw ("C: free space is {0:N2} GiB, below the 15 GiB launch gate." -f $FreeGiB)
    }
    Write-LauncherLog ("[visible] c_free_gib={0:N2}" -f $FreeGiB)
}

function Assert-CommonInputs {
    foreach ($Path in @($Text8Corpus, $BroadShardDir, $TinyShardDir, $RetentionCorpus)) {
        if (-not (Test-Path -LiteralPath $Path)) {
            throw "Missing required Stage 59 input: $Path"
        }
    }
}

function Invoke-VisiblePython {
    param(
        [string]$Label,
        [string[]]$RunArgs
    )
    $CommandText = "python " + ($RunArgs -join " ")
    Write-LauncherLog "[visible] command=$CommandText"
    "[phase6-visible] start $Label" | Tee-Object -FilePath $RunLog -Append
    python @RunArgs 2>&1 | Tee-Object -FilePath $RunLog -Append
    $ExitCode = $LASTEXITCODE
    "[phase6-visible] exit $Label code=$ExitCode" | Tee-Object -FilePath $RunLog -Append
    if ($ExitCode -ne 0) {
        throw "$Label exited with code $ExitCode"
    }
}

function Assert-FreshMixtureTarget {
    param(
        [string]$OutDir,
        [string]$MetaPath
    )
    if ((Test-Path -LiteralPath $OutDir) -or (Test-Path -LiteralPath $MetaPath)) {
        throw "Refusing to overwrite existing Stage 59 mixture evidence: $OutDir or $MetaPath"
    }
}

function Invoke-MixtureBuild {
    param(
        [string]$Label,
        [int]$TinyWeight,
        [int]$BroadWeight,
        [int64]$Chars,
        [string]$OutDir,
        [string]$MetaPath
    )
    Assert-FreshMixtureTarget -OutDir $OutDir -MetaPath $MetaPath
    $RunArgs = @(
        ".\experiments\tiny_language_lab\make_mixture_shards.py",
        "--tiny-dir", $TinyShardDir,
        "--broad-dir", $BroadShardDir,
        "--tiny-weight", [string]$TinyWeight,
        "--broad-weight", [string]$BroadWeight,
        "--total-chars", [string]$Chars,
        "--out-dir", $OutDir,
        "--metadata-out", $MetaPath
    )
    Invoke-VisiblePython -Label $Label -RunArgs $RunArgs
    $Meta = Get-Content -Raw -LiteralPath $MetaPath | ConvertFrom-Json
    if ([int64]$Meta.written_chars -ne $Chars) {
        throw "$Label wrote $($Meta.written_chars) chars, expected $Chars"
    }
    if ([int]$Meta.tiny_reader_wraps -ne 0 -or [int]$Meta.broad_reader_wraps -ne 0) {
        throw "$Label wrapped a source reader; increase source data or lower --total-chars"
    }
    Write-LauncherLog "[verified] label=$Label ratio=$($Meta.ratio) chars=$($Meta.written_chars) tiny_fraction=$($Meta.tiny_fraction) sha256=$($Meta.sha256)"
}

function Assert-Stage59FitGate {
    $FitOut = Join-Path $RunDir "stage59_mixing_law_fit.json"
    if (-not (Test-Path -LiteralPath $FitOut)) {
        throw "The frozen Stage 59 fit artifact must exist before any Part 2 action: $FitOut"
    }
    $Fit = Get-Content -Raw -LiteralPath $FitOut | ConvertFrom-Json
    if ([string]::IsNullOrWhiteSpace([string]$Fit.primary.kind)) {
        throw "Stage 59 fit artifact has no registered primary family"
    }
    if ($null -eq $Fit.predictions.predicted_85m_cost_at_w010.delta_bits_per_char -or $null -eq $Fit.predictions.w_star.dose) {
        throw "Stage 59 fit artifact is missing the two pre-registered Part 2 predictions"
    }
    Write-LauncherLog "[gate] fit_frozen=$FitOut created_utc=$($Fit.created_utc) primary=$($Fit.primary.kind) predicted_w010_cost_bits=$($Fit.predictions.predicted_85m_cost_at_w010.delta_bits_per_char) predicted_w_star=$($Fit.predictions.w_star.dose)"
}

function Assert-Seed7EscalationGate {
    if ($Seed -ne 7 -or -not $AllowSeed7Escalation) {
        throw "The registered seed-7 contingency requires -Seed 7 -AllowSeed7Escalation"
    }
    $VerdictPath = Join-Path $RunDir "stage59_verdict.json"
    if (-not (Test-Path -LiteralPath $VerdictPath)) {
        throw "Seed-7 escalation requires the preserved two-seed INCONCLUSIVE verdict artifact"
    }
    $Verdict = Get-Content -Raw -LiteralPath $VerdictPath | ConvertFrom-Json
    if ($Verdict.verdict -ne "INCONCLUSIVE") {
        throw "Seed-7 escalation is unauthorized because the preserved verdict is $($Verdict.verdict), not INCONCLUSIVE"
    }
    Write-LauncherLog "[gate] seed7_escalation_explicit=true source_verdict=$VerdictPath"
}

function Assert-Part2Seed {
    if ($Seed -in @(11, 19)) {
        return
    }
    if ($Seed -eq 7) {
        Assert-Seed7EscalationGate
        return
    }
    throw "Stage 59 Part 2 primary arms are seeds 11 and 19. Seed 7 is the only registered escalation seed."
}

function Assert-Part2Mixture {
    $MetaPath = Join-Path $CorpusDir "stage59_mix_w010_85m.meta.json"
    if (-not (Test-Path -LiteralPath $Part2MixDir) -or -not (Test-Path -LiteralPath $MetaPath)) {
        throw "Missing the dedicated Stage 59 Part 2 w=0.10 corpus or metadata. Build it only after the fit gate."
    }
    $Meta = Get-Content -Raw -LiteralPath $MetaPath | ConvertFrom-Json
    $RequiredChars = 20000L * 4096L
    if ([math]::Abs([double]$Meta.tiny_fraction - 0.10) -gt 1e-12) {
        throw "Part 2 mixture metadata is not the registered w=0.10 dose"
    }
    if ([int64]$Meta.written_chars -lt $RequiredChars) {
        throw "Part 2 mixture has $($Meta.written_chars) chars but a 20,000-step arm consumes $RequiredChars"
    }
    if ([int]$Meta.tiny_reader_wraps -ne 0 -or [int]$Meta.broad_reader_wraps -ne 0) {
        throw "Part 2 mixture metadata records source wrapping"
    }
    Write-LauncherLog "[gate] part2_mixture=$Part2MixDir chars=$($Meta.written_chars) ratio=$($Meta.ratio) sha256=$($Meta.sha256)"
}

function Assert-CheckpointWriteReady {
    param([string]$CheckpointDir)
    $ResolvedRoot = [IO.Path]::GetFullPath("C:\cassandra_runs")
    $ResolvedTarget = [IO.Path]::GetFullPath($CheckpointDir)
    if (-not $ResolvedTarget.StartsWith($ResolvedRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing checkpoint target outside C:\cassandra_runs: $ResolvedTarget"
    }
    New-Item -ItemType Directory -Force -Path $ResolvedTarget | Out-Null
    $ProbeStem = "phase6_checkpoint_probe_{0}" -f ([guid]::NewGuid().ToString("N"))
    $ProbeTemp = Join-Path $ResolvedTarget "$ProbeStem.pt.tmp"
    $ProbeFinal = Join-Path $ResolvedTarget "$ProbeStem.pt"
    try {
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
        Write-LauncherLog "[gate] checkpoint_write_probe=passed path=$ResolvedTarget"
    }
    finally {
        if (Test-Path -LiteralPath $ProbeTemp) {
            Remove-Item -LiteralPath $ProbeTemp -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path -LiteralPath $ProbeFinal) {
            Remove-Item -LiteralPath $ProbeFinal -Force -ErrorAction SilentlyContinue
        }
    }
}
"[visible] launcher_start=$(Get-Date -Format o)" | Set-Content -LiteralPath $LauncherLog
Write-LauncherLog "[visible] repo=$RepoRoot"
Write-LauncherLog "[visible] mode=$Mode"
Write-LauncherLog "[visible] run_log=$RunLog"
Assert-FreeSpaceForRun
Assert-CommonInputs

$KeepArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $ScriptDir "keep_awake_while_pid.ps1"),
    "-WatchPid", [string]$PID,
    "-LogPath", $KeepAwakeLog
)
Start-Process -FilePath "powershell.exe" -ArgumentList $KeepArgs -WindowStyle Hidden | Out-Null
Write-LauncherLog "[visible] keep_awake_log=$KeepAwakeLog watch_pid=$PID"

if ($Mode -eq "stage59-smoke-prep") {
    if ($SmokeTotalChars -le 0) {
        throw "-SmokeTotalChars must be positive"
    }
    $MetaPath = Join-Path $CorpusDir "stage59_mix_smoke_w010.meta.json"
    Invoke-MixtureBuild -Label "stage59_smoke_mix_w010" -TinyWeight 1 -BroadWeight 9 -Chars $SmokeTotalChars -OutDir $SmokeMixDir -MetaPath $MetaPath
}
elseif ($Mode -eq "stage59-proxy-smoke") {
    if (-not (Test-Path -LiteralPath $SmokeMixDir)) {
        throw "Missing smoke mixture directory $SmokeMixDir. Run stage59-smoke-prep first."
    }
    $RunArgs = @(
        ".\experiments\tiny_language_lab\cassandra_compare.py",
        "--corpus", $Text8Corpus,
        "--device", "cuda",
        "--steps", [string]$Budget,
        "--eval-interval", [string]$Budget,
        "--log-every", "50",
        "--seeds", "7",
        "--configs", "stage59_proxy_random_full",
        "--train-shard-dir", $SmokeMixDir,
        "--retention-eval-corpus", $RetentionCorpus,
        "--mixture-dose", "0.10",
        "--stream-train-eval-chars", "200000",
        "--prompt", "the history of ",
        "--max-new-tokens", "80",
        "--out", ".\experiments\tiny_language_lab\runs\stage59_proxy_throughput_smoke.jsonl",
        "--summary", ".\experiments\tiny_language_lab\runs\stage59_proxy_throughput_smoke.md",
        "--title", "Stage 59 H025 proxy throughput smoke"
    )
    Invoke-VisiblePython -Label "stage59_proxy_throughput_smoke" -RunArgs $RunArgs
}
elseif ($Mode -eq "stage59-build-mixtures") {
    if ($TotalChars -le 0) {
        throw "-TotalChars must be positive and must include registered sweep-budget headroom"
    }
    $Doses = @(
        [PSCustomObject]@{ Label = "w005"; Tiny = 1; Broad = 19 },
        [PSCustomObject]@{ Label = "w010"; Tiny = 1; Broad = 9 },
        [PSCustomObject]@{ Label = "w020"; Tiny = 1; Broad = 4 },
        [PSCustomObject]@{ Label = "w030"; Tiny = 3; Broad = 7 },
        [PSCustomObject]@{ Label = "w050"; Tiny = 1; Broad = 1 }
    )
    foreach ($Dose in $Doses) {
        $OutDir = Join-Path $CorpusDir ("stage59_mix_{0}" -f $Dose.Label)
        $MetaPath = Join-Path $CorpusDir ("stage59_mix_{0}.meta.json" -f $Dose.Label)
        Invoke-MixtureBuild -Label ("stage59_mix_{0}" -f $Dose.Label) -TinyWeight $Dose.Tiny -BroadWeight $Dose.Broad -Chars $TotalChars -OutDir $OutDir -MetaPath $MetaPath
    }
}
elseif ($Mode -eq "stage59-proxy-sweep") {
    if ($Budget -le 0) {
        throw "-Budget must be positive"
    }
    $Out = Join-Path $RunDir "stage59_proxy_sweep.jsonl"
    $Summary = Join-Path $RunDir "stage59_proxy_sweep.md"
    if ((Test-Path -LiteralPath $Out) -or (Test-Path -LiteralPath $Summary)) {
        throw "Refusing to overwrite existing Stage 59 proxy sweep artifacts. Preserve them and use suffixed recovery artifacts."
    }
    $Doses = @(
        [PSCustomObject]@{ Dose = 0.00; Dir = $BroadShardDir },
        [PSCustomObject]@{ Dose = 0.05; Dir = (Join-Path $CorpusDir "stage59_mix_w005") },
        [PSCustomObject]@{ Dose = 0.10; Dir = (Join-Path $CorpusDir "stage59_mix_w010") },
        [PSCustomObject]@{ Dose = 0.20; Dir = (Join-Path $CorpusDir "stage59_mix_w020") },
        [PSCustomObject]@{ Dose = 0.30; Dir = (Join-Path $CorpusDir "stage59_mix_w030") },
        [PSCustomObject]@{ Dose = 0.50; Dir = (Join-Path $CorpusDir "stage59_mix_w050") }
    )
    foreach ($Dose in $Doses) {
        if (-not (Test-Path -LiteralPath $Dose.Dir)) {
            throw "Missing registered Stage 59 dose directory: $($Dose.Dir)"
        }
    }
    Write-LauncherLog "[registered] proxy_steps_per_run=$Budget"
    Write-LauncherLog "[registered] chars_per_step=4096"
    Write-LauncherLog "[registered] runs=18 doses=0.00,0.05,0.10,0.20,0.30,0.50 seeds=7,11,19"
    Write-LauncherLog "[registered] provisional_retention_bound=mean_w030_retention_bits_plus_0.5"
    for ($Index = 0; $Index -lt $Doses.Count; $Index++) {
        $Dose = $Doses[$Index]
        $DoseText = $Dose.Dose.ToString("0.00", [Globalization.CultureInfo]::InvariantCulture)
        $RunArgs = @(
            ".\experiments\tiny_language_lab\cassandra_compare.py",
            "--corpus", $Text8Corpus,
            "--device", "cuda",
            "--steps", [string]$Budget,
            "--eval-interval", [string]$Budget,
            "--log-every", "100",
            "--seeds", "7", "11", "19",
            "--configs", "stage59_proxy_random_full",
            "--train-shard-dir", $Dose.Dir,
            "--retention-eval-corpus", $RetentionCorpus,
            "--mixture-dose", $DoseText,
            "--stream-train-eval-chars", "200000",
            "--prompt", "the history of ",
            "--max-new-tokens", "80",
            "--out", $Out,
            "--summary", $Summary,
            "--title", "Stage 59 H025 proxy dose sweep"
        )
        if ($Index -gt 0) {
            $RunArgs += "--append"
        }
        Invoke-VisiblePython -Label ("stage59_proxy_dose_{0}" -f $DoseText) -RunArgs $RunArgs
    }
}
elseif ($Mode -eq "stage59-fit") {
    $Sweep = Join-Path $RunDir "stage59_proxy_sweep.jsonl"
    $FitOut = Join-Path $RunDir "stage59_mixing_law_fit.json"
    $FitSummary = Join-Path $RunDir "stage59_mixing_law_fit.md"
    if (-not (Test-Path -LiteralPath $Sweep)) {
        throw "Missing completed proxy sweep: $Sweep"
    }
    if ((Test-Path -LiteralPath $FitOut) -or (Test-Path -LiteralPath $FitSummary)) {
        throw "Refusing to overwrite existing Stage 59 mixing-law fit artifacts"
    }
    $RunArgs = @(
        ".\experiments\tiny_language_lab\make_mixing_law_fit.py",
        "--input", $Sweep,
        "--out", $FitOut,
        "--summary", $FitSummary
    )
    Invoke-VisiblePython -Label "stage59_mixing_law_fit" -RunArgs $RunArgs
}
elseif ($Mode -eq "stage59-build-w10-85m") {
    Assert-Stage59FitGate
    if ($TotalChars -le 0) {
        throw "-TotalChars must be positive and include Part 2 headroom"
    }
    $OutDir = Join-Path $CorpusDir "stage59_mix_w010_85m"
    $MetaPath = Join-Path $CorpusDir "stage59_mix_w010_85m.meta.json"
    Invoke-MixtureBuild -Label "stage59_mix_w010_85m" -TinyWeight 1 -BroadWeight 9 -Chars $TotalChars -OutDir $OutDir -MetaPath $MetaPath
}
elseif ($Mode -eq "stage59-indicative-eval") {
    Assert-Stage59FitGate
    $MixtureCheckpoint = "C:\cassandra_runs\stage58_dev_mixture_checkpoints\stage58_dev_mixture_85m_b42000_seed7_random_full_seed7_step020000.pt"
    $ColdCheckpoint = "C:\cassandra_runs\stage58_dev_cold_checkpoints\stage58_dev_cold_85m_b42000_seed7_random_full_seed7_step020000.pt"
    foreach ($Checkpoint in @($MixtureCheckpoint, $ColdCheckpoint)) {
        if (-not (Test-Path -LiteralPath $Checkpoint)) {
            throw "Missing Stage 58 indicative checkpoint: $Checkpoint"
        }
    }
    $MixtureStem = Join-Path $RunDir "stage59_indicative_w298_step20000_text8_test"
    $ColdStem = Join-Path $RunDir "stage59_indicative_cold_step20000_text8_test"
    foreach ($Path in @("$MixtureStem.json", "$MixtureStem.md", "$ColdStem.json", "$ColdStem.md")) {
        if (Test-Path -LiteralPath $Path) {
            throw "Refusing to overwrite existing indicative evaluation evidence: $Path"
        }
    }
    $MixtureArgs = @(
        ".\experiments\tiny_language_lab\eval_text8.py",
        "--split", "test",
        "--device", "cuda",
        "--checkpoint", $MixtureCheckpoint,
        "--checkpoint-name", "stage58_mixture_w298_step20000_seed7",
        "--out-stem", $MixtureStem
    )
    Invoke-VisiblePython -Label "stage59_indicative_w298_step20000_text8" -RunArgs $MixtureArgs
    $ColdArgs = @(
        ".\experiments\tiny_language_lab\eval_text8.py",
        "--split", "test",
        "--device", "cuda",
        "--checkpoint", $ColdCheckpoint,
        "--checkpoint-name", "stage58_cold_step20000_of42000_seed7",
        "--out-stem", $ColdStem
    )
    Invoke-VisiblePython -Label "stage59_indicative_cold_step20000_text8" -RunArgs $ColdArgs
    $MixtureReport = Get-Content -Raw -LiteralPath "$MixtureStem.json" | ConvertFrom-Json
    $ColdReport = Get-Content -Raw -LiteralPath "$ColdStem.json" | ConvertFrom-Json
    $MixtureBits = [double](($MixtureReport.models.PSObject.Properties | Select-Object -First 1).Value.result.bits_per_char)
    $ColdBits = [double](($ColdReport.models.PSObject.Properties | Select-Object -First 1).Value.result.bits_per_char)
    $Toll = $MixtureBits - $ColdBits
    $Anchor = 0.027977
    Write-LauncherLog ("[indicative] paired_mid_cosine=true dose=0.298 step=20000 mixture_bits={0:F6} cold_bits={1:F6} toll_bits={2:+0.000000;-0.000000;0.000000} anchor_bits={3:+0.000000;-0.000000;0.000000} anchor_difference={4:+0.000000;-0.000000;0.000000} role=no_decision_line" -f $MixtureBits, $ColdBits, $Toll, $Anchor, ($Toll - $Anchor))
}
elseif ($Mode -eq "stage59-retention-baselines") {
    Assert-Stage59FitGate
    $Checkpoint11 = "C:\cassandra_runs\stage58_dev_cold_checkpoints\stage58_dev_cold_85m_b20000_seed11_random_full_seed11.pt"
    $Checkpoint19 = "C:\cassandra_runs\stage58_dev_cold_checkpoints\stage58_dev_cold_85m_b20000_seed19_random_full_seed19.pt"
    foreach ($Checkpoint in @($Checkpoint11, $Checkpoint19)) {
        if (-not (Test-Path -LiteralPath $Checkpoint)) {
            throw "Missing Stage 58 COLD retention baseline checkpoint: $Checkpoint"
        }
    }
    $OutStem = Join-Path $RunDir "stage59_cold_b20000_retention_baselines"
    if ((Test-Path -LiteralPath "$OutStem.json") -or (Test-Path -LiteralPath "$OutStem.md")) {
        throw "Refusing to overwrite existing retention baseline artifacts"
    }
    $RunArgs = @(
        ".\experiments\tiny_language_lab\eval_tinystories_retention.py",
        "--device", "cuda",
        "--corpus", $RetentionCorpus,
        "--checkpoint", $Checkpoint11,
        "--checkpoint", $Checkpoint19,
        "--out-stem", $OutStem
    )
    Invoke-VisiblePython -Label "stage59_cold_b20000_retention_baselines" -RunArgs $RunArgs
}

elseif ($Mode -eq "stage59-part2-arm") {
    Assert-Stage59FitGate
    Assert-Part2Seed
    Assert-Part2Mixture
    if ($Budget -ne 20000) {
        throw "H025 pins every Stage 59 Part 2 arm to exactly 20,000 steps"
    }
    if (-not (Test-Path -LiteralPath $UnionVocab)) {
        throw "Missing registered 33-character union vocabulary: $UnionVocab"
    }
    $OutStem = Join-Path $RunDir ("stage59_mixture_w10_85m_b20000_seed{0}" -f $Seed)
    if ((Test-Path -LiteralPath "$OutStem.jsonl") -or (Test-Path -LiteralPath "$OutStem.md")) {
        throw "Refusing to overwrite existing Part 2 arm evidence for seed $Seed"
    }
    $CheckpointPrefix = "stage59_mixture_w10_85m_b20000_seed{0}_random_full_seed{0}" -f $Seed
    $ExistingCheckpoints = @(Get-ChildItem -LiteralPath $Part2CheckpointDir -Filter "$CheckpointPrefix*.pt" -ErrorAction SilentlyContinue)
    if ($ResumeFrom -eq "" -and $ExistingCheckpoints.Count -gt 0) {
        throw "Seed $Seed already has checkpoint evidence. Resume explicitly or preserve it and use a suffixed recovery plan."
    }
    if ($ResumeFrom -ne "") {
        if (-not (Test-Path -LiteralPath $ResumeFrom)) {
            throw "Resume checkpoint does not exist: $ResumeFrom"
        }
        if ((Split-Path -Leaf $ResumeFrom) -notlike "$CheckpointPrefix*.pt") {
            throw "Resume checkpoint does not match the registered seed-$Seed Part 2 lineage"
        }
        Write-LauncherLog "[gate] resume_from=$ResumeFrom"
    }
    Assert-CheckpointWriteReady -CheckpointDir $Part2CheckpointDir
    $RunArgs = @(
        ".\experiments\tiny_language_lab\cassandra_compare.py",
        "--corpus", $Text8Corpus,
        "--device", "cuda",
        "--steps", "20000",
        "--eval-interval", "5000",
        "--log-every", "1000",
        "--seeds", [string]$Seed,
        "--configs", "random_full",
        "--block-size", "256",
        "--batch-size", "8",
        "--grad-accum-steps", "2",
        "--pos-encoding", "rope",
        "--activation-checkpoint",
        "--optimizer", "muon",
        "--muon-lr", "0.01",
        "--precision", "fp32",
        "--lr-schedule", "cosine",
        "--lr-final-frac", "0.1",
        "--lr-total-steps", "20000",
        "--vocab-chars-file", $UnionVocab,
        "--eval-mode", "sampled",
        "--eval-batches", "16",
        "--no-copy-train-marker",
        "--prompt", "the history of ",
        "--max-new-tokens", "240",
        "--n-layer", "12",
        "--n-head", "12",
        "--n-embd", "768",
        "--train-shard-dir", $Part2MixDir,
        "--stream-train-eval-chars", "200000",
        "--val-fraction", "0.05263157894736842",
        "--retention-eval-corpus", $RetentionCorpus,
        "--checkpoint-dir", $Part2CheckpointDir,
        "--checkpoint-every", "5000",
        "--checkpoint-keep", "0",
        "--out", "$OutStem.jsonl",
        "--summary", "$OutStem.md",
        "--title", "Stage 59 H025 MIXTURE w=0.10 85M 20000-step seed $Seed"
    )
    if ($ResumeFrom -ne "") {
        $RunArgs += @("--resume-from", $ResumeFrom)
    }
    Invoke-VisiblePython -Label ("stage59_mixture_w10_85m_b20000_seed{0}" -f $Seed) -RunArgs $RunArgs
    $Rows = @(Get-Content -LiteralPath "$OutStem.jsonl" | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_ | ConvertFrom-Json })
    if ($Rows.Count -ne 1) {
        throw "Part 2 seed $Seed output must contain exactly one decision row"
    }
    $Row = $Rows[0]
    if ([int]$Row.seed -ne $Seed -or [int]$Row.steps -ne 20000 -or [int64]$Row.parameters -ne 85106721) {
        throw "Part 2 seed $Seed row failed the seed/step/parameter verification"
    }
    if ($Row.eval_mode -ne "sampled" -or [int]$Row.eval_batches -ne 16) {
        throw "Part 2 seed $Seed row failed the sampled monitoring convention verification"
    }
    Write-LauncherLog "[verified] part2_seed=$Seed parameters=$($Row.parameters) final_sampled_broad_nll=$($Row.val_nll) final_sampled_broad_bits=$($Row.val_bits_per_char)"
}
elseif ($Mode -eq "stage59-part2-eval") {
    Assert-Stage59FitGate
    Assert-Part2Seed
    $CheckpointPrefix = "stage59_mixture_w10_85m_b20000_seed{0}_random_full_seed{0}" -f $Seed
    $FinalCheckpoint = Join-Path $Part2CheckpointDir "$CheckpointPrefix.pt"
    $StepCheckpoints = @(5000, 10000, 15000, 20000 | ForEach-Object { Join-Path $Part2CheckpointDir ("{0}_step{1:D6}.pt" -f $CheckpointPrefix, $_) })
    foreach ($Checkpoint in @($FinalCheckpoint) + $StepCheckpoints) {
        if (-not (Test-Path -LiteralPath $Checkpoint)) {
            throw "Missing required Part 2 seed-$Seed evaluation checkpoint: $Checkpoint"
        }
    }
    $Text8Stem = Join-Path $RunDir ("stage59_mixture_w10_85m_b20000_seed{0}_text8_test" -f $Seed)
    $RetentionStem = Join-Path $RunDir ("stage59_mixture_w10_85m_b20000_seed{0}_retention" -f $Seed)
    foreach ($Path in @("$Text8Stem.json", "$Text8Stem.md", "$RetentionStem.json", "$RetentionStem.md")) {
        if (Test-Path -LiteralPath $Path) {
            throw "Refusing to overwrite existing Part 2 evaluation evidence: $Path"
        }
    }
    $Text8Args = @(
        ".\experiments\tiny_language_lab\eval_text8.py",
        "--split", "test",
        "--device", "cuda",
        "--checkpoint", $FinalCheckpoint,
        "--checkpoint-name", ("stage59_mixture_w10_85m_b20000_seed{0}" -f $Seed),
        "--out-stem", $Text8Stem
    )
    Invoke-VisiblePython -Label ("stage59_part2_text8_test_seed{0}" -f $Seed) -RunArgs $Text8Args
    $RetentionArgs = @(
        ".\experiments\tiny_language_lab\eval_tinystories_retention.py",
        "--device", "cuda",
        "--corpus", $RetentionCorpus,
        "--checkpoint", $FinalCheckpoint
    )
    foreach ($Checkpoint in $StepCheckpoints) {
        $RetentionArgs += @("--checkpoint", $Checkpoint)
    }
    $RetentionArgs += @("--out-stem", $RetentionStem)
    Invoke-VisiblePython -Label ("stage59_part2_retention_seed{0}" -f $Seed) -RunArgs $RetentionArgs
}
elseif ($Mode -eq "stage59-verdict") {
    Assert-Stage59FitGate
    $VerdictOut = Join-Path $RunDir "stage59_verdict.json"
    $VerdictSummary = Join-Path $RunDir "stage59_verdict.md"
    if ((Test-Path -LiteralPath $VerdictOut) -or (Test-Path -LiteralPath $VerdictSummary)) {
        throw "Refusing to overwrite existing Stage 59 verdict evidence"
    }
    $RunArgs = @(
        ".\experiments\tiny_language_lab\make_stage59_verdict.py",
        "--out", $VerdictOut,
        "--summary", $VerdictSummary
    )
    Invoke-VisiblePython -Label "stage59_h025_verdict" -RunArgs $RunArgs
}
elseif ($Mode -eq "stage59-seed7-cold-arm") {
    Assert-Stage59FitGate
    Assert-Seed7EscalationGate
    if ($Budget -ne 20000) {
        throw "H025 pins the seed-7 escalation COLD arm to exactly 20,000 steps"
    }
    if (-not (Test-Path -LiteralPath $UnionVocab)) {
        throw "Missing registered 33-character union vocabulary: $UnionVocab"
    }
    $OutStem = Join-Path $RunDir "stage59_cold_85m_b20000_seed7"
    if ((Test-Path -LiteralPath "$OutStem.jsonl") -or (Test-Path -LiteralPath "$OutStem.md")) {
        throw "Refusing to overwrite existing seed-7 COLD escalation evidence"
    }
    $CheckpointPrefix = "stage59_cold_85m_b20000_seed7_random_full_seed7"
    $ExistingCheckpoints = @(Get-ChildItem -LiteralPath $Seed7ColdCheckpointDir -Filter "$CheckpointPrefix*.pt" -ErrorAction SilentlyContinue)
    if ($ResumeFrom -eq "" -and $ExistingCheckpoints.Count -gt 0) {
        throw "Seed-7 COLD already has checkpoint evidence. Resume explicitly or preserve it and use a suffixed recovery plan."
    }
    if ($ResumeFrom -ne "") {
        if (-not (Test-Path -LiteralPath $ResumeFrom)) {
            throw "Resume checkpoint does not exist: $ResumeFrom"
        }
        if ((Split-Path -Leaf $ResumeFrom) -notlike "$CheckpointPrefix*.pt") {
            throw "Resume checkpoint does not match the registered seed-7 COLD lineage"
        }
        Write-LauncherLog "[gate] resume_from=$ResumeFrom"
    }
    Assert-CheckpointWriteReady -CheckpointDir $Seed7ColdCheckpointDir
    $RunArgs = @(
        ".\experiments\tiny_language_lab\cassandra_compare.py",
        "--corpus", $Text8Corpus,
        "--device", "cuda",
        "--steps", "20000",
        "--eval-interval", "5000",
        "--log-every", "1000",
        "--seeds", "7",
        "--configs", "random_full",
        "--block-size", "256",
        "--batch-size", "8",
        "--grad-accum-steps", "2",
        "--pos-encoding", "rope",
        "--activation-checkpoint",
        "--optimizer", "muon",
        "--muon-lr", "0.01",
        "--precision", "fp32",
        "--lr-schedule", "cosine",
        "--lr-final-frac", "0.1",
        "--lr-total-steps", "20000",
        "--vocab-chars-file", $UnionVocab,
        "--eval-mode", "sampled",
        "--eval-batches", "16",
        "--no-copy-train-marker",
        "--prompt", "the history of ",
        "--max-new-tokens", "240",
        "--n-layer", "12",
        "--n-head", "12",
        "--n-embd", "768",
        "--train-shard-dir", $BroadShardDir,
        "--stream-train-eval-chars", "200000",
        "--val-fraction", "0.05263157894736842",
        "--retention-eval-corpus", $RetentionCorpus,
        "--checkpoint-dir", $Seed7ColdCheckpointDir,
        "--checkpoint-every", "5000",
        "--checkpoint-keep", "0",
        "--out", "$OutStem.jsonl",
        "--summary", "$OutStem.md",
        "--title", "Stage 59 H025 escalation COLD 85M 20000-step seed 7"
    )
    if ($ResumeFrom -ne "") {
        $RunArgs += @("--resume-from", $ResumeFrom)
    }
    Invoke-VisiblePython -Label "stage59_cold_85m_b20000_seed7" -RunArgs $RunArgs
    $Rows = @(Get-Content -LiteralPath "$OutStem.jsonl" | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_ | ConvertFrom-Json })
    if ($Rows.Count -ne 1) {
        throw "Seed-7 COLD output must contain exactly one decision row"
    }
    $Row = $Rows[0]
    if ([int]$Row.seed -ne 7 -or [int]$Row.steps -ne 20000 -or [int64]$Row.parameters -ne 85106721) {
        throw "Seed-7 COLD row failed the seed/step/parameter verification"
    }
    if ($Row.eval_mode -ne "sampled" -or [int]$Row.eval_batches -ne 16) {
        throw "Seed-7 COLD row failed the sampled monitoring convention verification"
    }
    Write-LauncherLog "[verified] escalation_arm=cold seed=7 parameters=$($Row.parameters) final_sampled_broad_nll=$($Row.val_nll) final_sampled_broad_bits=$($Row.val_bits_per_char)"
}
elseif ($Mode -eq "stage59-seed7-cold-eval") {
    Assert-Stage59FitGate
    Assert-Seed7EscalationGate
    $CheckpointPrefix = "stage59_cold_85m_b20000_seed7_random_full_seed7"
    $FinalCheckpoint = Join-Path $Seed7ColdCheckpointDir "$CheckpointPrefix.pt"
    $StepCheckpoints = @(5000, 10000, 15000, 20000 | ForEach-Object { Join-Path $Seed7ColdCheckpointDir ("{0}_step{1:D6}.pt" -f $CheckpointPrefix, $_) })
    foreach ($Checkpoint in @($FinalCheckpoint) + $StepCheckpoints) {
        if (-not (Test-Path -LiteralPath $Checkpoint)) {
            throw "Missing required seed-7 COLD evaluation checkpoint: $Checkpoint"
        }
    }
    $Text8Stem = Join-Path $RunDir "stage59_cold_85m_b20000_seed7_text8_test"
    $RetentionStem = Join-Path $RunDir "stage59_cold_85m_b20000_seed7_retention"
    foreach ($Path in @("$Text8Stem.json", "$Text8Stem.md", "$RetentionStem.json", "$RetentionStem.md")) {
        if (Test-Path -LiteralPath $Path) {
            throw "Refusing to overwrite existing seed-7 COLD evaluation evidence: $Path"
        }
    }
    $Text8Args = @(
        ".\experiments\tiny_language_lab\eval_text8.py",
        "--split", "test",
        "--device", "cuda",
        "--checkpoint", $FinalCheckpoint,
        "--checkpoint-name", "stage59_cold_85m_b20000_seed7",
        "--out-stem", $Text8Stem
    )
    Invoke-VisiblePython -Label "stage59_seed7_cold_text8_test" -RunArgs $Text8Args
    $RetentionArgs = @(
        ".\experiments\tiny_language_lab\eval_tinystories_retention.py",
        "--device", "cuda",
        "--corpus", $RetentionCorpus,
        "--checkpoint", $FinalCheckpoint
    )
    foreach ($Checkpoint in $StepCheckpoints) {
        $RetentionArgs += @("--checkpoint", $Checkpoint)
    }
    $RetentionArgs += @("--out-stem", $RetentionStem)
    Invoke-VisiblePython -Label "stage59_seed7_cold_retention" -RunArgs $RetentionArgs
}
elseif ($Mode -eq "stage59-seed7-majority-verdict") {
    Assert-Stage59FitGate
    Assert-Seed7EscalationGate
    $VerdictOut = Join-Path $RunDir "stage59_verdict_seed7_majority.json"
    $VerdictSummary = Join-Path $RunDir "stage59_verdict_seed7_majority.md"
    if ((Test-Path -LiteralPath $VerdictOut) -or (Test-Path -LiteralPath $VerdictSummary)) {
        throw "Refusing to overwrite existing seed-7 majority verdict evidence"
    }
    $RunArgs = @(
        ".\experiments\tiny_language_lab\make_stage59_verdict.py",
        "--include-seed7",
        "--out", $VerdictOut,
        "--summary", $VerdictSummary
    )
    Invoke-VisiblePython -Label "stage59_h025_seed7_majority_verdict" -RunArgs $RunArgs
}
Write-LauncherLog "[visible] launcher_end=$(Get-Date -Format o) status=success"
if ($KeepOpen) {
    Read-Host "Run finished. Press Enter to close this window"
}