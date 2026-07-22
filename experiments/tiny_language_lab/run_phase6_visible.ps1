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
        "stage59-seed7-majority-verdict",
        "stage60-circuit-matrix",
        "stage60-circuit-matrix-analysis",
        "stage61-arm-segment"
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
$DesktopRoot = Split-Path -Parent $RepoRoot
$MusahitNightlyCandidates = @(Get-ChildItem -LiteralPath $DesktopRoot -Directory | ForEach-Object {
    Join-Path $_.FullName "scripts/scheduling/run_nightly.ps1"
} | Where-Object { Test-Path -LiteralPath $_ })
if ($MusahitNightlyCandidates.Count -ne 1) {
    throw "Expected exactly one sibling MUSAHIT nightly launcher, found $($MusahitNightlyCandidates.Count)"
}
$MusahitNightly = $MusahitNightlyCandidates[0]
$MusahitSkipFlag = Join-Path (Split-Path -Parent $MusahitNightly) "SKIP_NEXT_RUN.flag"
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
    $Sweep = Join-Path $RunDir "stage59_proxy_sweep.jsonl"
    $FitOut = Join-Path $RunDir "stage59_mixing_law_fit.json"
    $FitSummary = Join-Path $RunDir "stage59_mixing_law_fit.md"
    foreach ($Path in @($Sweep, $FitOut, $FitSummary)) {
        if (-not (Test-Path -LiteralPath $Path)) {
            throw "The complete frozen Stage 59 sweep and fit artifacts must exist before any Part 2 action: $Path"
        }
        if ((Get-Item -LiteralPath $Path).Length -le 0) {
            throw "Stage 59 launch-gate artifact is empty: $Path"
        }
    }
    $Fit = Get-Content -Raw -LiteralPath $FitOut | ConvertFrom-Json
    if ([int]$Fit.rows -ne 18 -or [int]$Fit.parameters -ne 3176481 -or [int]$Fit.steps_per_run -ne 5000) {
        throw "Stage 59 fit artifact does not describe the registered 18-run, 3,176,481-parameter, 5,000-step sweep"
    }
    $ExpectedDoses = @(0.00, 0.05, 0.10, 0.20, 0.30, 0.50)
    $ExpectedSeeds = @(7, 11, 19)
    if ((@($Fit.expected_doses) -join ",") -ne ($ExpectedDoses -join ",") -or
        (@($Fit.expected_seeds) -join ",") -ne ($ExpectedSeeds -join ",")) {
        throw "Stage 59 fit artifact does not carry the registered dose and seed grid"
    }
    if ([string]$Fit.primary.kind -notin @("exponential", "power")) {
        throw "Stage 59 fit artifact has no valid registered primary family"
    }
    foreach ($Kind in @("exponential", "power")) {
        $Family = $Fit.fits.$Kind
        if ($null -eq $Family -or @($Family.residual_table).Count -ne 6) {
            throw "Stage 59 fit artifact is missing the complete $Kind fit and residual table"
        }
        foreach ($Metric in @($Family.rmse, $Family.leave_one_out.rmse)) {
            $Value = [double]$Metric
            if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value)) {
                throw "Stage 59 $Kind fit contains a non-finite residual metric"
            }
        }
    }
    $Cost = $Fit.predictions.predicted_85m_cost_at_w010
    $WStar = $Fit.predictions.w_star
    foreach ($Metric in @($Cost.delta_nll, $Cost.delta_bits_per_char, $WStar.dose, $WStar.predicted_broad_cost_delta_bits)) {
        if ($null -eq $Metric) {
            throw "Stage 59 fit artifact is missing one of the two pre-registered Part 2 predictions"
        }
        $Value = [double]$Metric
        if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value)) {
            throw "Stage 59 fit artifact contains a non-finite pre-registered prediction"
        }
    }
    if ([double]$WStar.dose -lt 0.0 -or [double]$WStar.dose -gt 0.5) {
        throw "Stage 59 fit artifact predicts w* outside the registered sweep range"
    }
    if ([IO.Path]::GetFullPath([string]$Fit.input) -ne [IO.Path]::GetFullPath($Sweep)) {
        throw "Stage 59 fit artifact points at the wrong sweep input"
    }
    $SweepSha256 = (Get-FileHash -LiteralPath $Sweep -Algorithm SHA256).Hash.ToLowerInvariant()
    if ([string]$Fit.input_sha256 -ne $SweepSha256) {
        throw "Stage 59 fit artifact is not frozen to the current sweep SHA-256"
    }
    $SummaryText = Get-Content -Raw -LiteralPath $FitSummary
    foreach ($Needle in @("Pre-registered Predictions Before Part 2", "predicted 85M broad cost", "predicted w")) {
        if (-not $SummaryText.Replace([string][char]96, "").Contains($Needle)) {
            throw "Stage 59 fit summary is missing required prediction text: $Needle"
        }
    }
    $FitSha256 = (Get-FileHash -LiteralPath $FitOut -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-LauncherLog "[gate] fit_frozen=$FitOut fit_sha256=$FitSha256 sweep_sha256=$SweepSha256 created_utc=$($Fit.created_utc) primary=$($Fit.primary.kind) predicted_w010_cost_bits=$($Cost.delta_bits_per_char) predicted_w_star=$($WStar.dose)"
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
    $ExpectedChars = 90112000L
    $RequiredChars = 20000L * 4096L
    $ExpectedTinyChars = 9011200L
    $ExpectedBroadChars = 81100800L
    if ($ExpectedChars -ne [math]::Ceiling($RequiredChars * 1.10)) {
        throw "Internal Part 2 corpus headroom registration is inconsistent"
    }
    if ([int64]$Meta.requested_total_chars -ne $ExpectedChars -or [int64]$Meta.written_chars -ne $ExpectedChars) {
        throw "Part 2 mixture is not the registered 90,112,000-character corpus"
    }
    if ([int64]$Meta.written_tiny_chars -ne $ExpectedTinyChars -or
        [int64]$Meta.written_broad_chars -ne $ExpectedBroadChars) {
        throw "Part 2 mixture has incorrect TinyStories or broad character counts"
    }
    if ([string]$Meta.ratio -ne "1:9" -or [math]::Abs([double]$Meta.tiny_fraction - 0.10) -gt 1e-12) {
        throw "Part 2 mixture metadata is not the registered w=0.10 dose"
    }
    if ([int]$Meta.tiny_reader_wraps -ne 0 -or [int]$Meta.broad_reader_wraps -ne 0) {
        throw "Part 2 mixture metadata records source wrapping"
    }
    if ([IO.Path]::GetFullPath([string]$Meta.tiny_source.dir) -ne [IO.Path]::GetFullPath($TinyShardDir) -or
        [IO.Path]::GetFullPath([string]$Meta.broad_source.dir) -ne [IO.Path]::GetFullPath($BroadShardDir)) {
        throw "Part 2 mixture metadata points at the wrong source corpus"
    }
    $ActualSha256 = Get-ConcatenatedShardSha256 -ShardDir $Part2MixDir
    if ([string]$Meta.sha256 -ne $ActualSha256) {
        throw "Part 2 mixture shard content does not match its metadata SHA-256"
    }
    Write-LauncherLog "[gate] part2_mixture=$Part2MixDir chars=$($Meta.written_chars) tiny_chars=$($Meta.written_tiny_chars) broad_chars=$($Meta.written_broad_chars) ratio=$($Meta.ratio) no_wrap=true sha256=$ActualSha256"
}

function Assert-85MTrainingRow {
    param(
        [object]$Row,
        [int]$ExpectedSeed,
        [string]$ExpectedTrainShardDir,
        [string]$Label
    )
    $Expected = [ordered]@{
        comparison_name = "random_full"
        seed = $ExpectedSeed
        steps = 20000
        formation_steps = 20000
        formation_forward_passes = 40000
        formation_parameters = 85106721
        training_target_steps = 20000
        planned_training_steps = 20000
        parameters = 85106721
        trainable_parameters = 85106721
        frozen_prior_parameters = 0
        n_layer = 12
        n_head = 12
        n_embd = 768
        block_size = 256
        batch_size = 8
        grad_accum_steps = 2
        effective_batch_size = 16
        pos_encoding = "rope"
        activation_checkpoint = $true
        dropout = 0.0
        precision = "fp32"
        optimizer = "muon"
        lr_schedule = "cosine"
        lr_final_frac = 0.1
        lr_total_steps = 20000
        lr_last_factor = 0.1
        vocab_size = 33
        vocab_chars_override = $true
        train_scope = "all"
        eval_mode = "sampled"
        eval_batches = 16
        retention_eval_mode = "sampled"
        retention_eval_batches = 16
        checkpoint_every = 5000
        checkpoint_keep = 0
        copy_train_marker = ""
    }
    $Mismatches = @()
    foreach ($Name in $Expected.Keys) {
        if ($Row.$Name -ne $Expected[$Name]) {
            $Mismatches += "$Name expected=$($Expected[$Name]) observed=$($Row.$Name)"
        }
    }
    if ($Mismatches.Count -gt 0) {
        throw "$Label training surface mismatch: $($Mismatches -join '; ')"
    }
    if ([IO.Path]::GetFullPath([string]$Row.corpus) -ne [IO.Path]::GetFullPath($Text8Corpus) -or
        [IO.Path]::GetFullPath([string]$Row.train_shard_dir) -ne [IO.Path]::GetFullPath($ExpectedTrainShardDir) -or
        [IO.Path]::GetFullPath([string]$Row.retention_eval_corpus) -ne [IO.Path]::GetFullPath($RetentionCorpus)) {
        throw "$Label training row points at the wrong corpus lineage"
    }
    if ([int]$Row.retention_eval_seed -ne $ExpectedSeed + 59000) {
        throw "$Label auxiliary retention evaluation seed is not registered"
    }
    foreach ($Metric in @($Row.val_nll, $Row.val_bits_per_char, $Row.retention_val_nll, $Row.retention_val_bits_per_char)) {
        if ($null -eq $Metric) {
            throw "$Label training row is missing a monitoring metric"
        }
        $Value = [double]$Metric
        if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value)) {
            throw "$Label training row contains a non-finite monitoring metric"
        }
    }
    if ([math]::Abs([double]$Row.val_bits_per_char - [double]$Row.val_nll / [math]::Log(2.0)) -gt 3e-6 -or
        [math]::Abs([double]$Row.retention_val_bits_per_char - [double]$Row.retention_val_nll / [math]::Log(2.0)) -gt 3e-6) {
        throw "$Label training row has inconsistent NLL and bits/char conversions"
    }
    $Curve = @($Row.loss_curve)
    foreach ($Step in @(5000, 10000, 15000, 20000)) {
        $Matches = @($Curve | Where-Object { [int]$_.step -eq $Step })
        if ($Matches.Count -ne 1) {
            throw "$Label sampled loss curve must contain exactly one step-$Step record"
        }
        foreach ($Metric in @($Matches[0].train_nll, $Matches[0].val_nll)) {
            $Value = [double]$Metric
            if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value)) {
                throw "$Label sampled loss curve contains a non-finite value at step $Step"
            }
        }
    }
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
function Assert-MusahitSkipReady {
    if (-not (Test-Path -LiteralPath $MusahitNightly)) {
        throw "Missing MUSAHIT nightly launcher: $MusahitNightly"
    }
    $Source = Get-Content -Raw -LiteralPath $MusahitNightly
    $RequiredGuardLines = @(
        'Join-Path $PSScriptRoot "SKIP_NEXT_RUN.flag"',
        'if (Test-Path $SkipFlag)',
        'Remove-Item $SkipFlag -Force',
        'exit 0'
    )
    foreach ($Needle in $RequiredGuardLines) {
        if (-not $Source.Contains($Needle)) {
            throw "MUSAHIT one-shot skip guard changed or is missing: $Needle"
        }
    }
    if (-not (Test-Path -LiteralPath $MusahitSkipFlag)) {
        throw "MUSAHIT one-shot skip flag is absent: $MusahitSkipFlag"
    }
    Write-LauncherLog "[gate] musahit_guard_verified=true skip_flag_ready=$MusahitSkipFlag"
}

function Assert-GpuIdleForLaunch {
    $Rows = @(& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null |
        Where-Object { $_.Trim() -ne "" })
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi compute-process preflight failed"
    }
    if ($Rows.Count -gt 0) {
        throw "GPU compute process already active; refusing concurrent launch: $($Rows -join ' | ')"
    }
    Write-LauncherLog "[gate] gpu_compute_processes=0"
}

function Assert-MusahitWindowReady {
    param(
        [double]$ExpectedHours,
        [string]$Label
    )
    $Now = Get-Date
    $NextFiring = $Now.Date.AddHours(2)
    if ($Now -ge $NextFiring) {
        $NextFiring = $NextFiring.AddDays(1)
    }
    $EstimatedFinish = $Now.AddHours($ExpectedHours)
    if ($EstimatedFinish -ge $NextFiring) {
        Assert-MusahitSkipReady
        Write-LauncherLog "[gate] musahit_collision=$true label=$Label expected_hours=$ExpectedHours estimated_finish=$($EstimatedFinish.ToString('o')) next_firing=$($NextFiring.ToString('o'))"
    }
    else {
        Write-LauncherLog "[gate] musahit_collision=$false label=$Label expected_hours=$ExpectedHours estimated_finish=$($EstimatedFinish.ToString('o')) next_firing=$($NextFiring.ToString('o'))"
    }
}

function Get-ConcatenatedShardSha256 {
    param([string]$ShardDir)
    $Files = @(Get-ChildItem -LiteralPath $ShardDir -File -Filter "train_*.txt" | Sort-Object Name)
    if ($Files.Count -eq 0) {
        throw "No train shards found while verifying $ShardDir"
    }
    $Hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $Utf8 = [Text.UTF8Encoding]::new($false)
        foreach ($File in $Files) {
            # Path.write_text() translates newlines on Windows after the generator
            # hashes each UTF-8 text piece. Normalize exactly as Python read_text()
            # so this re-verifies the generator's semantic content hash.
            $Text = [IO.File]::ReadAllText($File.FullName, $Utf8)
            $NormalizedText = $Text.Replace("`r`n", "`n").Replace("`r", "`n")
            $Bytes = $Utf8.GetBytes($NormalizedText)
            $null = $Hasher.TransformBlock($Bytes, 0, $Bytes.Length, $null, 0)
        }
        $null = $Hasher.TransformFinalBlock([byte[]]::new(0), 0, 0)
        return [Convert]::ToHexString($Hasher.Hash).ToLowerInvariant()
    }
    finally {
        $Hasher.Dispose()
    }
}

function Assert-ProxyMixtureInputs {
    $ExpectedChars = 22528000L
    $RequiredChars = 5000L * 4096L
    $Expected = @(
        [PSCustomObject]@{ Label = "w005"; Dose = 0.05; Tiny = 1126400L; Broad = 21401600L; Ratio = "1:19"; Sha256 = "d8d2b1ced7095e968ca4354464d32e8e857b377226591c6254b07f6f5bc66381" },
        [PSCustomObject]@{ Label = "w010"; Dose = 0.10; Tiny = 2252800L; Broad = 20275200L; Ratio = "1:9"; Sha256 = "7055b42dc55828ffe519adb9b356453b656f4a202b029d330155b8d8b5b9153d" },
        [PSCustomObject]@{ Label = "w020"; Dose = 0.20; Tiny = 4505600L; Broad = 18022400L; Ratio = "1:4"; Sha256 = "95f61408306bd8e6d0b89fd19a5b5930263bb08f60dc872acfa9699280714d93" },
        [PSCustomObject]@{ Label = "w030"; Dose = 0.30; Tiny = 6758400L; Broad = 15769600L; Ratio = "3:7"; Sha256 = "fa88ab9b16bb6562868e78c2d1aa1413fc5736e155867e7114e353cadca2bc4a" },
        [PSCustomObject]@{ Label = "w050"; Dose = 0.50; Tiny = 11264000L; Broad = 11264000L; Ratio = "1:1"; Sha256 = "e92e83e4ce1292f4bf8fe897bf7c5d64b52988fa18bcca25421c16bd960eb04a" }
    )
    if ($ExpectedChars -le $RequiredChars) {
        throw "Registered proxy corpora lack headroom over the 20,480,000 chars consumed per run"
    }
    foreach ($Item in $Expected) {
        $ShardDir = Join-Path $CorpusDir ("stage59_mix_{0}" -f $Item.Label)
        $MetaPath = Join-Path $CorpusDir ("stage59_mix_{0}.meta.json" -f $Item.Label)
        foreach ($Path in @($ShardDir, $MetaPath)) {
            if (-not (Test-Path -LiteralPath $Path)) {
                throw "Missing registered Stage 59 mixture input: $Path"
            }
        }
        $Meta = Get-Content -Raw -LiteralPath $MetaPath | ConvertFrom-Json
        if ([int64]$Meta.requested_total_chars -ne $ExpectedChars -or [int64]$Meta.written_chars -ne $ExpectedChars) {
            throw "$($Item.Label) does not contain the registered $ExpectedChars chars"
        }
        if ([int64]$Meta.written_tiny_chars -ne $Item.Tiny -or [int64]$Meta.written_broad_chars -ne $Item.Broad) {
            throw "$($Item.Label) has incorrect tiny/broad character counts"
        }
        if ([string]$Meta.ratio -ne $Item.Ratio -or [math]::Abs([double]$Meta.tiny_fraction - $Item.Dose) -gt 1e-12) {
            throw "$($Item.Label) has incorrect ratio or dose metadata"
        }
        if ([int]$Meta.tiny_reader_wraps -ne 0 -or [int]$Meta.broad_reader_wraps -ne 0) {
            throw "$($Item.Label) records a wrapped source reader"
        }
        if ([IO.Path]::GetFullPath([string]$Meta.tiny_source.dir) -ne [IO.Path]::GetFullPath($TinyShardDir) -or
            [IO.Path]::GetFullPath([string]$Meta.broad_source.dir) -ne [IO.Path]::GetFullPath($BroadShardDir)) {
            throw "$($Item.Label) metadata points at the wrong source corpus"
        }
        if ([string]$Meta.sha256 -ne $Item.Sha256) {
            throw "$($Item.Label) metadata SHA-256 differs from the frozen value"
        }
        $ActualSha256 = Get-ConcatenatedShardSha256 -ShardDir $ShardDir
        if ($ActualSha256 -ne $Item.Sha256) {
            throw "$($Item.Label) shard content SHA-256 differs from the frozen value"
        }
        Write-LauncherLog "[gate] proxy_mixture=$($Item.Label) dose=$($Item.Dose) ratio=$($Item.Ratio) chars=$ExpectedChars no_wrap=true sha256=$ActualSha256"
    }
}

function Assert-ProxySweepProgress {
    param(
        [string]$OutPath,
        [double[]]$CompletedDoses
    )
    $Rows = @(Get-Content -LiteralPath $OutPath | Where-Object { $_.Trim() -ne "" } |
        ForEach-Object { $_ | ConvertFrom-Json })
    $ExpectedCount = $CompletedDoses.Count * 3
    if ($Rows.Count -ne $ExpectedCount) {
        throw "Proxy sweep progress has $($Rows.Count) rows, expected $ExpectedCount after $($CompletedDoses.Count) doses"
    }
    $ErrorRows = @($Rows | Where-Object { $_.status -eq "error" })
    if ($ErrorRows.Count -gt 0) {
        $First = $ErrorRows[0]
        throw "Proxy sweep preserved an error row for dose=$($First.mixture_dose) seed=$($First.seed): $($First.error)"
    }
    $ExpectedKeys = @{}
    foreach ($Dose in $CompletedDoses) {
        foreach ($ExpectedSeed in @(7, 11, 19)) {
            $ExpectedKeys[("{0:F2}|{1}" -f $Dose, $ExpectedSeed)] = $true
        }
    }
    $ObservedKeys = @{}
    foreach ($Row in $Rows) {
        $Key = "{0:F2}|{1}" -f [double]$Row.mixture_dose, [int]$Row.seed
        if ($ObservedKeys.ContainsKey($Key)) {
            throw "Proxy sweep progress contains duplicate dose/seed key $Key"
        }
        $ObservedKeys[$Key] = $true
        if (-not $ExpectedKeys.ContainsKey($Key)) {
            throw "Proxy sweep progress contains unexpected dose/seed key $Key"
        }
        if ($Row.comparison_name -ne "stage59_proxy_random_full" -or
            [int]$Row.steps -ne 5000 -or [int64]$Row.parameters -ne 3176481 -or
            $Row.eval_mode -ne "sampled" -or [int]$Row.eval_batches -ne 16 -or
            $Row.retention_eval_mode -ne "sampled" -or [int]$Row.retention_eval_batches -ne 16) {
            throw "Proxy sweep progress contains a row outside the registered config, budget, or evaluation surface at $Key"
        }
    }
    foreach ($Key in $ExpectedKeys.Keys) {
        if (-not $ObservedKeys.ContainsKey($Key)) {
            throw "Proxy sweep progress is missing dose/seed key $Key"
        }
    }
    Write-LauncherLog "[verified] proxy_sweep_rows=$($Rows.Count) completed_doses=$($CompletedDoses -join ',') status=clean"
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

if ($Mode -eq "stage60-circuit-matrix") {
    $MatrixOut = Join-Path $RunDir "stage60_circuit_matrix.jsonl"
    $MatrixSummary = Join-Path $RunDir "stage60_circuit_matrix.md"
    $InventoryOut = Join-Path $RunDir "stage60_circuit_inventory.json"
    $PayloadOut = Join-Path $RunDir "stage60_circuit_matrix.payload.json"
    $ProbeDir = Join-Path $RunDir "stage60_probe_rows"
    foreach ($Path in @($MatrixOut, $MatrixSummary, $InventoryOut, $PayloadOut, $ProbeDir)) {
        if (Test-Path -LiteralPath $Path) {
            throw "Refusing to overwrite existing Stage 60 circuit-matrix evidence: $Path"
        }
    }
    Assert-GpuIdleForLaunch
    Assert-MusahitWindowReady -ExpectedHours 1.0 -Label "stage60_h026_circuit_matrix"
    $RunArgs = @(
        ".\experiments\tiny_language_lab\make_stage60_circuit_matrix.py",
        "--device", "cuda",
        "--out", $MatrixOut,
        "--summary", $MatrixSummary,
        "--inventory-out", $InventoryOut,
        "--probe-dir", $ProbeDir
    )
    Invoke-VisiblePython -Label "stage60_h026_circuit_matrix" -RunArgs $RunArgs
    foreach ($Path in @($MatrixOut, $MatrixSummary, $InventoryOut, $PayloadOut)) {
        if (-not (Test-Path -LiteralPath $Path) -or (Get-Item -LiteralPath $Path).Length -le 0) {
            throw "Stage 60 required output missing or empty: $Path"
        }
    }
    $Rows = @(Get-Content -LiteralPath $MatrixOut | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_ | ConvertFrom-Json })
    if ($Rows.Count -ne 82) {
        throw "Stage 60 matrix must preserve exactly 82 frozen inventory rows, found $($Rows.Count)"
    }
    $Anchor = $Rows[0]
    if ($Anchor.checkpoint_name -ne "stage58_dev_cold_85m_b42000_seed7_random_full_seed7.pt" -or
        $Anchor.status -ne "scored" -or [math]::Abs([double]$Anchor.choice_accuracy - 0.194336) -gt 0.0000005) {
        throw "Stage 60 deterministic anchor did not exactly reproduce 0.194336"
    }
    $HashMismatches = @($Rows | Where-Object { $_.status -eq "hash_mismatch_excluded" })
    $ProbeErrors = @($Rows | Where-Object { $_.status -eq "probe_error" })
    Write-LauncherLog "[verified] stage60_rows=$($Rows.Count) anchor_choice_accuracy=$($Anchor.choice_accuracy) hash_mismatch_excluded=$($HashMismatches.Count) probe_error_rows=$($ProbeErrors.Count)"
}
elseif ($Mode -eq "stage60-circuit-matrix-analysis") {
    $MatrixOut = Join-Path $RunDir "stage60_circuit_matrix.jsonl"
    $AnalysisOut = Join-Path $RunDir "stage60_circuit_matrix_analysis.json"
    $AnalysisSummary = Join-Path $RunDir "stage60_circuit_matrix_analysis.md"
    if (-not (Test-Path -LiteralPath $MatrixOut)) {
        throw "Stage 60 corrected analysis requires the preserved completed matrix: $MatrixOut"
    }
    $ExistingAnalysis = @((Test-Path -LiteralPath $AnalysisOut), (Test-Path -LiteralPath $AnalysisSummary))
    if ($ExistingAnalysis -contains $true -and $ExistingAnalysis -contains $false) {
        throw "Stage 60 corrected-analysis evidence is partial; preserve it and inspect before any recovery"
    }
    Assert-GpuIdleForLaunch
    if ($ExistingAnalysis -contains $false) {
        $RunArgs = @(
            ".\experiments\tiny_language_lab\make_stage60_circuit_matrix.py",
            "--analysis-from", $MatrixOut,
            "--analysis-out", $AnalysisOut,
            "--analysis-summary", $AnalysisSummary
        )
        Invoke-VisiblePython -Label "stage60_h026_corrected_analysis" -RunArgs $RunArgs
    }
    else {
        Write-LauncherLog "[recovery] revalidating existing corrected analysis without overwriting evidence"
    }
    foreach ($Path in @($AnalysisOut, $AnalysisSummary)) {
        if (-not (Test-Path -LiteralPath $Path) -or (Get-Item -LiteralPath $Path).Length -le 0) {
            throw "Stage 60 corrected-analysis output missing or empty: $Path"
        }
    }
    $Analysis = Get-Content -Raw -LiteralPath $AnalysisOut | ConvertFrom-Json
    $MatrixSha256 = (Get-FileHash -LiteralPath $MatrixOut -Algorithm SHA256).Hash.ToLowerInvariant()
    if ([string]$Analysis.analysis_input_sha256 -ne $MatrixSha256 -or
        [IO.Path]::GetFullPath([string]$Analysis.analysis_input) -ne [IO.Path]::GetFullPath($MatrixOut)) {
        throw "Stage 60 corrected analysis is not bound to the preserved matrix input"
    }
    if ($Analysis.verdict.decision_line -ne "E-gray" -or
        [math]::Abs([double]$Analysis.anchor.choice_accuracy - 0.194336) -gt 0.0000005 -or
        $null -eq $Analysis.verdict.seed7_domain_shift_accuracy_gain) {
        throw "Stage 60 corrected analysis failed its fixed-input verdict, anchor, or terminal-final checks"
    }
    Write-LauncherLog "[verified] stage60_corrected_analysis=true decision_line=$($Analysis.verdict.decision_line) phase2_gain=$($Analysis.verdict.seed7_domain_shift_accuracy_gain) matrix_sha256=$MatrixSha256"
}
elseif ($Mode -eq "stage59-smoke-prep") {
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
    if ($Budget -ne 5000) {
        throw "The registered Stage 59 proxy sweep budget is exactly 5,000 steps per run"
    }
    $Out = Join-Path $RunDir "stage59_proxy_sweep.jsonl"
    $Summary = Join-Path $RunDir "stage59_proxy_sweep.md"
    if ((Test-Path -LiteralPath $Out) -or (Test-Path -LiteralPath $Summary)) {
        throw "Refusing to overwrite existing Stage 59 proxy sweep artifacts. Preserve them and use suffixed recovery artifacts."
    }
    Assert-GpuIdleForLaunch
    Assert-MusahitSkipReady
    Assert-ProxyMixtureInputs
    Write-LauncherLog "[registered] conservative_smoke_seconds_per_step=0.31"
    Write-LauncherLog "[registered] conservative_training_hours=7.75 plus_per_run_eval_overhead=true"
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
        $CompletedDoses = @($Doses[0..$Index] | ForEach-Object { [double]$_.Dose })
        Assert-ProxySweepProgress -OutPath $Out -CompletedDoses $CompletedDoses
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
    Assert-Stage59FitGate
}
elseif ($Mode -eq "stage59-build-w10-85m") {
    Assert-Stage59FitGate
    if ($TotalChars -ne 90112000) {
        throw "The registered Part 2 corpus is exactly 90,112,000 characters, including 10 percent headroom"
    }
    $OutDir = Join-Path $CorpusDir "stage59_mix_w010_85m"
    $MetaPath = Join-Path $CorpusDir "stage59_mix_w010_85m.meta.json"
    Invoke-MixtureBuild -Label "stage59_mix_w010_85m" -TinyWeight 1 -BroadWeight 9 -Chars $TotalChars -OutDir $OutDir -MetaPath $MetaPath
    Assert-Part2Mixture
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
    Assert-GpuIdleForLaunch
    Assert-MusahitWindowReady -ExpectedHours 6.5 -Label ("stage59_part2_mixture_seed{0}" -f $Seed)
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
    Assert-85MTrainingRow -Row $Row -ExpectedSeed $Seed -ExpectedTrainShardDir $Part2MixDir -Label ("Part 2 MIXTURE seed {0}" -f $Seed)

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
    Assert-GpuIdleForLaunch
    Assert-MusahitWindowReady -ExpectedHours 6.5 -Label "stage59_seed7_cold"
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
    Assert-85MTrainingRow -Row $Row -ExpectedSeed 7 -ExpectedTrainShardDir $BroadShardDir -Label "Seed-7 COLD"

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
elseif ($Mode -eq "stage61-arm-segment") {
    # 2026-07-22: reconstructed by Claude after Codex's session ran out of
    # budget mid-flight. The stage61 launcher modes Codex used to reach
    # step 40,000 were not present in the working tree when this handoff
    # began (see runs/stage61_pure_broad_200m_seed7_pitstop_20260722.md).
    # This block reuses the file's existing, already-audited gate helpers
    # and reproduces the training command verbatim from
    # runs/phase6_stage61-arm-segment_20260722_201331_launcher.log, only
    # substituting -Budget and -ResumeFrom for the next segment.
    if ($Budget -le 0 -or $Budget % 5000 -ne 0) {
        throw "Stage 61 arm segments must be a positive multiple of 5,000 steps"
    }
    if ($ResumeFrom -eq "") {
        throw "Stage 61 arm segments require -ResumeFrom pointing at the last durable checkpoint"
    }
    if (-not (Test-Path -LiteralPath $ResumeFrom)) {
        throw "Stage 61 resume checkpoint does not exist: $ResumeFrom"
    }
    $Stage61CheckpointPrefix = "stage61_pure_broad_200m_seed7_random_full_seed7"
    if ((Split-Path -Leaf $ResumeFrom) -notlike "$Stage61CheckpointPrefix*.pt") {
        throw "Resume checkpoint does not match the registered Stage 61 pure-broad seed-7 lineage"
    }
    if (-not (Test-Path -LiteralPath $UnionVocab)) {
        throw "Missing registered 33-character union vocabulary: $UnionVocab"
    }
    $Stage61CheckpointDir = "C:\cassandra_runs\stage61_pure_broad_200m_checkpoints"
    $Stage61Out = Join-Path $RunDir "stage61_pure_broad_200m_seed7.jsonl"
    $Stage61Summary = Join-Path $RunDir "stage61_pure_broad_200m_seed7.md"
    Assert-GpuIdleForLaunch
    $ExpectedHours = [double]$Budget * 1.079669 / 3600.0
    Assert-MusahitWindowReady -ExpectedHours $ExpectedHours -Label "stage61_pure_broad_segment"
    Assert-CheckpointWriteReady -CheckpointDir $Stage61CheckpointDir
    $RunArgs = @(
        ".\experiments\tiny_language_lab\cassandra_compare.py",
        "--corpus", $Text8Corpus,
        "--device", "cuda",
        "--steps", [string]$Budget,
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
        "--lr-total-steps", "50000",
        "--vocab-chars-file", $UnionVocab,
        "--eval-mode", "sampled",
        "--eval-batches", "16",
        "--no-copy-train-marker",
        "--prompt", "the history of ",
        "--max-new-tokens", "240",
        "--n-layer", "16",
        "--n-head", "16",
        "--n-embd", "1024",
        "--train-shard-dir", $BroadShardDir,
        "--stream-train-eval-chars", "200000",
        "--val-fraction", "0.05263157894736842",
        "--out", $Stage61Out,
        "--summary", $Stage61Summary,
        "--title", "Stage 61 pure-broad Recipe v2",
        "--checkpoint-dir", $Stage61CheckpointDir,
        "--checkpoint-every", "5000",
        "--checkpoint-keep", "12",
        "--append",
        "--resume-from", $ResumeFrom,
        "--resume-steps-additional"
    )
    Invoke-VisiblePython -Label "stage61_pure_broad_segment" -RunArgs $RunArgs
    $InstArgs = @(
        ".\experiments\tiny_language_lab\make_stage61_instrumentation.py",
        "--checkpoint-dir", $Stage61CheckpointDir,
        "--prefix", $Stage61CheckpointPrefix,
        "--jsonl", (Join-Path $RunDir "stage61_instrumentation.jsonl"),
        "--summary", (Join-Path $RunDir "stage61_instrumentation.md"),
        "--minimum-age-seconds", "0"
    )
    Invoke-VisiblePython -Label "stage61_instrumentation" -RunArgs $InstArgs
    Write-LauncherLog "[verified] stage61_arm_segment_budget=$Budget resume_from=$ResumeFrom"
}
Write-LauncherLog "[visible] launcher_end=$(Get-Date -Format o) status=success"
if ($KeepOpen) {
    Read-Host "Run finished. Press Enter to close this window"
}