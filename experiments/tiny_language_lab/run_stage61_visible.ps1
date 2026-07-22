param(
    [ValidateSet(
        "stage61-size-gate",
        "stage61-throughput",
        "stage61-resume-drill",
        "stage61-arm-segment",
        "stage61-instrument-final",
        "stage61-final-eval",
        "stage61-user-samples"
    )]
    [string]$Mode = "stage61-size-gate",
    [int]$TargetSteps = 0,
    [int]$CurrentStep = 0,
    [switch]$KeepOpen
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$RunDir = Join-Path $ScriptDir "runs"
$CorpusDir = Join-Path $ScriptDir "corpus"
$Text8Corpus = Join-Path $CorpusDir "text8_char_seed.txt"
$Text8ShardDir = Join-Path $CorpusDir "text8_char_shards"
$Text8MetaPath = Join-Path $CorpusDir "text8_char_shards.meta.json"
$UnionVocab = Join-Path $CorpusDir "phase5_union_vocab.txt"
$RetentionCorpus = Join-Path $CorpusDir "tinystories_char_shards_500mb\val.txt"
$CheckpointRoot = "C:\cassandra_runs"
$ThroughputCheckpointDir = Join-Path $CheckpointRoot "stage61_throughput_checkpoints"
$ResumeCheckpointDir = Join-Path $CheckpointRoot "stage61_resume_drill_checkpoints"
$ArmCheckpointDir = Join-Path $CheckpointRoot "stage61_pure_broad_200m_checkpoints"
$SizeOut = Join-Path $RunDir "stage61_size_gate.jsonl"
$SizeSummary = Join-Path $RunDir "stage61_size_gate.md"
$ThroughputOut = Join-Path $RunDir "stage61_throughput.jsonl"
$ThroughputSummary = Join-Path $RunDir "stage61_throughput.md"
$BudgetOut = Join-Path $RunDir "stage61_throughput_gate.json"
$BudgetSummary = Join-Path $RunDir "stage61_throughput_gate.md"
$ResumeInitialOut = Join-Path $RunDir "stage61_resume_drill_initial.jsonl"
$ResumeInitialSummary = Join-Path $RunDir "stage61_resume_drill_initial.md"
$ResumeResumedOut = Join-Path $RunDir "stage61_resume_drill_resumed.jsonl"
$ResumeResumedSummary = Join-Path $RunDir "stage61_resume_drill_resumed.md"
$ArmOut = Join-Path $RunDir "stage61_pure_broad_200m_seed7.jsonl"
$ArmSummary = Join-Path $RunDir "stage61_pure_broad_200m_seed7.md"
$Instrumentation = Join-Path $RunDir "stage61_instrumentation.jsonl"
$InstrumentationSummary = Join-Path $RunDir "stage61_instrumentation.md"
$Text8Out = Join-Path $RunDir "stage61_text8_test.json"
$Text8Summary = Join-Path $RunDir "stage61_text8_test.md"
$PublishBarsOut = Join-Path $RunDir "stage61_publish_bars.json"
$PublishBarsSummary = Join-Path $RunDir "stage61_publish_bars.md"
$UserSamplesOut = Join-Path $RunDir "stage61_user_samples.json"
$UserSamplesSummary = Join-Path $RunDir "stage61_user_samples.md"
$ArmStem = "stage61_pure_broad_200m_seed7"
$ArmCheckpointPrefix = "${ArmStem}_random_full_seed7"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunLog = Join-Path $RunDir ("phase6_{0}_{1}.log" -f $Mode, $Timestamp)
$LauncherLog = Join-Path $RunDir ("phase6_{0}_{1}_launcher.log" -f $Mode, $Timestamp)
$KeepAwakeLog = Join-Path $RunDir ("keep_awake_{0}_{1}.log" -f $Mode, $Timestamp)
$DesktopRoot = Split-Path -Parent $RepoRoot
$MusahitNightlyCandidates = @(Get-ChildItem -LiteralPath $DesktopRoot -Directory | ForEach-Object {
    Join-Path $_.FullName "scripts/scheduling/run_nightly.ps1"
} | Where-Object { Test-Path -LiteralPath $_ })
if ($MusahitNightlyCandidates.Count -ne 1) {
    throw "Expected exactly one sibling MUSAHIT nightly launcher, found $($MusahitNightlyCandidates.Count)"
}
$MusahitNightly = $MusahitNightlyCandidates[0]
$MusahitSkipFlag = Join-Path (Split-Path -Parent $MusahitNightly) "SKIP_NEXT_RUN.flag"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
Set-Location $RepoRoot
trap {
    $Failure = "[error] $($_.Exception.GetType().Name): $($_.Exception.Message)"
    Write-Host $Failure
    try { Add-Content -LiteralPath $LauncherLog -Value $Failure } catch {}
    exit 1
}

function Write-LauncherLog {
    param([string]$Message)
    Write-Host $Message
    Add-Content -LiteralPath $LauncherLog -Value $Message
}

function Invoke-VisiblePython {
    param([string]$Label, [string[]]$RunArgs)
    Write-LauncherLog ("[visible] command=python " + ($RunArgs -join " "))
    "[stage61-visible] start $Label" | Tee-Object -FilePath $RunLog -Append
    python @RunArgs 2>&1 | Tee-Object -FilePath $RunLog -Append
    $ExitCode = $LASTEXITCODE
    "[stage61-visible] exit $Label code=$ExitCode" | Tee-Object -FilePath $RunLog -Append
    if ($ExitCode -ne 0) {
        throw "$Label exited with code $ExitCode"
    }
}

function Assert-FreeSpaceForRun {
    $FreeGiB = (Get-PSDrive -Name C).Free / 1GB
    if ($FreeGiB -lt 15) {
        throw ("C: free space is {0:N2} GiB, below the 15 GiB Stage 61 launch gate." -f $FreeGiB)
    }
    Write-LauncherLog ("[gate] c_free_gib={0:N2}" -f $FreeGiB)
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

function Assert-CheckpointWriteReady {
    param([string]$CheckpointDir)
    $ResolvedRoot = [IO.Path]::GetFullPath($CheckpointRoot)
    $ResolvedTarget = [IO.Path]::GetFullPath($CheckpointDir)
    if (-not $ResolvedTarget.StartsWith($ResolvedRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing checkpoint target outside ${CheckpointRoot}: $ResolvedTarget"
    }
    New-Item -ItemType Directory -Force -Path $ResolvedTarget | Out-Null
    $ProbeStem = "stage61_checkpoint_probe_{0}" -f ([guid]::NewGuid().ToString("N"))
    $ProbeTemp = Join-Path $ResolvedTarget "$ProbeStem.pt.tmp"
    $ProbeFinal = Join-Path $ResolvedTarget "$ProbeStem.pt"
    try {
        $ProbeCode = "import os, torch; from pathlib import Path; tmp=Path(r'$ProbeTemp'); final=Path(r'$ProbeFinal'); torch.save({'probe': torch.tensor([1])}, tmp); os.replace(tmp, final); assert final.exists() and final.stat().st_size > 0"
        python -c $ProbeCode
        if ($LASTEXITCODE -ne 0) {
            throw "PyTorch checkpoint probe exited with code $LASTEXITCODE"
        }
        Write-LauncherLog "[gate] checkpoint_write_probe=passed path=$ResolvedTarget"
    }
    finally {
        foreach ($Path in @($ProbeTemp, $ProbeFinal)) {
            if (Test-Path -LiteralPath $Path) {
                Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

function Get-Sha256 {
    param([string]$Path)
    $Hasher = [Security.Cryptography.SHA256]::Create()
    $Stream = [IO.File]::OpenRead($Path)
    try {
        return ([BitConverter]::ToString($Hasher.ComputeHash($Stream))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $Stream.Dispose()
        $Hasher.Dispose()
    }
}
function Get-MuonLearningRate {
    param([object]$Row, [string]$Label)
    $Report = $Row.optimizer_report
    if ($null -eq $Report -or [string]$Report.optimizer -ne "muon") {
        throw "$Label does not carry a Muon optimizer report"
    }
    $Rate = [double]$Report.muon_lr
    if ([double]::IsNaN($Rate) -or [double]::IsInfinity($Rate)) {
        throw "$Label has a non-finite Muon learning rate"
    }
    return $Rate
}

function Assert-Stage61Inputs {
    foreach ($Path in @($Text8Corpus, $Text8ShardDir, $Text8MetaPath, $UnionVocab, $RetentionCorpus)) {
        if (-not (Test-Path -LiteralPath $Path)) {
            throw "Missing required Stage 61 input: $Path"
        }
    }
    $Meta = Get-Content -Raw -LiteralPath $Text8MetaPath | ConvertFrom-Json
    if ([int64]$Meta.train_chars -ne 90000000L -or [int64]$Meta.val_chars -ne 5000000L -or
        [int64]$Meta.test_chars -ne 5000000L -or [int64]$Meta.seed_chars -ne 95000000L -or
        [string]$Meta.seed_sha256 -ne "a27d3a863dbd8102fb14618aa65e966c40117d9ce852ac02a290b4a5078e0aeb") {
        throw "text8 shard metadata does not describe the registered 90M/5M/5M pure-broad split"
    }
    $Train = @(Get-ChildItem -LiteralPath $Text8ShardDir -File -Filter "train_*.txt" | Sort-Object Name)
    if ($Train.Count -ne 9 -or @($Train | Where-Object { $_.Length -ne 10000000 }).Count -ne 0) {
        throw "text8 training shards are not the registered nine 10M-character files"
    }
    if ((Get-Item -LiteralPath (Join-Path $Text8ShardDir "val.txt")).Length -ne 5000000 -or
        (Get-Item -LiteralPath $Text8Corpus).Length -ne 95000000) {
        throw "text8 seed or validation file length differs from the registered split"
    }
    $VocabHash = Get-Sha256 -Path $UnionVocab
    if ($VocabHash -ne "05d8a05cbdfdfab6be9f5aff6251eb3dec2b166cf634fc4acc7579bf138bef43" -or
        ([IO.File]::ReadAllText((Resolve-Path $UnionVocab))).Length -ne 34) {
        throw "The registered 33-character Phase 5 union vocabulary changed"
    }
    Write-LauncherLog "[gate] pure_text8_train_chars=90000000 val_chars=5000000 test_chars=5000000 union_vocab_sha256=$VocabHash"
}

function Assert-RowRecipe {
    param(
        [object]$Row,
        [int]$ExpectedSteps,
        [int]$ExpectedLrTotalSteps,
        [string]$Label,
        [int]$ExpectedResumeStep = 0,
        [bool]$ExpectedResumeLoaded = $false,
        [int]$ExpectedCheckpointEvery = 5000
    )
    $Expected = @{
        comparison_name = "random_full"; seed = 7; parameters = 201609249; block_size = 256;
        batch_size = 8; grad_accum_steps = 2; n_layer = 16; n_head = 16; n_embd = 1024;
        pos_encoding = "rope"; activation_checkpoint = $true; optimizer = "muon"; precision = "fp32";
        lr_schedule = "cosine"; checkpoint_every = $ExpectedCheckpointEvery; checkpoint_keep = 12; steps = $ExpectedSteps;
        lr_total_steps = $ExpectedLrTotalSteps
    }
    foreach ($Key in $Expected.Keys) {
        if ($Row.$Key -ne $Expected[$Key]) {
            throw "$Label has $Key=$($Row.$Key), expected $($Expected[$Key])"
        }
    }
    $MuonLearningRate = Get-MuonLearningRate -Row $Row -Label $Label
    foreach ($Value in @($Row.val_nll, $Row.val_bits_per_char, $Row.seconds, $Row.peak_cuda_memory_mib, $MuonLearningRate, $Row.lr_final_frac)) {
        $Number = [double]$Value
        if ([double]::IsNaN($Number) -or [double]::IsInfinity($Number)) {
            throw "$Label contains a non-finite metric"
        }
    }
    if ([math]::Abs($MuonLearningRate - 0.01) -gt 1e-12 -or [math]::Abs([double]$Row.lr_final_frac - 0.1) -gt 1e-12) {
        throw "$Label differs from the registered Muon/cosine surface"
    }
    if ([IO.Path]::GetFullPath([string]$Row.train_shard_dir) -ne [IO.Path]::GetFullPath($Text8ShardDir) -or $Row.copy_train_marker -notin @("", $null)) {
        throw "$Label is not a pure text8 random-full run"
    }
    if ([int]$Row.resume_step -ne $ExpectedResumeStep -or [bool]$Row.resume_loaded -ne $ExpectedResumeLoaded) {
        throw "$Label has an unexpected resume state"
    }
    if ([int]$Row.formation_steps -ne ($ExpectedResumeStep + $ExpectedSteps) -or [int]$Row.formation_forward_passes -le 0) {
        throw "$Label has an incorrect formation-step or forward-pass accounting"
    }
}

function Assert-Stage61SizeGate {
    foreach ($Path in @($SizeOut, $SizeSummary)) {
        if (-not (Test-Path -LiteralPath $Path) -or (Get-Item -LiteralPath $Path).Length -le 0) {
            throw "Stage 61 size gate is missing or empty: $Path"
        }
    }
    $Rows = @(Get-Content -LiteralPath $SizeOut | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_ | ConvertFrom-Json })
    if ($Rows.Count -ne 1) { throw "Stage 61 size gate must contain exactly one row" }
    $Row = $Rows[0]
    if ([int64]$Row.parameters -ne 201609249 -or [int]$Row.n_layer -ne 16 -or [int]$Row.n_head -ne 16 -or [int]$Row.n_embd -ne 1024) {
        throw "Stage 61 size gate does not carry the exact 201,609,249-parameter Recipe v2 shape"
    }
    $Gpu = @(& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null)
    if ($LASTEXITCODE -ne 0 -or $Gpu.Count -lt 1) { throw "nvidia-smi memory-total query failed" }
    $TotalMiB = [double]$Gpu[0].Trim()
    if ([double]$Row.peak_cuda_memory_mib -ge ($TotalMiB - 1024.0)) {
        throw "Stage 61 size gate leaves less than the required 1 GiB CUDA allocation headroom"
    }
    Write-LauncherLog "[gate] size_gate=passed parameters=$($Row.parameters) peak_cuda_mib=$($Row.peak_cuda_memory_mib) gpu_total_mib=$TotalMiB"
}

function Get-Stage61Budget {
    foreach ($Path in @($BudgetOut, $BudgetSummary)) {
        if (-not (Test-Path -LiteralPath $Path) -or (Get-Item -LiteralPath $Path).Length -le 0) {
            throw "Stage 61 throughput-budget decision is missing or empty: $Path"
        }
    }
    $Budget = Get-Content -Raw -LiteralPath $BudgetOut | ConvertFrom-Json
    if ($Budget.decision.status -ne "CLEARED" -or $null -eq $Budget.decision.target_steps) {
        throw "Stage 61 budget is not CLEARED. Preserve it and request the user's size-versus-budget decision."
    }    if ([IO.Path]::GetFullPath([string]$Budget.throughput_path) -ne [IO.Path]::GetFullPath($ThroughputOut) -or
        [string]$Budget.throughput_sha256 -ne (Get-Sha256 -Path $ThroughputOut)) {
        throw "Stage 61 throughput budget is not bound to the current measured throughput row"
    }
    $Target = [int]$Budget.decision.target_steps
    if ($Target -lt 30000 -or $Target -gt 50000 -or $Target % 5000 -ne 0) {
        throw "Stage 61 budget has an invalid target step count: $Target"
    }
    return $Budget
}

function Assert-Stage61ThroughputGate {
    foreach ($Path in @($ThroughputOut, $ThroughputSummary, $BudgetOut, $BudgetSummary)) {
        if (-not (Test-Path -LiteralPath $Path) -or (Get-Item -LiteralPath $Path).Length -le 0) {
            throw "Stage 61 throughput gate is missing or empty: $Path"
        }
    }
    $Rows = @(Get-Content -LiteralPath $ThroughputOut | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_ | ConvertFrom-Json })
    if ($Rows.Count -ne 1) { throw "Stage 61 throughput must contain exactly one row" }
    Assert-RowRecipe -Row $Rows[0] -ExpectedSteps 5000 -ExpectedLrTotalSteps 5000 -Label "Stage 61 throughput"
    $null = Get-Stage61Budget
    Write-LauncherLog "[gate] throughput_budget=passed seconds=$($Rows[0].seconds)"
}

function Assert-Stage61ResumeGate {
    foreach ($Path in @($ResumeInitialOut, $ResumeInitialSummary, $ResumeResumedOut, $ResumeResumedSummary)) {
        if (-not (Test-Path -LiteralPath $Path) -or (Get-Item -LiteralPath $Path).Length -le 0) {
            throw "Stage 61 resume drill is missing or empty: $Path"
        }
    }
    $Budget = Get-Stage61Budget
    $Target = [int]$Budget.decision.target_steps
    $Initial = @(Get-Content -LiteralPath $ResumeInitialOut | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_ | ConvertFrom-Json })
    $Resumed = @(Get-Content -LiteralPath $ResumeResumedOut | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_ | ConvertFrom-Json })
    if ($Initial.Count -ne 1 -or $Resumed.Count -ne 1) { throw "Stage 61 resume drill must retain exactly one initial and one resumed row" }
    Assert-RowRecipe -Row $Initial[0] -ExpectedSteps 200 -ExpectedLrTotalSteps $Target -Label "Stage 61 resume drill initial" -ExpectedCheckpointEvery 200
    Assert-RowRecipe -Row $Resumed[0] -ExpectedSteps 200 -ExpectedLrTotalSteps $Target -Label "Stage 61 resume drill resumed" -ExpectedResumeStep 200 -ExpectedResumeLoaded $true -ExpectedCheckpointEvery 200
    Write-LauncherLog "[gate] resume_drill=passed resumed_formation_steps=$($Resumed[0].formation_steps)"
}

function Assert-MusahitGuard {
    $Source = Get-Content -Raw -LiteralPath $MusahitNightly
    foreach ($Needle in @('Join-Path $PSScriptRoot "SKIP_NEXT_RUN.flag"', 'if (Test-Path $SkipFlag)', 'Remove-Item $SkipFlag -Force', 'exit 0')) {
        if (-not $Source.Contains($Needle)) { throw "MUSAHIT one-shot skip guard changed or is missing: $Needle" }
    }
}

function Prepare-MusahitForRemainingRun {
    param([double]$RemainingHours, [string]$Label)
    $Now = Get-Date
    $Finish = $Now.AddHours($RemainingHours)
    $Firing = $Now.Date.AddHours(2)
    if ($Firing -le $Now) { $Firing = $Firing.AddDays(1) }
    $Collisions = 0
    while ($Firing -le $Finish) {
        $Collisions += 1
        $Firing = $Firing.AddDays(1)
    }
    if ($Collisions -gt 1) {
        throw "Stage 61 would cross $Collisions MUSAHIT firings. A live re-arming monitor is required before launch; do not consume a one-shot flag speculatively."
    }
    if ($Collisions -eq 1) {
        Assert-MusahitGuard
        if (-not (Test-Path -LiteralPath $MusahitSkipFlag)) {
            New-Item -ItemType File -Path $MusahitSkipFlag -Force | Out-Null
            Write-LauncherLog "[gate] musahit_skip_flag_created=$MusahitSkipFlag"
        }
        else {
            Write-LauncherLog "[gate] musahit_skip_flag_already_present=$MusahitSkipFlag"
        }
    }
    Write-LauncherLog "[gate] musahit_collisions=$Collisions label=$Label remaining_hours=$RemainingHours finish=$($Finish.ToString('o'))"
}

function Get-RecipeArgs {
    param([int]$Steps, [int]$LrTotalSteps, [string]$CheckpointDir, [int]$CheckpointEvery, [string]$Out, [string]$Summary, [bool]$Append, [string]$ResumeFrom = "", [bool]$ResumeAdditional = $false)
    $Args = @(
        ".\experiments\tiny_language_lab\cassandra_compare.py",
        "--corpus", $Text8Corpus,
        "--device", "cuda",
        "--steps", [string]$Steps,
        "--eval-interval", [string]$CheckpointEvery,
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
        "--lr-total-steps", [string]$LrTotalSteps,
        "--vocab-chars-file", $UnionVocab,
        "--eval-mode", "sampled",
        "--eval-batches", "16",
        "--no-copy-train-marker",
        "--prompt", "the history of ",
        "--max-new-tokens", "240",
        "--n-layer", "16",
        "--n-head", "16",
        "--n-embd", "1024",
        "--train-shard-dir", $Text8ShardDir,
        "--stream-train-eval-chars", "200000",
        "--val-fraction", "0.05263157894736842",
        "--out", $Out,
        "--summary", $Summary,
        "--title", "Stage 61 pure-broad Recipe v2"
    )
    if ($CheckpointDir -ne "") {
        $Args += @("--checkpoint-dir", $CheckpointDir, "--checkpoint-every", [string]$CheckpointEvery, "--checkpoint-keep", "12")
    }
    if ($Append) { $Args += "--append" }
    if ($ResumeFrom -ne "") { $Args += @("--resume-from", $ResumeFrom) }
    if ($ResumeAdditional) { $Args += "--resume-steps-additional" }
    return $Args
}

"[visible] launcher_start=$(Get-Date -Format o)" | Set-Content -LiteralPath $LauncherLog
Write-LauncherLog "[visible] repo=$RepoRoot"
Write-LauncherLog "[visible] mode=$Mode"
Write-LauncherLog "[visible] run_log=$RunLog"
Assert-FreeSpaceForRun
Assert-Stage61Inputs

$KeepArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $ScriptDir "keep_awake_while_pid.ps1"), "-WatchPid", [string]$PID, "-LogPath", $KeepAwakeLog)
Start-Process -FilePath "powershell.exe" -ArgumentList $KeepArgs -WindowStyle Hidden | Out-Null
Write-LauncherLog "[visible] keep_awake_log=$KeepAwakeLog watch_pid=$PID"

if ($Mode -eq "stage61-size-gate") {
    $Existing = @((Test-Path -LiteralPath $SizeOut), (Test-Path -LiteralPath $SizeSummary))
    if ($Existing -contains $true -and $Existing -contains $false) { throw "Stage 61 size-gate evidence is partial; preserve and inspect it" }
    if ($Existing -contains $false) {
        Assert-GpuIdleForLaunch
        $Args = Get-RecipeArgs -Steps 20 -LrTotalSteps 5000 -CheckpointDir "" -CheckpointEvery 20 -Out $SizeOut -Summary $SizeSummary -Append $false
        Invoke-VisiblePython -Label "stage61_size_gate" -RunArgs $Args
    }
    else {
        Write-LauncherLog "[recovery] revalidating preserved Stage 61 size-gate evidence without overwriting it"
    }
    Assert-Stage61SizeGate
}
elseif ($Mode -eq "stage61-throughput") {
    Assert-Stage61SizeGate
    foreach ($Path in @($ThroughputOut, $ThroughputSummary, $BudgetOut, $BudgetSummary)) { if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite Stage 61 throughput evidence: $Path" } }
    if (@(Get-ChildItem -LiteralPath $ThroughputCheckpointDir -Filter "stage61_throughput*.pt" -ErrorAction SilentlyContinue).Count -gt 0) { throw "Stage 61 throughput checkpoint evidence already exists; preserve it" }
    Assert-GpuIdleForLaunch
    Assert-CheckpointWriteReady -CheckpointDir $ThroughputCheckpointDir
    $Args = Get-RecipeArgs -Steps 5000 -LrTotalSteps 5000 -CheckpointDir $ThroughputCheckpointDir -CheckpointEvery 5000 -Out $ThroughputOut -Summary $ThroughputSummary -Append $false
    Invoke-VisiblePython -Label "stage61_throughput_5000" -RunArgs $Args
    $Rows = @(Get-Content -LiteralPath $ThroughputOut | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_ | ConvertFrom-Json })
    if ($Rows.Count -ne 1) { throw "Stage 61 throughput must contain exactly one row" }
    Assert-RowRecipe -Row $Rows[0] -ExpectedSteps 5000 -ExpectedLrTotalSteps 5000 -Label "Stage 61 throughput"
    $Prefix = "stage61_throughput_random_full_seed7"
    foreach ($Checkpoint in @((Join-Path $ThroughputCheckpointDir "$Prefix.pt"), (Join-Path $ThroughputCheckpointDir "${Prefix}_step005000.pt"))) {
        if (-not (Test-Path -LiteralPath $Checkpoint) -or (Get-Item -LiteralPath $Checkpoint).Length -le 0) { throw "Stage 61 throughput checkpoint missing or empty: $Checkpoint" }
    }
    Invoke-VisiblePython -Label "stage61_throughput_budget" -RunArgs @(".\experiments\tiny_language_lab\make_stage61_budget.py", "--throughput", $ThroughputOut, "--text8-shards", $Text8ShardDir, "--out", $BudgetOut, "--summary", $BudgetSummary)
    $null = Get-Stage61Budget
    Write-LauncherLog "[verified] throughput_seconds=$($Rows[0].seconds) budget_target_steps=$((Get-Stage61Budget).decision.target_steps)"
}
elseif ($Mode -eq "stage61-resume-drill") {
    Assert-Stage61ThroughputGate
    $Budget = Get-Stage61Budget
    $Target = [int]$Budget.decision.target_steps
    foreach ($Path in @($ResumeInitialOut, $ResumeInitialSummary, $ResumeResumedOut, $ResumeResumedSummary)) { if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite Stage 61 resume-drill evidence: $Path" } }
    if (@(Get-ChildItem -LiteralPath $ResumeCheckpointDir -Filter "stage61_resume_drill*.pt" -ErrorAction SilentlyContinue).Count -gt 0) { throw "Stage 61 resume-drill checkpoints already exist; preserve and use pitstop recovery" }
    Assert-GpuIdleForLaunch
    Assert-CheckpointWriteReady -CheckpointDir $ResumeCheckpointDir
    $InitialArgs = Get-RecipeArgs -Steps 200 -LrTotalSteps $Target -CheckpointDir $ResumeCheckpointDir -CheckpointEvery 200 -Out $ResumeInitialOut -Summary $ResumeInitialSummary -Append $false
    Invoke-VisiblePython -Label "stage61_resume_drill_initial_200" -RunArgs $InitialArgs
    $ResumeCheckpoint = Join-Path $ResumeCheckpointDir "stage61_resume_drill_initial_random_full_seed7_step000200.pt"
    if (-not (Test-Path -LiteralPath $ResumeCheckpoint)) { throw "Stage 61 resume drill did not produce its step-200 anchor" }
    $ResumedArgs = Get-RecipeArgs -Steps 200 -LrTotalSteps $Target -CheckpointDir $ResumeCheckpointDir -CheckpointEvery 200 -Out $ResumeResumedOut -Summary $ResumeResumedSummary -Append $false -ResumeFrom $ResumeCheckpoint -ResumeAdditional $true
    Invoke-VisiblePython -Label "stage61_resume_drill_resumed_400" -RunArgs $ResumedArgs
    Assert-Stage61ResumeGate
}
elseif ($Mode -eq "stage61-arm-segment") {
    Assert-Stage61ThroughputGate
    Assert-Stage61ResumeGate
    $Budget = Get-Stage61Budget
    $RegisteredTarget = [int]$Budget.decision.target_steps
    if ($TargetSteps -ne $RegisteredTarget) { throw "-TargetSteps must equal the frozen throughput budget $RegisteredTarget" }
    if ($CurrentStep -lt 0 -or $CurrentStep -ge $TargetSteps -or $CurrentStep % 5000 -ne 0) { throw "-CurrentStep must be a non-negative 5,000-step rung below the target" }
    $ExpectedRowsBefore = [int]($CurrentStep / 5000)
    if ($CurrentStep -eq 0) {
        foreach ($Path in @($ArmOut, $ArmSummary)) { if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite initial Stage 61 arm evidence: $Path" } }
        if (@(Get-ChildItem -LiteralPath $ArmCheckpointDir -Filter "$ArmCheckpointPrefix*.pt" -ErrorAction SilentlyContinue).Count -gt 0) { throw "Stage 61 arm checkpoint evidence already exists; preserve and use explicit pitstop recovery" }
    }
    else {
        foreach ($Path in @($ArmOut, $ArmSummary)) { if (-not (Test-Path -LiteralPath $Path)) { throw "Stage 61 resume segment is missing prior arm evidence: $Path" } }
        $PriorRows = @(Get-Content -LiteralPath $ArmOut | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_ | ConvertFrom-Json })
        if ($PriorRows.Count -ne $ExpectedRowsBefore) { throw "Stage 61 arm has $($PriorRows.Count) rows, expected $ExpectedRowsBefore before rung $CurrentStep" }
    }
    Assert-GpuIdleForLaunch
    Assert-CheckpointWriteReady -CheckpointDir $ArmCheckpointDir
    $RemainingHours = [double]$Budget.decision.seconds_per_step * ($TargetSteps - $CurrentStep) / 3600.0
    Prepare-MusahitForRemainingRun -RemainingHours $RemainingHours -Label "stage61_segment_$CurrentStep"
    $ResumeFrom = ""
    $ResumeAdditional = $false
    if ($CurrentStep -gt 0) {
        $ResumeFrom = Join-Path $ArmCheckpointDir ("{0}_step{1:D6}.pt" -f $ArmCheckpointPrefix, $CurrentStep)
        if (-not (Test-Path -LiteralPath $ResumeFrom)) { throw "Stage 61 required rung checkpoint is absent: $ResumeFrom" }
        $ResumeAdditional = $true
        Write-LauncherLog "[gate] resume_from=$ResumeFrom"
    }
    $Args = Get-RecipeArgs -Steps 5000 -LrTotalSteps $TargetSteps -CheckpointDir $ArmCheckpointDir -CheckpointEvery 5000 -Out $ArmOut -Summary $ArmSummary -Append ($CurrentStep -gt 0) -ResumeFrom $ResumeFrom -ResumeAdditional $ResumeAdditional
    Invoke-VisiblePython -Label ("stage61_pure_broad_segment_{0:D6}_{1:D6}" -f $CurrentStep, ($CurrentStep + 5000)) -RunArgs $Args
    $Rows = @(Get-Content -LiteralPath $ArmOut | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_ | ConvertFrom-Json })
    if ($Rows.Count -ne ($ExpectedRowsBefore + 1)) { throw "Stage 61 arm row count is $($Rows.Count), expected $($ExpectedRowsBefore + 1)" }
    Assert-RowRecipe -Row $Rows[-1] -ExpectedSteps 5000 -ExpectedLrTotalSteps $TargetSteps -Label "Stage 61 arm segment" -ExpectedResumeStep $CurrentStep -ExpectedResumeLoaded ($CurrentStep -gt 0)
    $NewStep = $CurrentStep + 5000
    $NewCheckpoint = Join-Path $ArmCheckpointDir ("{0}_step{1:D6}.pt" -f $ArmCheckpointPrefix, $NewStep)
    if (-not (Test-Path -LiteralPath $NewCheckpoint) -or (Get-Item -LiteralPath $NewCheckpoint).Length -le 0) { throw "Stage 61 checkpoint rung missing or empty: $NewCheckpoint" }
    Invoke-VisiblePython -Label ("stage61_instrument_step_{0:D6}" -f $NewStep) -RunArgs @(".\experiments\tiny_language_lab\make_stage61_instrumentation.py", "--checkpoint-dir", $ArmCheckpointDir, "--prefix", $ArmCheckpointPrefix, "--jsonl", $Instrumentation, "--summary", $InstrumentationSummary, "--minimum-age-seconds", "0")
    $InstrumentRows = @(Get-Content -LiteralPath $Instrumentation | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_ | ConvertFrom-Json })
    $RungRows = @($InstrumentRows | Where-Object { $_.status -eq "complete" -and $_.checkpoint.path -eq [IO.Path]::GetFullPath($NewCheckpoint) })
    if ($RungRows.Count -ne 1 -or [int]$RungRows[0].tinystories_retention.chars_evaluated -ne 1499904 -or [int]$RungRows[0].letters_probe.choice_cases -ne 1024) {
        throw "Stage 61 rung instrumentation is absent or does not satisfy the frozen probe/retention contract"
    }
    Write-LauncherLog "[verified] arm_segment_end_step=$NewStep letters_choice_accuracy=$($RungRows[0].letters_probe.choice_accuracy) retention_bits=$($RungRows[0].tinystories_retention.bits_per_char)"
}
elseif ($Mode -eq "stage61-instrument-final") {
    Assert-Stage61ThroughputGate
    Assert-Stage61ResumeGate
    $Budget = Get-Stage61Budget
    $Target = [int]$Budget.decision.target_steps
    if ($TargetSteps -ne $Target) { throw "-TargetSteps must equal the frozen throughput budget $Target" }
    Assert-GpuIdleForLaunch
    $Final = Join-Path $ArmCheckpointDir "$ArmCheckpointPrefix.pt"
    if (-not (Test-Path -LiteralPath $Final) -or (Get-Item -LiteralPath $Final).Length -le 0) { throw "Stage 61 final unsuffixed checkpoint is absent: $Final" }
    Invoke-VisiblePython -Label "stage61_instrument_final" -RunArgs @(".\experiments\tiny_language_lab\make_stage61_instrumentation.py", "--checkpoint-dir", $ArmCheckpointDir, "--prefix", $ArmCheckpointPrefix, "--jsonl", $Instrumentation, "--summary", $InstrumentationSummary, "--minimum-age-seconds", "0", "--include-final")
    $Rows = @(Get-Content -LiteralPath $Instrumentation | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_ | ConvertFrom-Json })
    $Complete = @($Rows | Where-Object { $_.status -eq "complete" })
    if ($Complete.Count -ne (($Target / 5000) + 1)) { throw "Stage 61 instrumentation must contain every rung plus the final, found $($Complete.Count) complete rows" }
    $FinalRows = @($Complete | Where-Object { $_.checkpoint.path -eq [IO.Path]::GetFullPath($Final) })
    if ($FinalRows.Count -ne 1) { throw "Stage 61 final checkpoint lacks exactly one instrumentation row" }
    Write-LauncherLog "[verified] final_instrumentation_rows=$($Complete.Count) final_choice_accuracy=$($FinalRows[0].letters_probe.choice_accuracy)"
}
elseif ($Mode -eq "stage61-final-eval") {
    Assert-Stage61ThroughputGate
    Assert-Stage61ResumeGate
    $Budget = Get-Stage61Budget
    $Target = [int]$Budget.decision.target_steps
    if ($TargetSteps -ne $Target) { throw "-TargetSteps must equal the frozen throughput budget $Target" }
    foreach ($Path in @($Text8Out, $Text8Summary, $PublishBarsOut, $PublishBarsSummary)) {
        if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite Stage 61 final-evaluation evidence: $Path" }
    }
    $TrainingRows = @(Get-Content -LiteralPath $ArmOut | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_ | ConvertFrom-Json })
    if ($TrainingRows.Count -ne ($Target / 5000)) { throw "Stage 61 final evaluation requires the full $Target-step training ladder" }
    for ($Index = 0; $Index -lt $TrainingRows.Count; $Index += 1) {
        $PriorStep = $Index * 5000
        Assert-RowRecipe -Row $TrainingRows[$Index] -ExpectedSteps 5000 -ExpectedLrTotalSteps $Target -Label ("Stage 61 final ladder row {0}" -f $Index) -ExpectedResumeStep $PriorStep -ExpectedResumeLoaded ($PriorStep -gt 0)
        $Rung = Join-Path $ArmCheckpointDir ("{0}_step{1:D6}.pt" -f $ArmCheckpointPrefix, ($PriorStep + 5000))
        if (-not (Test-Path -LiteralPath $Rung) -or (Get-Item -LiteralPath $Rung).Length -le 0) { throw "Stage 61 checkpoint ladder is missing or empty: $Rung" }
    }
    Assert-GpuIdleForLaunch
    $Final = Join-Path $ArmCheckpointDir "$ArmCheckpointPrefix.pt"
    if (-not (Test-Path -LiteralPath $Final) -or (Get-Item -LiteralPath $Final).Length -le 0) { throw "Stage 61 final unsuffixed checkpoint is absent: $Final" }
    Invoke-VisiblePython -Label "stage61_instrument_final_for_eval" -RunArgs @(".\experiments\tiny_language_lab\make_stage61_instrumentation.py", "--checkpoint-dir", $ArmCheckpointDir, "--prefix", $ArmCheckpointPrefix, "--jsonl", $Instrumentation, "--summary", $InstrumentationSummary, "--minimum-age-seconds", "0", "--include-final")
    Invoke-VisiblePython -Label "stage61_text8_test_final" -RunArgs @(".\experiments\tiny_language_lab\eval_text8.py", "--split", "test", "--device", "cuda", "--checkpoint", $Final, "--checkpoint-name", "stage61_pure_broad_200m_seed7", "--out-stem", (Join-Path $RunDir "stage61_text8_test"))
    Invoke-VisiblePython -Label "stage61_publish_bars" -RunArgs @(".\experiments\tiny_language_lab\make_stage61_publish_bars.py", "--target-steps", [string]$Target, "--training", $ArmOut, "--instrumentation", $Instrumentation, "--text8", $Text8Out, "--checkpoint-dir", $ArmCheckpointDir, "--prefix", $ArmCheckpointPrefix, "--out", $PublishBarsOut, "--summary", $PublishBarsSummary)
    $Bars = Get-Content -Raw -LiteralPath $PublishBarsOut | ConvertFrom-Json
    Write-LauncherLog "[verified] text8_bar=$($Bars.bars.text8_bar) text8_bpc=$($Bars.text8_test.bits_per_char) margin=$($Bars.bars.text8_margin_vs_bar) instrumentation=$($Bars.bars.instrumentation_bar) packaging=$($Bars.bars.packaging_status)"
}
elseif ($Mode -eq "stage61-user-samples") {
    foreach ($Path in @($PublishBarsOut, $PublishBarsSummary)) {
        if (-not (Test-Path -LiteralPath $Path) -or (Get-Item -LiteralPath $Path).Length -le 0) { throw "Stage 61 user samples require completed publication-bar evidence: $Path" }
    }
    foreach ($Path in @($UserSamplesOut, $UserSamplesSummary)) {
        if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite Stage 61 user-review evidence: $Path" }
    }
    $Bars = Get-Content -Raw -LiteralPath $PublishBarsOut | ConvertFrom-Json
    if ($Bars.bars.text8_bar -ne "PASS" -or $Bars.bars.instrumentation_bar -ne "PASS") {
        throw "Stage 61 user samples are not a publication-recovery path after a failed required bar"
    }
    Assert-GpuIdleForLaunch
    $Final = Join-Path $ArmCheckpointDir "$ArmCheckpointPrefix.pt"
    Invoke-VisiblePython -Label "stage61_user_sample_sheet" -RunArgs @(".\experiments\tiny_language_lab\make_stage61_user_samples.py", "--checkpoint", $Final, "--out", $UserSamplesOut, "--summary", $UserSamplesSummary)
    $Samples = Get-Content -Raw -LiteralPath $UserSamplesOut | ConvertFrom-Json
    if ($Samples.review_status -ne "PENDING_USER_REVIEW" -or @($Samples.samples).Count -ne 8) { throw "Stage 61 user sample sheet is incomplete" }
    Write-LauncherLog "[verified] user_sample_sheet=true samples=$(@($Samples.samples).Count) review_status=$($Samples.review_status)"
}

Write-LauncherLog "[visible] launcher_end=$(Get-Date -Format o) status=success"
if ($KeepOpen) { Read-Host "Run finished. Press Enter to close this window" }
