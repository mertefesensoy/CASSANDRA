param(
    [Parameter(Mandatory = $true)]
    [int]$WatchPid,
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"

if ($LogPath -eq "") {
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $LogPath = Join-Path $ScriptDir "runs\keep_awake.log"
}

$LogDir = Split-Path -Parent $LogPath
if ($LogDir -ne "" -and -not (Test-Path -LiteralPath $LogDir)) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}

function Write-KeepAwakeLog {
    param([string]$Message)
    $Stamp = Get-Date -Format o
    "[$Stamp] $Message" | Out-File -LiteralPath $LogPath -Encoding utf8 -Append
}

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class ExecutionState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$ES_CONTINUOUS = [Convert]::ToUInt32("80000000", 16)
$ES_SYSTEM_REQUIRED = [Convert]::ToUInt32("00000001", 16)
$ES_DISPLAY_REQUIRED = [Convert]::ToUInt32("00000002", 16)
$Flags = $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_DISPLAY_REQUIRED

Write-KeepAwakeLog "start watch_pid=$WatchPid flags=system+display"

try {
    while (Get-Process -Id $WatchPid -ErrorAction SilentlyContinue) {
        $Result = [ExecutionState]::SetThreadExecutionState($Flags)
        if ($Result -eq 0) {
            Write-KeepAwakeLog "warning SetThreadExecutionState returned 0"
        } else {
            Write-KeepAwakeLog "tick watch_pid=$WatchPid"
        }
        Start-Sleep -Seconds 30
    }
    Write-KeepAwakeLog "watched pid exited"
}
finally {
    [void][ExecutionState]::SetThreadExecutionState($ES_CONTINUOUS)
    Write-KeepAwakeLog "stop"
}
