param(
    [string]$Run = 'runs/programmatic-search-8h',
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
$root = Join-Path $PSScriptRoot $Run
$progressPython = $null

function Format-ActiveBatch($batch) {
    if ($null -ne $batch.active) {
        return ' | batch {0} in progress ({1:N0}/{2:N0} trials)' -f $batch.active, $batch.done, $batch.total
    }
    return ''
}

function Read-RunJson([string]$Path) {
    for ($attempt = 0; $attempt -lt 3; $attempt++) {
        try {
            return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        } catch [System.Management.Automation.ItemNotFoundException] {
            return $null
        } catch [System.IO.IOException] {
            if ($attempt -eq 2) { throw }
            Start-Sleep -Milliseconds 100
        }
    }
}

do {
    $status = Read-RunJson (Join-Path $root 'status.json')
    $protocol = Read-RunJson (Join-Path $root 'protocol.json')
    Write-Host ""
    Write-Host (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
    if ($null -eq $status) {
        Write-Host "Waiting for run status in $root"
    } else {
        $batchStats = $null
        if ($protocol.PSObject.Properties['search_version']) {
            if (-not $progressPython) {
                $process = Get-Process -Id $status.pid -ErrorAction SilentlyContinue
                $hint = Read-RunJson (Join-Path $root 'monitor-runtime.json')
                $candidates = @($process.Path, $hint.python)
                $candidates += @(Get-ChildItem "$env:ProgramFiles/WindowsApps/OpenAI.CodexPrimaryRuntime*/dependencies/python/python.exe" -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending | Select-Object -ExpandProperty FullName)
                $candidates += (Join-Path $PSScriptRoot '.venv/Scripts/python.exe')
                foreach ($candidate in $candidates) {
                    if ($candidate -and (Test-Path -LiteralPath $candidate)) {
                        try {
                            $output = & $candidate (Join-Path $PSScriptRoot 'strategy_finder/monitor_progress.py') $root 2>$null
                        } catch {
                            continue
                        }
                        if ($LASTEXITCODE -eq 0) {
                            $progressPython = $candidate
                            $batchStats = $output | ConvertFrom-Json
                            break
                        }
                    }
                }
            } else {
                $output = & $progressPython (Join-Path $PSScriptRoot 'strategy_finder/monitor_progress.py') $root
                if ($LASTEXITCODE -eq 0) { $batchStats = $output | ConvertFrom-Json }
            }
        }
        Write-Host "Phase $($status.phase) | PID $($status.pid)"
        if ($null -ne $status.reserve_seconds) {
            Write-Host "Holdout reserve $($status.reserve_seconds)s | Estimated unique candidates $($status.holdout_candidates) | Time left $($status.remaining_seconds)s"
        }
        if ($status.deadline) {
            $deadline = [DateTimeOffset]::Parse($status.deadline.ToString()).ToLocalTime()
            Write-Host "Deadline $($deadline.ToString('yyyy-MM-dd HH:mm:ss zzz'))"
        }
        foreach ($market in $protocol.market_order) {
            Write-Host $market
            $shared = $batchStats.$market.shared
            if ($null -ne $shared) {
                $warmup = if ($protocol.warmup_batches) { $protocol.warmup_batches } else { 2 }
                Write-Host ('  Shared warm-up       {0}/{1} batches complete{2}' -f $shared.complete, $warmup, (Format-ActiveBatch $shared))
            }
            $marketRoot = Join-Path $root $market
            $clock = Read-RunJson (Join-Path $marketRoot 'budget-clock.json')
            $marketStatus = Read-RunJson (Join-Path $marketRoot 'status.json')
            if ($clock -and ($status.market -eq $market -or $marketStatus.phase -in @('shared_warmup', 'discovery'))) {
                $deadline = [DateTimeOffset]::FromUnixTimeSeconds([long][Math]::Floor($clock.deadline))
                $remaining = [Math]::Max(0, ($deadline - [DateTimeOffset]::UtcNow).TotalSeconds)
                Write-Host ('  Market time remaining {0:hh\:mm\:ss}' -f [TimeSpan]::FromSeconds($remaining))
            }
            if ($null -ne $marketStatus.reserve_seconds -and $marketStatus.phase -eq 'discovery') {
                Write-Host "  Holdout reserve $($marketStatus.reserve_seconds)s | Estimated unique candidates $($marketStatus.holdout_candidates)"
            }
            foreach ($arm in $protocol.arms) {
                $directory = Join-Path (Join-Path $root $market) $arm
                $endpoint = Read-RunJson (Join-Path $directory 'endpoint.json')
                $progress = Read-RunJson (Join-Path $directory 'progress.json')
                $message = 'waiting'
                if ($endpoint) {
                    $message = '{0:N0} backtests, {1} batches, frozen' -f $endpoint.backtests, $endpoint.completed_batches
                } elseif ($progress) {
                    $message = '{0:N0} backtests, {1} batches' -f $progress.backtests, $progress.completed_batches
                }
                $liveBatch = $batchStats.$market.$arm
                if ($null -ne $liveBatch -and -not $endpoint) {
                    $message = '{0} adaptive batches complete{1}' -f $liveBatch.complete, (Format-ActiveBatch $liveBatch)
                }
                Write-Host ('  {0,-20} {1}' -f $arm, $message)
            }
        }
        if ($status.phase -eq 'complete') {
            Write-Host "Report $(Join-Path $root $status.report)"
            break
        }
        if ($status.phase -eq 'failed') {
            Write-Host "Stopped. $($status.error)"
            break
        }
    }
    if ($Once) { break }
    Write-Host 'Refreshes every 30 seconds. Ctrl+C closes this monitor.'
    Start-Sleep -Seconds 30
} while ($true)
