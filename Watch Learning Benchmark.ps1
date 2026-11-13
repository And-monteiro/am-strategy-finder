param(
    [string]$Run = 'runs/programmatic-search-comparison',
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
$root = Join-Path $PSScriptRoot $Run

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
