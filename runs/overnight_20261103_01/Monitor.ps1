# Read-only status display. Ctrl+C closes this display, not the training process.
while ($true) {
    Clear-Host
    Write-Host 'Unattended VGG16 + ResNet50 training'
    Write-Host (Get-Date)
    $pdStatusPath = Join-Path $PSScriptRoot 'queue_status.json'
    if (Test-Path -LiteralPath $pdStatusPath) {
        $pdQueueState = Get-Content -LiteralPath $pdStatusPath -Raw | ConvertFrom-Json
        $pdQueueState | Select-Object status,job,attempt,available_ram_gib,updated_utc,message | Format-List
        if ($pdQueueState.log) {
            Write-Host 'Current training log:'
            Get-Content -LiteralPath $pdQueueState.log -Tail 8 -ErrorAction SilentlyContinue
        }
        $pdLatestHistory = Get-ChildItem -LiteralPath $PSScriptRoot -Filter 'training_log.csv' -Recurse -File |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($pdLatestHistory) {
            Write-Host "Latest completed-epoch history: $($pdLatestHistory.FullName)"
            Get-Content -LiteralPath $pdLatestHistory.FullName -Tail 3
        }
        if ($pdQueueState.status -in @('COMPLETE','FAILED','INTERRUPTED','STOPPED_BY_USER')) { break }
    }
    Start-Sleep -Seconds 15
}
