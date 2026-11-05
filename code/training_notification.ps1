param([Parameter(Mandatory=$true)][string]$Message)
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$pdNotice = New-Object System.Windows.Forms.NotifyIcon
$pdNotice.Icon = [System.Drawing.SystemIcons]::Information
$pdNotice.Visible = $true
$pdNotice.BalloonTipTitle = 'Partial discharge training'
$pdNotice.BalloonTipText = $Message
$pdNotice.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Info
try {
    [System.Media.SystemSounds]::Exclamation.Play()
    $pdNotice.ShowBalloonTip(30000)
    Start-Sleep -Seconds 30
} finally {
    $pdNotice.Visible = $false
    $pdNotice.Dispose()
}
