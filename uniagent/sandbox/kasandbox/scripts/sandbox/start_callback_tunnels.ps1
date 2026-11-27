# Keep this PC terminal open for the training job. Requires working SSH keys.
[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$GatewayCount = 1,
    [ValidateRange(1, 65535)][int]$CallbackPortBase = 29193,
    [ValidateRange(1, 65535)][int]$RelayPortBase = 29193,
    [Parameter(Mandatory)][string]$PcIp,
    [Parameter(Mandatory)][string]$RelayTarget,
    [ValidateRange(1, 65535)][int]$SshPort = 22
)
$ErrorActionPreference = "Stop"
if (($CallbackPortBase + $GatewayCount - 1) -gt 65535 -or
    ($RelayPortBase + $GatewayCount - 1) -gt 65535) {
    throw "Callback port range exceeds 65535"
}
$sshExecutable = (Get-Command ssh.exe -ErrorAction Stop).Source
$callbackProcesses = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()
try {
    for ($gatewayIndex = 0; $gatewayIndex -lt $GatewayCount; $gatewayIndex++) {
        $pcPort = $CallbackPortBase + $gatewayIndex
        $relayPort = $RelayPortBase + $gatewayIndex
        $forward = "${PcIp}:${pcPort}:127.0.0.1:${relayPort}"
        $sshArguments = @(
            "-p", "$SshPort", "-g", "-N", "-T",
            "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
            "-L", $forward, $RelayTarget
        )
        $callbackProcess = Start-Process -FilePath $sshExecutable -ArgumentList $sshArguments `
            -WindowStyle Hidden -PassThru
        $callbackProcesses.Add($callbackProcess)
        Write-Host "gateway $gatewayIndex : $PcIp`:$pcPort -> relay 127.0.0.1:$relayPort"
    }
    Write-Host "Keep this terminal open. Ctrl+C closes only these callback SSH processes."
    while ($true) {
        foreach ($callbackProcess in $callbackProcesses) {
            if ($callbackProcess.HasExited) {
                throw "Callback SSH process $($callbackProcess.Id) exited. Check key authentication, known_hosts and ports."
            }
        }
        Start-Sleep -Seconds 1
    }
}
finally {
    foreach ($callbackProcess in $callbackProcesses) {
        if (-not $callbackProcess.HasExited) {
            Stop-Process -Id $callbackProcess.Id -ErrorAction SilentlyContinue
        }
        $callbackProcess.Dispose()
    }
}
