# Run on Windows and keep open throughout training. Requires SSH key access to both hosts.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$AgentEnvTarget,
    [Parameter(Mandatory)][string]$RelayTarget,
    [Parameter(Mandatory)][string]$AgentEnvIp,
    [Parameter(Mandatory)][string]$AgentEnvWorkDir,
    [Parameter(Mandatory)][string]$RelayWorkDir,
    [int]$AgentEnvSshPort = 22,
    [int]$RelaySshPort = 22,
    [int]$AgentEnvApiPort = 8000,
    [int]$GatewayCount = 8,
    [int]$LocalApiPort = 40190,
    [int]$RelayApiPort = 40191,
    [int]$RelayApiLanePortBase = 31090,
    [int]$CallbackPortBase = 40193,
    [int]$RelayCallbackPortBase = 30193,
    [int]$AgentEnvReversePortBase = 31293,
    [int]$LocalCallbackPortBase = 40393
)
$ErrorActionPreference = 'Stop'
foreach ($port in @($LocalApiPort,$RelayApiPort,$RelayApiLanePortBase,$CallbackPortBase,$RelayCallbackPortBase,$AgentEnvReversePortBase,$LocalCallbackPortBase)) {
    if ($port -lt 1 -or $GatewayCount -lt 1 -or $port+$GatewayCount-1 -gt 65535) { throw 'Invalid port range' }
}
if ($AgentEnvWorkDir -notmatch '^/[a-zA-Z0-9_./-]+$' -or $RelayWorkDir -notmatch '^/[a-zA-Z0-9_./-]+$' -or $AgentEnvIp -notmatch '^[a-zA-Z0-9.-]+$') { throw 'Invalid path/address' }
$children = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()
$logPumps = [System.Collections.Generic.List[System.Threading.Tasks.Task]]::new()
if (-not ('AgentEnvSshLogPump' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
public static class AgentEnvSshLogPump {
    public static Task Start(Process process, string path) {
        return Task.Run(async () => {
            using (var output = new StreamWriter(path, true)) {
                output.AutoFlush = true;
                string line;
                while ((line = await process.StandardError.ReadLineAsync()) != null) {
                    await output.WriteLineAsync(DateTime.UtcNow.AddHours(8).ToString("yyyy-MM-dd HH:mm:ss.fff") + " +0800 " + line);
                }
            }
        });
    }
}
'@
}
$logDir = Join-Path $PSScriptRoot ("artifacts/bridge/" + [DateTime]::UtcNow.AddHours(8).ToString('yyyyMMdd_HHmmss'))
$null = New-Item -ItemType Directory -Force -Path $logDir
$bridgeLog = Join-Path $logDir 'bridge.log'
function Write-BridgeLog([string]$Message) {
    ([DateTime]::UtcNow.AddHours(8).ToString('yyyy-MM-dd HH:mm:ss.fff') + ' +0800 ' + $Message) | Add-Content -LiteralPath $bridgeLog
}
$ssh = (Get-Command ssh.exe).Source
$common = @('-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
            '-o','ExitOnForwardFailure=yes','-o','ConnectTimeout=15',
            '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3')
function Start-BridgeSsh([string[]]$SshArguments, [string]$LogName) {
    $info = [System.Diagnostics.ProcessStartInfo]::new($ssh)
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true
    $info.RedirectStandardError = $true
    foreach ($arg in (@('-v') + $SshArguments)) { $info.ArgumentList.Add($arg) }
    $child = [System.Diagnostics.Process]::Start($info)
    $children.Add($child)
    $logPumps.Add([AgentEnvSshLogPump]::Start($child, (Join-Path $logDir ($LogName + '.ssh.log'))))
    Write-BridgeLog "started pid=$($child.Id) channel=$LogName"
    return $child
}
try {
    # A separate SSH pair per gateway prevents one Windows process from
    # accumulating all 256 rollouts' API streams and model callbacks.
    for ($i=0; $i -lt $GatewayCount; $i++) {
        $relayCallback = $RelayCallbackPortBase+$i
        $reverse = $AgentEnvReversePortBase+$i
        $localPort = $LocalCallbackPortBase+$i
        $apiLocal = $LocalApiPort+$i
        $apiLane = $RelayApiLanePortBase+$i
        $agentArgs = $common + @('-N',
            '-L',"127.0.0.1:${apiLocal}:127.0.0.1:${AgentEnvApiPort}",
            '-R',"127.0.0.1:${reverse}:127.0.0.1:${localPort}")
        $relayArgs = $common + @('-N',
            '-R',"127.0.0.1:${apiLane}:127.0.0.1:${apiLocal}",
            '-L',"127.0.0.1:${localPort}:127.0.0.1:${relayCallback}")
        $null = Start-BridgeSsh ($agentArgs + @('-p', $AgentEnvSshPort, $AgentEnvTarget)) "agentenv-$reverse"
        $null = Start-BridgeSsh ($relayArgs + @('-p', $RelaySshPort, $RelayTarget)) "relay-$relayCallback"
        Start-Sleep -Milliseconds 200
    }
    $relayCommand = "exec python3 -u $RelayWorkDir/api_relay.py --port $RelayApiPort --upstream-start $RelayApiLanePortBase --count $GatewayCount"
    $relay = Start-BridgeSsh ($common + @('-p',$RelaySshPort,$RelayTarget,$relayCommand)) 'api-relay-control'
    $remoteCommand = "exec python3 -u $AgentEnvWorkDir/callback_proxy.py --bind $AgentEnvIp --port $CallbackPortBase --upstream-port $AgentEnvReversePortBase --count $GatewayCount"
    $agent = Start-BridgeSsh ($common + @('-p',$AgentEnvSshPort,$AgentEnvTarget,$remoteCommand)) 'callback-proxy-control'
    Write-Host "SSH logs: $logDir"
    Write-Host "API: relay localhost:$RelayApiPort -> $GatewayCount independent Windows SSH lanes -> AgentENV:8000"
    Write-Host "Callbacks: ${AgentEnvIp}:$CallbackPortBase..$($CallbackPortBase+$GatewayCount-1) -> Windows -> relay -> training gateways"
    Write-Host 'Keep open throughout training. Ctrl+C closes this bridge and its AgentENV listeners.'
    while ($true) {
        foreach ($child in $children) {
            if ($child.HasExited) {
                Write-BridgeLog "exited pid=$($child.Id) code=$($child.ExitCode)"
                throw "Bridge SSH process $($child.Id) exited ($($child.ExitCode))"
            }
        }
        foreach ($pump in $logPumps) {
            if ($pump.IsFaulted) { throw "SSH log writer failed: $($pump.Exception.Message)" }
        }
        $agent.StandardInput.WriteLine('heartbeat')
        $agent.StandardInput.Flush()
        $relay.StandardInput.WriteLine('heartbeat')
        $relay.StandardInput.Flush()
        Start-Sleep -Seconds 5
    }
} finally {
    # Close both control-session inputs first so remote listeners shut down.
    # Bound cleanup for the whole group, rather than waiting 3s per SSH lane.
    foreach ($child in $children) {
        if (-not $child.HasExited) {
            $child.StandardInput.Close()
        }
    }
    $cleanupDeadline = [DateTime]::UtcNow.AddSeconds(5)
    while ([DateTime]::UtcNow -lt $cleanupDeadline -and @($children | Where-Object { -not $_.HasExited }).Count) {
        Start-Sleep -Milliseconds 100
    }
    foreach ($child in $children) {
        if (-not $child.HasExited) { $child.Kill() }
        Write-BridgeLog "cleanup pid=$($child.Id)"
    }
    foreach ($pump in $logPumps) { try { $null = $pump.Wait(3000) } catch { Write-Warning $_ } }
    foreach ($child in $children) { $child.Dispose() }
}
