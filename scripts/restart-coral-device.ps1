#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$devices = @(Get-PnpDevice -PresentOnly | Where-Object InstanceId -Match '^USB\\VID_(18D1&PID_9302|1A6E&PID_089A)\\')
if ($devices.Count -ne 1) { throw 'Expected exactly one Coral USB device; no devices were restarted.' }
Start-Transcript -Path "$root\reports\coral-device-restart.log" -Force
try {
    & pnputil.exe /restart-device $devices[0].InstanceId
    $LASTEXITCODE | Set-Content "$root\reports\coral-device-restart-exit.txt"
} finally { Stop-Transcript }
