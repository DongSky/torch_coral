$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force "$root\reports" | Out-Null
$source = Join-Path $root '.tools\libusb-1.0.29'
$archive = Join-Path $root 'downloads\libusb-1.0.29.zip'
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest 'https://github.com/libusb/libusb/archive/refs/tags/v1.0.29.zip' -OutFile $archive
}
$expected = (Get-Content "$root\native\libusb-source-sha256.txt").Trim()
if ((Get-FileHash -LiteralPath $archive).Hash.ToLowerInvariant() -ne $expected) { throw 'libusb source checksum mismatch' }
if (-not (Test-Path -LiteralPath $source)) { Expand-Archive -LiteralPath $archive -DestinationPath "$root\.tools" }
$file = Join-Path $source 'libusb\os\windows_common.c'
$text = [IO.File]::ReadAllText($file)
if (-not $text.Contains('TORCH_CORAL_FORCE_WINUSB')) {
    $needle = 'if (option == LIBUSB_OPTION_USE_USBDK) {'
    if (($text.Split(@($needle), [StringSplitOptions]::None)).Count -ne 2) { throw 'Upstream patch location changed' }
    $patch = [IO.File]::ReadAllText("$root\native\winusb-option.txt").TrimEnd()
    [IO.File]::WriteAllText($file, $text.Replace($needle, $patch), [Text.UTF8Encoding]::new($false))
}
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio C++ tools required to build the project-local libusb DLL' }
& "$vs\MSBuild\Current\Bin\MSBuild.exe" "$source\msvc\libusb_dll.vcxproj" /p:Configuration=Release /p:Platform=x64 /p:PlatformToolset=v143 /v:minimal /nologo
if ($LASTEXITCODE -ne 0) { throw 'libusb build failed' }
$dll = @(Get-ChildItem -LiteralPath $source -Recurse -Filter libusb-1.0.dll | Where-Object FullName -Match 'Release.*dll')
if ($dll.Count -ne 1) { throw 'Expected exactly one compiled libusb DLL' }
$dest = Join-Path $root '.tools\coral-native'
New-Item -ItemType Directory -Force $dest | Out-Null
Copy-Item -LiteralPath $dll[0].FullName -Destination "$dest\libusb-1.0.dll" -Force
Copy-Item -LiteralPath "$root\.tools\edgetpu_runtime\libedgetpu\throttled\x64_windows\edgetpu.dll" -Destination "$dest\edgetpu.dll" -Force
Copy-Item -LiteralPath "$source\COPYING" -Destination "$dest\COPYING-libusb" -Force
Get-FileHash "$dest\*.dll" | Select-Object Hash,Path | ConvertTo-Json | Set-Content -Encoding utf8 "$root\reports\native-runtime-sha256.json"
$runtimeFiles = @{}
foreach ($item in (Get-ChildItem -LiteralPath $dest -Filter '*.dll')) { $runtimeFiles[$item.Name] = (Get-FileHash -LiteralPath $item.FullName).Hash.ToLowerInvariant() }
@{ backend='winusb'; libusb_version='1.0.29'; patch='TORCH_CORAL_FORCE_WINUSB'; files=$runtimeFiles } | ConvertTo-Json | Set-Content -Encoding utf8 "$dest\torch-coral-runtime.json"
