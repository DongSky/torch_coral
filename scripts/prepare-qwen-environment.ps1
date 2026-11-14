$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$frontendPython = Join-Path $projectRoot '.venv-torch/Scripts/python.exe'
$isaDirectory = Join-Path $projectRoot '.venv-isa'
$isaPython = Join-Path $isaDirectory 'Scripts/python.exe'
$tinycoralRoot = Join-Path $projectRoot '.tools/tinycoral'
$pinnedCommit = '60794943cca3ddc63a4438cfff91b520b12bcfef'
if (-not (Test-Path -LiteralPath $frontendPython)) { throw 'Create the Python 3.12 frontend environment first.' }
if (-not (Test-Path -LiteralPath $isaPython)) {
    & python -m uv venv $isaDirectory --python $frontendPython
    if ($LASTEXITCODE) { throw 'ISA environment creation failed.' }
}
& python -m uv pip install --python $isaPython -r (Join-Path $projectRoot 'torch_coral/dependencies/isa-py312.txt')
if ($LASTEXITCODE) { throw 'ISA dependency installation failed.' }
if (-not (Test-Path -LiteralPath $tinycoralRoot)) {
    & git clone https://github.com/FrederikRogalski/tinycoral.git $tinycoralRoot
    if ($LASTEXITCODE) { throw 'tinycoral clone failed.' }
    & git -C $tinycoralRoot checkout $pinnedCommit
    if ($LASTEXITCODE) { throw 'tinycoral checkout failed.' }
}
$actualCommit = (& git -C $tinycoralRoot rev-parse HEAD).Trim()
if ($actualCommit -ne $pinnedCommit) { throw 'Unexpected tinycoral revision; use the documented pinned source.' }
$patchPath = Join-Path $projectRoot 'native/tinycoral-windows.patch'
& git -C $tinycoralRoot apply --reverse --check $patchPath 2>$null
if ($LASTEXITCODE) {
    & git -C $tinycoralRoot apply --check $patchPath
    if ($LASTEXITCODE) { throw 'Windows patch does not apply cleanly; existing files were preserved.' }
    & git -C $tinycoralRoot apply $patchPath
    if ($LASTEXITCODE) { throw 'Windows patch failed.' }
}
& python -m uv pip install --python $isaPython --no-deps -e $tinycoralRoot
if ($LASTEXITCODE) { throw 'tinycoral installation failed.' }
Write-Output 'ISA environment prepared. Runtime DLLs, model weights and tokenizer must be supplied separately.'
