$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$chatPython = Join-Path $projectRoot '.venv-torch/Scripts/pythonw.exe'
if (-not (Test-Path -LiteralPath $chatPython)) { throw 'Create the frontend environment before starting the chat GUI.' }
Start-Process -FilePath $chatPython -ArgumentList @('-m','torch_coral.qwen_chat_gui') -WorkingDirectory $projectRoot -WindowStyle Normal
