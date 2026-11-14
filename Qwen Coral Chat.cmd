@echo off
if not exist "%~dp0.venv-torch\Scripts\pythonw.exe" (
  echo Create the Python frontend environment first.
  pause
  exit /b 1
)
start "" /D "%~dp0" "%~dp0.venv-torch\Scripts\pythonw.exe" -m torch_coral.qwen_chat_gui
