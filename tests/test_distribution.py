import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from torch_coral import settings


def test_config_and_environment_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCH_CORAL_HOME', str(tmp_path))
    monkeypatch.delenv('CORAL_WORKER_PYTHON', raising=False)
    first, second = tmp_path / 'first-python', tmp_path / 'second-python'
    settings.write_config({'worker_python': str(first), 'compiler16': '/opt/coral/compiler'})
    assert settings.python_executable('worker') == os.path.normcase(str(first))
    monkeypatch.setenv('CORAL_WORKER_PYTHON', str(second))
    assert settings.python_executable('worker') == os.path.normcase(str(second))
    assert settings.python_executable('worker', str(first)) == os.path.normcase(str(first))
    assert settings.compiler_path(strategy='coral') == '/opt/coral/compiler'
    with pytest.raises(ValueError, match='Unknown'):
        settings.write_config({'unexpected': 'value'})


def test_installed_default_does_not_write_into_site_packages(tmp_path, monkeypatch):
    monkeypatch.delenv('TORCH_CORAL_HOME', raising=False)
    monkeypatch.setattr(settings, 'source_root', lambda: None)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path))
    assert settings.workspace().name == 'torch-coral'


def test_worker_helpers_parse_as_python39():
    root = Path(settings.__file__).parent
    for name in ('worker.py', 'protocol.py', 'shared.py', 'native_runtime.py', 'settings.py'):
        ast.parse((root / name).read_text(encoding='utf-8'), feature_version=(3, 9))
    for path in root.parent.joinpath('torch_coral_runtime').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 9))


def test_requirements_are_available_without_importing_torch():
    result = subprocess.run([sys.executable, '-c',
        "import sys,torch_coral;from pathlib import Path;print(torch_coral.__version__);assert 'torch' not in sys.modules;assert Path(torch_coral.__file__).with_name('dependencies').joinpath('convert.txt').is_file()"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_configure_cli_writes_only_selected_home(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCH_CORAL_HOME', str(tmp_path))
    run = subprocess.run([sys.executable, '-m', 'torch_coral', 'configure', '--wsl-distro', 'ExampleDistro'],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert json.loads((tmp_path / 'config.json').read_text()) == {'wsl_distro': 'ExampleDistro'}
