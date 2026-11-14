"""Installation-safe configuration, also importable by the Python 3.9 worker."""
import json
import os
from pathlib import Path
import shutil
import sys


def source_root():
    candidate = Path(__file__).resolve().parents[1]
    if (candidate / 'pyproject.toml').is_file() and (candidate / 'torch_coral').is_dir():
        return candidate
    return None


def workspace():
    override = os.environ.get('TORCH_CORAL_HOME')
    if override:
        return Path(override).expanduser().resolve()
    checkout = source_root()
    if checkout is not None:
        return checkout
    if os.name == 'nt':
        base = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData/Local')))
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library/Application Support'
    else:
        base = Path(os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local/share')))
    return base / 'torch-coral'


def read_config():
    path = workspace() / 'config.json'
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('torch-coral config.json must contain a JSON object')
    return value


def write_config(values):
    allowed = {'worker_python', 'converter_python', 'compiler', 'compiler16', 'runtime_dir', 'wsl_distro'}
    if set(values) - allowed:
        raise ValueError('Unknown torch-coral configuration key')
    config = read_config()
    config.update({key: str(value) for key, value in values.items() if value is not None})
    path = workspace() / 'config.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(config, indent=2), encoding='utf-8')
    temporary.replace(path)
    return path


def python_executable(kind, explicit=None, required=True):
    if kind not in ('worker', 'converter'):
        raise ValueError('Unknown interpreter role')
    key = kind + '_python'
    env = 'CORAL_WORKER_PYTHON' if kind == 'worker' else 'CORAL_CONVERTER_PYTHON'
    chosen = explicit or os.environ.get(env) or read_config().get(key)
    if chosen:
        found = shutil.which(str(chosen))
        return os.path.normcase(str(Path(found or chosen).expanduser().resolve()))
    directory = '.venv-coral' if kind == 'worker' else '.venv-convert'
    executable = 'Scripts/python.exe' if os.name == 'nt' else 'bin/python'
    roots = [r for r in (source_root(), workspace()) if r is not None]
    for root in roots:
        candidate = root / directory / executable
        if candidate.is_file():
            return os.path.normcase(str(candidate.resolve()))
    if kind == 'converter':
        import importlib.util
        if importlib.util.find_spec('onnx2tf') is not None and importlib.util.find_spec('ai_edge_litert') is not None:
            return sys.executable
    if required:
        raise RuntimeError('Configure %s with %s or `torch-coral configure --%s-python PATH`; see the installation guide.' % (key, env, kind))
    return None


def compiler_path(explicit=None, strategy=None):
    key = 'compiler16' if strategy == 'coral' else 'compiler'
    env = 'CORAL_COMPILER16' if strategy == 'coral' else 'CORAL_COMPILER'
    chosen = explicit or os.environ.get(env) or read_config().get(key)
    if chosen:
        return str(chosen)
    suffix = '.tools/compiler16/edgetpu_compiler' if strategy == 'coral' else '.tools/compiler/x86_64/edgetpu_compiler'
    for root in [r for r in (source_root(), workspace()) if r is not None]:
        candidate = root / suffix
        if candidate.exists():
            return str(candidate)
    if os.name != 'nt':
        found = shutil.which('edgetpu_compiler')
        if found:
            return found
    raise FileNotFoundError('Configure the Edge TPU compiler with %s or `torch-coral configure --%s PATH`. v16 is required for operator_strategy="coral".' % (env, key))


def runtime_directory():
    chosen = os.environ.get('CORAL_RUNTIME_DIR') or read_config().get('runtime_dir')
    if chosen:
        return Path(chosen).expanduser().resolve()
    checkout = source_root()
    if checkout is not None:
        return checkout / '.tools/coral-native'
    return workspace() / 'runtime'
