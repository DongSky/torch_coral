"""Host-side contract tests. No Coral device is opened by these tests."""
import ast
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading


def test_worker_starts_with_open_stdin_and_exits_without_claiming_usb():
    env = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    process = subprocess.Popen([sys.executable, '-u', '-m', 'torch_coral.qwen_chat_worker'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', env=env)
    lines = queue.Queue()
    def read():
        for line in process.stdout:
            lines.put(json.loads(line))
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        assert lines.get(timeout=20)['event'] == 'ready'
        process.stdin.write(json.dumps({'action': 'shutdown'}) + '\n')
        process.stdin.flush()
        assert lines.get(timeout=10)['event'] == 'closed'
        process.wait(timeout=10)
        assert process.returncode == 0, process.stderr.read()
    finally:
        # No load command was sent, so this process cannot own a TPU handle.
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        for stream in (process.stdin, process.stdout, process.stderr): stream.close()


def test_inference_module_contains_no_cpu_tensor_arithmetic_fallback():
    import torch_coral.qwen_tpu as module
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    allowed_numpy = {'float32', 'stack', 'concatenate', 'split', 'empty_like', 'zeros'}
    for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        for method in (node for node in cls.body if isinstance(node, ast.FunctionDef)):
            if method.name == '__init__':
                continue  # immutable RoPE tables are prepared before inference
            for node in ast.walk(method):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'np':
                        assert node.func.attr in allowed_numpy, (method.name, node.func.attr)
    imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert 'torch' not in imports
