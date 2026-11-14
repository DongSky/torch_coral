"""Export a strict allowlist and audit publication artifacts. No upload code."""
import argparse
import csv
import datetime
import getpass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STATIC = ['pyproject.toml', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'MANIFEST.in', '.gitignore', '.gitattributes',
          '.github/workflows/ci.yml', 'docs/execution-strategies.md', 'docs/releasing.md', 'docs/development.md',
          'docs/runtime-v2.md', 'docs/custom-kernels.md',
          'examples/direct_inference.py', 'examples/prepared_inference.py', 'tests/test_runtime_v2.py',
          'scripts/benchmark-runtime.py',
          'scripts/verify-runtime-v2.py',
          'docs/qwen-full-tpu.md', 'scripts/start-qwen-chat.ps1', 'scripts/prepare-qwen-environment.ps1',
          'scripts/inspect-gguf-header.py', 'scripts/probe-isa-fp32.py', 'scripts/verify-isa-scalar.py',
          'scripts/run-qwen-tpu.py', 'scripts/validate-qwen-reference.py', 'tests/test_isa_scalar.py',
          'tests/test_qwen_control.py',
          'docs/matrix-acceleration.md', 'tests/test_isa_mxu.py',
          'scripts/benchmark-mxu.py', 'scripts/benchmark-qwen-chat.py',
          'scripts/verify-mxu-recurrent.py', 'scripts/verify-mxu-readout.py',
          'scripts/probe-mxu.py', 'scripts/validate-qwen-chat-reference.py',
          'docs/chain-optimization.md', 'tests/test_isa_optimized.py',
          'scripts/profile-qwen-chain.py', 'scripts/verify-optimized-fusions.py',
          'scripts/benchmark-fast-equivalence.py',
          'docs/turbo-v2.md', 'tests/test_isa_streamed.py', 'tests/test_isa_v2_fusions.py',
          'scripts/probe-scalar-sram.py', 'scripts/tune-recurrent-tiles.py',
          'scripts/verify-streamed-recurrent.py', 'scripts/verify-v2-fusions.py',
          'scripts/benchmark-turbo-v2.py',
          'docs/turbo-v3.md', 'tests/test_isa_schedule.py', 'tests/test_speculative.py',
          'scripts/probe-v3-native-batch.py', 'scripts/benchmark-v3-projections.py',
          'scripts/verify-v3-state-capture.py', 'scripts/verify-v3-convolution.py',
          'scripts/verify-v3-schedule.py', 'scripts/benchmark-v3-speculative.py',
          'scripts/verify-v3-native-retries.py',
          'docs/turbo-v4.md', 'tests/test_usb_buffers.py', 'scripts/benchmark-usb-output-copy.py',
          'tests/test_usb_dispatch.py', 'scripts/benchmark-v4-dispatch.py',
          'tests/test_isa_v4_fusions.py',
          'scripts/verify-v4-fusions.py',
          'docs/turbo-v4-a9.md', 'docs/v4-io-windows.md', 'tests/test_v4_compute.py', 'tests/test_speculative_blocks.py',
          'scripts/verify-v4-compute.py', 'scripts/verify-v4-prefixes.py', 'scripts/verify-v4-blocks.py',
          'scripts/tune-v4-native.py', 'scripts/benchmark-v4-compute.py', 'scripts/benchmark-v4-blocks.py', 'scripts/probe-v4-io.py',
          'docs/chat-studio.md', 'tests/test_chat_metrics.py', 'tests/test_qwen_gui.py',
          'Qwen Coral Chat.cmd',
          'native/tinycoral-windows.patch', 'native/TINYCORAL-LICENSE',
          'examples/basic.py', 'tests/test_distribution.py', 'tests/test_release_audit.py', 'tests/test_experimental_ops.py',
          'tests/test_operator_strategies.py', 'native/README.md', 'native/winusb-option.txt',
          'native/COPYING-LIBUSB', 'native/libusb-source-sha256.txt', 'scripts/build-winusb.ps1',
          'scripts/fetch-compiler.py', 'scripts/fetch-compiler16.py', 'scripts/restart-coral-device.ps1',
          'scripts/prepare-release.py', 'scripts/check-pypi-name.py']
DENIED = {'.git', '.ssh', '.tools', '.venv', '.gui', '.vscode', '.release-tests', '__pycache__', 'artifacts', 'downloads', 'reports'}
PRIVATE_SUFFIXES = {'.dll', '.exe', '.pyc', '.pyo', '.onnx', '.tflite', '.npy', '.npz', '.log', '.gguf', '.safetensors', '.bin', '.pem', '.key'}


def inspect(name, data):
    path = PurePosixPath(name.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe archive path: ' + name)
    if any(p.lower() in DENIED or p.lower().startswith('.venv') for p in path.parts) or path.suffix.lower() in PRIVATE_SUFFIXES:
        raise ValueError('Private file in release: ' + name)
    if (path.name.lower() in ('.env', '.pypirc', 'config.json', 'credentials.json', '.git-credentials')
            or path.name.lower().startswith(('.env.', 'secrets.'))):
        raise ValueError('Private configuration in release: ' + name)
    text = data.decode('utf-8', errors='replace')
    if path.name == 'RECORD' and any(p.endswith('.dist-info') for p in path.parts):
        rows = list(csv.reader(text.splitlines()))
        if any(len(row) != 3 or (row[1] and not re.fullmatch(r'sha256=[A-Za-z0-9_-]{43}', row[1]))
               or (row[2] and not row[2].isdigit()) for row in rows):
            raise ValueError('Invalid wheel RECORD')
        # Digest bytes can coincidentally resemble credential prefixes. Every
        # actual archive member is scanned separately; scan RECORD paths here.
        text = '\n'.join(row[0] for row in rows)
    patterns = [r'(?<![A-Za-z0-9_\-])(?:sk-proj-|hf_|ghp_|github_pat_)[A-Za-z0-9_\-]{20,}',
                r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                r'https?://[^\s/@:]+:[^\s/@]+@', r'[A-Za-z]:[\\/]Users[\\/][^\s/\\]+',
                r'[A-Za-z0-9_.+%\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}',
                r'/' + r'home/[^\s/]+/', r'/' + r'Users/[^\s/]+/']
    identity = getpass.getuser()
    if len(identity) > 3:
        patterns.append(re.escape(identity))
    patterns.extend(re.escape(p) for p in (str(ROOT), ROOT.as_posix()))
    for index, pattern in enumerate(patterns):
        if re.search(pattern, text + '\n' + name, flags=re.I):
            # Do not print a matched secret or local path.
            raise ValueError('Privacy pattern %d matched file %s; content redacted' % (index, name))


def export(destination):
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Use an empty destination; existing trees are never deleted')
    files = STATIC + [p.relative_to(ROOT).as_posix() for p in (ROOT / 'torch_coral').glob('*.py')]
    files += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'torch_coral/dependencies').glob('*.txt')]
    files += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'torch_coral_runtime').glob('*.py')]
    records = []
    for name in sorted(set(files)):
        if (ROOT / name).is_symlink():
            raise ValueError('Source symlinks are not permitted: ' + name)
        data = (ROOT / name).read_bytes()
        inspect(name, data)
        target = destination / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({'path': name, 'sha256': hashlib.sha256(data).hexdigest()})
    return records


def audit_archive(path):
    names = []
    if path.suffix == '.whl' or path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                if not item.is_dir():
                    inspect(item.filename, archive.read(item))
                    names.append(item.filename)
    else:
        with tarfile.open(path) as archive:
            for item in archive.getmembers():
                if item.issym() or item.islnk():
                    raise ValueError('Archive links are not permitted')
                if item.isfile():
                    inspect(item.name, archive.extractfile(item).read())
                    if item.uname or item.gname:
                        raise ValueError('sdist contains owner names; normalize tar headers before publishing')
                    names.append(item.name)
    return {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'members': names, 'privacy': 'PASS'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', default=str(ROOT / 'release/torch-coral'))
    parser.add_argument('--audit', nargs='*')
    parser.add_argument('--report', default=str(ROOT / 'reports/release-audit.json'))
    args = parser.parse_args()
    if args.audit:
        result = {'archives': [audit_archive(Path(p)) for p in args.audit]}
    else:
        result = {'source_files': export(Path(args.destination).resolve()), 'privacy': 'PASS'}
    result['checked_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = Path(args.report); report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Release audit PASS:', len(result.get('source_files', result.get('archives', []))), 'items')


if __name__ == '__main__': main()
