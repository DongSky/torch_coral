"""Local configuration and diagnostics; never installs drivers or publishes."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from . import __version__
from . import settings


def main(argv=None):
    parser = argparse.ArgumentParser(prog='torch-coral')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    configure = sub.add_parser('configure', help='Save local environment paths')
    for key in ('worker-python', 'converter-python', 'compiler', 'compiler16', 'runtime-dir', 'wsl-distro'):
        configure.add_argument('--' + key)
    doctor = sub.add_parser('doctor', help='Inspect configuration; no USB claim by default')
    doctor.add_argument('--probe', action='store_true', help='Enumerate devices using the isolated worker')
    dependencies = sub.add_parser('dependencies', help='Print the path of a packaged environment requirements file')
    dependencies.add_argument('role', choices=['convert', 'worker-windows-py39', 'isa-py312'])
    args = parser.parse_args(argv)
    if args.command == 'dependencies':
        print(Path(__file__).with_name('dependencies') / (args.role + '.txt'))
        return 0
    if args.command == 'configure':
        values = {k: v for k, v in vars(args).items() if k != 'command' and v is not None}
        if not values:
            parser.error('configure requires at least one option')
        path = settings.write_config(values)
        print(json.dumps({'config_file': str(path), 'configured_keys': sorted(values)}, indent=2))
        return 0
    report = {'version': __version__, 'python': sys.version.split()[0],
              'home': str(settings.workspace()), 'config_keys': sorted(settings.read_config()),
              'worker_python': settings.python_executable('worker', required=False),
              'converter_python': settings.python_executable('converter', required=False),
              'runtime_dir': str(settings.runtime_directory())}
    for key, strategy in [('compiler', None), ('compiler16', 'coral')]:
        try:
            report[key] = settings.compiler_path(strategy=strategy)
        except FileNotFoundError:
            report[key] = None
    if args.probe:
        python = settings.python_executable('worker')
        run = subprocess.run([python, str(Path(__file__).with_name('worker.py')), ''],
                             input=b'', capture_output=True, timeout=30)
        from .protocol import decode, read_packet
        import io
        try:
            _, metadata = decode(read_packet(io.BytesIO(run.stdout)))
            report['worker_probe'] = metadata
        except Exception:
            report['worker_probe'] = {'error': 'Worker did not return a valid packet', 'exit_code': run.returncode}
    print(json.dumps(report, indent=2))
    return int(bool(args.probe and not report.get('worker_probe', {}).get('ready')))


if __name__ == '__main__':
    sys.exit(main())
