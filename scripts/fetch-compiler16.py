"""Install a pinned, isolated v16 compiler from the Ultralytics mirror.

This is a third-party mirror of the legacy Google binary, not a new open-source
compiler. Preserve bundled notices and leave the default v14 installation alone.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/edgetpu-compiler_16.0_amd64.tar.gz'
SHA256 = 'f9a97adbf2ff621d24a91721191264f2abc3c1db31e63d9a266d988f907269ce'


def main():
    archive = ROOT / 'downloads/edgetpu-compiler_16.0_amd64.tar.gz'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        if os.name == 'nt':
            data = subprocess.check_output(['curl.exe', '--fail', '--location', '--silent', '--show-error', URL])
        else:
            with urllib.request.urlopen(URL, timeout=120) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError('Compiler archive SHA256 mismatch')
        archive.write_bytes(data)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Cached compiler archive SHA256 mismatch')
    destination = ROOT / '.tools/compiler16'
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as package:
        package.extractall(destination, filter='data')
    wrapper = destination / 'edgetpu_compiler'
    wrapper.write_text('#!/bin/sh\nd=$(CDPATH= cd -- "$(dirname -- "$0")/usr/bin/edgetpu_compiler_bin" && pwd)\nexec "$d/ld-linux-x86-64.so.2" --library-path "$d" "$d/edgetpu_compiler" "$@"\n', encoding='utf-8', newline='\n')
    paths = [wrapper, destination / 'usr/bin/edgetpu_compiler_bin/edgetpu_compiler',
             destination / 'usr/bin/edgetpu_compiler_bin/ld-linux-x86-64.so.2']
    if os.name == 'nt':
        converted = [subprocess.check_output(['wsl', '-d', 'Ubuntu', '--exec', 'wslpath', '-a', p.as_posix()], text=True).strip() for p in paths]
        subprocess.run(['wsl', '-d', 'Ubuntu', '--exec', 'chmod', '+x', *converted], check=True)
        command = ['wsl', '-d', 'Ubuntu', '--exec', 'bash', converted[0], '--version']
    else:
        for p in paths:
            p.chmod(p.stat().st_mode | 0o111)
        command = [str(wrapper), '--version']
    version = subprocess.check_output(command, text=True).strip()
    if version != 'Edge TPU Compiler version 16.0.384591198':
        raise ValueError('Unexpected compiler version: ' + version)
    report = ROOT / 'reports/compiler16-source.json'
    record = json.loads(report.read_text(encoding='utf-8')) if report.exists() else {}
    record.update(source=URL, publisher='Ultralytics mirror of Google Edge TPU compiler package',
                  sha256=SHA256, version='16.0.384591198', isolated=True,
                  default_compiler_unchanged='14.1.317412892')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(version + '; verified archive SHA256; v14 default unchanged.')


if __name__ == '__main__':
    main()
