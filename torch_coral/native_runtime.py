"""Python 3.9-compatible private DLL loading; no torch imports."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
try:
    from .settings import runtime_directory
except ImportError:
    from settings import runtime_directory

_handles = []

def configure():
    folder = runtime_directory()
    if os.name == "nt" and folder.is_dir():
        marker = folder / 'torch-coral-runtime.json'
        verified = False
        if marker.is_file():
            record = json.loads(marker.read_text(encoding='utf-8-sig'))
            for name in ('libusb-1.0.dll', 'edgetpu.dll'):
                expected = record.get('files', {}).get(name)
                if not expected or hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError('Private Coral runtime checksum mismatch: ' + name)
            verified = record.get('backend') == 'winusb' and record.get('patch') == 'TORCH_CORAL_FORCE_WINUSB'
        if verified:
            os.environ["TORCH_CORAL_FORCE_WINUSB"] = "1"
        if not _handles:
            _handles.extend([os.add_dll_directory(str(folder)),
                ctypes.WinDLL(str(folder / "libusb-1.0.dll"), winmode=8),
                ctypes.WinDLL(str(folder / "edgetpu.dll"), winmode=8)])
        return "WinUSB (verified local libusb)" if verified else 'Custom runtime (USB backend selected by supplied libraries)'
    return "system runtime"
