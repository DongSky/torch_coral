"""Explicit physical-device probe of reverse-engineered scalar FP32 instructions.

Requires the pinned tinycoral checkout with native/tinycoral-windows.patch.
Never run simultaneously with libedgetpu or another USB owner.
"""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from torch_coral.native_runtime import configure
    from torch_coral.settings import runtime_directory
    backend = configure()
    if os.name == 'nt':
        os.environ['LIBUSB_PATH'] = str(runtime_directory() / 'libusb-1.0.dll')
    os.environ['CORAL_ASYNC'] = '0'
    from torch_coral.isa_scalar import ScalarTPU
    report = {'scope': 'FP32 scalar instruction hardware probes only; not full-model support',
              'backend': backend, 'results': []}
    tpu = ScalarTPU()
    try:
        for kind, constant, expected in [('add', 1.5, lambda x: x + np.float32(1.5)),
                                          ('mul', 0.25, lambda x: x * np.float32(.25)),
                                          ('exp', None, np.exp), ('recip', None, np.reciprocal)]:
            x = np.linspace(.5, 2.0, 32, dtype=np.float32)
            start = time.perf_counter()
            actual = tpu.op(kind, x, np.float32(constant) if constant is not None else None)
            elapsed = (time.perf_counter() - start) * 1000
            reference = expected(x)
            error = float(np.max(np.abs(actual - reference)))
            np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-5)
            report['results'].append(dict(instruction=kind, status='PASS', max_abs_error=error, elapsed_ms=elapsed))
            print(kind, 'PASS', error, elapsed, flush=True)
    finally:
        tpu.close()
    (ROOT / 'reports/isa-fp32-probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
