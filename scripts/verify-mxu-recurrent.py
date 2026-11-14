"""Physical FP32 recurrent fusion versus the original physical TPU operations."""
from pathlib import Path
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch_coral.isa_mxu import MatrixTPU


def main():
    rng = np.random.default_rng(71)
    rows = []
    with MatrixTPU('.tools/models/qwen35-gguf/int8-matrices', components=2, output_digits=3) as b:
        for trial in range(3):
            state = rng.normal(size=(128, 128)).astype(np.float32)
            k, q, v = rng.normal(size=(3, 128)).astype(np.float32)
            beta, decay = np.float32(.3), np.float32(.9)
            t = time.perf_counter()
            current = b.op('mul', state, decay)
            memory = b.matvec(current.T, k)
            delta = b.op('mul', b.op('sub', v, memory), beta)
            current = b.op('add', current, b.op('mul', k[:, None], delta[None, :]))
            expected = b.matvec(current.T, q)
            original_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            updated, actual = b.recurrent(state, k, q, v, beta, decay)
            fused_ms = (time.perf_counter() - t) * 1000
            np.testing.assert_array_equal(updated, current)
            np.testing.assert_array_equal(actual, expected)
            row = dict(trial=trial, original_ms=original_ms, fused_ms=fused_ms, status='bitwise PASS')
            rows.append(row)
            print(row, flush=True)
    Path('reports/mxu-recurrent.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__': main()
