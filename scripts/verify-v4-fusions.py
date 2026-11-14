"""Compare wide primary quantization with the original chain on the Coral."""
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from torch_coral.isa_io import IOOptimizedMatrixTPU
from torch_coral.isa_streamed import StreamedMatrixTPU


def main():
    rows = []
    rng = np.random.default_rng(914)
    with IOOptimizedMatrixTPU(ROOT/'.tools/models/qwen35-gguf/int8-matrices', components=1, output_digits=2, clock='max') as b:
        for length in (1028, 2048, 2052, 3584, 4000):
            for pattern in ('random', 'zero', 'outlier'):
                values = rng.normal(size=length).astype(np.float32)
                if pattern == 'zero':
                    values[:] = 0
                elif pattern == 'outlier':
                    values *= np.float32(1e-4); values[-1] = np.float32(-1e6)
                before = b.calls; begin = time.perf_counter()
                expected = StreamedMatrixTPU.quantize_projection_input(b, values)
                old_time, old_calls = time.perf_counter()-begin, b.calls-before
                before = b.calls; begin = time.perf_counter()
                actual = b.quantize_projection_input(values)
                new_time, new_calls = time.perf_counter()-begin, b.calls-before
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1].view(np.uint32), expected[1].view(np.uint32))
                assert actual[2] is None and new_calls == 1
                row = dict(length=length, pattern=pattern, bitwise=True,
                           original_calls=old_calls, optimized_calls=new_calls,
                           original_ms=old_time*1000, optimized_ms=new_time*1000)
                rows.append(row); print(json.dumps(row), flush=True)
    report = dict(status='bitwise PASS', clock='max', cases=rows)
    (ROOT/'reports/turbo-v4-wide-quantization.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
