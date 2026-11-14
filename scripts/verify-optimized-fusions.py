"""Physical bitwise comparisons against the original FP32 instruction chains."""
from pathlib import Path
import json
import argparse
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch_coral.isa_optimized import OptimizedMatrixTPU
from torch_coral.isa_mxu import MatrixTPU
from torch_coral.qwen_tpu import QwenTPU


def assert_bits(actual, expected):
    np.testing.assert_array_equal(np.ascontiguousarray(actual).view(np.uint8), np.ascontiguousarray(expected).view(np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--async-usb', action='store_true')
    args = parser.parse_args()
    rows = []
    rng = np.random.default_rng(61)
    with OptimizedMatrixTPU('.tools/models/qwen35-gguf/int8-matrices', components=1, output_digits=2, async_usb=args.async_usb) as b:
        class Original:
            op = b.op
            reduce = b.reduce
        old = object.__new__(QwenTPU); old.backend, old.eps, old.cancel = Original(), 1e-6, None
        for length in (128, 1024):
            x = rng.normal(size=(2, length)).astype(np.float32); x[0] = 0
            w = rng.normal(size=length).astype(np.float32)
            for l2 in (False, True):
                for weight in (None, w):
                    expected = old.norm(x, weight, l2=l2)
                    actual = b.normalize(x, weight, l2, old.eps)
                    assert_bits(actual, expected)
                    rows.append(dict(operation='norm', length=length, l2=l2, weighted=weight is not None))
            for values in (x[0], x[1]):
                expected = MatrixTPU.quantize_input(b, values)
                actual = b.quantize_input(values)
                for a, e in zip(actual, expected): assert_bits(a, e)
                rows.append(dict(operation='quantize', length=length))
        x = np.concatenate([rng.normal(size=128).astype(np.float32), np.linspace(-100, 100, 128, dtype=np.float32)])
        for kind in ('sigmoid', 'silu'):
            expected = getattr(old, kind)(x)
            actual = b.activation(kind, x)
            assert_bits(actual, expected)
            rows.append(dict(operation=kind, length=len(x)))
        window = rng.normal(size=(4, 256)).astype(np.float32)
        kernel = rng.normal(size=(256, 4)).astype(np.float32)
        expected = b.op('mul', window[0], kernel[:, 0])
        for tap in range(1, 4): expected = b.op('add', expected, b.op('mul', window[tap], kernel[:, tap]))
        expected = old.silu(expected)
        assert_bits(b.convolution_silu(window, kernel), expected)
        rows.append(dict(operation='conv_silu', length=256))
        state = rng.normal(size=(128, 128)).astype(np.float32)
        k, q, v = rng.normal(size=(3, 128)).astype(np.float32)
        beta, decay = np.float32(.3), np.float32(.9)
        current = b.op('mul', state, decay)
        delta = b.op('mul', b.op('sub', v, b.matvec(current.T, k)), beta)
        current = b.op('add', current, b.op('mul', k[:, None], delta[None, :]))
        expected = b.matvec(current.T, q)
        actual_state, actual = b.recurrent(state, k, q, v, beta, decay)
        assert_bits(actual_state, current)
        assert_bits(actual, expected)
        rows.append(dict(operation='recurrent_13_columns', length=128))
    suffix = '-async' if args.async_usb else ''
    Path('reports/optimized-fusions-physical' + suffix + '.json').write_text(json.dumps(dict(status='bitwise PASS', cases=rows), indent=2))
    print('Physical bitwise comparisons PASS:', len(rows))


if __name__ == '__main__': main()
