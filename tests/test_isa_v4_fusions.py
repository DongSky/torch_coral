"""Bitwise wide-quantization scale/bytes and in-place aliasing contracts."""
import numpy as np
import pytest
pytest.importorskip('coral.isa.scalar_core')
from coral.isa.scalar_core import run
from torch_coral.isa_io import IOOptimizedMatrixTPU
from torch_coral.isa_streamed import StreamedMatrixTPU
from torch_coral.isa_v4_fusions import wide_primary_quantize_program


class Emulated:
    def __init__(self):
        self.components = 1
        self.wide_projection_quantization = True
        self._quant_scope = None
        self.matrix_stats = dict(wide_quantization_calls=0)

    def _run(self, program, payload, count):
        _, words, base = program
        memory = bytearray(16384)
        memory[base:base + len(payload)] = payload
        run(words, memory, load_latency=5)
        return np.frombuffer(memory, '<f4', count=count).copy()

    def _inplace_result(self, program, payload, count):
        _, words, base, output_base = program
        memory = bytearray(16384)
        memory[base:base + len(payload)] = payload
        run(words, memory, load_latency=5)
        return np.frombuffer(memory, '<f4', count=count, offset=output_base).copy()


class Original(Emulated, StreamedMatrixTPU):
    pass


class Optimized(Emulated, IOOptimizedMatrixTPU):
    pass


@pytest.mark.parametrize('length', [1028, 2048, 2052, 3584, 4000])
@pytest.mark.parametrize('pattern', ['random', 'zero', 'outlier'])
def test_primary_bytes_and_scale_match_original_wide_chain(length, pattern):
    values = np.random.default_rng(102).normal(size=length).astype(np.float32)
    if pattern == 'zero':
        values[:] = 0
    elif pattern == 'outlier':
        values *= np.float32(1e-4)
        values[-1] = np.float32(-1e6)
    backend = Optimized()
    actual = backend.quantize_projection_input(values)
    expected = Original().quantize_projection_input(values)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1].view(np.uint32), expected[1].view(np.uint32))
    assert actual[2] is None
    assert backend.matrix_stats['wide_quantization_calls'] == 1


def test_wide_kernel_bounds_and_residual_profiles_keep_original_path():
    for length in (0, 3, 1025, 4004):
        with pytest.raises(ValueError):
            wide_primary_quantize_program(length)
    backend = Optimized()
    backend.components = 2
    values = np.linspace(-2, 2, 1028, dtype=np.float32)
    actual = backend.quantize_projection_input(values)
    expected = Original().quantize_input(values)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
    assert backend.matrix_stats['wide_quantization_calls'] == 0
