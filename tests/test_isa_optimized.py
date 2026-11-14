"""Offline instruction checks, not evidence of physical speed or support."""
import numpy as np
import pytest
pytest.importorskip('coral.isa.scalar_core')
from coral.isa.scalar_core import run
from torch_coral.isa_scalar import ScalarTPU
from torch_coral.isa_mxu import MatrixTPU
from torch_coral.isa_optimized import OptimizedMatrixTPU
from torch_coral.qwen_tpu import QwenTPU


class Emulated:
    def __init__(self): pass
    def _run(self, program, payload, count):
        _, words, base = program
        memory = bytearray(16384)
        memory[base:base + len(payload)] = payload
        run(words, memory, load_latency=5)
        return np.frombuffer(memory, '<f4', count=count).copy()


class Old(Emulated, MatrixTPU): pass
class New(Emulated, OptimizedMatrixTPU): pass


def engine(backend):
    model = object.__new__(QwenTPU)
    model.backend, model.eps, model.cancel = backend, 1e-6, None
    return model


@pytest.mark.parametrize('kind', ['sigmoid', 'silu'])
def test_activation_bitwise_matches_old_instruction_path(kind):
    x = np.concatenate([np.linspace(-100, 100, 64, dtype=np.float32), np.zeros(4, np.float32)])
    expected = getattr(engine(Old()), kind)(x)
    actual = getattr(engine(New()), kind)(x)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('l2', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_normalization_preserves_rounding_and_zero_rows(l2, weighted):
    x = np.random.default_rng(1).normal(size=(3, 128)).astype(np.float32)
    x[0] = 0
    weight = np.linspace(.1, 2., 128, dtype=np.float32) if weighted else None
    expected = engine(Old()).norm(x, weight, l2=l2)
    actual = engine(New()).norm(x, weight, l2=l2)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('length', [16, 128, 1024])
def test_fused_input_quantization_matches_bytes_scale_and_residual(length):
    x = np.random.default_rng(2).normal(size=length).astype(np.float32)
    for values in (x, np.zeros_like(x)):
        old = Old().quantize_input(values)
        new = New().quantize_input(values)
        for actual, expected in zip(new, old): np.testing.assert_array_equal(actual, expected)


def test_parameter_storage_is_bounded_and_reuses_unchanged_bytes():
    from collections import OrderedDict
    class Program:
        def params(self, w, b, zp): return w.tobytes()
    backend = New()
    backend.parameter_cache_limit = 16
    backend._parameter_cache = OrderedDict()
    backend._parameter_cache_size = 0
    backend.matrix_stats = dict(host_cache_hits=0, host_cache_misses=0, host_cache_bytes=0)
    w = np.zeros((8, 1), np.uint8)
    first = backend.parameter_blob(Program(), 'a', w)
    assert backend.parameter_blob(Program(), 'a', w) is first
    backend.parameter_blob(Program(), 'b', w)
    backend.parameter_blob(Program(), 'c', w)
    assert backend._parameter_cache_size == 16
    assert 'a' not in backend._parameter_cache
    assert backend.matrix_stats['host_cache_hits'] == 1


def test_convolution_silu_order_is_bitwise_identical():
    rng = np.random.default_rng(81)
    window = rng.normal(size=(4, 256)).astype(np.float32)
    kernel = rng.normal(size=(256, 4)).astype(np.float32)
    old = Old(); expected = old.op('mul', window[0], kernel[:, 0])
    for tap in range(1, 4): expected = old.op('add', expected, old.op('mul', window[tap], kernel[:, tap]))
    expected = engine(old).silu(expected)
    np.testing.assert_array_equal(New().convolution_silu(window, kernel), expected)
