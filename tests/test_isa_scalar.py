"""Offline instruction execution checks, separate from physical-device tests."""
import numpy as np
import pytest
pytest.importorskip('coral.isa.scalar_core')
from coral.isa.scalar_core import run
from torch_coral.isa_scalar import elementwise, reduction, gemv, gemm, argmax_program


def execute(program, values, count):
    _, words, base = program
    memory = bytearray(16384)
    payload = np.asarray(values, dtype='<f4').tobytes()
    memory[base:base + len(payload)] = payload
    run(words, memory, load_latency=5)
    return np.frombuffer(memory, dtype='<f4', count=count).copy()


@pytest.mark.parametrize('kind,operation', [('add', np.add), ('sub', np.subtract),
    ('mul', np.multiply), ('max', np.maximum), ('min', np.minimum)])
def test_binary_instructions(kind, operation):
    rng = np.random.default_rng(41)
    x, y = rng.normal(size=(2, 31)).astype(np.float32)
    actual = execute(elementwise(kind, len(x)), np.concatenate([x, y]), len(x))
    np.testing.assert_array_equal(actual, operation(x, y))


@pytest.mark.parametrize('kind,operation', [('copy', lambda x: x), ('exp', np.exp),
    ('recip', np.reciprocal), ('rsqrt', lambda x: np.reciprocal(np.sqrt(x)))])
def test_unary_instructions(kind, operation):
    x = np.geomspace(.001, 4, 29).astype(np.float32)
    actual = execute(elementwise(kind, len(x)), x, len(x))
    np.testing.assert_allclose(actual, operation(x), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('kind', ['sum', 'max'])
def test_reduction_instructions(kind):
    x = np.random.default_rng(13).normal(size=33).astype(np.float32)
    actual = execute(reduction(kind, len(x)), x, 1)
    expected = np.sum(x) if kind == 'sum' else np.max(x)
    np.testing.assert_allclose(actual[0], expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('rows,columns', [(1, 3), (4, 16), (6, 256)])
def test_matvec_instructions_and_accumulation(rows, columns):
    rng = np.random.default_rng(17)
    x = rng.normal(size=columns).astype(np.float32)
    w = rng.normal(size=(rows, columns)).astype(np.float32)
    seed = rng.normal(size=rows).astype(np.float32)
    values = np.concatenate([x, w.ravel(), seed])
    actual = execute(gemv(rows, columns), values, rows)
    np.testing.assert_allclose(actual, w @ x + seed, atol=2e-5, rtol=2e-5)


def test_program_limits_reject_excessive_sram():
    with pytest.raises(ValueError, match='scratch-buffer'):
        gemv(100, 100)


@pytest.mark.parametrize('batch,rows,columns', [(2, 3, 7), (8, 10, 16), (16, 32, 32)])
def test_batched_projection_outer_loop(batch, rows, columns):
    rng = np.random.default_rng(18)
    x = rng.normal(size=(batch, columns)).astype(np.float32)
    w = rng.normal(size=(rows, columns)).astype(np.float32)
    seed = rng.normal(size=(batch, rows)).astype(np.float32)
    actual = execute(gemm(batch, rows, columns), np.concatenate([x.ravel(), w.ravel(), seed.ravel()]), batch * rows)
    np.testing.assert_allclose(actual.reshape(batch, rows), x @ w.T + seed, atol=2e-5, rtol=2e-5)


def test_argmax_and_logarithm_instructions():
    x = np.array([-2., 4., 4., -10., 3.], np.float32)
    actual = execute(argmax_program(len(x)), x, 2)
    np.testing.assert_array_equal(actual, [4., 1.])
    values = np.linspace(1, 2, 23, dtype=np.float32)
    actual = execute(elementwise('log_unit', len(values)), values, len(values))
    np.testing.assert_allclose(actual, np.log(values), atol=3e-6, rtol=2e-5)
