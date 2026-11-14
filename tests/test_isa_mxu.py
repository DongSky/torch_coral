"""Offline checks for the INT8 bridge, not evidence of matrix-array hardware."""
import numpy as np
import pytest
pytest.importorskip('coral.isa.scalar_core')
from coral.isa.scalar_core import run
from torch_coral.isa_mxu import absmax_program, quantize_program, correction_program, reconstruct_program
from torch_coral.isa_mxu import prepare_matrix_weights, recurrent_program


def execute(program, payload, nbytes):
    _, words, base = program
    memory = bytearray(16384)
    memory[base:base + len(payload)] = payload
    run(words, memory, load_latency=5)
    return bytes(memory[:nbytes])


def test_activation_quantization_and_residual():
    x = np.linspace(-2., 2., 32, dtype=np.float32)
    scale = np.float32(2 / 127)
    inverse = np.float32(1 / scale)
    payload = x.tobytes() + scale.tobytes() + inverse.tobytes()
    output = execute(quantize_program(len(x)), payload, len(x) * 5)
    q = np.frombuffer(output[:len(x)], np.uint8)
    expected = np.rint(x * inverse).astype(np.int32)
    np.testing.assert_array_equal(q, (expected + 128).astype(np.uint8))
    residual = np.frombuffer(output[len(x):], '<f4')
    np.testing.assert_allclose(residual, x - expected.astype(np.float32) * scale, atol=1e-7)
    maximum = np.frombuffer(execute(absmax_program(len(x)), x.tobytes(), 4), '<f4')
    np.testing.assert_array_equal(maximum, [2.])


@pytest.mark.parametrize('digits', [1, 2, 3])
def test_bias_correction_and_output_reconstruction(digits):
    n = 32
    rng = np.random.default_rng(3)
    accumulator = rng.integers(-200000, 200000, size=n, dtype=np.int32)
    old_bias = np.zeros(n, '<i4')
    outputs, coefficients = [], []
    scale = 4096.
    for digit in range(digits):
        q = (np.rint((accumulator.astype(np.float64) + old_bias) / scale) + 128).astype(np.uint8)
        outputs.append(q); coefficients.append(scale)
        if digit + 1 < digits:
            payload = q.tobytes() + old_bias.tobytes()
            new_bias = np.frombuffer(execute(correction_program(n, scale), payload, n * 4), '<i4').copy()
            np.testing.assert_array_equal(new_bias, old_bias - (q.astype(np.int32) - 128) * int(scale))
            old_bias = new_bias
        scale /= 128
    row_scale = np.linspace(.01, .02, n, dtype=np.float32)
    input_scale = np.array([.03], np.float32)
    payload = b''.join(v.tobytes() for v in outputs) + row_scale.tobytes() + input_scale.tobytes()
    output = np.frombuffer(execute(reconstruct_program(n, tuple(coefficients)), payload, n * 4), '<f4')
    expected = sum((q.astype(np.float32) - 128) * s for q, s in zip(outputs, coefficients))
    expected = (expected * row_scale) * input_scale
    np.testing.assert_array_equal(output, expected)
    if digits == 3:
        np.testing.assert_array_equal(sum((q.astype(np.float32) - 128) * s for q, s in zip(outputs, coefficients)), accumulator)


def test_weight_cache_roundtrip_and_residual_precision(tmp_path):
    class Store:
        manifest = dict(source_sha256='fixture', tensors={'linear.weight': dict(shape=[128, 2048])})
        def __getitem__(self, name): return weights
    weights = np.random.default_rng(2).normal(size=(128, 2048)).astype(np.float32)
    weights[0] = 0
    manifest = prepare_matrix_weights(Store(), tmp_path, progress=lambda s: None)
    q0 = np.load(tmp_path / 'linear.weight.q0.npy', allow_pickle=False)
    q1 = np.load(tmp_path / 'linear.weight.q1.npy', allow_pickle=False)
    scales = np.load(tmp_path / 'linear.weight.scales.npy', allow_pickle=False)
    assert q0.dtype == q1.dtype == np.uint8 and q0.shape == weights.shape
    actual = (q0.astype(np.float32) - 128) * scales[0, :, None] + (q1.astype(np.float32) - 128) * scales[1, :, None]
    assert np.all(np.abs(actual - weights) <= scales[1, :, None] * .51 + 1e-6)
    assert manifest['format_version'] == 2


def test_fused_fp32_recurrent_update():
    rng = np.random.default_rng(7)
    length, columns = 128, 12
    state = rng.normal(size=(columns, length)).astype(np.float32)
    k, q = rng.normal(size=(2, length)).astype(np.float32)
    v = rng.normal(size=columns).astype(np.float32)
    beta, decay = np.float32(.3), np.float32(.9)
    payload = state.tobytes() + k.tobytes() + q.tobytes() + v.tobytes() + beta.tobytes() + decay.tobytes()
    out = np.frombuffer(execute(recurrent_program(columns, length), payload, (state.size + columns) * 4), '<f4')
    current = state * decay
    delta = (v - current @ k) * beta
    current += delta[:, None] * k
    np.testing.assert_allclose(out[:state.size].reshape(state.shape), current, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(out[state.size:], current @ q, atol=2e-4, rtol=2e-5)
