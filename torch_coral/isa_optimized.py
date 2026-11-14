"""FP32 operation fusion and bounded host parameter storage for the MXU path.

No model arithmetic is moved to the host. Fused programs preserve the original
FP32 operation order; physical comparisons are still required for new devices.
"""
from collections import OrderedDict
from functools import lru_cache
import numpy as np
from .isa_scalar import _parts, _wrap
from .isa_mxu import MatrixTPU


def _reciprocal(S, value, result, temporary, two):
    words = [S.mov(result, value), S.NOP, S.sfu(S.RECIP, result, result)] + [S.NOP] * 4
    for _ in range(2):
        words += [S.alu(S.FMUL, temporary, value, reg=result), S.movi(two, 0x40000000),
                  S.alu(S.FSUB, temporary, two, reg=temporary), S.alu(S.FMUL, result, result, reg=temporary)]
    return words


def _sigmoid(S):
    # x remains in r8; sigmoid result in r10. Same clipping/SFU/Newton order.
    words = [S.fimm(S.FMUL, 10, 8, -1.), S.fimm(S.FMAX, 10, 10, -80.),
             S.fimm(S.FMIN, 10, 10, 80.), S.fimm(S.FMUL, 10, 10, 1.4426950408889634),
             S.NOP, S.sfu(S.EXP2, 10, 10)] + [S.NOP] * 4
    words += [S.fimm(S.FADD, 11, 10, 1.)]
    return words + _reciprocal(S, 11, 10, 12, 13)


@lru_cache(maxsize=128)
def activation_program(kind, length):
    _, _, _, S = _parts()
    if kind not in ('sigmoid', 'silu'): raise ValueError('Unknown fused activation')
    def body(base):
        loop = [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + _sigmoid(S)
        if kind == 'silu': loop += [S.alu(S.FMUL, 10, 8, reg=10)]
        loop += [S.st(6, 10), S.alu(S.ADD, 6, 6, imm=1)]
        return [S.movi(4, base), S.movi(6, 0), S.loop(length, len(loop))] + loop
    return _wrap(length, length, body)


@lru_cache(maxsize=32)
def convolution_silu_program(length):
    _, _, _, S = _parts()
    def body(base):
        words = [S.movi(25, 0)]
        for tap in range(4):
            words += [S.movi(4 + tap, base + tap * length), S.movi(21 + tap, base + (4 + tap) * length)]
        loop = []
        for tap in range(4):
            loop += [S.ld(8, 4 + tap, inc=True), S.ld(9, 21 + tap, inc=True)] + [S.NOP] * 4 + [
                S.alu(S.FMUL, 10 if tap == 0 else 11, 8, reg=9)]
            if tap: loop += [S.alu(S.FADD, 10, 10, reg=11)]
        loop += [S.mov(8, 10)] + _sigmoid(S) + [S.alu(S.FMUL, 10, 8, reg=10),
            S.st(25, 10), S.alu(S.ADD, 25, 25, imm=1)]
        return words + [S.loop(length, len(loop))] + loop
    return _wrap(length * 8, length, body)


@lru_cache(maxsize=128)
def norm_program(length, weighted, l2, epsilon):
    _, _, _, S = _parts()
    def body(base):
        loop = [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + [S.alu(S.FMUL, 12, 8, reg=8),
            S.NOP, S.alu(S.FADD, 10, 10, reg=12), S.NOP]
        words = [S.movi(4, base), S.movi(10, 0), S.loop(length, len(loop))] + loop
        if not l2: words += [S.fimm(S.FMUL, 10, 10, 1 / length)]
        words += [S.fimm(S.FADD, 10, 10, epsilon), S.alu(S.SHR, 14, 10, imm=1),
            S.movi(11, 0x5f3759df), S.alu(S.SUB, 14, 11, reg=14), S.fimm(S.FMUL, 11, 10, .5)]
        for _ in range(4):
            words += [S.alu(S.FMUL, 12, 14, reg=14), S.alu(S.FMUL, 12, 12, reg=11),
                S.movi(13, 0x3fc00000), S.alu(S.FSUB, 12, 13, reg=12), S.alu(S.FMUL, 14, 14, reg=12)]
        output = [S.ld(8, 4, inc=True)]
        if weighted: output += [S.ld(9, 5, inc=True)]
        output += [S.NOP] * 4 + [S.alu(S.FMUL, 10, 8, reg=14)]
        if weighted: output += [S.alu(S.FMUL, 10, 10, reg=9)]
        output += [S.st(6, 10), S.alu(S.ADD, 6, 6, imm=1)]
        return words + [S.movi(4, base), S.movi(5, base + length), S.movi(6, 0), S.loop(length, len(output))] + output
    return _wrap(length * (2 if weighted else 1), length, body)


@lru_cache(maxsize=64)
def auto_quantize_program(length):
    """Absmax, scale, inverse, INT8 bytes and FP32 residual in one TPU call."""
    _, _, _, S = _parts()
    assert length % 4 == 0
    packed = length // 4
    def body(base):
        maximum = [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + [S.alu(S.AND, 8, 8, imm=0x7fffffff),
            S.alu(S.FMAX, 15, 15, reg=8), S.NOP]
        words = [S.movi(4, base), S.movi(15, 0), S.loop(length, len(maximum))] + maximum
        words += [S.fimm(S.FMAX, 15, 15, 1e-20), S.fimm(S.FMUL, 15, 15, 1 / 127)]
        words += _reciprocal(S, 15, 16, 17, 18)
        words += [S.movi(6, packed + length), S.st(6, 15), S.movi(4, base), S.movi(6, 0), S.movi(14, packed)]
        loop = [S.movi(20, 0)]
        for shift in (0, 8, 16, 24):
            loop += [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + [S.alu(S.FMUL, 10, 8, reg=16),
                S.fimm(S.FMAX, 10, 10, -127.), S.fimm(S.FMIN, 10, 10, 127.), S.f2i(10, 10)] + [S.NOP] * 2 + [
                S.i2f(11, 10), S.NOP, S.alu(S.FMUL, 11, 11, reg=15), S.alu(S.FSUB, 11, 8, reg=11),
                S.st(14, 11), S.alu(S.ADD, 14, 14, imm=1), S.alu(S.ADD, 10, 10, imm=128),
                S.alu(S.SHL, 10, 10, imm=shift), S.alu(S.OR, 20, 20, reg=10)]
        loop += [S.st(6, 20), S.alu(S.ADD, 6, 6, imm=1)]
        return words + [S.loop(packed, len(loop))] + loop
    return _wrap(length, packed + length + 1, body)


class OptimizedMatrixTPU(MatrixTPU):
    recurrent_columns = 13
    def __init__(self, *args, parameter_cache_bytes=1024**3, async_usb=True, **kwargs):
        self.parameter_cache_limit = int(parameter_cache_bytes)
        if self.parameter_cache_limit < 0: raise ValueError('Negative parameter cache limit')
        self._parameter_cache = OrderedDict()
        self._parameter_cache_size = 0
        super().__init__(*args, async_usb=async_usb, **kwargs)
        self.matrix_stats.update(host_cache_hits=0, host_cache_misses=0, host_cache_bytes=0)

    def parameter_blob(self, prog, key, qweight):
        if key in self._parameter_cache:
            self.matrix_stats['host_cache_hits'] += 1
            self._parameter_cache.move_to_end(key)
            return self._parameter_cache[key]
        self.matrix_stats['host_cache_misses'] += 1
        blob = super().parameter_blob(prog, key, qweight)
        if len(blob) <= self.parameter_cache_limit:
            while self._parameter_cache_size + len(blob) > self.parameter_cache_limit:
                _, old = self._parameter_cache.popitem(last=False)
                self._parameter_cache_size -= len(old)
            self._parameter_cache[key] = blob
            self._parameter_cache_size += len(blob)
            self.matrix_stats['host_cache_bytes'] = self._parameter_cache_size
        return blob

    def quantize_input(self, array):
        x = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
        if len(x) > 1024: return super().quantize_input(x)
        out = self._run(auto_quantize_program(len(x)), x.tobytes(), len(x) + len(x) // 4 + 1)
        return out[:len(x) // 4].view(np.uint8).copy(), out[-1:].copy(), out[len(x) // 4:-1].copy()

    def activation(self, kind, array):
        x = np.asarray(array, np.float32)
        result = np.empty(x.shape, np.float32)
        flat, output = np.ascontiguousarray(x).reshape(-1), result.reshape(-1)
        for offset in range(0, len(flat), 2048):
            part = flat[offset:offset + 2048]
            output[offset:offset + len(part)] = self._run(activation_program(kind, len(part)), part.tobytes(), len(part))
        return result

    def normalize(self, array, weight, l2, epsilon):
        x = np.asarray(array, np.float32)
        length = x.shape[-1]
        if length > 1024: raise ValueError('Fused normalization supports at most 1024 columns')
        output = np.empty(x.shape, np.float32).reshape(-1, length)
        for row, values in enumerate(x.reshape(-1, length)):
            payload = values.tobytes() + (weight.tobytes() if weight is not None else b'')
            output[row] = self._run(norm_program(length, weight is not None, l2, float(epsilon)), payload, length)
        return output.reshape(x.shape)

    def convolution_silu(self, window, kernel):
        output = np.empty(window.shape[1], np.float32)
        for offset in range(0, len(output), 256):
            part = window[:, offset:offset + 256]
            coefficients = kernel[offset:offset + 256].T
            payload = part.tobytes() + coefficients.tobytes()
            output[offset:offset + part.shape[1]] = self._run(convolution_silu_program(part.shape[1]), payload, part.shape[1])
        return output

    def close(self):
        super().close()
        self._parameter_cache.clear()
        self._parameter_cache_size = 0
