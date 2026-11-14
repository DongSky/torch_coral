"""Larger primary-only activation quantization in verified scalar SRAM."""
from functools import lru_cache
from .isa_scalar import _parts
from .isa_optimized import _reciprocal
from .isa_v2_fusions import _inplace


def selection_words(S, length, count, source, destination):
    """Same stable exclusion passes as V3 top-k, at caller-selected addresses."""
    words = []
    for rank in range(count):
        words += [S.movi(4, source), S.movi(7, 0), S.movi(10, 0xff800000), S.movi(11, 0)]
        loop = [S.ld(8, 4, inc=True)] + [S.NOP] * 4
        for previous in range(rank):
            loop += [S.alu(S.NE, 0, 7, reg=20 + previous),
                     S.alu(S.MOV, 8, imm=0xff800000, gate=1, pred_reg=0, yes_pred=0)]
        loop += [S.alu(S.FMAX, 12, 10, reg=8), S.alu(S.NE, 0, 12, reg=10), S.mov(10, 12),
                 S.alu(S.MOV, 11, reg=7, gate=1, pred_reg=0, yes_pred=1), S.alu(S.ADD, 7, 7, imm=1)]
        words += [S.loop(length, len(loop))] + loop + [S.mov(20 + rank, 11), S.i2f(11, 11),
                  S.movi(6, destination + rank * 2), S.st(6, 10),
                  S.movi(6, destination + rank * 2 + 1), S.st(6, 11)]
    return words


def primary_words(S, length, source, destination):
    packed = length // 4
    maximum = [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + [
        S.alu(S.AND, 8, 8, imm=0x7fffffff), S.alu(S.FMAX, 15, 15, reg=8), S.NOP]
    words = [S.movi(4, source), S.movi(15, 0), S.loop(length, len(maximum))] + maximum
    words += [S.fimm(S.FMAX, 15, 15, 1e-20), S.fimm(S.FMUL, 15, 15, 1 / 127)]
    words += _reciprocal(S, 15, 16, 17, 18)
    words += [S.movi(4, source), S.movi(6, destination)]
    loop = [S.movi(20, 0)]
    for shift in (0, 8, 16, 24):
        loop += [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + [
            S.alu(S.FMUL, 10, 8, reg=16), S.fimm(S.FMAX, 10, 10, -127.),
            S.fimm(S.FMIN, 10, 10, 127.), S.f2i(10, 10)] + [S.NOP] * 2 + [
            S.alu(S.ADD, 10, 10, imm=128), S.alu(S.SHL, 10, 10, imm=shift),
            S.alu(S.OR, 20, 20, reg=10)]
    loop += [S.st(6, 20), S.alu(S.ADD, 6, 6, imm=1)]
    return words + [S.loop(packed, len(loop))] + loop + [S.movi(6, destination + packed), S.st(6, 15)]


@lru_cache(maxsize=32)
def silu_multiply_program(length):
    from .isa_optimized import _sigmoid
    if not 1 <= length <= 2000:
        raise ValueError('Fused SiLU multiplication supports 1..2000 elements')
    _, _, _, S = _parts()
    def body(base):
        loop = [S.ld(8, 4, inc=True), S.ld(19, 5, inc=True)] + [S.NOP] * 4
        loop += _sigmoid(S) + [S.alu(S.FMUL, 10, 8, reg=10),
                              S.alu(S.FMUL, 10, 10, reg=19), S.st(6, 10), S.alu(S.ADD, 6, 6, imm=1)]
        return [S.movi(4, base), S.movi(5, base + length), S.movi(6, base), S.loop(length, len(loop))] + loop
    return _inplace(2 * length, length, 0, body)


@lru_cache(maxsize=16)
def normalize_quantize_program(epsilon):
    from .isa_scalar import _wrap
    from .isa_v2_fusions import norm_rows_program
    _, _, _, S = _parts()
    length = 1024
    # Input layout and base match the original weighted one-row normalization.
    norm_words = norm_rows_program(length, 1, True, False, epsilon)[1]
    return _wrap(2 * length, length + length // 4 + 1,
                 lambda base: list(norm_words) + primary_words(S, length, 0, length))


@lru_cache(maxsize=32)
def wide_primary_quantize_program(length):
    """Read every input before overwriting consumed words with packed INT8.

    Keep the scale in r15 until the entire packing pass finishes. Storing it
    beside the packed result earlier would overwrite a not-yet-consumed input.
    Absmax, reciprocal refinement, clipping and rounding retain V4's order.
    """
    if not 4 <= length <= 4000 or length % 4:
        raise ValueError('Wide primary quantization requires 4..4000 values, divisible by four')
    _, _, _, S = _parts()
    packed = length // 4
    def body(base):
        maximum = [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + [
            S.alu(S.AND, 8, 8, imm=0x7fffffff), S.alu(S.FMAX, 15, 15, reg=8), S.NOP]
        words = [S.movi(4, base), S.movi(15, 0), S.loop(length, len(maximum))] + maximum
        words += [S.fimm(S.FMAX, 15, 15, 1e-20), S.fimm(S.FMUL, 15, 15, 1 / 127)]
        words += _reciprocal(S, 15, 16, 17, 18)
        words += [S.movi(4, base), S.movi(6, base)]
        loop = [S.movi(20, 0)]
        for shift in (0, 8, 16, 24):
            loop += [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + [
                S.alu(S.FMUL, 10, 8, reg=16), S.fimm(S.FMAX, 10, 10, -127.),
                S.fimm(S.FMIN, 10, 10, 127.), S.f2i(10, 10)] + [S.NOP] * 2 + [
                S.alu(S.ADD, 10, 10, imm=128), S.alu(S.SHL, 10, 10, imm=shift),
                S.alu(S.OR, 20, 20, reg=10)]
        loop += [S.st(6, 20), S.alu(S.ADD, 6, 6, imm=1)]
        return words + [S.loop(packed, len(loop))] + loop + [S.movi(6, base + packed), S.st(6, 15)]
    return _inplace(length, packed + 1, 0, body)
