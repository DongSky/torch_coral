"""FP32 Turbo V2 kernels; preserve reduction and Newton iteration order."""
from functools import lru_cache
from .isa_scalar import _parts, _wrap
from .isa_optimized import _reciprocal


def _inplace(input_words, output_words, output_offset, body):
    Emitter,E,SC,_ = _parts()
    size=(input_words*4+7)//8*8
    base=E.SMEM_TOP-size
    out_size=(output_words*4+7)//8*8
    out_base=base//4+output_offset
    if size>16000 or out_base*4+out_size>E.SMEM_TOP:
        raise ValueError('In-place kernel exceeds verified SRAM limits')
    words=body(base//4)
    e=Emitter(); e.scalar(SC.exe_prologue(),seqs=1)
    tiles=E._sm_single_row(e,size,base)
    e.scalar(E.S_RING_OUTFEED_RESET); e.scalar(SC.output_wait(e.seq,tiles),seqs=2)
    e.scalar(words); e.scalar(E.smem_to_host(out_base,output_words,out_size)); e.sync(SC.epilogue)
    return E.executable(e.program(),size,out_size),words,base,out_base*4


@lru_cache(maxsize=64)
def primary_quantize_program(length):
    """One-component projections never consume activation residuals."""
    _,_,_,S=_parts(); packed=length//4
    if length%4: raise ValueError('Quantization requires a multiple of four')
    def body(base):
        maximum=[S.ld(8,4,inc=True)]+[S.NOP]*4+[S.alu(S.AND,8,8,imm=0x7fffffff),S.alu(S.FMAX,15,15,reg=8),S.NOP]
        words=[S.movi(4,base),S.movi(15,0),S.loop(length,len(maximum))]+maximum
        words += [S.fimm(S.FMAX,15,15,1e-20),S.fimm(S.FMUL,15,15,1/127)]
        words += _reciprocal(S,15,16,17,18)
        words += [S.movi(6,packed),S.st(6,15),S.movi(4,base),S.movi(6,0)]
        loop=[S.movi(20,0)]
        for shift in (0,8,16,24):
            loop += [S.ld(8,4,inc=True)]+[S.NOP]*4+[S.alu(S.FMUL,10,8,reg=16),
                S.fimm(S.FMAX,10,10,-127.),S.fimm(S.FMIN,10,10,127.),S.f2i(10,10)]+[S.NOP]*2+[
                S.alu(S.ADD,10,10,imm=128),S.alu(S.SHL,10,10,imm=shift),S.alu(S.OR,20,20,reg=10)]
        loop += [S.st(6,20),S.alu(S.ADD,6,6,imm=1)]
        return words+[S.loop(packed,len(loop))]+loop
    return _wrap(length,packed+1,body)


@lru_cache(maxsize=128)
def correction_inplace_program(length, coefficient):
    _,_,_,S=_parts(); packed=length//4
    if length%4: raise ValueError('Correction length must be divisible by four')
    def body(base):
        words=[S.movi(4,base),S.movi(5,base+packed),S.movi(6,base+packed)]
        loop=[S.ld(8,4,inc=True)]+[S.ld(r,5,inc=True) for r in (20,21,22,23)]+[S.NOP]*4
        for shift,previous in zip((0,8,16,24),(20,21,22,23)):
            loop += [S.alu(S.SHR,9,8,imm=shift),S.alu(S.AND,9,9,imm=255),
                S.alu(S.SUB,9,9,imm=128),S.i2f(9,9),S.NOP,
                S.fimm(S.FMUL,9,9,coefficient),S.f2i(9,9)]+[S.NOP]*2+[
                S.alu(S.SUB,previous,previous,reg=9),S.st(6,previous),S.alu(S.ADD,6,6,imm=1)]
        return words+[S.loop(packed,len(loop))]+loop
    return _inplace(packed+length,length,packed,body)


@lru_cache(maxsize=128)
def reconstruct_inplace_program(length, coefficients, select_count=0):
    _,_,_,S=_parts(); packed=length//4; digits=len(coefficients)
    if length%4 or not 1<=digits<=3 or select_count not in (0,4): raise ValueError('Invalid reconstruction dimensions')
    def body(base):
        words=[S.movi(6,base+digits*packed),S.movi(7,base+digits*packed),
            S.movi(17,base+digits*packed+length),S.ld(15,17)]+[S.NOP]*4
        for digit in range(digits): words += [S.movi(20+digit,base+digit*packed)]
        loop=[S.ld(8+digit,20+digit,inc=True) for digit in range(digits)]
        loop += [S.ld(r,7,inc=True) for r in (24,25,26,27)]+[S.NOP]*4
        for shift,row_scale in zip((0,8,16,24),(24,25,26,27)):
            loop += [S.movi(12,0)]
            for digit,coefficient in enumerate(coefficients):
                loop += [S.alu(S.SHR,11,8+digit,imm=shift),S.alu(S.AND,11,11,imm=255),
                    S.alu(S.SUB,11,11,imm=128),S.i2f(11,11),S.NOP,
                    S.fimm(S.FMUL,11,11,coefficient),S.alu(S.FADD,12,12,reg=11)]
            loop += [S.alu(S.FMUL,12,12,reg=row_scale),S.alu(S.FMUL,12,12,reg=15),
                S.st(6,12),S.alu(S.ADD,6,6,imm=1)]
        words += [S.loop(packed,len(loop))]+loop
        if select_count:
            from .isa_v4_fusions import selection_words
            words += selection_words(S,length,select_count,base+digits*packed,base+digits*packed+length)
        return words
    return _inplace(digits*packed+length+1+2*select_count,length+2*select_count,digits*packed,body)


@lru_cache(maxsize=128)
def norm_rows_program(length, rows, weighted, l2, epsilon):
    _, _, _, S = _parts()
    def body(base):
        words = []
        for row in range(rows):
            start = base + row * length
            loop = [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + [S.alu(S.FMUL, 12, 8, reg=8),
                S.NOP, S.alu(S.FADD, 10, 10, reg=12), S.NOP]
            words += [S.movi(4, start), S.movi(10, 0), S.loop(length, len(loop))] + loop
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
            words += [S.movi(4, start), S.movi(5, base + rows * length), S.movi(6, row * length),
                      S.loop(length, len(output))] + output
        return words
    return _wrap(length * (rows + int(weighted)), length * rows, body)


@lru_cache(maxsize=128)
def softmax_program(length):
    _, _, _, S = _parts()
    def body(base):
        maximum = [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + [S.alu(S.FMAX, 15, 15, reg=8), S.NOP]
        words = [S.movi(4, base), S.movi(15, 0xff800000), S.loop(length, len(maximum))] + maximum
        exponential = [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + [S.alu(S.FSUB, 10, 8, reg=15),
            S.fimm(S.FMUL, 10, 10, 1.4426950408889634), S.NOP, S.sfu(S.EXP2, 10, 10)] + [S.NOP] * 4 + [
            S.st(6, 10), S.alu(S.ADD, 6, 6, imm=1), S.alu(S.FADD, 16, 16, reg=10), S.NOP]
        words += [S.movi(4, base), S.movi(6, 0), S.movi(16, 0), S.loop(length, len(exponential))] + exponential
        words += _reciprocal(S, 16, 14, 11, 13)
        output = [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + [S.alu(S.FMUL, 10, 8, reg=14),
            S.st(6, 10), S.alu(S.ADD, 6, 6, imm=1)]
        return words + [S.movi(4, 0), S.movi(6, 0), S.loop(length, len(output))] + output
    return _wrap(length, length, body)


@lru_cache(maxsize=128)
def rope_program(half, rows):
    """Input: split-half rows, shared cosine and sine. Output: split-half rows."""
    _, _, _, S = _parts()
    def body(base):
        words = []
        for row in range(rows):
            words += [S.movi(4, base + row * 2 * half), S.movi(5, base + (row * 2 + 1) * half),
                S.movi(6, base + rows * 2 * half), S.movi(7, base + (rows * 2 + 1) * half),
                S.movi(20, row * 2 * half), S.movi(21, (row * 2 + 1) * half)]
            loop = [S.ld(8, 4, inc=True), S.ld(9, 5, inc=True), S.ld(10, 6, inc=True), S.ld(11, 7, inc=True)] + [S.NOP] * 4 + [
                S.alu(S.FMUL, 12, 8, reg=10), S.alu(S.FMUL, 13, 9, reg=11), S.alu(S.FSUB, 12, 12, reg=13),
                S.st(20, 12), S.alu(S.ADD, 20, 20, imm=1),
                S.alu(S.FMUL, 12, 8, reg=11), S.alu(S.FMUL, 13, 9, reg=10), S.alu(S.FADD, 12, 12, reg=13),
                S.st(21, 12), S.alu(S.ADD, 21, 21, imm=1)]
            words += [S.loop(half, len(loop))] + loop
        return words
    return _wrap((rows + 1) * 2 * half, rows * 2 * half, body)
