"""Experimental FP32 programs for the original Edge TPU scalar core.

Uses the pinned tinycoral instruction encoders; no CPU arithmetic fallback.
Host code packs data and schedules tiles. This is separate from the TFLite ABI.
"""
from functools import lru_cache
import os
import threading
import time
import numpy as np


def _parts():
    from coral.codegen import Emitter
    from coral.codegen import eltops as E
    from coral.isa import scalar as SC, scalar_core as S
    return Emitter, E, SC, S


def _wrap(input_words, output_words, body):
    Emitter, E, SC, S = _parts()
    nbytes = (input_words * 4 + 7) // 8 * 8
    if nbytes > 8192 or output_words * 4 > 8192:
        raise ValueError('Scalar program exceeds verified scratch-buffer limits')
    base = E.SMEM_TOP - nbytes
    e = Emitter()
    e.scalar(SC.exe_prologue(), seqs=1)
    tiles = E._sm_single_row(e, nbytes, base)
    e.scalar(E.S_RING_OUTFEED_RESET)
    e.scalar(SC.output_wait(e.seq, tiles), seqs=2)
    words = body(base // 4)
    e.scalar(words)
    out_bytes = (output_words * 4 + 7) // 8 * 8
    e.scalar(E.smem_to_host(0, output_words, out_bytes))
    e.sync(SC.epilogue)
    return E.executable(e.program(), nbytes, out_bytes), words, base


@lru_cache(maxsize=256)
def elementwise(kind, length):
    _, _, _, S = _parts()
    binary = kind in ('add', 'sub', 'mul', 'max', 'min')
    def body(base):
        words = [S.movi(4, base), S.movi(5, base + length), S.movi(6, 0)]
        loop = [S.ld(8, 4, inc=True)]
        if binary:
            loop += [S.ld(9, 5, inc=True)]
        loop += [S.NOP] * 4
        if binary:
            op = dict(add=S.FADD, sub=S.FSUB, mul=S.FMUL, max=S.FMAX, min=S.FMIN)[kind]
            loop += [S.alu(op, 10, 8, reg=9), S.NOP]
        elif kind == 'copy':
            loop += [S.mov(10, 8)]
        elif kind == 'exp':
            loop += [S.fimm(S.FMUL, 10, 8, 1.4426950408889634), S.NOP,
                     S.sfu(S.EXP2, 10, 10)] + [S.NOP] * 4
        elif kind == 'recip':
            loop += [S.mov(10, 8), S.NOP, S.sfu(S.RECIP, 10, 10)] + [S.NOP] * 4
            for _ in range(2):
                loop += [S.alu(S.FMUL, 11, 8, reg=10), S.movi(12, 0x40000000),
                         S.alu(S.FSUB, 11, 12, reg=11), S.alu(S.FMUL, 10, 10, reg=11)]
        elif kind == 'rsqrt':
            # Positive finite inputs: bit estimate followed by four FP32 Newton steps.
            loop += [S.alu(S.SHR, 10, 8, imm=1), S.movi(11, 0x5f3759df),
                     S.alu(S.SUB, 10, 11, reg=10), S.fimm(S.FMUL, 11, 8, .5)]
            for _ in range(4):
                loop += [S.alu(S.FMUL, 12, 10, reg=10), S.alu(S.FMUL, 12, 12, reg=11),
                         S.movi(13, 0x3fc00000), S.alu(S.FSUB, 12, 13, reg=12),
                         S.alu(S.FMUL, 10, 10, reg=12)]
        elif kind == 'log_unit':
            # log(x), x in [1,2]: 2*atanh((x-1)/(x+1)), nine odd terms.
            loop += [S.fimm(S.FSUB, 10, 8, 1.), S.fimm(S.FADD, 11, 8, 1.),
                     S.sfu(S.RECIP, 11, 11)] + [S.NOP] * 4
            loop += [S.alu(S.FMUL, 10, 10, reg=11), S.alu(S.FMUL, 11, 10, reg=10),
                     S.fimm(S.FADD, 12, 0, 1 / 17)]
            # register zero is not a hardwired zero; initialize the polynomial explicitly.
            loop[-1] = S.movi(12, int(np.float32(1 / 17).view(np.uint32)))
            for denominator in (15, 13, 11, 9, 7, 5, 3, 1):
                loop += [S.alu(S.FMUL, 12, 12, reg=11), S.fimm(S.FADD, 12, 12, 1 / denominator)]
            loop += [S.alu(S.FMUL, 10, 10, reg=12), S.fimm(S.FMUL, 10, 10, 2.)]
        else:
            raise ValueError('Unsupported scalar operation: ' + kind)
        loop += [S.st(6, 10), S.alu(S.ADD, 6, 6, imm=1)]
        return words + [S.loop(length, len(loop))] + loop
    return _wrap(length * (2 if binary else 1), length, body)


@lru_cache(maxsize=256)
def reduction(kind, length):
    _, _, _, S = _parts()
    def body(base):
        initial = 0 if kind == 'sum' else 0xff800000
        op = S.FADD if kind == 'sum' else S.FMAX
        loop = [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + [S.alu(op, 10, 10, reg=8), S.NOP]
        return [S.movi(4, base), S.movi(10, initial), S.loop(length, len(loop))] + loop + [S.movi(6, 0), S.st(6, 10)]
    if kind not in ('sum', 'max'):
        raise ValueError('Unknown reduction')
    return _wrap(length, 1, body)


@lru_cache(maxsize=256)
def gemv(rows, columns):
    """FP32 y = W @ x + seed, including partial-sum accumulation on the device."""
    _, _, _, S = _parts()
    def body(base):
        words = [S.movi(5, base + columns)]
        for row in range(rows):
            words += [S.movi(4, base), S.movi(6, base + columns + rows * columns + row),
                      S.ld(10, 6)] + [S.NOP] * 4
            loop = [S.ld(8, 4, inc=True), S.ld(9, 5, inc=True)] + [S.NOP] * 4 + [
                S.alu(S.FMUL, 11, 8, reg=9), S.NOP, S.alu(S.FADD, 10, 10, reg=11), S.NOP]
            words += [S.loop(columns, len(loop))] + loop + [S.movi(6, row), S.st(6, 10)]
        return words
    return _wrap(columns + rows * columns + rows, rows, body)


@lru_cache(maxsize=64)
def argmax_program(length):
    _, _, _, S = _parts()
    def body(base):
        words = [S.movi(4, base), S.movi(7, 0), S.movi(10, 0xff800000), S.movi(11, 0)]
        loop = [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + [S.alu(S.FMAX, 12, 10, reg=8),
            S.alu(S.NE, 0, 12, reg=10), S.mov(10, 12),
            S.alu(S.MOV, 11, reg=7, gate=1, pred_reg=0, yes_pred=1), S.alu(S.ADD, 7, 7, imm=1)]
        # Return maximum and index as FP32, so host only assembles index offsets.
        return words + [S.loop(length, len(loop))] + loop + [S.i2f(11, 11), S.movi(6, 0),
            S.st(6, 10), S.movi(6, 1), S.st(6, 11)]
    return _wrap(length, 2, body)


@lru_cache(maxsize=256)
def gemm(batch, rows, columns):
    """Batched FP32 projection, reusing every weight tile across prompt positions."""
    _, _, _, S = _parts()
    def body(base):
        words = [S.movi(20, 0), S.movi(21, base + batch * columns),
                 S.movi(22, base + batch * columns + rows * columns)]
        top = len(words)
        for token in range(batch):
            words += [S.movi(4, base + token * columns), S.mov(5, 21),
                S.alu(S.ADD, 6, 22, imm=token * rows), S.ld(10, 6)] + [S.NOP] * 4
            loop = [S.ld(8, 4, inc=True), S.ld(9, 5, inc=True)] + [S.NOP] * 4 + [
                S.alu(S.FMUL, 11, 8, reg=9), S.NOP, S.alu(S.FADD, 10, 10, reg=11), S.NOP]
            words += [S.loop(columns, len(loop))] + loop + [S.alu(S.ADD, 6, 20, imm=token * rows), S.st(6, 10)]
        words += [S.alu(S.ADD, 20, 20, imm=1), S.alu(S.ADD, 21, 21, imm=columns),
                  S.alu(S.ADD, 22, 22, imm=1), S.alu(S.LT, 0, 20, imm=rows)]
        words += [S.branch(top - len(words))]
        return words
    return _wrap(batch * columns + rows * columns + batch * rows, batch * rows, body)


class ScalarTPU:
    """Strict synchronous executor; unsupported arithmetic raises.

    Every numerical operation calls run_executable on the physical USB device.
    This reference path prioritizes coverage/precision over throughput.
    """
    def __init__(self, *, async_usb=False, clock='high'):
        if clock not in ('high','max'):
            raise ValueError('clock must be high (250 MHz) or max (500 MHz)')
        from .native_runtime import configure
        from .settings import runtime_directory
        configure()
        if os.name == 'nt':
            os.environ['LIBUSB_PATH'] = str(runtime_directory() / 'libusb-1.0.dll')
        os.environ['CORAL_ASYNC'] = str(int(async_usb))
        from coral.device import EdgeTPU
        from coral.dfu import APP_VID, APP_PID
        from coral.usbdev import find
        if not find(APP_VID, APP_PID):
            raise RuntimeError('Coral is unavailable or still in DFU mode. Initialize firmware and complete USB '
                               're-enumeration first; close other Coral applications. See docs/qwen-full-tpu.md.')
        self.lock = threading.RLock()
        self.device = EdgeTPU.__new__(EdgeTPU)
        try:
            self.device.__init__(clock=clock)
        except Exception:
            self.device.close()
            raise
        self.closed = False
        # tinycoral uses one process-wide transport flag. Its device lock permits
        # only one raw owner; set it again for an already-imported runtime module.
        from coral import runtime
        runtime.ASYNC = int(async_usb)
        self.async_usb = bool(async_usb)
        self.calls = 0
        self.device_ms = 0.0
        self.cancel_event = None

    def _run_executable(self, exe, inputs=b'', parameters=None):
        from coral.runtime import run_executable
        return run_executable(self.device, exe, inputs, parameters)

    def _run(self, program, payload, count):
        with self.lock:
            if self.closed:
                raise RuntimeError('Scalar TPU is closed')
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise InterruptedError('Cancelled between TPU programs')
            if len(payload) % 8:
                payload += b'\0' * (8 - len(payload) % 8)
            if len(payload) > 8192:
                raise ValueError('Scalar input exceeds verified DMA limit')
            start = time.perf_counter()
            try:
                data = self._run_executable(program[0], payload)
            except Exception:
                # Do not continue dispatching after a failed hardware program.
                self.close()
                raise
            self.calls += 1
            self.device_ms += (time.perf_counter() - start) * 1000
            if len(data) < count * 4:
                self.close()
                raise RuntimeError('Short TPU result')
            return np.frombuffer(data, dtype='<f4', count=count).copy()

    def op(self, kind, a, b=None):
        a = np.asarray(a, dtype=np.float32)
        if b is not None:
            a, b = np.broadcast_arrays(a, np.asarray(b, dtype=np.float32))
        shape = a.shape
        a = np.ascontiguousarray(a).reshape(-1)
        if b is not None:
            b = np.ascontiguousarray(b).reshape(-1)
        result = np.empty_like(a)
        limit = 1024 if b is not None else 2048
        for offset in range(0, a.size, limit):
            x = a[offset:offset + limit]
            payload = x.tobytes() + (b[offset:offset + limit].tobytes() if b is not None else b'')
            result[offset:offset + len(x)] = self._run(elementwise(kind, len(x)), payload, len(x))
        return result.reshape(shape)

    def reduce(self, kind, array):
        values = np.asarray(array, dtype=np.float32).reshape(-1)
        if not len(values):
            raise ValueError('Cannot reduce an empty tensor')
        while len(values) > 1:
            values = np.concatenate([self._run(reduction(kind, len(x)), x.tobytes(), 1)
                for x in (values[i:i + 2048] for i in range(0, len(values), 2048))])
        return values

    def matvec(self, weights, x):
        weights = np.asarray(weights, dtype=np.float32)
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if weights.ndim != 2 or weights.shape[1] != len(x):
            raise ValueError('Matvec dimensions do not match')
        output = np.empty(weights.shape[0], dtype=np.float32)
        # Small fixed tiles keep activations, weights and partial sums in scalar SRAM.
        columns = min(256, len(x))
        rows = max(1, (2048 - columns) // (columns + 1))
        for r in range(0, weights.shape[0], rows):
            count = min(rows, weights.shape[0] - r)
            partial = np.zeros(count, np.float32)
            for k in range(0, len(x), columns):
                xv = np.ascontiguousarray(x[k:k + columns])
                w = np.ascontiguousarray(weights[r:r + count, k:k + columns])
                payload = xv.tobytes() + w.tobytes() + partial.tobytes()
                partial = self._run(gemv(count, len(xv)), payload, count)
            output[r:r + count] = partial
        return output

    def argmax(self, array):
        values = np.asarray(array, dtype=np.float32).reshape(-1)
        if not len(values):
            raise ValueError('Cannot select from an empty tensor')
        maxima, indices = [], []
        for offset in range(0, len(values), 2048):
            x = values[offset:offset + 2048]
            pair = self._run(argmax_program(len(x)), x.tobytes(), 2)
            maxima.append(pair[:1])
            indices.append(offset + int(pair[1]))
        if len(maxima) == 1:
            return indices[0]
        winner = self.argmax(np.concatenate(maxima))
        return indices[winner]

    def matmul(self, weights, inputs):
        weights = np.asarray(weights, dtype=np.float32)
        inputs = np.asarray(inputs, dtype=np.float32)
        if inputs.ndim != 2 or weights.ndim != 2 or weights.shape[1] != inputs.shape[1]:
            raise ValueError('Batched projection dimensions do not match')
        output = np.empty((len(inputs), len(weights)), np.float32)
        columns = min(32, weights.shape[1])
        for b in range(0, len(inputs), 16):
            count = min(16, len(inputs) - b)
            rows = min(32, (2048 - count * columns) // (columns + count))
            for r in range(0, len(weights), rows):
                n = min(rows, len(weights) - r)
                partial = np.zeros((count, n), np.float32)
                for k in range(0, weights.shape[1], columns):
                    x = np.ascontiguousarray(inputs[b:b + count, k:k + columns])
                    w = np.ascontiguousarray(weights[r:r + n, k:k + columns])
                    payload = x.tobytes() + w.tobytes() + partial.tobytes()
                    partial = self._run(gemm(count, n, x.shape[1]), payload, count * n).reshape(count, n)
                output[b:b + count, r:r + n] = partial
        return output

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.device.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
