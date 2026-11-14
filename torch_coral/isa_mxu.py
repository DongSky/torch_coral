"""INT8 tensor-array projections with TPU quantization and residual readout.

Weight quantization is offline preparation. Activation quantization, matrix
arithmetic, bias corrections and reconstruction execute on the physical TPU.
"""
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
from .isa_scalar import ScalarTPU, _wrap, _parts


@lru_cache(maxsize=128)
def absmax_program(length):
    _, _, _, S = _parts()
    def body(base):
        loop = [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + [S.alu(S.AND, 8, 8, imm=0x7fffffff),
            S.alu(S.FMAX, 10, 10, reg=8), S.NOP]
        return [S.movi(4, base), S.movi(10, 0), S.loop(length, len(loop))] + loop + [S.movi(6, 0), S.st(6, 10)]
    return _wrap(length, 1, body)


@lru_cache(maxsize=128)
def quantize_program(length):
    """Return packed unsigned bytes followed by FP32 input residuals."""
    _, _, _, S = _parts()
    assert length % 4 == 0
    packed = length // 4
    def body(base):
        words = [S.movi(4, base), S.movi(6, 0), S.movi(14, packed), S.movi(7, base + length),
                 S.ld(15, 7, inc=True), S.ld(16, 7)] + [S.NOP] * 4
        loop = [S.movi(20, 0)]
        for shift in (0, 8, 16, 24):
            loop += [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + [
                S.alu(S.FMUL, 10, 8, reg=16), S.fimm(S.FMAX, 10, 10, -127.),
                S.fimm(S.FMIN, 10, 10, 127.), S.f2i(10, 10)] + [S.NOP] * 2 + [
                S.i2f(11, 10), S.NOP, S.alu(S.FMUL, 11, 11, reg=15), S.alu(S.FSUB, 11, 8, reg=11),
                S.st(14, 11), S.alu(S.ADD, 14, 14, imm=1), S.alu(S.ADD, 10, 10, imm=128),
                S.alu(S.SHL, 10, 10, imm=shift), S.alu(S.OR, 20, 20, reg=10)]
        loop += [S.st(6, 20), S.alu(S.ADD, 6, 6, imm=1)]
        return words + [S.loop(packed, len(loop))] + loop
    return _wrap(length + 2, packed + length, body)


@lru_cache(maxsize=256)
def correction_program(length, coefficient):
    """New int32 biases = old biases - (output_byte-128)*coefficient."""
    _, _, _, S = _parts()
    packed = (length + 3) // 4
    def body(base):
        words = []
        # Unroll four channels, share each loaded packed word.
        words += [S.movi(4, base), S.movi(5, base + packed), S.movi(6, 0)]
        loop = [S.ld(8, 4, inc=True)] + [S.NOP] * 4
        for shift in (0, 8, 16, 24):
            loop += [S.alu(S.SHR, 9, 8, imm=shift), S.alu(S.AND, 9, 9, imm=255),
                     S.alu(S.SUB, 9, 9, imm=128), S.i2f(9, 9), S.NOP,
                     S.fimm(S.FMUL, 9, 9, coefficient), S.f2i(9, 9)] + [S.NOP] * 2 + [
                     S.ld(10, 5, inc=True)] + [S.NOP] * 4 + [S.alu(S.SUB, 10, 10, reg=9),
                     S.st(6, 10), S.alu(S.ADD, 6, 6, imm=1)]
        return words + [S.loop(packed, len(loop))] + loop
    assert length % 4 == 0
    return _wrap(packed + length, length, body)


@lru_cache(maxsize=256)
def reconstruct_program(length, coefficients):
    _, _, _, S = _parts()
    assert length % 4 == 0
    packed = length // 4
    digits = len(coefficients)
    def body(base):
        words = [S.movi(6, 0), S.movi(7, base + digits * packed),
                 S.movi(17, base + digits * packed + length), S.ld(15, 17)] + [S.NOP] * 4
        for digit in range(digits): words += [S.movi(20 + digit, base + digit * packed)]
        loop = []
        for digit in range(digits): loop += [S.ld(8 + digit, 20 + digit, inc=True)]
        loop += [S.NOP] * 4
        for shift in (0, 8, 16, 24):
            loop += [S.movi(12, 0)]
            for digit, coefficient in enumerate(coefficients):
                loop += [S.alu(S.SHR, 11, 8 + digit, imm=shift), S.alu(S.AND, 11, 11, imm=255),
                    S.alu(S.SUB, 11, 11, imm=128), S.i2f(11, 11), S.NOP,
                    S.fimm(S.FMUL, 11, 11, coefficient), S.alu(S.FADD, 12, 12, reg=11)]
            loop += [S.ld(13, 7, inc=True)] + [S.NOP] * 4 + [S.alu(S.FMUL, 12, 12, reg=13),
                S.alu(S.FMUL, 12, 12, reg=15), S.st(6, 12), S.alu(S.ADD, 6, 6, imm=1)]
        return words + [S.loop(packed, len(loop))] + loop
    return _wrap(digits * packed + length + 1, length, body)


@lru_cache(maxsize=32)
def recurrent_program(columns, length):
    """Fuse FP32 decay, two GEMVs and rank-one update for a state-column tile."""
    _, _, _, S = _parts()
    def body(base):
        end = base + columns * length
        words = [S.movi(7, end + 2 * length + columns), S.ld(15, 7, inc=True),
                 S.ld(16, 7)] + [S.NOP] * 4  # beta, decay
        for column in range(columns):
            start = base + column * length
            words += [S.movi(4, start), S.movi(5, end), S.movi(10, 0)]
            dot = [S.ld(8, 4, inc=True), S.ld(9, 5, inc=True)] + [S.NOP] * 4 + [
                S.alu(S.FMUL, 8, 8, reg=16), S.alu(S.FMUL, 11, 8, reg=9),
                S.NOP, S.alu(S.FADD, 10, 10, reg=11), S.NOP]
            words += [S.loop(length, len(dot))] + dot
            words += [S.movi(7, end + 2 * length + column), S.ld(12, 7)] + [S.NOP] * 4 + [
                S.alu(S.FSUB, 12, 12, reg=10), S.alu(S.FMUL, 12, 12, reg=15),
                S.movi(4, start), S.movi(5, end), S.movi(7, end + length),
                S.movi(6, column * length), S.movi(10, 0)]
            update = [S.ld(8, 4, inc=True), S.ld(9, 5, inc=True), S.ld(13, 7, inc=True)] + [S.NOP] * 4 + [
                S.alu(S.FMUL, 8, 8, reg=16), S.alu(S.FMUL, 9, 9, reg=12),
                S.alu(S.FADD, 8, 8, reg=9), S.st(6, 8), S.alu(S.ADD, 6, 6, imm=1),
                S.alu(S.FMUL, 11, 8, reg=13), S.NOP, S.alu(S.FADD, 10, 10, reg=11), S.NOP]
            words += [S.loop(length, len(update))] + update + [S.movi(6, columns * length + column), S.st(6, 10)]
        return words
    return _wrap(columns * length + 2 * length + columns + 2, columns * length + columns, body)


@lru_cache(maxsize=128)
def bias_executable(n, k, parameter_offset=0):
    from coral.codegen import caching_program
    from coral.codegen.fc import FCGeom
    from coral.isa import ttu
    from coral.programs import _exe
    from coral.executable import Hint
    g = FCGeom(max(64, n), k)
    parts = []
    for tile in range(g.T):
        groups = g.gt(tile)
        offset = tile * g.P * 256
        rc = dict(tile_mask=1 << tile, addr=2 + parameter_offset // 64, sdims=int(groups > 1), **ttu('', [1], [groups], 4))
        parts.append((rc, offset, groups * 256, 4))
    size = g.G * 256
    code = caching_program(size, [parts])
    return _exe('PARAMETER_CACHING', code, [Hint('dma', 'INFEED', 'PARAMETER', None, 0, size)])


def prepare_matrix_weights(store, directory, *, progress=print, rebuild=False):
    """Offline row-wise two-component INT8 weight representation."""
    directory = Path(directory).resolve()
    manifest_path = directory / 'matrices.json'
    if manifest_path.exists() and not rebuild:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('source_sha256') != store.manifest['source_sha256'] or manifest.get('format_version') != 2:
            raise ValueError('Matrix cache does not match source weights')
        return manifest
    directory.mkdir(parents=True, exist_ok=True)
    matrices = {}
    for name, record in store.manifest['tensors'].items():
        shape = record['shape']
        if len(shape) != 2 or shape[0] * shape[1] < 262144: continue
        weight = store[name]
        scales = np.empty((2, shape[0]), np.float32)
        arrays = [np.lib.format.open_memmap(directory / (name + '.q%d.npy' % i), mode='w+', dtype=np.uint8, shape=tuple(shape)) for i in range(2)]
        for start in range(0, shape[0], 256):
            residual = np.array(weight[start:start + 256], dtype=np.float32)
            for digit in range(2):
                scale = np.maximum(np.max(np.abs(residual), axis=1), np.float32(1e-20)) / np.float32(127)
                q = np.clip(np.rint(residual / scale[:, None]), -127, 127).astype(np.int16)
                arrays[digit][start:start + len(residual)] = (q + 128).astype(np.uint8)
                scales[digit, start:start + len(residual)] = scale
                residual = residual - q.astype(np.float32) * scale[:, None]
        for array in arrays: array.flush()
        np.save(directory / (name + '.scales.npy'), scales)
        matrices[name] = dict(shape=shape)
        progress('INT8 weights: ' + name)
    manifest = dict(format_version=2, source_sha256=store.manifest['source_sha256'], matrices=matrices)
    temp = manifest_path.with_suffix('.tmp'); temp.write_text(json.dumps(manifest, indent=2)); temp.replace(manifest_path)
    return manifest


class MatrixTPU(ScalarTPU):
    def __init__(self, matrix_directory, *, components=2, output_digits=3, chunk_rows=6144, async_usb=False, clock='high'):
        if components not in (1, 2) or output_digits not in (1, 2, 3): raise ValueError('Invalid precision profile')
        self.matrix_directory = Path(matrix_directory).resolve()
        self.matrix_manifest = json.loads((self.matrix_directory / 'matrices.json').read_text())
        if self.matrix_manifest.get('format_version') != 2:
            raise ValueError('Rebuild the matrix cache with the current preparation code')
        if chunk_rows not in (512, 1024, 2048, 4096, 6144):
            raise ValueError('chunk_rows must be 512, 1024, 2048, 4096 or 6144')
        self.components, self.output_digits, self.chunk_rows = components, output_digits, chunk_rows
        self._matrices = {}
        self._resident = None
        self._allocation = None
        self.matrix_stats = dict(projections=0, weight_uploads=0, weight_bytes=0, mxu_calls=0, bias_uploads=0, range_retries=0)
        super().__init__(async_usb=async_usb, clock=clock)

    def recurrent(self, state, k, q, v, beta, decay):
        """Return updated state and projection; all arithmetic is physical FP32."""
        length = len(k)
        columns = min(getattr(self, 'recurrent_columns', 12), (2048 - 2 * length - 2) // (length + 1))
        if columns < 1: raise ValueError('Recurrent state exceeds verified scalar SRAM tile')
        updated, output = [], []
        for offset in range(0, len(v), columns):
            count = min(columns, len(v) - offset)
            tile = np.ascontiguousarray(state[:, offset:offset + count].T, np.float32)
            payload = (tile.tobytes() + k.tobytes() + q.tobytes() + v[offset:offset + count].tobytes()
                       + np.float32(beta).tobytes() + np.float32(decay).tobytes())
            values = self._run(recurrent_program(count, length), payload, count * length + count)
            updated.append(values[:count * length].reshape(count, length).T)
            output.append(values[count * length:])
        return np.concatenate(updated, axis=1), np.concatenate(output)

    def _execute(self, exe, *, inputs=b'', parameters=None):
        with self.lock:
            if self.closed: raise RuntimeError('Matrix TPU is closed')
            if self.cancel_event is not None and self.cancel_event.is_set(): raise InterruptedError('Cancelled')
            start = time.perf_counter()
            try: result = self._run_executable(exe, inputs, parameters)
            except Exception as exc:
                from coral.device import mark_hung
                mark_hung('MXU execution failed: ' + type(exc).__name__)
                self.close(); raise
            self.calls += 1; self.device_ms += (time.perf_counter() - start) * 1000
            return result

    def quantize_input(self, array):
        x = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
        maxima = [self._run(absmax_program(len(v)), v.tobytes(), 1)
                  for v in (x[i:i + 2048] for i in range(0, len(x), 2048))]
        maximum = self.reduce('max', np.concatenate(maxima))
        scale = self.op('mul', self.op('max', maximum, np.float32(1e-20)), np.float32(1 / 127))
        inverse = self.op('recip', scale)
        quantized, residual = [], []
        for i in range(0, len(x), 1024):
            v = x[i:i + 1024]
            if len(v) % 4: raise ValueError('INT8 projections require K divisible by four')
            values = self._run(quantize_program(len(v)), v.tobytes() + scale.tobytes() + inverse.tobytes(), len(v) // 4 + len(v))
            quantized.append(values[:len(v) // 4].view(np.uint8).copy())
            residual.append(values[len(v) // 4:])
        return np.concatenate(quantized), scale, np.concatenate(residual)

    def _upload_bias(self, n, k, bias, parameter_offset=0):
        padded = np.zeros(((n + 63) // 64) * 64, '<i4'); padded[:n] = bias
        self._execute(bias_executable(n, k, parameter_offset), parameters=padded.tobytes())
        self.matrix_stats['bias_uploads'] += 1

    def _correct(self, q, bias, coefficient):
        parts = []
        for i in range(0, len(q), 1024):
            v = q[i:i + 1024]
            values = self._run(correction_program(len(v), float(coefficient)), v.tobytes() + bias[i:i + len(v)].tobytes(), len(v))
            parts.append(values.view('<i4').copy())
        return np.concatenate(parts)

    def _reconstruct(self, outputs, coefficients, row_scale, input_scale):
        result = []
        for i in range(0, len(row_scale), 1024):
            scales = np.ascontiguousarray(row_scale[i:i + 1024], np.float32)
            payload = b''.join(q[i:i + len(scales)].tobytes() for q in outputs) + scales.tobytes() + input_scale.tobytes()
            result.append(self._run(reconstruct_program(len(scales), tuple(coefficients)), payload, len(scales)))
        return np.concatenate(result)

    def project(self, name, weights, inputs):
        with self.lock:
            return self._project(name, weights, inputs)

    def parameter_blob(self, prog, key, qweight):
        return prog.params(qweight, np.zeros(len(qweight), np.int32), 128)

    def _project(self, name, weights, inputs):
        if name not in self.matrix_manifest['matrices']:
            return self.matmul(weights, inputs) if inputs.ndim == 2 else self.matvec(weights, inputs)
        from coral.programs import fc_program, Quant
        if name not in self._matrices:
            self._matrices[name] = ([np.load(self.matrix_directory / (name + '.q%d.npy' % i), mmap_mode='r', allow_pickle=False) for i in range(2)],
                                    np.load(self.matrix_directory / (name + '.scales.npy'), mmap_mode='r', allow_pickle=False))
        arrays, row_scales = self._matrices[name]
        if tuple(weights.shape) != tuple(arrays[0].shape):
            raise ValueError('INT8 matrix shape does not match model weights')
        inputs = np.asarray(inputs, np.float32)
        is_vector = inputs.ndim == 1
        batch = inputs.reshape(-1, inputs.shape[-1])
        quantized = []
        for x in batch:
            digits = []; residual = x
            for _ in range(self.components):
                quantize = getattr(self, 'quantize_projection_input', self.quantize_input)
                q, scale, residual = quantize(residual); digits.append((q, scale))
            quantized.append(digits)
        result = np.zeros((len(batch), arrays[0].shape[0]), np.float32)
        n_total, k = arrays[0].shape
        for offset in range(0, n_total, self.chunk_rows):
            n = min(self.chunk_rows, n_total - offset)
            if n % 4: raise ValueError('INT8 output channels must be divisible by four')
            prog = fc_program(n, k)
            if not prog.fits(): raise ValueError('Selected MXU tile exceeds parameter SRAM')
            slots = self.components if prog.tile_bytes * self.components <= prog.limit else 1
            allocation = (prog.tile_bytes, slots)
            if self._allocation != allocation or self._resident is None:
                self._resident = {}
                self._allocation = allocation
            for weight_digit in range(self.components):
                key = (name, offset, weight_digit, n, k)
                parameter_offset = (weight_digit if slots > 1 else 0) * prog.tile_bytes
                if self._resident.get(parameter_offset) != key:
                    qweight = np.ascontiguousarray(arrays[weight_digit][offset:offset + n])
                    blob = self.parameter_blob(prog, key, qweight)
                    pc, _ = prog.programs(0, parameter_offset, Quant(128, 128, 128, 1 / 4096))
                    self._execute(pc, parameters=blob)
                    self._resident[parameter_offset] = key
                    self.matrix_stats['weight_uploads'] += 1; self.matrix_stats['weight_bytes'] += len(blob)
                    clean = True
                else: clean = False
                for b, input_digits in enumerate(quantized):
                    for input_digit, (qinput, input_scale) in enumerate(input_digits):
                        bias = np.zeros(n, '<i4')
                        if not clean: self._upload_bias(n, k, bias, parameter_offset)
                        clean = False
                        first_scale = 4096.0
                        for attempt in range(8):
                            scale = first_scale
                            outputs, coefficients = [], []
                            retry = False
                            for digit in range(self.output_digits):
                                _, exe = prog.programs(0, parameter_offset, Quant(128, 128, 128, 1 / scale))
                                raw = self._execute(exe, inputs=prog.input_bytes(qinput[None, :], 128))
                                q = prog.output_array(raw)[0].copy()
                                self.matrix_stats['mxu_calls'] += 1
                                # Validate the byte stream for saturation; host only
                                # chooses another precomputed quantization range.
                                if digit == 0 and (0 in q or 255 in q):
                                    retry = True
                                    first_scale *= 4
                                    self.matrix_stats['range_retries'] += 1
                                    break
                                outputs.append(q); coefficients.append(scale)
                                if digit + 1 < self.output_digits:
                                    bias = self._correct(q, bias, scale)
                                    self._upload_bias(n, k, bias, parameter_offset)
                                scale /= 128
                            if not retry: break
                        else: raise RuntimeError('MXU quantization range could not avoid clipping')
                        value = self._reconstruct(outputs, coefficients, row_scales[weight_digit, offset:offset + n], input_scale)
                        result[b, offset:offset + n] = (value if weight_digit == input_digit == 0 else
                            self.op('add', result[b, offset:offset + n], value))
        self.matrix_stats['projections'] += 1
        return result[0] if is_vector else result
