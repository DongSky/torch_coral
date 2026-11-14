"""Turbo V4: prepared transport and wider in-place primary quantization."""
from .isa_batched import BatchedMatrixTPU
from .usb_buffers import enable_direct_readback
from .usb_dispatch import PinnedAsyncBatch, PreparedRuntime
import numpy as np
from collections import OrderedDict
from contextlib import contextmanager
import weakref


class IOOptimizedMatrixTPU(BatchedMatrixTPU):
    def __init__(self, *args, prepared_usb=True, wide_quantization=True,
                 reuse_quantization=True, compute_fusions=True, head_fusion=True, active_bias=True,
                 reuse_decode_calibration=False, **kwargs):
        if not kwargs.get('async_usb', True):
            raise ValueError('Turbo V4 requires asynchronous USB')
        self._legacy_readback_buffers = None
        self.direct_usb_readback = False
        self.prepared_usb_dispatch = False
        self._prepared_runtime = None
        self._legacy_v4_batch = None
        self.wide_projection_quantization = bool(wide_quantization)
        self.reuse_projection_quantization = bool(reuse_quantization)
        self.compute_fusions = bool(compute_fusions)
        self._quant_scope = None
        self._normalized_quantization = OrderedDict()
        self.fused_head_selection = bool(head_fusion)
        self._head_collection = None
        self._head_summaries = OrderedDict()
        self.native_active_bias = bool(active_bias)
        self.native_large_min = 8
        self.reuse_decode_calibration = bool(reuse_decode_calibration)
        self.decode_calibration = {}
        kwargs.setdefault('native_batch_min',3)
        super().__init__(*args, **kwargs)
        try:
            with self.lock:
                self._legacy_readback_buffers = enable_direct_readback(self.device.batch)
                self.direct_usb_readback = True
                self.matrix_stats['direct_usb_readback'] = True
                if prepared_usb:
                    self._legacy_v4_batch = self.device.batch
                    self.device.batch = PinnedAsyncBatch(self.device.batch)
                    self._prepared_runtime = PreparedRuntime(self.device)
                    self.prepared_usb_dispatch = True
                self.matrix_stats['prepared_usb_dispatch'] = self.prepared_usb_dispatch
                self.matrix_stats.update(wide_projection_quantization=self.wide_projection_quantization,
                                         wide_quantization_calls=0, reused_quantization_rows=0,
                                         fused_normalization_rows=0, fused_silu_multiply_calls=0,
                                         fused_head_tiles=0, fused_head_argmax_hits=0)
        except Exception:
            self.close()
            raise

    def _run_executable(self, exe, inputs=b'', parameters=None):
        if self._prepared_runtime is not None:
            return self._prepared_runtime.run(exe, inputs, parameters)
        return super()._run_executable(exe, inputs, parameters)

    def quantize_projection_input(self, array):
        scope = self._quant_scope
        key = None
        if scope is not None and self.components == 1:
            owner, cache = scope
            address = array.__array_interface__['data'][0]
            start = owner.__array_interface__['data'][0]
            width = owner.shape[-1]
            if (array.dtype == np.float32 and array.flags.c_contiguous and array.size == width
                    and start <= address < start + owner.nbytes and (address - start) % (width * 4) == 0):
                key = (address - start) // (width * 4)
                if key in cache:
                    self.matrix_stats['reused_quantization_rows'] += 1
                    return cache[key]
        result = self._quantize_uncached(array)
        if key is not None:
            for value in result:
                if isinstance(value,np.ndarray):value.setflags(write=False)
            cache[key] = result
        return result

    def _quantize_uncached(self, array):
        if self.wide_projection_quantization and self.components == 1 and 1024 < array.size <= 4000:
            from .isa_v4_fusions import wide_primary_quantize_program
            values = np.ascontiguousarray(array, np.float32).reshape(-1)
            output = self._inplace_result(wide_primary_quantize_program(len(values)), values.tobytes(), len(values) // 4 + 1)
            self.matrix_stats['wide_quantization_calls'] += 1
            return output[:-1].view(np.uint8).copy(), output[-1:].copy(), None
        return super().quantize_projection_input(array)

    @contextmanager
    def projection_scope(self, inputs):
        if not self.reuse_projection_quantization or self.components != 1:
            yield inputs
            return
        with self.lock:
            original = self._quant_scope
            frozen = np.array(inputs, np.float32, order='C', copy=True)
            frozen.setflags(write=False)
            cache = {}
            record = self._normalized_quantization.get(id(inputs))
            if record is not None and record[0]() is inputs and frozen.tobytes() == record[1]:
                cache = dict(enumerate(record[2]))
            self._quant_scope = frozen, cache
            try:
                yield frozen
            finally:
                self._quant_scope = original

    def project(self, name, weights, inputs):
        if self._quant_scope is None and id(inputs) in self._normalized_quantization:
            with self.projection_scope(inputs) as frozen:
                return super().project(name, weights, frozen)
        return super().project(name, weights, inputs)

    def _project(self, name, weights, inputs):
        x = np.asarray(inputs, np.float32)
        rows = 1 if x.ndim == 1 else len(x)
        saved_large=self.native_large_min
        if name=='token_embd.weight':
            self.native_large_min=12
        native_threshold=max(self.native_batch_min,self.native_large_min)
        enabled = (self.fused_head_selection and name == 'token_embd.weight' and self.components == 1
                   and self.output_digits == 2 and rows < native_threshold and self.chunk_rows % 2048 == 0)
        previous = self._head_collection
        if enabled:
            self._head_collection = []
        try:
            result = super()._project(name, weights, inputs)
            if enabled:
                columns = weights.shape[0]
                expected = ((columns + self.chunk_rows - 1) // self.chunk_rows) * rows
                if len(self._head_collection) != expected:
                    raise RuntimeError('Head reconstruction summary layout mismatch')
                row_summaries = [[] for _ in range(rows)]
                for entry, summaries in enumerate(self._head_collection):
                    row = entry % rows
                    offset = (entry // rows) * self.chunk_rows
                    row_summaries[row].extend((offset + start, pair) for start, pair in summaries)
                for row, values in enumerate(result.reshape(rows, columns)):
                    address = values.__array_interface__['data'][0]
                    self._head_summaries[address] = (weakref.ref(result), values.tobytes(), row_summaries[row])
                    self._head_summaries.move_to_end(address)
                while len(self._head_summaries) > 8:
                    self._head_summaries.popitem(last=False)
            return result
        finally:
            self._head_collection = previous
            self.native_large_min=saved_large

    def _reconstruct(self, outputs, coefficients, row_scale, input_scale):
        if self._head_collection is None:
            return super()._reconstruct(outputs, coefficients, row_scale, input_scale)
        from .isa_v2_fusions import reconstruct_inplace_program
        parts, summaries = [], []
        for start in range(0, len(row_scale), 2048):
            scales = np.ascontiguousarray(row_scale[start:start + 2048], np.float32)
            length = len(scales)
            payload = (b''.join(q[start:start + length].tobytes() for q in outputs)
                       + scales.tobytes() + input_scale.tobytes() + b'\0' * 32)
            data = self._inplace_result(reconstruct_inplace_program(length, tuple(coefficients), 4), payload, length + 8)
            parts.append(data[:length])
            summaries.append((start, data[length:].reshape(4, 2).copy()))
            self.matrix_stats['fused_head_tiles'] += 1
        self._head_collection.append(summaries)
        return np.concatenate(parts)

    def argmax(self, array):
        values = np.asarray(array, np.float32).reshape(-1)
        record = self._head_summaries.get(values.__array_interface__['data'][0]) if self.fused_head_selection else None
        if record is None or record[0]() is None or values.tobytes() != record[1]:
            return super().argmax(array)
        maxima = np.asarray([pair[0, 0] for _, pair in record[2]], np.float32)
        winner = super().argmax(maxima) if len(maxima) > 1 else 0
        self.draft_candidate_ids = tuple(offset + int(item[1]) for offset, pair in record[2] for item in pair)
        self.matrix_stats['fused_head_argmax_hits'] += 1
        offset, pair = record[2][winner]
        return offset + int(pair[0, 1])

    def normalize_project_input(self, array, weight, l2, epsilon):
        x = np.asarray(array, np.float32)
        if (not self.compute_fusions or not self.reuse_projection_quantization or self.components != 1
                or l2 or weight is None or x.shape[-1] != 1024):
            return self.normalize(array, weight, l2, epsilon)
        from .isa_v4_fusions import normalize_quantize_program
        rows = x.reshape(-1, 1024)
        result = np.empty_like(x)
        destination = result.reshape(-1, 1024)
        prepared = []
        for index, row in enumerate(rows):
            data = self._run(normalize_quantize_program(float(epsilon)), row.tobytes() + weight.tobytes(), 1281)
            destination[index] = data[:1024]
            packed,scale=data[1024:-1].view(np.uint8).copy(),data[-1:].copy()
            packed.setflags(write=False);scale.setflags(write=False)
            prepared.append((packed,scale,None))
            self.matrix_stats['fused_normalization_rows'] += 1
        self._normalized_quantization[id(result)] = (weakref.ref(result), result.tobytes(), prepared)
        self._normalized_quantization.move_to_end(id(result))
        while len(self._normalized_quantization) > 4:
            self._normalized_quantization.popitem(last=False)
        return result

    def silu_multiply(self, gate, up):
        if not self.compute_fusions:
            return self.op('mul', self.activation('silu', gate), up)
        from .isa_v4_fusions import silu_multiply_program
        a, b = np.broadcast_arrays(np.asarray(gate, np.float32), np.asarray(up, np.float32))
        shape = a.shape
        a, b = np.ascontiguousarray(a).reshape(-1), np.ascontiguousarray(b).reshape(-1)
        result = np.empty_like(a)
        for start in range(0, len(a), 2000):
            part = a[start:start + 2000]
            payload = part.tobytes() + b[start:start + len(part)].tobytes()
            result[start:start + len(part)] = self._inplace_result(silu_multiply_program(len(part)), payload, len(part))
            self.matrix_stats['fused_silu_multiply_calls'] += 1
        return result.reshape(shape)

    def close(self):
        super().close()
        self._legacy_readback_buffers = None
        if self._prepared_runtime is not None:
            self._prepared_runtime.clear()
        self._prepared_runtime = self._legacy_v4_batch = None
        self._quant_scope = None
        self._normalized_quantization.clear()
        self._head_summaries.clear()
        self._head_collection = None
        self.decode_calibration.clear()
