"""Prepared, instance-local dispatch for the pinned tinycoral async protocol.

The numerical programs and DMA boundaries do not change. Immutable OUT packets
remain alive until all transfers complete; libusb reads directly from those
bytes instead of first copying them into a second ctypes allocation.
"""
from collections import OrderedDict
import ctypes
import struct


class OutPacket:
    """Own an immutable USB packet and its stable address for the whole run."""
    __slots__ = ('data', 'pointer', 'address')

    def __init__(self, data):
        if not isinstance(data, bytes):
            raise TypeError('Pinned USB OUT packets require immutable bytes')
        self.data = data
        self.pointer = ctypes.c_char_p(data)
        self.address = ctypes.cast(self.pointer, ctypes.c_void_p).value

    def __len__(self):
        return len(self.data)


class PinnedAsyncBatch:
    """Reuse a vendor batch's transfers/callback, pinning OUT packet storage.

    All INs are submitted before OUTs. Short INs continue at their received
    offset, and any failure drains submitted transfers before storage can be
    reused. No extra commands are pipelined past a program's completion.
    """
    def __init__(self, original, library=None):
        if original.pending.value:
            raise RuntimeError('Cannot prepare USB dispatch during a transfer')
        if library is None:
            from coral.usbdev import _lib
            library = _lib
        self.original = original
        self.lib = library
        self.dev, self.pending = original.dev, original.pending
        self.xfers, self.bufs = original.xfers, original.bufs
        self.done_flags, self.ctx = original.done_flags, original.ctx
        self.cb = original.cb
        self._handle = ctypes.cast(self.dev.handle, ctypes.c_void_p).value
        callback = ctypes.cast(self.cb, ctypes.c_void_p).value
        for transfer in self.xfers:
            item = transfer.contents
            item.dev_handle, item.callback, item.type = self._handle, callback, 2
            item.flags, item.num_iso_packets = 0, 0

    def _drain(self, submitted, done):
        for index in submitted:
            if not self.done_flags[index]:
                self.lib.libusb_cancel_transfer(self.xfers[index])
        while self.pending.value:
            self.lib.libusb_handle_events_completed(self.ctx, ctypes.byref(done))

    def run(self, ops, timeout=6000):
        if self.pending.value:
            raise RuntimeError('USB batch is already active')
        if len(ops) > len(self.xfers):
            raise ValueError('Too many transfers in a USB segment')
        inputs, outputs, owners = [], [], []
        # Validate/retain every buffer before submitting anything. OUT owners
        # must also survive cancellation and completion callbacks on failures.
        for index, (endpoint, data) in enumerate(ops):
            item = self.xfers[index].contents
            if endpoint & 0x80:
                if not isinstance(data, int) or data < 0:
                    raise ValueError('USB IN requires a nonnegative byte count')
                size = data
                if self.bufs[index] is None or len(self.bufs[index]) < size:
                    self.bufs[index] = (ctypes.c_ubyte * max(size, 1024))()
                address = ctypes.addressof(self.bufs[index])
                inputs.append(index)
            else:
                owner = data if isinstance(data, OutPacket) else OutPacket(data)
                owners.append(owner)
                size, address = len(owner), owner.address
                outputs.append(index)
            item.endpoint, item.timeout, item.length = endpoint, timeout, size
            item.buffer, item.status, item.actual_length = address, 0, 0
            self.done_flags[index] = False
        got, counted, submitted = [0] * len(ops), [False] * len(ops), []
        done = ctypes.c_int(0)
        try:
            for index in inputs + outputs:
                self.pending.value += 1
                code = self.lib.libusb_submit_transfer(self.xfers[index])
                if code < 0:
                    self.pending.value -= 1
                    raise RuntimeError('USB submit failed %d' % code)
                submitted.append(index)
            while self.pending.value:
                code = self.lib.libusb_handle_events_completed(self.ctx, ctypes.byref(done))
                if code < 0 and code != -10:  # LIBUSB_ERROR_INTERRUPTED is retryable.
                    raise RuntimeError('USB event handling failed %d' % code)
                for index in submitted:
                    if self.done_flags[index] and self.xfers[index].contents.status:
                        raise RuntimeError('USB transfer failed with status %d' % self.xfers[index].contents.status)
                for index in inputs:
                    if not self.done_flags[index] or counted[index]:
                        continue
                    item = self.xfers[index].contents
                    counted[index] = True
                    received = item.actual_length
                    if received < 0 or got[index] + received > ops[index][1]:
                        raise RuntimeError('Invalid USB IN transfer length')
                    got[index] += received
                    if got[index] < ops[index][1]:
                        if not received:
                            raise RuntimeError('USB IN made no progress')
                        item.buffer = ctypes.addressof(self.bufs[index]) + got[index]
                        item.length = ops[index][1] - got[index]
                        item.status = item.actual_length = 0
                        self.done_flags[index] = counted[index] = False
                        self.pending.value += 1
                        code = self.lib.libusb_submit_transfer(self.xfers[index])
                        if code < 0:
                            self.pending.value -= 1
                            # This transfer was not submitted again; leave it
                            # complete so draining only touches live requests.
                            self.done_flags[index] = True
                            raise RuntimeError('USB IN resubmit failed %d' % code)
            # The last completion may have been a failure; every submitted
            # transfer must be checked even when pending has just reached zero.
            for index in submitted:
                if self.xfers[index].contents.status:
                    raise RuntimeError('USB transfer failed with status %d' % self.xfers[index].contents.status)
                if not ops[index][0] & 0x80:
                    item = self.xfers[index].contents
                    if item.actual_length != item.length:
                        raise RuntimeError('Short USB OUT transfer')
            return [ctypes.string_at(ctypes.addressof(self.bufs[index]), got[index])
                    if endpoint & 0x80 else None for index, (endpoint, _) in enumerate(ops)]
        except BaseException:
            self._drain(submitted, done)
            raise


class DispatchPlan:
    """Freeze immutable hint structure and instruction framing once per exe."""
    def __init__(self, exe):
        self.exe = exe  # Also prevents object-id reuse in the bounded cache.
        self.segments = []
        self.input_min_bytes = self.parameter_min_bytes = 0
        ins, outs = [], []
        for hint in exe.hints:
            if hint.kind == 'instruction' or (hint.kind == 'dma' and hint.direction == 'INFEED'):
                if ins:
                    self.segments.append((tuple(ins), tuple(outs)))
                    ins, outs = [], []
                if hint.kind == 'instruction':
                    data = exe.bitstreams[hint.chunk].data
                    packet = OutPacket(struct.pack('<II', len(data), 0) + data)
                    outs.append((packet, None, 0, 0))
                else:
                    if hint.desc not in ('INPUT', 'PARAMETER'):
                        raise ValueError('Unsupported infeed descriptor')
                    if hint.offset < 0 or hint.size < 0:
                        raise ValueError('Invalid infeed range')
                    if hint.desc == 'INPUT':
                        self.input_min_bytes = max(self.input_min_bytes, hint.offset + hint.size)
                    else:
                        self.parameter_min_bytes = max(self.parameter_min_bytes, hint.offset + hint.size)
                    header = struct.pack('<II', hint.size, 1 if hint.desc == 'INPUT' else 2)
                    outs.append((header, hint.desc, hint.offset, hint.size))
            elif hint.kind == 'dma' and hint.direction == 'OUTFEED':
                ins.append((0x81, hint.size))
            elif hint.kind == 'interrupt':
                ins.append((0x82, 16))
            else:
                raise ValueError('Unsupported dispatch hint')
        if ins or outs:
            self.segments.append((tuple(ins), tuple(outs)))


class PreparedRuntime:
    def __init__(self, device, *, cache_entries=512):
        if cache_entries < 1:
            raise ValueError('Dispatch cache must have at least one entry')
        self.device, self.cache_entries = device, cache_entries
        self.plans = OrderedDict()
        self.hits = self.misses = 0

    def run(self, exe, inputs=b'', parameters=None):
        from coral.runtime import TRAFFIC, run_executable
        key = id(exe)
        plan = self.plans.get(key)
        if plan is None:
            # Other hint types retain the pinned runtime's original semantics.
            if not all(h.kind in ('instruction', 'dma', 'interrupt') for h in exe.hints):
                return run_executable(self.device, exe, inputs, parameters)
            plan = DispatchPlan(exe)
            self.plans[key] = plan
            self.misses += 1
            if len(self.plans) > self.cache_entries:
                self.plans.popitem(last=False)
        else:
            self.plans.move_to_end(key)
            self.hits += 1
        params = exe.parameters if parameters is None else parameters
        # Validate the complete call before any segment changes device state.
        if len(inputs) < plan.input_min_bytes or len(params) < plan.parameter_min_bytes:
            raise ValueError('Invalid DMA length for prepared executable')
        results = []
        for ins, outs in plan.segments:
            ops = list(ins)
            outgoing_bytes = 0
            for packet, source, offset, size in outs:
                if source is not None:
                    payload = (inputs if source == 'INPUT' else params)[offset:offset + size]
                    if len(payload) != size:
                        raise ValueError('%s has an invalid DMA length' % source)
                    packet = OutPacket(packet + bytes(payload))
                ops.append((0x01, packet))
                outgoing_bytes += len(packet)
            try:
                received = self.device.batch.run(ops)
            except RuntimeError as exc:
                from coral.device import mark_hung
                mark_hung('%s %r: %s' % (exe.type, exe.name, exc))
                raise
            results.extend(data for (endpoint, _), data in zip(ins, received) if endpoint == 0x81)
            TRAFFIC['to_device'] += outgoing_bytes
            TRAFFIC['from_device'] += sum(size for endpoint, size in ins if endpoint == 0x81)
        return b''.join(results)

    def clear(self):
        self.plans.clear()
