"""USB ownership, segment barriers and failure draining without physical I/O."""
from collections import deque
import ctypes
import gc
import struct
import sys
from types import ModuleType, SimpleNamespace as NS
import pytest
from torch_coral.usb_dispatch import DispatchPlan, OutPacket, PinnedAsyncBatch, PreparedRuntime


class FakeLibrary:
    def __init__(self, original, payloads=None, submit_codes=(), statuses=()):
        self.original = original
        self.payloads = {i: deque(chunks) for i, chunks in (payloads or {}).items()}
        self.submit_codes, self.statuses = deque(submit_codes), deque(statuses)
        self.active, self.submitted, self.cancelled, self.out_bytes = [], [], [], []

    def index(self, transfer):
        return next(i for i, item in enumerate(self.original.xfers) if item is transfer)

    def libusb_submit_transfer(self, transfer):
        code = self.submit_codes.popleft() if self.submit_codes else 0
        if code < 0:
            return code
        index = self.index(transfer)
        self.submitted.append(index)
        self.active.append(transfer)
        return 0

    def libusb_cancel_transfer(self, transfer):
        self.cancelled.append(self.index(transfer))
        transfer.contents.status = 3
        return 0

    def libusb_handle_events_completed(self, ctx, done):
        # Simulate real asynchronous reads of every OUT owner, after submission
        # and garbage collection rather than while preparing the descriptor.
        gc.collect()
        transfer = self.active.pop(0)
        index, item = self.index(transfer), transfer.contents
        if not item.status and self.statuses:
            item.status = self.statuses.popleft()
        if not item.status:
            if item.endpoint & 0x80:
                data = self.payloads[index].popleft()
                ctypes.memmove(item.buffer, data, len(data))
                item.actual_length = len(data)
            else:
                self.out_bytes.append(ctypes.string_at(item.buffer, item.length))
                item.actual_length = item.length
        self.original.done_flags[index] = True
        self.original.pending.value -= 1
        return 0


def fake_batch(payloads=None, **kwargs):
    callback = ctypes.CFUNCTYPE(None)(lambda: None)
    original = NS(dev=NS(handle=ctypes.c_void_p(1)), pending=ctypes.c_int(0),
                  xfers=[NS(contents=NS()) for _ in range(8)], bufs=[None] * 8,
                  done_flags=[False] * 8, ctx=None, cb=callback)
    library = FakeLibrary(original, payloads, **kwargs)
    return PinnedAsyncBatch(original, library=library), library


def test_pinned_out_is_identical_and_in_results_own_storage():
    batch, lib = fake_batch({1: [b'A\0B\xff'], 2: [b'EVNT']})
    packet = OutPacket(b'header\0binary\xffpayload')
    assert ctypes.string_at(packet.address, len(packet)) == packet.data
    result = batch.run([(1, packet), (0x81, 4), (0x82, 4)])
    assert result == [None, b'A\0B\xff', b'EVNT']
    assert lib.submitted == [1, 2, 0]  # All INs before OUTs.
    assert lib.out_bytes == [packet.data]
    snapshot = result[1]
    ctypes.memset(batch.bufs[1], 0, 4)
    assert snapshot == b'A\0B\xff'
    assert batch.pending.value == 0


def test_segmented_in_continues_at_the_received_offset():
    payload = b'first\0' + b'second\xff'
    batch, lib = fake_batch({0: [payload[:6], payload[6:]], 1: [b'event']})
    result = batch.run([(0x81, len(payload)), (0x82, 5), (1, b'OUT')])
    assert result == [payload, b'event', None]
    assert lib.submitted == [0, 1, 2, 0]
    assert lib.out_bytes == [b'OUT']
    assert batch.pending.value == 0


@pytest.mark.parametrize('codes', [(0, -1), (0, 0, -1)])
def test_submission_and_resubmission_failures_drain_live_transfers(codes):
    batch, lib = fake_batch({0: [b'A']}, submit_codes=codes)
    with pytest.raises(RuntimeError, match='submit'):
        batch.run([(0x81, 2), (1, OutPacket(b'packet'))])
    assert batch.pending.value == 0 and not lib.active
    assert lib.cancelled


def test_transfer_error_cancels_other_transfers_even_on_last_completion():
    batch, lib = fake_batch({0: [b'A']}, statuses=[4])
    with pytest.raises(RuntimeError, match='status 4'):
        batch.run([(0x81, 1), (1, b'out')])
    assert lib.cancelled == [1] and batch.pending.value == 0
    only, _ = fake_batch(statuses=[4])
    with pytest.raises(RuntimeError, match='status 4'):
        only.run([(1, b'out')])
    assert only.pending.value == 0


def test_zero_progress_short_in_fails_and_drains_out_owner():
    batch, lib = fake_batch({0: [b'']})
    with pytest.raises(RuntimeError, match='no progress'):
        batch.run([(0x81, 2), (1, b'owned')])
    assert lib.cancelled == [1] and not lib.active and batch.pending.value == 0


def test_invalid_ops_fail_before_any_submission():
    batch, lib = fake_batch()
    for ops in ([(0x81, -1)], [(0x81, b'bad')], [(1, bytearray(b'mutable'))], [(1, b'a')] * 9):
        with pytest.raises((ValueError, TypeError)):
            batch.run(ops)
    assert not lib.submitted
    batch.pending.value = 1
    with pytest.raises(RuntimeError, match='active'):
        batch.run([])


def hint(kind, direction='INFEED', **kwargs):
    values = dict(kind=kind, direction=direction, desc=None, size=0, offset=0, chunk=0)
    values.update(kwargs)
    return NS(**values)


def executable(hints):
    return NS(hints=hints, bitstreams=[NS(data=b'INSTRUCTIONS')], parameters=b'PARAMETERS', type='TEST', name='test')


@pytest.fixture
def runtime_modules(monkeypatch):
    parent = ModuleType('coral'); parent.__path__ = []
    runtime = ModuleType('coral.runtime')
    runtime.TRAFFIC = dict(to_device=0, from_device=0)
    runtime.run_executable = lambda *args: b'fallback'
    device = ModuleType('coral.device'); device.mark_hung = lambda reason: None
    for name, module in [('coral', parent), ('coral.runtime', runtime), ('coral.device', device)]:
        monkeypatch.setitem(sys.modules, name, module)
    return runtime


def test_prepared_plan_preserves_dma_barriers_offsets_frames_and_traffic(runtime_modules):
    calls = []
    def run(ops):
        calls.append([(ep, value.data if isinstance(value, OutPacket) else value) for ep, value in ops])
        return [b'X' * value if ep == 0x81 else (b'E' * value if ep == 0x82 else None) for ep, value in ops]
    device = NS(batch=NS(run=run))
    exe = executable([hint('instruction'), NS(kind='dma', direction='INFEED', desc='INPUT', offset=1, size=3),
                      hint('dma', 'OUTFEED', size=4),
                      NS(kind='dma', direction='INFEED', desc='PARAMETER', offset=2, size=4),
                      hint('interrupt')])
    runtime = PreparedRuntime(device, cache_entries=1)
    for _ in range(2):
        assert runtime.run(exe, b'input', b'parameters') == b'XXXX'
    assert len(calls) == 4
    assert calls[0] == [(0x81, 4), (1, struct.pack('<II', 12, 0) + b'INSTRUCTIONS'),
                        (1, struct.pack('<II', 3, 1) + b'npu')]
    assert calls[1] == [(0x82, 16), (1, struct.pack('<II', 4, 2) + b'rame')]
    assert runtime.hits == runtime.misses == 1
    assert runtime_modules.TRAFFIC == dict(to_device=86, from_device=8)
    runtime.run(executable([hint('instruction')]))
    assert len(runtime.plans) == 1
    runtime.clear()
    assert not runtime.plans


def test_wrong_input_length_rejected_and_fences_use_original_runtime(runtime_modules):
    runtime = PreparedRuntime(NS(batch=NS(run=lambda ops: pytest.fail('must not submit'))))
    with pytest.raises(ValueError, match='DMA length'):
        runtime.run(executable([hint('dma', desc='INPUT', size=4)]), b'abc')
    assert runtime.run(executable([hint('fence')])) == b'fallback'
    # A bad later segment must fail before the first segment changes the chip.
    with pytest.raises(ValueError, match='DMA length'):
        runtime.run(executable([hint('instruction'), hint('dma', 'OUTFEED', size=4),
                                hint('dma', desc='INPUT', size=4)]), b'abc')


def test_short_out_is_rejected_after_all_transfers_are_drained():
    batch, lib = fake_batch()
    handle = lib.libusb_handle_events_completed
    def short(ctx, done):
        result = handle(ctx, done)
        batch.xfers[0].contents.actual_length -= 1
        return result
    lib.libusb_handle_events_completed = short
    with pytest.raises(RuntimeError, match='Short USB OUT'):
        batch.run([(1, b'payload')])
    assert batch.pending.value == 0 and not lib.active


def test_event_error_drains_active_transfers():
    batch, lib = fake_batch({0: [b'in']})
    handle = lib.libusb_handle_events_completed
    calls = 0
    def event(ctx, done):
        nonlocal calls
        calls += 1
        return -1 if calls == 1 else handle(ctx, done)
    lib.libusb_handle_events_completed = event
    with pytest.raises(RuntimeError, match='event handling'):
        batch.run([(0x81, 2), (1, b'out')])
    assert batch.pending.value == 0 and not lib.active and lib.cancelled == [0, 1]


def test_dispatch_runtime_has_bounded_cache_and_rejects_other_infeeds():
    with pytest.raises(ValueError):
        PreparedRuntime(None, cache_entries=0)
    with pytest.raises(ValueError, match='descriptor'):
        DispatchPlan(executable([hint('dma', desc='SCRATCH', size=4)]))
