"""Receive-buffer ownership and segmented-read contracts; no device is opened."""
import ctypes
import gc
from types import SimpleNamespace
import pytest
from torch_coral.usb_buffers import DirectReadBuffers, enable_direct_readback


def batch_with(data):
    array = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    return SimpleNamespace(bufs=[array, None], pending=SimpleNamespace(value=0))


@pytest.mark.parametrize('length', [16, 512, 8192, 16384, 98304])
def test_received_bytes_are_exact_and_survive_next_dma(length):
    data = (bytes(range(256)) * ((length + 255) // 256))[:length]
    batch = batch_with(data)
    original = enable_direct_readback(batch)
    assert batch.bufs is not original
    assert isinstance(batch.bufs, DirectReadBuffers)
    view = batch.bufs[0]
    snapshot = bytes(view[:length])
    assert snapshot == data and isinstance(view[:length], bytes)
    ctypes.memset(view, 0, length)
    assert snapshot == data
    assert bytes(view[:length]) == b'\0' * length
    assert isinstance(original[0][:length], list)


def test_original_allocation_stays_pinned_after_owner_is_discarded():
    batch = batch_with(b'prefix\0payload')
    original = enable_direct_readback(batch)
    original.clear()
    gc.collect()
    assert batch.bufs[0][:] == b'prefix\0payload'
    assert ctypes.cast(batch.bufs[0], ctypes.c_void_p).value


def test_reallocation_and_short_packet_append_use_the_same_ctypes_contract():
    batch = batch_with(b'old')
    enable_direct_readback(batch)
    batch.bufs[0] = (ctypes.c_ubyte * 8192)()
    del batch.bufs.original[:]
    gc.collect()
    first, second = b'A' * 3072, b'B\0' * 2560
    ctypes.memmove(batch.bufs[0], first, len(first))
    ctypes.memmove(ctypes.byref(batch.bufs[0], len(first)), second, len(second))
    assert batch.bufs[0][:8192] == first + second


def test_slice_bounds_strides_and_single_byte_access():
    batch = batch_with(bytes(range(100)))
    enable_direct_readback(batch)
    view = batch.bufs[0]
    data = bytes(range(100))
    for index in (slice(None), slice(-8, None), slice(5, 200), slice(80, 10), slice(100, 100), slice(None, None, -1), slice(1, 90, 3)):
        assert bytes(view[index]) == data[index]
    assert view[-1] == 99 and view[0] == 0
    with pytest.raises(ValueError): view[::0]


def test_installation_is_idle_only_idempotent_and_instance_local():
    active = batch_with(b'abcd'); active.pending.value = 1
    before = active.bufs
    with pytest.raises(RuntimeError): enable_direct_readback(active)
    assert active.bufs is before
    first, other = batch_with(b'abcd'), batch_with(b'efgh')
    original = enable_direct_readback(first); pool = first.bufs
    assert enable_direct_readback(first) is original and first.bufs is pool
    assert not isinstance(other.bufs, DirectReadBuffers)
    assert isinstance(other.bufs[0][:], list)


def test_unsupported_buffer_types_fail_before_replacing_the_pool():
    batch = SimpleNamespace(bufs=[(ctypes.c_int * 2)()], pending=SimpleNamespace(value=0))
    original = batch.bufs
    with pytest.raises(TypeError): enable_direct_readback(batch)
    assert batch.bufs is original


def test_v4_rejects_synchronous_mode_before_loading_weights_or_claiming_usb():
    from torch_coral.isa_io import IOOptimizedMatrixTPU
    with pytest.raises(ValueError, match='asynchronous'):
        IOOptimizedMatrixTPU('unused', async_usb=False)
