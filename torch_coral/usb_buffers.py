"""Direct immutable reads from the pinned raw USB driver's ctypes buffers.

The existing driver completes/cancels transfers and resubmits short IN reads.
Only its final ``bytes(buffer[:received])`` conversion changes: contiguous
slices already return bytes, avoiding a temporary Python integer list. The
returned bytes own their storage and survive reuse of the USB receive buffer.
"""
import ctypes
from functools import lru_cache


@lru_cache(maxsize=64)
def _direct_buffer_type(length):
    class DirectByteBuffer(ctypes.Array):
        _type_ = ctypes.c_ubyte
        _length_ = length

        def __getitem__(self, index):
            if isinstance(index, slice):
                start, stop, step = index.indices(len(self))
                if step == 1:
                    return ctypes.string_at(ctypes.addressof(self) + start, max(0, stop - start))
            return super().__getitem__(index)

    return DirectByteBuffer


def _direct_view(buffer):
    if buffer is None:
        return None
    if not isinstance(buffer, ctypes.Array) or buffer._type_ is not ctypes.c_ubyte:
        raise TypeError('Direct USB readback requires ctypes unsigned-byte arrays')
    # from_buffer pins the original allocation. This remains a real ctypes
    # array, so cast(), byref(offset), incoming writes and buffer sizes retain
    # their normal semantics, including segmented device outputs.
    return _direct_buffer_type(len(buffer)).from_buffer(buffer)


class DirectReadBuffers(list):
    """Wrap current and future per-transfer allocations without copying data."""
    def __init__(self, original):
        self.original = original
        super().__init__(_direct_view(buffer) for buffer in original)

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            value = [_direct_view(buffer) for buffer in value]
        else:
            value = _direct_view(value)
        super().__setitem__(index, value)


def enable_direct_readback(batch):
    """Enable direct reads for an idle batch; return its legacy buffer pool.

    This is instance-local. No driver method or process-global ctypes function
    is replaced. Call between completed programs under the backend's lock.
    """
    if batch.pending.value:
        raise RuntimeError('Cannot replace USB buffers while transfers are pending')
    if isinstance(batch.bufs, DirectReadBuffers):
        return batch.bufs.original
    original = batch.bufs
    batch.bufs = DirectReadBuffers(original)
    return original
