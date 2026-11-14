"""Owned host shared-memory arrays for persistent worker bindings (Python 3.9+)."""
import math
from multiprocessing import shared_memory
import os
import sys
import numpy as np

MAX_SHARED = 256 * 1024 * 1024


def layout(description):
    entries, offset = {}, 0
    for kind in ('inputs', 'outputs'):
        entries[kind] = []
        for spec in description[kind]:
            dtype = np.dtype(spec['dtype'])
            shape = spec['shape']
            if dtype.kind not in 'biuf' or len(shape) > 32 or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError('Invalid shared buffer specification')
            offset = (offset + 63) // 64 * 64
            size = math.prod(shape) * dtype.itemsize
            entries[kind].append(dict(shape=shape, dtype=dtype.str, offset=offset, size=size))
            offset += size
    if offset > MAX_SHARED:
        raise ValueError('Shared buffers exceed 256 MiB')
    return entries, max(1, offset)


class SharedArrays:
    def __init__(self, description, *, name=None, untrack=False):
        self.entries, size = layout(description)
        self.owner = name is None
        kwargs = dict(create=self.owner, size=size) if self.owner else dict(name=name)
        # The independent worker must not unlink the parent's segment on POSIX.
        if sys.version_info >= (3, 13):
            kwargs['track'] = self.owner or not untrack
        self.memory = shared_memory.SharedMemory(**kwargs)
        if untrack and not self.owner and os.name != 'nt' and sys.version_info < (3, 13):
            from multiprocessing import resource_tracker
            resource_tracker.unregister(self.memory._name, 'shared_memory')
        if self.memory.size < size:
            self.memory.close()
            raise ValueError('Shared memory block is too small')
        self.arrays = {kind: [np.ndarray(s['shape'], dtype=s['dtype'], buffer=self.memory.buf,
                                      offset=s['offset']) for s in specs]
                       for kind, specs in self.entries.items()}
        if self.owner:
            for arrays in self.arrays.values():
                for array in arrays:
                    array.fill(0)

    def close(self):
        self.arrays.clear()
        self.memory.close()
        if self.owner:
            try:
                self.memory.unlink()
            except FileNotFoundError:
                pass
