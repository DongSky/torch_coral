"""Buffer access without exporting borrowed interpreter/shared-memory views."""
import numpy as np


class Buffer:
    """A checked buffer handle. Reads return copies; writes never retain views.

    The owner lock also protects invocation and close. Keeping this handle after
    close is safe: subsequent access raises instead of touching freed memory.
    """
    def __init__(self, owner, getter, *, writable=True):
        self._owner, self._getter, self._writable = owner, getter, writable

    def _array(self):
        self._owner._check_open()
        return self._getter()

    @property
    def shape(self):
        with self._owner.lock:
            return self._array().shape

    @property
    def dtype(self):
        with self._owner.lock:
            return self._array().dtype

    def copy_from(self, value):
        with self._owner.lock:
            if not self._writable:
                raise TypeError("Output buffer is read-only")
            array = self._array()
            validate_array(value, array.shape, array.dtype)
            np.copyto(array, value, casting="no")

    def read_into(self, destination):
        with self._owner.lock:
            array = self._array()
            validate_array(destination, array.shape, array.dtype, writable=True)
            np.copyto(destination, array, casting="no")
        return destination

    def numpy(self):
        with self._owner.lock:
            return self._array().copy()

    def __getitem__(self, key):
        with self._owner.lock:
            value = self._array()[key]
            return value.copy() if isinstance(value, np.ndarray) else value

    def __setitem__(self, key, value):
        with self._owner.lock:
            if not self._writable:
                raise TypeError("Output buffer is read-only")
            self._array()[key] = value


def validate_array(array, shape, dtype, *, writable=False):
    if not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype):
        raise TypeError("Expected a NumPy array with dtype %s" % np.dtype(dtype))
    if tuple(array.shape) != tuple(shape):
        raise ValueError("Expected shape %s, got %s" % (tuple(shape), array.shape))
    if writable and not array.flags.writeable:
        raise ValueError("Output array must be writable")


def quantize(array, detail):
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.integer):
        scale, zero = detail["quantization"]
        if scale <= 0:
            raise ValueError("Missing per-tensor quantization parameters")
        limits = np.iinfo(dtype)
        return np.clip(np.rint(array / scale) + zero, limits.min, limits.max).astype(dtype)
    return array.astype(dtype)


def dequantize(array, detail):
    if np.issubdtype(array.dtype, np.integer):
        scale, zero = detail["quantization"]
        if scale <= 0:
            raise ValueError("Missing per-tensor quantization parameters")
        return (array.astype(np.float32) - zero) * scale
    return array.astype(np.float32)
