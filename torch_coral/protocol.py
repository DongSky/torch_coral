"""Length-delimited, pickle-free ndarray packets shared with the Python 3.9 worker."""
import io
import json
import struct
import math
import numpy as np

MAX_PACKET = 256 * 1024 * 1024

def encode(arrays=(), **metadata):
    values, specs, offset = [], [], 0
    for array in arrays:
        array = np.asarray(array)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        if array.dtype.kind not in 'biufc':
            raise TypeError("Only numeric arrays may cross the Coral pipe")
        specs.append(dict(shape=list(array.shape), dtype=array.dtype.str, offset=offset, size=array.nbytes))
        values.append(array)
        offset += array.nbytes
    header = json.dumps(dict(metadata=metadata, arrays=specs), separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(header) > 1024 * 1024 or 8 + len(header) + offset > MAX_PACKET:
        raise ValueError("Coral IPC packet exceeds size limit")
    return b'TC02' + struct.pack('<I', len(header)) + header + b''.join(a.tobytes() for a in values)

def decode(payload):
    if len(payload) > MAX_PACKET:
        raise ValueError("Coral IPC packet exceeds 256 MiB")
    if payload.startswith(b'TC02'):
        if len(payload) < 8:
            raise ValueError("Truncated Coral IPC header")
        count = struct.unpack('<I', payload[4:8])[0]
        if count > 1024 * 1024 or 8 + count > len(payload):
            raise ValueError("Invalid Coral IPC header length")
        header = json.loads(payload[8:8 + count])
        if not isinstance(header.get('metadata'), dict) or not isinstance(header.get('arrays'), list):
            raise ValueError("Invalid Coral IPC metadata")
        data = memoryview(payload)[8 + count:]
        arrays, offset = [], 0
        for spec in header['arrays']:
            dtype = np.dtype(spec['dtype'])
            shape = spec['shape']
            if dtype.kind not in 'biufc' or not isinstance(shape, list) or len(shape) > 32:
                raise ValueError("Invalid Coral IPC array type or shape")
            if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape):
                raise ValueError("Invalid Coral IPC dimensions")
            size = math.prod(shape) * dtype.itemsize
            if spec['offset'] != offset or spec['size'] != size or offset + size > len(data):
                raise ValueError("Invalid Coral IPC array bounds")
            arrays.append(np.frombuffer(data, dtype=dtype, count=math.prod(shape), offset=offset).reshape(shape))
            offset += size
        if offset != len(data):
            raise ValueError("Trailing Coral IPC data")
        return arrays, header['metadata']
    # Accept 0.1 packets for offline readers; new workers use TC02 both ways.
    with np.load(io.BytesIO(payload), allow_pickle=False) as packet:
        metadata = json.loads(str(packet["__meta__"]))
        arrays = [packet["tensor_%d" % i] for i in range(len(packet.files) - 1)]
    return arrays, metadata

def read_exact(stream, count):
    chunks = []
    while count:
        chunk = stream.read(count)
        if not chunk:
            raise EOFError("Coral worker closed its pipe")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)

def read_packet(stream):
    size = struct.unpack("<Q", read_exact(stream, 8))[0]
    if size > MAX_PACKET:
        raise ValueError("Coral IPC packet exceeds 256 MiB")
    return read_exact(stream, size)

def write_packet(stream, payload):
    if len(payload) > MAX_PACKET:
        raise ValueError("Coral IPC packet exceeds 256 MiB")
    stream.write(struct.pack("<Q", len(payload)))
    stream.write(payload)
    stream.flush()
