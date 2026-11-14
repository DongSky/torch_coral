"""Host runtime tests; SDK mock checks ownership, not physical TPU support."""
import hashlib
import io
import json
from multiprocessing import shared_memory
from pathlib import Path
import struct
import threading
import numpy as np
import pytest
from torch_coral.protocol import encode, decode, read_packet, write_packet
from torch_coral.shared import SharedArrays
from torch_coral_runtime import DirectSession, normalize_options
from torch_coral_runtime import direct


@pytest.mark.parametrize('dtype', [np.float32, np.int8, np.uint8, np.int32, np.bool_])
def test_protocol_noncontiguous_and_empty(dtype):
    arrays = [np.arange(12).reshape(3, 4).astype(dtype).T, np.empty((0, 2), dtype=dtype), np.array(1, dtype=dtype)]
    stream = io.BytesIO()
    write_packet(stream, encode(arrays, action='test'))
    stream.seek(0)
    values, meta = decode(read_packet(stream))
    assert meta == {'action': 'test'}
    for actual, expected in zip(values, arrays):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype


def test_protocol_rejects_invalid_bounds_and_pickle():
    header = json.dumps(dict(metadata={}, arrays=[dict(dtype='|i1', shape=[8], offset=0, size=8)])).encode()
    with pytest.raises(ValueError, match='bounds'):
        decode(b'TC02' + struct.pack('<I', len(header)) + header + b'xx')
    with pytest.raises(TypeError, match='numeric'):
        encode([np.array([object()], dtype=object)])
    with pytest.raises(ValueError, match='length'):
        decode(b'TC02' + struct.pack('<I', 2**30))
    with pytest.raises(EOFError):
        read_packet(io.BytesIO(b'x'))


def test_legacy_npz_reader():
    stream = io.BytesIO()
    np.savez(stream, __meta__=np.array('{"legacy": true}'), tensor_0=np.ones(2))
    arrays, metadata = decode(stream.getvalue())
    assert metadata['legacy']
    np.testing.assert_array_equal(arrays[0], np.ones(2))


def test_shared_memory_ownership_and_bounds():
    desc = dict(inputs=[dict(shape=[1, 4], dtype='|i1')], outputs=[dict(shape=[1, 4], dtype='<f4')])
    owner = SharedArrays(desc)
    name = owner.memory.name
    attached = SharedArrays(desc, name=name)
    try:
        owner.arrays['inputs'][0][:] = 7
        np.testing.assert_array_equal(attached.arrays['inputs'][0], np.full((1, 4), 7))
        assert owner.entries['outputs'][0]['offset'] % 64 == 0
    finally:
        attached.close()
        owner.close()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)
    with pytest.raises(ValueError, match='256 MiB'):
        SharedArrays(dict(inputs=[dict(shape=[2**30], dtype='|i1')], outputs=[]))


def test_shared_attachment_process_does_not_unlink_owner():
    import subprocess
    import sys
    desc = dict(inputs=[dict(shape=[1, 4], dtype='|i1')], outputs=[])
    owner = SharedArrays(desc)
    try:
        script = ('import json,sys;from torch_coral.shared import SharedArrays;'
                  'm=SharedArrays(json.loads(sys.argv[1]),name=sys.argv[2],untrack=True);'
                  'm.arrays["inputs"][0][:]=11;m.close()')
        result = subprocess.run([sys.executable, '-c', script, json.dumps(desc), owner.memory.name],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert 'resource_tracker' not in result.stderr
        np.testing.assert_array_equal(owner.arrays['inputs'][0], np.full((1, 4), 11))
    finally:
        owner.close()


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    calls = []
    class FakeInterpreter:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.arrays = [np.zeros((1, 4), np.int8), np.zeros((1, 4), np.int8)]
            self.invocations = 0
        def allocate_tensors(self): pass
        def get_input_details(self):
            return [dict(name='x', index=0, dtype=np.int8, shape=np.array([1, 4]), quantization=(0.5, 0))]
        def get_output_details(self):
            return [dict(name='y', index=1, dtype=np.int8, shape=np.array([1, 4]), quantization=(0.5, 0))]
        def tensor(self, i): return lambda: self.arrays[i]
        def get_tensor(self, i): return self.arrays[i].copy()
        def set_tensor(self, i, array): self.arrays[i][:] = array
        def invoke(self):
            self.invocations += 1
            self.arrays[1][:] = self.arrays[0]
    monkeypatch.setattr(direct, '_sdk', lambda: (FakeInterpreter, lambda opts: calls.append(opts) or object(),
        dict(devices=[dict(type='usb', path='test')], runtime='mock', usb_backend='mock')))
    model = tmp_path / 'model.tflite'
    model.write_bytes(b'test model')
    manifest = dict(format_version=1, model_file=model.name, sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        inputs=[dict(name='x', tflite_name='x', shape=[1, 4], perm=[0, 1])],
        outputs=[dict(name='y', tflite_name='y', shape=[1, 4], inverse_perm=[0, 1])],
        compiler_report=dict(cpu_operations=0))
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return tmp_path, calls


def test_direct_buffers_outputs_and_close(sdk):
    folder, calls = sdk
    with DirectSession(folder, delegate_options={'Performance': 'High'}, num_threads=2) as session:
        handle = session.input_buffer()
        output = session.output_buffer()
        handle.copy_from(np.array([[1, 2, 3, 4]], np.int8))
        session.invoke()
        np.testing.assert_array_equal(output.numpy(), [[1, 2, 3, 4]])
        with pytest.raises(TypeError, match='read-only'):
            output[:] = 0
        copy = output.numpy()
        copy[:] = 0
        assert output[0, 0] == 1
        destination = np.empty((1, 4), np.int8)
        assert session.run_raw([np.full((1, 4), 2, np.int8)], out=[destination])[0] is destination
        with session.borrow_interpreter() as interpreter:
            assert interpreter is session.interpreter
        before = interpreter.invocations
        with pytest.raises(TypeError):
            session.run_raw([np.zeros((1, 4), np.float32)])
        with pytest.raises(ValueError, match='shape'):
            session.run_raw([np.zeros(4, np.int8)])
        assert interpreter.invocations == before
    with pytest.raises(RuntimeError, match='closed'):
        handle.numpy()
    assert calls[0] == {'device': 'usb:0', 'Performance': 'High'}
    assert calls[1]['num_threads'] == 2


def test_float_constraints_and_option_conflicts(sdk):
    folder, _ = sdk
    with DirectSession(folder) as first, DirectSession(folder) as second:
        assert first.lock is second.lock
        with pytest.raises(ValueError, match='different delegate options'):
            DirectSession(folder, delegate_options={'Performance': 'Low'})
        x = np.array([[0.0, 0.5, -0.5, 1.0]], np.float32)
        np.testing.assert_array_equal(first.run_float([x])[0], x)
        with pytest.raises(ValueError, match='NaN'):
            first.run_float([np.full((1, 4), np.nan, np.float32)])
        first.manifest['operator_rewrites'] = dict(input_constraints=[dict(name='x', min=0, max=3, integer_values=True)])
        with pytest.raises(ValueError, match='integer-valued'):
            first.run_float([x])
    with DirectSession(folder, delegate_options={'Performance': 'Low'}): pass
    with pytest.raises(ValueError, match='indexed device'):
        DirectSession(folder, device='usb', delegate_options={'Performance': 'Low'})
    assert normalize_options({'Usb.AlwaysDfu': False}) == {'Usb.AlwaysDfu': 'False'}


@pytest.mark.parametrize('options', [{'device': 'usb:0'}, {'Performance': 'Turbo'},
    {'Usb.MaxBulkInQueueLength': -1}, {'Usb.MaxBulkInQueueLength': 'xx'}, {'x': object()}])
def test_invalid_delegate_options(options):
    with pytest.raises((ValueError, TypeError)):
        normalize_options(options)


def test_direct_calls_are_serialized(sdk):
    from concurrent.futures import ThreadPoolExecutor
    folder, _ = sdk
    with DirectSession(folder) as session:
        def task(value):
            return session.run_raw([np.full((1, 4), value, np.int8)])[0].copy()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(task, range(20)))
        for value, result in enumerate(results):
            np.testing.assert_array_equal(result, np.full((1, 4), value))
