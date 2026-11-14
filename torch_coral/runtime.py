import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import weakref
import time
import uuid
import numpy as np
import torch
from .protocol import encode, decode, read_packet, write_packet
from .settings import workspace, python_executable
from .shared import SharedArrays
from torch_coral_runtime import DirectSession, normalize_options
from torch_coral_runtime.buffers import Buffer, validate_array

ROOT = workspace()
_SESSIONS = weakref.WeakValueDictionary()
_SESSIONS_LOCK = threading.RLock()

def unflatten(tree, leaves):
    kind = tree["kind"]
    if kind == "tensor":
        return leaves[tree["index"]]
    values = [unflatten(v, leaves) for v in tree["children"]]
    if kind == "tuple": return tuple(values)
    if kind == "list": return values
    if kind == "dict": return dict(zip(tree["keys"], values))
    raise ValueError("Unknown output structure: " + kind)

def _read_responses(stream, responses):
    try:
        while True:
            responses.put(read_packet(stream))
    except Exception as exc:
        responses.put(exc)

def _stop(process, log):
    if process.poll() is None:
        # Let interpreter/delegate destructors release UsbDk before exiting.
        # Killing the owning process can leave the Windows USB device unusable.
        if process.stdin: process.stdin.close()
        try: process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    for stream in (process.stdin, process.stdout):
        if stream: stream.close()
    log.close()

class _Session:
    """One process owns a device, sharing interpreters across modules and threads.
    Multiple processes cannot independently claim the same Windows USB device.
    """
    def __init__(self, python, device, timeout, folder, options):
        self.lock = threading.RLock()
        self.closed = False
        self.timeout = timeout
        self.responses = queue.Queue()
        self.options = options
        log = (folder / "worker.log").open("ab")
        try:
            self.process = subprocess.Popen([python, "-u", str(Path(__file__).with_name("worker.py")), device,
                                            json.dumps(options)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except Exception:
            log.close()
            raise
        self.finalizer = weakref.finalize(self, _stop, self.process, log)
        threading.Thread(target=_read_responses, args=(self.process.stdout, self.responses), daemon=True).start()
        try:
            _, self.hardware = self.receive(timeout)
            if not self.hardware.get("ready"): raise RuntimeError("Worker did not report ready")
            self.hardware['delegate_options'] = dict(options)
        except Exception:
            self.close()
            raise

    def close(self):
        self.closed = True
        if self.finalizer.alive: self.finalizer()

    def receive(self, timeout):
        try: payload = self.responses.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError("Edge TPU worker timed out; see worker.log") from None
        if isinstance(payload, Exception):
            self.close()
            raise RuntimeError("Edge TPU worker exited; see worker.log") from payload
        arrays, metadata = decode(payload)
        if "error" in metadata: raise RuntimeError("Edge TPU worker error:\n" + metadata["error"])
        return arrays, metadata

    def request(self, folder, arrays=(), *, timeout=60, **metadata):
        with self.lock:
            if self.closed: raise RuntimeError("Edge TPU session is closed; retry to start a new worker")
            try:
                write_packet(self.process.stdin, encode(arrays, artifact_dir=str(folder), **metadata))
                return self.receive(timeout)
            except (BrokenPipeError, OSError):
                self.close()
                raise RuntimeError("Edge TPU worker pipe failed; see worker.log") from None


def _release_binding(session, memory, folder, token, timeout):
    try:
        if not session.closed:
            session.request(folder, action='release_shared', token=token, timeout=timeout)
    finally:
        memory.close()


class PreparedInference:
    """Persistent shared host buffers. TPU execution remains synchronous.

    Use run() for an atomic copy/invoke/read across threads. Manual buffer
    operations and invoke() can be grouped under transaction().
    """
    def __init__(self, module, raw):
        self.closed = False
        self._session = module._session
        self.lock = self._session.lock
        self.folder, self.timeout = module.artifact_dir, module.timeout
        self.raw = raw
        _, meta = self._session.request(self.folder, action='describe', raw=raw,
                                       num_threads=module.num_threads, timeout=self.timeout)
        self.description = meta['description']
        self._memory = SharedArrays(self.description)
        self.token = uuid.uuid4().hex
        try:
            self._session.request(self.folder, action='register_shared', token=self.token, raw=raw,
                num_threads=module.num_threads, shared_name=self._memory.memory.name, timeout=self.timeout)
        except Exception:
            self._memory.close()
            raise
        self._finalizer = weakref.finalize(self, _release_binding, self._session, self._memory,
                                         self.folder, self.token, self.timeout)
        self.inputs = [Buffer(self, lambda i=i: self._memory.arrays['inputs'][i])
                       for i in range(len(self.description['inputs']))]
        self.outputs = [Buffer(self, lambda i=i: self._memory.arrays['outputs'][i], writable=False)
                        for i in range(len(self.description['outputs']))]
        self.last_timing = {}

    def _check_open(self):
        if self.closed or self._session.closed:
            raise RuntimeError('Prepared inference is closed')

    def transaction(self):
        return self.lock

    def invoke(self):
        with self.lock:
            self._check_open()
            start = time.perf_counter()
            _, metadata = self._session.request(self.folder, action='invoke_shared', token=self.token,
                                                timeout=self.timeout)
            self.last_timing = dict(invoke_ms=metadata['invoke_ms'],
                                    roundtrip_ms=(time.perf_counter() - start) * 1000)
            return dict(self.last_timing)

    def run(self, inputs, *, out=None):
        with self.lock:
            self._check_open()
            if len(inputs) != len(self.inputs):
                raise ValueError('Input count mismatch')
            if out is not None:
                DirectSession._validate_outputs(out, self.description['outputs'])
            for value, buffer in zip(inputs, self.inputs):
                validate_array(value, buffer.shape, buffer.dtype)
            for value, buffer in zip(inputs, self.inputs):
                buffer.copy_from(value)
            self.invoke()
            if out is None:
                return [buffer.numpy() for buffer in self.outputs]
            for dest, buffer in zip(out, self.outputs):
                buffer.read_into(dest)
            return out

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                self._finalizer()
            finally:
                self._session = None

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()

class CoralModule(torch.nn.Module):
    """Inference-only Module. Persistent legacy worker executes on the physical TPU.

    Float32 tensors in PyTorch layout in/out. CUDA tensors are copied through host
    memory; outputs return to the first input device. Autograd is never simulated.
    """
    def __init__(self, artifact_dir, *, worker_python=None, device=None, timeout=60,
                 execution='worker', transport='pipe', delegate_options=None, num_threads=1):
        super().__init__()
        self.training = False
        if execution not in ('worker', 'direct') or transport not in ('pipe', 'shared_memory'):
            raise ValueError('execution must be worker/direct; transport must be pipe/shared_memory')
        if execution == 'direct' and transport != 'pipe':
            raise ValueError('Direct execution already bypasses IPC; use its interpreter buffers')
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
            raise ValueError('num_threads must be a positive integer')
        self.execution, self.transport = execution, transport
        self.delegate_options = normalize_options(delegate_options)
        self.num_threads = num_threads
        self.artifact_dir = Path(artifact_dir).resolve()
        self.manifest = json.loads((self.artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != 1:
            raise ValueError("Unsupported Coral artifact version")
        model = self.artifact_dir / self.manifest["model_file"]
        if hashlib.sha256(model.read_bytes()).hexdigest() != self.manifest["sha256"]:
            raise ValueError("Compiled model checksum mismatch")
        self.worker_python = python_executable('worker', worker_python, required=False)
        self.device = device or "usb:0"
        self.timeout = timeout
        self._session = None
        self._direct = None
        self._prepared = {}
        self._bindings = weakref.WeakSet()
        self._lock = threading.RLock()
        self.last_timing = {}
        self.hardware = {}

    def train(self, mode=True):
        if mode:
            raise RuntimeError("Coral is inference-only. Train the source PyTorch model on CPU/CUDA, then recompile.")
        return super().train(False)

    def _start(self):
        if self.execution == 'direct':
            if self._direct is None or self._direct.closed:
                self._direct = DirectSession(self.artifact_dir, device=self.device,
                    delegate_options=self.delegate_options, num_threads=self.num_threads)
                self.hardware = self._direct.hardware
            return
        if self._session is not None and not self._session.closed: return
        if self.worker_python is None:
            self.worker_python = python_executable('worker')
        key = (self.worker_python, self.device)
        with _SESSIONS_LOCK:
            session = _SESSIONS.get(key)
            if session is None or session.closed:
                session = _Session(self.worker_python, self.device, self.timeout, self.artifact_dir, self.delegate_options)
                _SESSIONS[key] = session
            elif session.options != self.delegate_options:
                raise ValueError('Device already open with different delegate options; close its modules first')
            self._session = session
            self.hardware = dict(session.hardware, num_threads=self.num_threads,
                cpu_operations=self.manifest.get('compiler_report', {}).get('cpu_operations'))

    @property
    def direct_session(self):
        if self.execution != 'direct':
            raise RuntimeError('Interpreter access requires execution="direct" in a compatible SDK environment')
        with self._lock:
            self._start()
            return self._direct

    def prepare(self, *, raw=True):
        """Allocate persistent shared worker buffers, independent of transport default."""
        if self.execution != 'worker':
            raise RuntimeError('Use direct_session.input_buffer/output_buffer in direct mode')
        with self._lock:
            self._start()
            binding = PreparedInference(self, raw)
            self._bindings.add(binding)
            return binding

    def _infer(self, arrays, raw, out=None):
        self._start()
        start = time.perf_counter()
        if self.execution == 'direct':
            run = self._direct.run_raw if raw else self._direct.run_float
            outputs = run(arrays, out=out)
            metadata = self._direct.last_timing
        elif self.transport == 'shared_memory':
            binding = self._prepared.get(raw)
            if binding is None or binding.closed or binding._session.closed:
                binding = self.prepare(raw=raw)
                self._prepared[raw] = binding
            outputs = binding.run(arrays, out=out)
            metadata = binding.last_timing
        else:
            if out is not None:
                _, desc = self._session.request(self.artifact_dir, action='describe', raw=raw,
                    num_threads=self.num_threads, timeout=self.timeout)
                DirectSession._validate_outputs(out, desc['description']['outputs'])
            outputs, metadata = self._session.request(self.artifact_dir, arrays, raw=raw,
                num_threads=self.num_threads, timeout=self.timeout)
            if out is not None:
                for dest, value in zip(out, outputs):
                    np.copyto(dest, value, casting='no')
                outputs = out
        self.last_timing = {'invoke_ms': metadata['invoke_ms'],
                            'roundtrip_ms': (time.perf_counter() - start) * 1000}
        return outputs

    def run_raw(self, *inputs, out=None):
        """NumPy I/O in native TFLite dtype/layout; skip float quantization and Torch."""
        with self._lock:
            return self._infer(inputs, raw=True, out=out)

    def forward(self, *inputs):
        if len(inputs) != len(self.manifest["inputs"]):
            raise ValueError("Expected %d positional tensor inputs" % len(self.manifest["inputs"]))
        arrays = []
        for value, spec in zip(inputs, self.manifest["inputs"]):
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
                raise TypeError("Coral inputs must be float32 torch tensors")
            if list(value.shape) != spec["shape"]:
                raise ValueError("Static input shape %s required for %s" % (spec["shape"], spec["name"]))
            if value.requires_grad and torch.is_grad_enabled():
                raise RuntimeError("Edge TPU cannot backpropagate. Use torch.no_grad() for inference.")
            array = value.detach().cpu().contiguous().numpy()
            if not np.isfinite(array).all(): raise ValueError("Input contains NaN or infinity")
            for constraint in self.manifest.get('operator_rewrites', {}).get('input_constraints', []):
                if constraint['name'] == spec['name']:
                    if constraint.get('integer_values') and np.any(array != np.rint(array)):
                        raise ValueError('This Coral strategy requires integer-valued IDs for ' + spec['name'])
                    if np.any(array < constraint['min']) or np.any(array > constraint['max']):
                        raise ValueError('Index outside the bounded Coral table for ' + spec['name'])
            arrays.append(array)
        with self._lock:
            outputs = self._infer(arrays, raw=False)
        tensors = [torch.from_numpy(array.copy()).to(inputs[0].device) for array in outputs]
        return unflatten(self.manifest["output_tree"], tensors)

    def close(self):
        with self._lock:
            error = None
            try:
                for binding in list(self._bindings):
                    try:
                        binding.close()
                    except Exception as exc:
                        error = exc
                self._prepared.clear()
                if self._direct is not None:
                    self._direct.close()
                    self._direct = None
            finally:
                # Last reference closes the shared session through its finalizer.
                self._session = None
            if error is not None:
                raise error

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

def load(artifact_dir, **kwargs):
    """Load a compiled artifact without running conversion again."""
    return CoralModule(artifact_dir, **kwargs)
