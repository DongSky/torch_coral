"""Thin, explicit interpreter ownership. No PyTorch or model conversion imports."""
from contextlib import contextmanager
import hashlib
import json
import re
from pathlib import Path
import threading
import time
import weakref
import numpy as np
from .buffers import Buffer, validate_array, quantize, dequantize

_STATES = weakref.WeakValueDictionary()
_LOCK = threading.RLock()


def normalize_options(options=None):
    if options is not None and not isinstance(options, dict):
        raise TypeError('delegate_options must be a dictionary')
    result = {}
    for key, value in (options or {}).items():
        if not isinstance(key, str) or not key or key == "device":
            raise ValueError("Use the device argument; option names must be nonempty strings")
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError("Delegate option values must be strings or scalar values")
        result[key] = str(value)
    if "Performance" in result and result["Performance"] not in ("Low", "Medium", "High", "Max"):
        raise ValueError("Performance must be Low, Medium, High or Max")
    if "Usb.MaxBulkInQueueLength" in result:
        try:
            size = int(result["Usb.MaxBulkInQueueLength"])
        except ValueError:
            raise ValueError("Usb.MaxBulkInQueueLength must be an integer") from None
        if not 0 <= size <= 255:
            raise ValueError("Usb.MaxBulkInQueueLength must be between 0 and 255")
    if 'Usb.AlwaysDfu' in result and result['Usb.AlwaysDfu'] not in ('True', 'False'):
        raise ValueError('Usb.AlwaysDfu must be True or False')
    return result


class _DeviceState:
    def __init__(self, options):
        self.lock = threading.RLock()
        self.options = options
        self.sessions = weakref.WeakSet()


def _sdk():
    from torch_coral.native_runtime import configure
    backend = configure()
    try:
        from pycoral.utils.edgetpu import load_edgetpu_delegate, list_edge_tpus, get_runtime_version
        from tflite_runtime.interpreter import Interpreter
    except ImportError as exc:
        raise RuntimeError("Direct execution requires a compatible PyCoral/TFLite SDK in this interpreter. "
                           "On Windows use the Python 3.9 runtime environment, or execution='worker'.") from exc
    return Interpreter, load_edgetpu_delegate, {
        "devices": list_edge_tpus(), "runtime": get_runtime_version(), "usb_backend": backend}


class DirectSession:
    """Own one allocated interpreter. Accept an artifact directory or compiled TFLite.

    raw I/O uses the interpreter's dtype and layout. Float I/O requires a manifest.
    num_threads controls TFLite CPU threads, not TPU cores. Delegate options are
    passed unchanged after string normalization; acceptance is runtime dependent.
    """
    def __init__(self, model, *, device="usb:0", delegate_options=None, num_threads=1):
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
            raise ValueError("num_threads must be a positive integer")
        self.device = device
        self.options = normalize_options(delegate_options)
        if self.options and (not isinstance(device, str) or not re.fullmatch(r'(?:usb|pci)?:[0-9]+', device)):
            raise ValueError('Delegate options require an indexed device such as usb:0; upstream ignores them for usb/pci/any')
        self.closed = True
        self._interpreter = self._delegate = None
        self.manifest = None
        path = Path(model).resolve()
        if path.is_dir():
            self.manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            if self.manifest.get("format_version") != 1:
                raise ValueError("Unsupported Coral artifact version")
            model_path = (path / self.manifest["model_file"]).resolve()
            if not model_path.is_relative_to(path):
                raise ValueError("Model path escapes artifact directory")
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != self.manifest["sha256"]:
                raise ValueError("Compiled model checksum mismatch")
        else:
            model_path = path
        Interpreter, load_delegate, self.hardware = _sdk()
        if not self.hardware["devices"]:
            raise RuntimeError("No real Edge TPU found; CPU fallback is disabled")
        with _LOCK:
            state = _STATES.get(device)
            if state is None:
                state = _DeviceState(self.options)
                _STATES[device] = state
            if state.sessions and state.options != self.options:
                raise ValueError("Device already open with different delegate options; close its sessions first")
            state.options = self.options
            self._state, self.lock = state, state.lock
            with self.lock:
                try:
                    self._delegate = load_delegate(dict(self.options, device=device))
                    self._interpreter = Interpreter(model_path=str(model_path),
                        experimental_delegates=[self._delegate], num_threads=num_threads)
                    self._interpreter.allocate_tensors()
                    self.inputs = self._interpreter.get_input_details()
                    self.outputs = self._interpreter.get_output_details()
                    # A raw model may contain host operators. Report that explicitly.
                    self.cpu_operations = (self.manifest or {}).get("compiler_report", {}).get("cpu_operations")
                    self.hardware = dict(self.hardware, delegate_options=self.options, num_threads=num_threads,
                                         cpu_operations=self.cpu_operations)
                    self.closed = False
                    self._state.sessions.add(self)
                except Exception:
                    self._interpreter = self._delegate = None
                    raise
        self.last_timing = {}

    def _check_open(self):
        if self.closed:
            raise RuntimeError("DirectSession is closed")

    @property
    def interpreter(self):
        """Raw SDK escape hatch; use borrow_interpreter for serialized access."""
        self._check_open()
        return self._interpreter

    @contextmanager
    def borrow_interpreter(self):
        with self.lock:
            self._check_open()
            yield self._interpreter

    def input_buffer(self, index=0):
        return self._buffer(self.inputs[index], writable=True)

    def output_buffer(self, index=0):
        return self._buffer(self.outputs[index], writable=False)

    def _buffer(self, detail, writable):
        return Buffer(self, lambda: self._interpreter.tensor(detail["index"])(), writable=writable)

    def describe(self, raw=True):
        def details(items):
            return [dict(shape=d["shape"].tolist(), dtype=np.dtype(d["dtype"]).str,
                         quantization=list(d["quantization"]), name=d["name"]) for d in items]
        if raw:
            return {"inputs": details(self.inputs), "outputs": details(self.outputs)}
        if self.manifest is None:
            raise ValueError("Float I/O requires a torch-coral artifact manifest")
        return {kind: [dict(shape=s["shape"], dtype=np.dtype(np.float32).str, name=s["name"])
                       for s in self.manifest[kind]] for kind in ("inputs", "outputs")}

    def invoke(self):
        with self.lock:
            self._check_open()
            start = time.perf_counter()
            self._interpreter.invoke()
            self.last_timing = {"invoke_ms": (time.perf_counter() - start) * 1000}
        return dict(self.last_timing)

    def run_raw(self, inputs, *, out=None):
        with self.lock:
            self._check_open()
            if len(inputs) != len(self.inputs):
                raise ValueError("Input count mismatch")
            # Validate all inputs/outputs before modifying buffers or invoking.
            for array, d in zip(inputs, self.inputs):
                validate_array(array, d["shape"], d["dtype"])
            self._validate_outputs(out, self.outputs)
            for i, array in enumerate(inputs):
                self.input_buffer(i).copy_from(array)
            self.invoke()
            if out is None:
                return [self.output_buffer(i).numpy() for i in range(len(self.outputs))]
            for i, dest in enumerate(out):
                self.output_buffer(i).read_into(dest)
            return out

    @staticmethod
    def _validate_outputs(out, details):
        if out is not None:
            if len(out) != len(details):
                raise ValueError("Output count mismatch")
            for array, d in zip(out, details):
                validate_array(array, d["shape"], d["dtype"], writable=True)

    def run_float(self, inputs, *, out=None):
        with self.lock:
            self._check_open()
            desc = self.describe(raw=False)
            self._validate_outputs(out, desc["outputs"])
            if len(inputs) != len(self.manifest["inputs"]):
                raise ValueError("Input count mismatch")
            by_name = {d["name"]: d for d in self.inputs}
            for array, spec in zip(inputs, self.manifest["inputs"]):
                validate_array(array, spec["shape"], np.float32)
                if not np.isfinite(array).all():
                    raise ValueError("Input contains NaN or infinity")
                for constraint in self.manifest.get("operator_rewrites", {}).get("input_constraints", []):
                    if constraint["name"] == spec["name"]:
                        if constraint.get("integer_values") and np.any(array != np.rint(array)):
                            raise ValueError("This Coral strategy requires integer-valued IDs")
                        if np.any(array < constraint["min"]) or np.any(array > constraint["max"]):
                            raise ValueError("Index outside the bounded Coral table")
            for array, spec in zip(inputs, self.manifest["inputs"]):
                d = by_name[spec["tflite_name"]]
                self._interpreter.set_tensor(d["index"], quantize(array.transpose(spec["perm"]), d))
            self.invoke()
            output_details = {d["name"]: d for d in self.outputs}
            results = []
            for i, spec in enumerate(self.manifest["outputs"]):
                d = output_details[spec["tflite_name"]]
                value = dequantize(self._interpreter.get_tensor(d["index"]), d).transpose(spec["inverse_perm"])
                if out is None:
                    results.append(value)
                else:
                    np.copyto(out[i], value, casting="no")
            return results if out is None else out

    def close(self):
        if not hasattr(self, "lock"):
            return
        with self.lock:
            self.closed = True
            self._interpreter = None  # release interpreter before its delegate
            self._delegate = None
            self._state.sessions.discard(self)

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
