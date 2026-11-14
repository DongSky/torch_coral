# Runtime controls in 0.2

The default `torch_coral.load()` still uses a persistent worker and synchronous
float32 Tensor I/O. Existing artifacts and operator strategies remain compatible.
This release adds explicit control of the host execution path.

| Interface | Environment | Per-call path |
|---|---|---|
| `load(..., transport="pipe")` | PyTorch frontend | Tensor, binary ndarray packet, worker, interpreter |
| `load(..., transport="shared_memory")` | PyTorch frontend | Tensor, persistent shared buffers, control packet, interpreter |
| `model.prepare(raw=True)` | PyTorch frontend | Persistent native INT8 buffers, control packet, interpreter |
| `DirectSession` | Compatible PyCoral/TFLite SDK | In-process interpreter, no worker or IPC |

## Direct interpreter and buffer access

Install the same wheel into the SDK environment. On Python 3.9 through 3.11,
package dependency markers install NumPy without the modern PyTorch/export stack.
The frontend requires Python 3.12+. PyCoral, TFLite and libedgetpu remain separate
dependencies; use compatible upstream builds for the OS and architecture.

```python
from torch_coral_runtime import DirectSession

with DirectSession("artifacts/classifier", device="usb:0",
                   delegate_options={"Performance": "High"}, num_threads=1) as session:
    # q_input: correctly preprocessed NumPy array in native dtype/layout.
    session.input_buffer(0).copy_from(q_input)
    session.invoke()
    session.output_buffer(0).read_into(preallocated_output)
    with session.borrow_interpreter() as interpreter:
        print(interpreter.get_input_details())
```

`run_raw([q_input], out=[output])` validates native dtype/shape and reuses output
storage. `run_float([float_input], out=[float_output])` uses artifact layout and
quantization metadata. Float input constraints, including bounded integer
embedding IDs, remain enforced. Raw input callers own preprocessing and domain
validation. A bare compiled TFLite file supports raw I/O; without a manifest the
compiler's CPU-operation count is unknown and float I/O is unavailable.

Buffer handles lock the session and never export borrowed NumPy views. Reads
return copies or fill caller storage. Handles become invalid after close.
`interpreter` is an explicit raw SDK escape hatch: retained `tensor()()` views
can prevent TFLite invocation, and interpreter/delegate references retained by
the caller can delay device release. Release them before closing the session.
Resizing/reallocating tensors through that escape hatch invalidates cached
descriptions; create a new session afterward.

## Persistent worker buffers

```python
import torch_coral

with torch_coral.load("artifacts/classifier", transport="shared_memory") as model:
    with model.prepare(raw=True) as prepared:
        prepared.run([q_input], out=[preallocated_output])
        print(prepared.last_timing)
```

`prepare(raw=False)` instead uses float32 arrays in the frontend's layout. The
normal Tensor forward path caches a float binding with `shared_memory` transport.
Bindings attach once; subsequent invocation packets carry identifiers, not tensor
payloads. `prepared.run()` serializes copy, invoke and output read. For manual
buffer operations across threads, group them under `prepared.transaction()`.
Closing the module closes its bindings; stale handles raise. Worker failure
invalidates bindings; create new ones after reconnecting.

This removes ndarray serialization across host processes. It does not provide
USB DMA zero-copy, CUDA-device zero-copy, asynchronous TPU execution or resident
intermediate tensors spanning separately compiled models. The direct raw path
still copies into/out of TFLite's arena. Float mode still performs layout and
quantization work. TC02 pipe packets replace NPZ; the decoder also reads legacy
0.1 NPZ packets, but old and new live worker protocols are not interchangeable.

## Runtime options and ownership

Both `load()` and `DirectSession` accept `delegate_options` and `num_threads`.
The latter controls TFLite CPU threads, not TPU parallelism. Dictionary values
are converted to strings. Known checks cover `Performance` (`Low`, `Medium`,
`High`, `Max`), `Usb.AlwaysDfu` (`True`, `False`), and
`Usb.MaxBulkInQueueLength` (0..255). Other options are passed to the supplied
runtime, which may reject or ignore them. Reported options are requested values,
not measured effective clock settings.

Use an indexed selector such as `usb:0`. The upstream plugin only forwards
non-device options on indexed selectors; the direct API rejects an unindexed
selector with options. A device already open in this process cannot change
options: close all its sessions/bindings first. One worker per frontend
interpreter/device serializes model invocations; direct sessions sharing the
same selector share a lock. Use the same selector spelling consistently, and do
not claim one physical device from independent processes. Raw interpreter calls
must use `borrow_interpreter()` if concurrent callers exist.

The GUI exposes shared/pipe transport, CPU threads and a delegate-options JSON
field for Coral jobs. Windows' modern frontend still needs its Python 3.9 worker;
setting `execution="direct"` in an environment without the matching SDK raises a
clear error rather than changing execution backend.

Sources: [PyCoral interpreter/delegate factory](https://github.com/google-coral/pycoral/blob/master/pycoral/utils/edgetpu.py),
[Edge TPU plugin option forwarding](https://github.com/google-coral/libedgetpu/blob/master/tflite/edgetpu_delegate_for_custom_op_tflite_plugin.cc),
[TFLite tensor-view ownership](https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/lite/python/interpreter.py).
