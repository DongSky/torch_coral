# torch-coral

PyTorch integration and experimental language-model inference for the
**first-generation Coral USB Edge TPU**. Current version: **0.3.0a9 (alpha)**.

| Component | Execution path | Documentation |
|---|---|---|
| PyTorch compile/load API | Static INT8 TFLite graphs through Google's delegate | [Execution strategies](docs/execution-strategies.md) |
| Direct runtime | Interpreter, delegate and reusable native buffers | [Runtime controls](docs/runtime-v2.md) |
| Qwen3.5-0.8B text executor | Experimental INT8 matrix array and FP32 scalar instructions | [Model and setup](docs/qwen-full-tpu.md) |
| Coral Studio | Local chat, settings, live token/s and phase visualizations | [Chat interface](docs/chat-studio.md) |

[Turbo V4 Alpha 9](docs/turbo-v4-a9.md) combines prepared asynchronous USB
dispatch, shared quantization, fused scalar kernels and verified MTP decoding.
Earlier [V2](docs/turbo-v2.md) and [V3](docs/turbo-v3.md) profiles remain available
for comparison. All numerical operations in the Qwen executor run on the device;
the host tokenizes, stores weights/state and schedules transfers. This is not
full on-chip residency. GGUF weights are prepared offline; the chip does not
directly execute GGUF/Q4_K_M files.

Interactive generation remains slow. A full V4 request with a 512-token context
limit and 256-token output limit generated 59 visible tokens at **0.335 decode
token/s** (173.2 s for 58 tokens after the first). Short or repetitive benchmarks
can be faster and should not be treated as a general chat rate. Physical A/B
checks compare full logits and cached states bit for bit within the configured
quantization profile; they do not establish broad model-quality parity.

This is an independent community project. It does not add hardware instructions,
provide a training backend, or guarantee arbitrary Hugging Face model support.
The release is alpha. The frontend is tested on Python 3.12; the legacy
worker uses a separate compatible environment, tested with Python 3.9 on Windows.

## Install

The distribution name is `torch-coral`; the import is `torch_coral`.
PyPI publication is separate from this source repository. From a checkout,
using an activated Python 3.12 environment:

```sh
python -m pip install -e .
# Optional native desktop GUI / Hugging Face integration:
python -m pip install -e '.[gui]'
# Standalone text-chat frontend:
python -m pip install -e '.[chat]'
```

Install PyTorch's appropriate CPU/CUDA build before this package if you need a
specific GPU configuration. The package does not install USB drivers, Google
runtime binaries, the closed Edge TPU compiler, or model weights.

Three roles are intentionally separated:

| Role | Environment | Responsibility |
|---|---|---|
| Frontend | Python 3.12, PyTorch | Export, Tensor API, optional GUI |
| Converter | Python 3.12, pinned onnx2tf 2.6.9 | Float parity, INT8 calibration, compilation |
| Worker | Python 3.9 with compatible original PyCoral / TFLite | Own the device and execute compiled graphs |

Version 0.2 also includes `torch_coral_runtime.DirectSession` for direct
interpreter and buffer access inside the compatible SDK environment. Installing
on Python 3.9 through 3.11 installs this runtime tier without the modern
PyTorch/export dependencies. The PyTorch frontend requires Python 3.12+.

Create the two auxiliary environments yourself. Packaged requirement-file paths
are printed by `torch-coral dependencies convert` and
`torch-coral dependencies worker-windows-py39`. Install those requirements with
the corresponding environment's pip. The worker file targets Windows x86-64;
on Linux, install a compatible official PyCoral/TFLite stack and libedgetpu for
the target architecture instead. Linux hardware execution is not validated by
the initial Windows test run.

Configure paths after installing:

```sh
torch-coral configure --worker-python /path/to/worker/python --converter-python /path/to/converter/python
torch-coral configure --compiler /path/to/edgetpu_compiler --compiler16 /path/to/compiler16
torch-coral doctor
torch-coral doctor --probe
```

On Windows, the compilers execute through WSL (default distribution: `Ubuntu`).
Use Windows paths to executable compiler wrappers, or absolute Linux paths for
compilers installed inside WSL. `--wsl-distro` changes the distribution. The
optional repository download helpers preserve upstream notices. v16 is obtained
from a pinned third-party mirror and verified by SHA256; it is not bundled.

`TORCH_CORAL_HOME` overrides the data/config directory. Installed packages use
the user's application-data directory; a source checkout can discover its own
development environments. Explicit API arguments override environment variables,
which override saved configuration. Supported environment variables:
`CORAL_WORKER_PYTHON`, `CORAL_CONVERTER_PYTHON`, `CORAL_COMPILER`,
`CORAL_COMPILER16`, `CORAL_RUNTIME_DIR`, and `CORAL_WSL_DISTRO`.

For an explicitly supplied Windows runtime directory, both `edgetpu.dll` and
`libusb-1.0.dll` must be available. The optional source-repository WinUSB build
generates a checksum manifest. Only that verified build is identified as the
WinUSB compatibility backend; arbitrary supplied DLLs are not labelled WinUSB.

## PyTorch API

```python
import torch
import torch_coral

model = torch.nn.Sequential(
    torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2)
).eval()
example = torch.rand(1, 4)
# Illustrative only: replace with real, correctly preprocessed representative data.
calibration = [torch.rand(1, 4) for _ in range(64)]

with torch_coral.compile(model, example, calibration,
                         artifact_dir="artifacts/classifier") as coral:
    with torch.no_grad():
        result = coral(example)
    print(coral.last_timing)
    print(coral.manifest["compiler_report"])

# Reuse the compiled artifact without conversion:
with torch_coral.load("artifacts/classifier", transport="shared_memory") as coral:
    with torch.no_grad():
        result = coral(example)
```

`load()` accepts `delegate_options`, `num_threads`, `execution` and `transport`.
Use `model.prepare(raw=True)` for persistent native buffers and preallocated
outputs, or `DirectSession` to bypass the worker entirely. Direct mode requires
the matching SDK in the calling interpreter. See [runtime controls](docs/runtime-v2.md)
and the [custom kernel feasibility assessment](docs/custom-kernels.md).

Inputs and floating outputs are float32 tensors with static shapes and batch 1.
Tuple/list/string-keyed dictionary outputs are supported. CUDA inputs are copied
through the host; outputs return to the first input's device. Autograd and
training are rejected. Weight changes require recompilation. `make_backend`
provides an opt-in `torch.compile` backend for the same export/compile pipeline.

By default, any operations left on the CPU cause compilation to fail.
`allow_partial=True` is an explicit alternative and is reported in the manifest.
Zero CPU operators refers to the TFLite model graph: preprocessing, layout
conversion, quantization, IPC and orchestration still execute on the host.

## Extended operator strategy

Set `operator_strategy="coral"` to select v16 and enable static graph rewrites:

- Dynamic INT8 BMM and native RSQRT quantization fixes.
- RMSNorm, LayerNorm, GroupNorm, fixed Cumsum and constant-index selection.
- Value-only TopK via Min/Max sorting networks; power-of-two last-axis length <=32.
- Integer embedding tables with 2..32 rows via on-device indicator generation.
- Static nearest-neighbor spatial Resize via selector matrix multiplication.

Nonlinear approximations require an explicit bounded configuration:

```python
options = {"Sin": {"range": [-3.141593, 3.141593], "segments": 32}}
# Pass operator_strategy="coral", approximations=options to compile().
```

Sin, Cos, Exp, Softplus and Erf/GELU can use normalized soft lookup. Inputs outside
the declared function domain saturate at its endpoints. Default combined rewrite
and INT8 tolerances are `atol=0.15, rtol=0.15`; tighten them for the application.
Artifacts are rejected if the checked representative inputs exceed that budget.
This check is not a substitute for independent task-accuracy evaluation.

The implementation still rejects arbitrary dynamic control flow, general
Scatter, TopK index outputs and large dynamic embedding tables. Static expansion
can increase memory and arithmetic dramatically. No speedup is promised.

Fifteen small strategies, including a small Transformer block, were validated on
a real USB Edge TPU. This does not establish full LLM or diffusion deployment.
See `docs/execution-strategies.md` for the relationship to official examples.

## GUI and development

See [development setup](docs/development.md) for isolated environments, offline
tests and the public-source export procedure.

Install the `gui` extra, then run `coral-workbench`. Jobs run outside the UI
process. The interface accepts supported Hugging Face sources and local inputs;
selecting a source does not guarantee it can compile for Coral. Remote custom
model code is not executed. Bounded approximations are explicitly enabled.

The host shares a persistent worker per interpreter/device and serializes calls;
the worker caches at most 16 model interpreters. Independent applications should
not claim the same Windows USB device simultaneously. Windows cold initialization
can require a real device reset; see `native/README.md` in the source repository.
Normal inference does not require administrator privileges.

```sh
python -m pip install -e '.[test,release]'
python -m pytest tests
python -m build
python -m twine check dist/*
```

No local reports, device identifiers, calibrations, credentials, environments or
native binaries belong in a release. Release preparation uses an allowlist and
scans both source and distributions. See `docs/releasing.md`.

Python package: MIT. Optional libusb source patch: LGPL-2.1-or-later. See
`THIRD_PARTY_NOTICES.md`. No telemetry or automatic publishing is implemented.
