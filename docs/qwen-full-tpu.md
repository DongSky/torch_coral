# Experimental full-text-model Qwen3.5 on the original Edge TPU

This is an experimental executor, still slow for interactive LLM use. Alpha 2
adds [INT8 matrix acceleration and fused FP32 updates](matrix-acceleration.md)
to the original scalar reference. Alpha 3 adds [whole-chain fast-path optimization](chain-optimization.md)
with bitwise physical comparisons. It uses reverse-engineered instructions on the physical
Coral, independently of Google's TFLite delegate. Never use the two stacks on the
same device simultaneously. The existing 0.2 TFLite API remains separate.

## Actual source and quantization

The tested weight source is
[Unsloth Qwen3.5-0.8B-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.5-0.8B-MTP-GGUF/blob/cf8a611f6ed2c2060046219a19f12cd3d5ecd67c/Qwen3.5-0.8B-Q4_K_M.gguf),
549,698,976 bytes, SHA256
`ac7c9d7a1b3e3695bb3bd50f8ceaa97f9c93e99ccc3d3d1a620301b6dd6d3d86`.
Its inspected header contains 335 tensors: 113 Q4_K, 18 Q5_K, 27 Q6_K,
one Q8_0, and 176 F32. It is a mixed-precision quantized weight container,
not a uniform INT8 accelerator executable.

One-time `gguf_weights.prepare_weights()` decompresses weights into immutable
FP32 memory maps. This preserves the values represented by the quantized GGUF;
it does not restore the original unquantized model. The larger prepared storage
and host-to-device transfers remain a limitation. `MatrixTPU` additionally
prepares row-wise INT8 components and uses the tensor MAC array for projections;
the original `ScalarTPU` path remains available for FP32 reference checks.

## Execution boundary

All numerical model operations in `qwen_tpu.py` dispatch to the physical backend
(`MatrixTPU` by default in the GUI, or the original `ScalarTPU`):
embedding copy, projections, convolution, RMS/L2 normalization, RoPE arithmetic,
full causal attention, Gated DeltaNet state updates, SiLU/softplus/sigmoid,
residual additions, FFN, final normalization, LM head, and greedy selection.
Unsupported operations raise. There is no Torch/NumPy arithmetic fallback.

The host tokenizes, prepares constant weights/RoPE tables, allocates/stores
buffers, reshapes/copies data and schedules programs. KV and recurrent state are
stored in host memory between kernels; their arithmetic runs on the device.
This is full numerical offload, not full on-chip residency. A separate PyTorch
script is used only as an independent validation reference, never by the worker.

The main generator uses all 24 trunk layers. The GGUF additionally contains a
NextN/MTP layer. `mtp_draft()` executes its normalization, concatenation
projection, attention, FFN and shared output head on TPU. Earlier profiles offer
this as an optional diagnostic, disabled by default. Turbo V3/V4 instead use
fully verified speculative decoding, enabled by default, and accept a draft only
after checking it through the full trunk and output head. See
[Turbo V3](turbo-v3.md) and [Turbo V4 Alpha 9](turbo-v4-a9.md).
This distinction follows the separate
[main and MTP graphs in llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp).
The selected GGUF contains no vision projector; this GUI is text-only.

## Local setup and launch

Use Python 3.12 for both frontend and isolated ISA worker. The frontend needs
PySide6; the worker needs the pinned requirements in
`torch_coral/dependencies/isa-py312.txt`, tinycoral commit
`60794943cca3ddc63a4438cfff91b520b12bcfef`, and the Windows compatibility patch.
`scripts/prepare-qwen-environment.ps1` prepares that worker from source. It does
not install a USB driver, firmware or libusb DLL. Preserve upstream licenses.

Supply a compatible WinUSB libusb/runtime directory using the existing local
configuration. A locally built compatibility runtime is verified by its checksum
manifest. The ISA process selects that same libusb explicitly. The tinycoral
patch adds Windows locking and explicit device release; the reference uses
synchronous USB transfers and the standard 250 MHz clock setting.

The Coral must be in application mode (`18d1:9302`). A cold USB connection may
start in DFU (`1a6e:089a`). On this Windows setup, writing firmware followed by
libusb's pipe reset does not complete device re-enumeration; the existing
administrator-only `scripts/restart-coral-device.ps1` can restart exactly one
matching Coral after firmware upload. A denied/cancelled UAC does not perform
the restart. The chat executor reports DFU/unavailable instead of repeatedly
attempting firmware changes. Normal inference uses ordinary permissions.

Download the pinned GGUF and the original Qwen tokenizer files separately.
Required tokenizer files are `tokenizer.json` and `chat_template.jinja` from
Qwen/Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`.
The GUI's paths can be changed to local copies. Weights, tokenizers and runtime
binaries are excluded from the public source and distributions.

Run `scripts/start-qwen-chat.ps1`, or:

```powershell
.venv-torch/Scripts/python.exe -m torch_coral.qwen_chat_gui
```

Choose the GGUF, tokenizer folder and prepared-weight folder, then load. Text
streams after real TPU generation. Stop requests are read on a separate input
thread and observed between hardware programs; partial recurrent state is reset.
Closing the window waits for orderly worker/device release instead of killing
the USB owner. Conversation text is not saved by the application.

## Validation and performance limits

`scripts/verify-isa-scalar.py` tests real scalar arithmetic, reductions, tiled
matrix-vector products, batched projections, logarithm and argmax.
`tests/test_isa_scalar.py` separately checks emitted instructions with the offline
ISA interpreter. An offline test does not establish hardware support.

`scripts/run-qwen-tpu.py` runs the complete text graph and writes actual coverage
and timing. Add `--mtp` to exercise the extra draft head.
`scripts/validate-qwen-reference.py` loads identical decompressed GGUF weights
into the independent HF implementation for numerical comparison.

The first complete physical trunk test used one input token, 519,121 TPU program
calls and about 140 seconds. Its predicted token matched the HF reference;
logits relative L2 error was approximately 8.14e-6. This is one numerical case,
not a broad accuracy evaluation. A separate physical dialogue test processed
13 prompt tokens and generated two tokens, replying `你好！` to `你好`; this
matched an independent HF greedy reference using the same GGUF weights. It took
708.44 seconds and 1,546,214 TPU program calls. This dialogue test covered the
trunk; MTP was checked separately. The MTP single-position graph, driven by an
HF trunk hidden-state fixture, matched the independent Torch reference's top
token with relative L2 error approximately 4.45e-6. These are narrow correctness
checks, not a broad quality benchmark. Alpha 2 now implements INT8 matrix-array
projections, residual readouts, parameter residency and fused FP32 state updates.
The [matrix report](matrix-acceleration.md) compares speed and quantization error;
it does not claim advertised peak throughput or broad language-quality parity.
