# Custom kernel feasibility for the original USB Edge TPU

Assessment date: 2026-11-11. This concerns the first-generation USB Accelerator,
not the newer, distinct Coral NPU designs.

The earlier statement that public tools make custom Edge TPU code generation
impossible was too broad. Google's legacy compiler offers no documented
CUDA/Triton-style user kernel interface. However, a public reverse-engineering
effort now supplies a concrete experimental route.

## Evidence and its limits

[tinycoral](https://github.com/FrederikRogalski/tinycoral), inspected at commit
`60794943cca3ddc63a4438cfff91b520b12bcfef`, publishes instruction encoders,
code generators and a separate USB runtime under an
[MIT license](https://github.com/FrederikRogalski/tinycoral/blob/60794943cca3ddc63a4438cfff91b520b12bcfef/LICENSE).
Its authors report compiler-byte comparisons and physical-device tests. Their
normal tinygrad backend maps selected quantized kernels and leaves other work
on the host; a separate whole-transformer program has broader fusion. Reported
measurements use macOS and one USB device. These are upstream claims, not
torch-coral hardware validation or Windows support guarantees.

The inspected [FC generator](https://github.com/FrederikRogalski/tinycoral/blob/60794943cca3ddc63a4438cfff91b520b12bcfef/coral/codegen/fc.py)
contains explicit tile geometry, weight placement and instruction emission.
The [architecture notes](https://github.com/FrederikRogalski/tinycoral/blob/60794943cca3ddc63a4438cfff91b520b12bcfef/docs/ARCHITECTURE.md)
describe scalar control, compute tiles, memory, transfers and synchronization,
including experimentally observed alignment and USB ordering restrictions.
The [USB implementation](https://github.com/FrederikRogalski/tinycoral/blob/60794943cca3ddc63a4438cfff91b520b12bcfef/coral/usbdev.py)
drives libusb directly. This is a replacement execution stack, not a small
extension to PyCoral's operator registration.

## What can change

| Layer | Feasibility | Present torch-coral status |
|---|---|---|
| Graph decomposition, bounded approximations, static fusion | Available through the official compiler | Implemented operator strategies |
| Interpreter lifetime, arena copies, parameters, IPC | Available through existing runtime APIs | Implemented in 0.2 |
| Instruction selection, scheduling, tile placement, resident model fusion | Experimental reverse-ISA compiler/runtime route | Research only; no production backend included |
| New physical instructions, more SRAM, GPU-like universal FP16/BF16 execution | Cannot be added by a Python wrapper or compiler flag | Hardware constraints |

Our inference: a restricted kernel DSL and code generator are technically
plausible. A CUDA-equivalent promise of arbitrary kernels is unjustified.
Every operation still has to lower to existing compute, scalar, transfer and
synchronization primitives within available memory. Dynamic indexing, scatter,
precision-sensitive functions and large transformer caches need individual
proofs. A TFLite custom CPU op does not become a TPU instruction by registration.
The official [delegate implementation](https://github.com/google-coral/libedgetpu/blob/master/tflite/edgetpu_delegate_for_custom_op.cc)
binds compiled Edge TPU custom-op payloads; it does not compile arbitrary host code.

## Concrete implementation sequence for a future experimental backend

1. Pin the external source and dependencies; reproduce offline ISA encode/decode
   and compiler-oracle comparisons. Preserve license notices for any reused code.
2. Implement a separate, explicit `experimental_isa` backend with its own artifact
   version and provenance, distinct from official-compiler artifacts. Start with
   fixed-shape INT8 matmul/conv plus exact quantization, a CPU reference and strict
   shape, address-range and resource validation. Reject unsupported lowering.
3. Validate Windows libusb device discovery, firmware initialization, transfer
   ordering and deterministic release on this machine. A canary program must pass
   after initialization and recovery; timeouts must stop dispatch and quarantine
   that session. Never load it concurrently with libedgetpu on the same device.
4. Compare randomized and adversarial kernel results against reference arithmetic
   and official programs, including saturation, rounding ties, uneven shapes and
   memory boundaries. Byte equality alone is insufficient for hardware acceptance.
5. Add tile-memory allocation, fusion and persistent weights, then prototype one
   attention/norm block. Measure full-model accuracy and end-to-end latency with
   USB traffic included. Only then consider a user kernel DSL or PyTorch lowering.

The 0.2 release completes the existing-runtime integration. It does not ship
the external driver, execute downloaded reverse-engineering code, advertise
unverified custom kernels, or assert arbitrary Hugging Face LLM/diffusion support.
This keeps experimental ISA work a separately testable backend while current
applications gain direct control immediately.
