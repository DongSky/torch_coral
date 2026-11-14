# Turbo V2

Turbo V2 is a separate `turbo-v2` profile in Coral Studio and the command-line
runner. The original `turbo` profile remains available. Both use the same
one-component INT8 projections and two output readouts. Turbo V2 preserves the
FP32 operation order and performs all numerical model work on the physical
Edge TPU. Tokenization, copies, packing and scheduling stay on the host.

## Implemented changes

- Keep recurrent state columns in scalar SRAM across several prompt positions.
  An SRAM-aware planner minimizes launch count; 13 positions use 7+6 temporal
  chunks. State updates remain sequential in time.
- Use one input DMA per program and update the state in place. An earlier
  multi-input experiment stalled physically and is not part of Turbo V2.
- Loop over columns on the device instead of retransmitting the same instruction
  body. Pipeline four rows and combine independent scalar ALU and store slots.
- Batch normalization rows with shared weights. Fuse softmax and split-half
  RoPE while retaining the original reductions and Newton iterations.
- Correct INT32 biases in place in tiles of up to 3072 rows; reconstruct FP32
  outputs in place in tiles of up to 2560 rows for the two-readout profile.
- Omit unused activation residuals inside one-component projections. The public
  quantization API and compensated profiles still return/use residuals.

## Transfer limits and tile sizes

The scalar SRAM contains 16384 bytes. Physical round trips of 1024, 2048, 4096,
8192, 12288, 16000 and 16384 bytes per direction each passed 100 checked timing
repetitions after warmup. A 16384-byte round trip had median latency 0.498 ms
in that run. This is end-to-end scalar-copy latency, not raw USB bandwidth.

16384 bytes is the limit of this whole-tensor scalar-SRAM path, not a general
limit on the USB protocol, matrix weights, or model size. The MXU parameter
path already transfers larger buffers. Input, output and scratch share SRAM;
arbitrary separate input/output tensors cannot each consume all 16384 bytes.
Turbo V2 kernels retain a conservative 16000-byte DMA ceiling and explicitly
check the actual live-memory layout. The original scalar wrapper keeps its
existing 8192-byte-per-direction bounds.

Larger tiles are not monotonically faster. They trade fewer launches against
working space, temporal reuse, padding and instruction bytes. The measured
decode threshold reached a plateau at about 26--29 state columns. Prefill uses
a different time/column tradeoff. Tune the final instruction kernel on the
target device rather than maximizing the payload unconditionally.

The final 500 MHz scan used the same 16 heads and checked state/output bits on
every run, with two warmups and five timing repeats for each of 12 column caps.
Decode at caps 26/28/29 took 45.54/44.99/44.93 ms respectively. For 13 positions,
caps 16 and above selected the same 7+6 schedule, with 256 launches and about
218--222 ms. Higher caps did not increase the chosen temporal tile sizes.

## Clock selection

The prior backend fixed the device at 250 MHz (`high`). Turbo V2 in the GUI and
CLI defaults to the supported 500 MHz (`max`) setting. Select 250 MHz in
Settings -> Model and device -> Turbo V2 frequency, or pass `--clock high` to
the CLI. Other GUI modes retain 250 MHz. Direct Python backend construction
defaults to 250 MHz unless `clock='max'` is supplied.

Maximum frequency increases power and enclosure temperature; maintain cooling
and avoid touching the hot metal enclosure. This follows the
[official Coral frequency guidance](https://coral.ai/docs/accelerator/get-started/#install-with-maximum-operating-frequency-optional).
The current USB backend does not report a measured chip temperature; passing
these inference tests does not certify a thermal limit.

## Physical graph comparison

Same Windows host, USB 3 SuperSpeed, Qwen3.5-0.8B Q4_K_M prepared as the existing
INT8 cache, 13-token greeting, then three decode steps. Each profile uses a new
backend and a cold host weight cache for its first prompt. Program-generator
caches are process-wide; the separate GUI test also measures a fresh worker.
Prefill excludes model loading and the final greedy selection. Decode timings
include greedy selection; the first output token belongs to prefill/TTFT.

| Mode | Clock | Prefill | Median decode | Decode speed |
|---|---:|---:|---:|---:|
| Turbo | 250 MHz | 37.30 s | 4.779 s/token | 0.209 token/s |
| Turbo V2 | 250 MHz | 20.59 s | 4.046 s/token | 0.247 token/s |
| Turbo V2 | 500 MHz | 16.46 s | 3.654 s/token | 0.274 token/s |

At the same frequency, software changes reduced prefill time by 44.8% and
decode time by 15.3%. Including the maximum frequency setting, reductions were
55.9% and 23.5% (2.27x prefill and 1.31x decode throughput). These are measured
short-context cases, not a universal performance guarantee or hardware peak.
The previous estimates were 19--26 seconds prefill and 3.2--3.8 seconds/token
decode. Prefill exceeded that estimate; decode reached its upper part.

Three prompt lengths (1, 13, 17) each produced four complete logit vectors,
including the prefill output, and all final attention/recurrent states.
Both V2 clock settings matched Turbo bit for bit in every comparison. Generated
greedy IDs also matched. These tests did not execute the optional MTP draft.

## Reproduce

Close other Coral owners before physical tests. From the workspace root:

```powershell
.venv-isa/Scripts/python.exe scripts/probe-scalar-sram.py
.venv-isa/Scripts/python.exe scripts/tune-recurrent-tiles.py
.venv-isa/Scripts/python.exe scripts/verify-streamed-recurrent.py --full
.venv-isa/Scripts/python.exe scripts/verify-v2-fusions.py
.venv-isa/Scripts/python.exe scripts/benchmark-turbo-v2.py --extended --include-max
.venv-isa/Scripts/python.exe scripts/benchmark-qwen-chat.py --profile turbo-v2
.venv-isa/Scripts/python.exe scripts/run-qwen-tpu.py --matrix-profile turbo-v2 --clock high
```

The graph comparison checks every FP32 logit and all final attention/recurrent
caches, including a 17-token prompt crossing the 16-token prefill boundary.
Precision equivalence is relative to Turbo; it does not make Turbo's existing
INT8 approximation equivalent to FP32 weights or establish language quality.

A separate fresh GUI/worker run on the greeting returned the same `你好！`
tokens. Live telemetry measured 16.941 s prefill, 16.972 s TTFT, 3.698 s decode
and 20.671 s overall for two visible output tokens (46090 TPU calls). The
displayed decode rate was 0.270 token/s; overall output rate was 0.097 token/s.
No CPU/CUDA numerical model fallback was used. Both phase progress displays and
the completed-decode chart sample were checked through the real GUI event path.
