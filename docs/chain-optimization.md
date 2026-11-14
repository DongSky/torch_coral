# Full-chain optimization, 0.3.0a3

This page records the original Turbo mode. See [Turbo V2](turbo-v2.md) for the
new temporal/in-place kernels, transfer-boundary probe and clock comparison.

The `turbo` profile accelerates the previous `fast` profile without changing its
weights, activation quantization, two-digit MXU readout, model layers, or FP32
rounding order. Numerical model work remains on the physical Edge TPU. The host
stores immutable weights, packs/copies buffers, tokenizes and schedules work.
No CPU/CUDA model-arithmetic fallback was added.

## Measurements

Tests used the same Windows host, pinned ISA dependencies, direct USB 3
SuperSpeed connection and 250 MHz setting as alpha 2. Prepared GGUF/INT8 files
already existed. A first execution below means the first graph execution in a
new process with an empty packed-weight RAM cache; it is not model download,
initial conversion, USB firmware setup, or a cold operating-system disk cache.

| Same one-token graph, ID 9707 | Original fast | Turbo |
|---|---:|---:|
| First graph execution | 9.33 s | 6.63 s |
| Repeated execution with reset model state | 8.69 s | 4.79–4.80 s |
| Hardware program invocations per graph | 15,596 | 8,368 |
| Weight upload bytes | 753,895,424 | 753,895,424 |
| MXU execution calls | 384 | 384 |

The measured gains are approximately 1.4x for first execution and 1.8x for
repeated execution; hardware invocations fell 46.3%. Additional instrumentation
repeated the turbo result at 6.68 s first and 4.82–4.86 s repeated. The earliest
alpha-2 fast result was 9.11 s; comparisons above use a new same-session baseline.
Timing includes real device execution and host orchestration, not just a matrix
kernel. Graph reset does not clear immutable packed-weight storage.

All following cases ran two autoregressive selections. Each corresponding
full-vocabulary logits array was compared as raw uint32 bits. All eight arrays
and all selected token IDs matched the original fast path exactly, including
the previous release's saved ID-9707 logits. The short outputs below reflect
the two-token test limit, not a claim that complete answers were evaluated.

| Input case | Prompt tokens | Original fast | Turbo | Speedup |
|---|---:|---:|---:|---:|
| Raw token ID 9707 | 1 | 17.14 s | 11.52 s | 1.49x |
| `你好` with chat template | 13 | 73.13 s | 40.73 s | 1.80x |
| `1+1等于几？` with chat template | 18 | 98.97 s | 54.98 s | 1.80x |
| `Explain matrix multiplication.` with chat template | 17 | 93.88 s | 52.34 s | 1.79x |

The greeting produced `你好！` in both cases. These are equivalence checks,
not a broad language-quality evaluation. Turbo retains fast's existing precision
limits: the earlier single-token comparison to the same GGUF FP32 reference had
about 32.94% logits relative-L2 error. Turbo adds no error in these tested cases;
select `precise` when lower quantization error is required.

## What the profiler found

The initial original-fast profile covered two executions totaling 18.03 s.
Its exclusive wall-time attribution was:

| Operation | Share of measured graph wall time |
|---|---:|
| Reading results / waiting for the device | 24.6% |
| Sending parameters | 22.3% |
| Repacking static weights on the host | 14.4% |
| Sending instructions | 11.2% |
| Sending activation data | 9.9% |
| Reading completion events | 8.2% |
| Other orchestration and code generation | about 9% |

USB API time includes device work and waiting. It does not isolate wire time
from chip compute time. Parent-stage timings overlap their children and must
not be added to these exclusive rows. Async transfers overlap within a batch,
so individual sync-transfer percentages are not reused for the async result.

After optimization, a repeated one-token graph takes about 4.8 s. Across three
instrumented runs, parameter caching/upload programs averaged 1.89 s per graph,
and all scalar runtime programs averaged 2.70 s including transport/waiting.
Within the latter, recurrent updates accounted for 1.60 s. Actual MXU execution
programs averaged only 0.109 s. Thus weight streaming and recurrent-state work
remain the main targets; speeding up the INT8 matrix instructions alone has
limited impact on total latency. The measured timings include host API overhead
and do not establish raw hardware peak throughput.

## Changes implemented

- Immutable parameter layout is cached in a bounded host RAM LRU. The default
  limit is 1 GiB; a main-graph fast pass stores about 719 MiB (753,895,424 bytes).
  Cache hits reuse the exact original packed bytes. This avoids about 1.3 s of
  repeated CPU weight-layout preparation. Host storage is not TPU SRAM residency;
  the same weight bytes still traverse USB on each full graph.
- RMS/L2 normalization preserves the original FP32 sum, epsilon, reciprocal-root
  iterations and weight multiplication order, in one program per row.
- Sigmoid/SiLU preserves clipping, EXP2 and the two reciprocal Newton steps,
  without returning every intermediate to the host.
- For input vectors up to 1,024 elements, absmax, scale, reciprocal, quantized
  bytes and activation residual are produced by one TPU program.
- Four-tap convolution plus SiLU is fused with the original tap-addition order.
- Recurrent column tiles increase from 12 to 13 while remaining within the
  verified 8 KiB input/output limits. FP32 state and projection values are
  preserved; this is not a lower-precision state cache.
- The pinned runtime's asynchronous USB batches queue required reads before
  writes within each supported segment. Dependency boundaries remain intact,
  and each batch completes before the next dependent program. Arbitrary bulk
  concatenation is not used: the existing firmware/parser does not support
  putting multiple DMA descriptors in one transfer. A raw device owner remains
  exclusive, and cancellation is observed between completed programs.

`ScalarTPU`/`MatrixTPU` retain synchronous transport by default.
`OptimizedMatrixTPU` uses the validated async path and bounded parameter cache.
tinycoral exposes transport as a process-wide flag; construction sets the flag
for the one raw device owner, including when its module was already imported.
The legacy TFLite worker remains a separate process/API.

## Remaining opportunities and limits

1. Batch recurrent-state tiles into fewer hardware launches and keep temporary
   values in device buffers. This can preserve FP32 rounding and targets the
   roughly 1.6 s recurrent stage. It needs new DMA/staging programs and physical
   validation; no projected speedup is included in current results.
2. Prepare persistent packed-weight files to reduce first-execution layout work.
   The current optimization caches bytes in RAM until the worker is closed.
   Persistent files need layout/version and integrity checks.
3. Reduce streamed weight bytes or amortize them across independent requests.
   Current-generation compiler parameter regions total about 7.6 MiB, whereas a
   fast main pass uploads about 719 MiB. Dense weights cannot all stay in those
   regions. Lowering precision further would require a new accuracy evaluation;
   it was not used to obtain these results.
4. Improve multi-position MXU programs for prefill. The current implementation
   shares weights across positions but still performs separate residual readouts.
   Batching must preserve each position's bias correction and output precision.

In the measured warm case, parameter-upload programs alone consume about 39%
of wall time. Eliminating every other cost while leaving that stage unchanged
would still leave about 1.9 s per graph. That is an attribution bound for this
implementation and workload, not a hard physical device limit or a promised
future speed.

## Reproduce

The GUI offers `快速优化（与快速档同精度）`. Its worker uses the same optimized
backend as these commands. Other profiles remain available for comparisons.

```powershell
.venv-isa/Scripts/python.exe scripts/profile-qwen-chain.py --output reports/baseline.json
.venv-isa/Scripts/python.exe scripts/profile-qwen-chain.py --optimized --async-usb --steps 3 --output reports/turbo.json
.venv-isa/Scripts/python.exe scripts/verify-optimized-fusions.py --async-usb
.venv-isa/Scripts/python.exe scripts/benchmark-fast-equivalence.py
.venv-isa/Scripts/python.exe scripts/run-qwen-tpu.py --matrix-profile turbo --output reports/turbo-run.json
.venv-isa/Scripts/python.exe scripts/benchmark-qwen-chat.py --profile turbo --output reports/turbo-chat.json
.venv-isa/Scripts/python.exe -m pytest tests/test_isa_scalar.py tests/test_isa_mxu.py tests/test_isa_optimized.py tests/test_qwen_control.py -q
```

Close other Coral applications before physical tests. The offline interpreter
checks instruction semantics; only the separate physical comparisons establish
the observed device behavior. All timings are machine/workload specific.
