# Turbo V4: prepared USB dispatch and in-place quantization

**Alpha 9 update:** [shared quantization, compute fusion and verified long blocks](turbo-v4-a9.md)
extend this same mode. [Windows I/O measurements](v4-io-windows.md) distinguish
payload throughput from CPU, driver and device waiting. Alpha 7/8 measurements
below remain historical comparisons.

Select **Turbo V4 · I/O 加速** in Coral Studio, or pass
`--matrix-profile turbo-v4` to the CLI. V3 remains available for comparisons;
new preferences retain V2 as their default. V4 uses V3's native batch kernels,
two-digit readout, FP32 arithmetic order and verified speculative decoding.

## Alpha 8 implementation

The existing V4 mode now uses three additional optimizations:

- `usb_dispatch.PreparedRuntime` caches an executable's immutable hint layout,
  instruction frames and DMA segment boundaries in a bounded per-device cache.
  Dynamic input and parameter ranges are validated before submitting any part
  of the call. Other hint types retain the pinned runtime's original path.
- `usb_dispatch.PinnedAsyncBatch` lets libusb read OUT packets directly from
  immutable bytes. Each packet and its pointer remain owned until completion
  or cancellation has drained every submitted transfer. This removes the
  additional OUT copy into a ctypes staging buffer. Framing and IN results
  still involve copies; this is not an end-to-end zero-copy pipeline.
- `isa_v4_fusions.wide_primary_quantize_program` quantizes 1028--4000 FP32
  activation values in-place within the verified 16000-byte input limit.
  It preserves absmax, scale, reciprocal refinements, clipping and rounding.
  Only the single-component projection path omits the unused residual. Other
  precision profiles and larger inputs keep their existing implementation.

The wide kernel reads all inputs during absmax, then packs each four consumed
values into one word. Its scale is stored only after packing finishes, so it
cannot overwrite an unconsumed input. A 3584-value FFN input needs one TPU call
instead of ten. The packed bytes and FP32 scale matched bit for bit in fifteen
physical cases: five lengths, each with random, zero and outlier inputs.

The transport keeps all-IN-before-OUT submission, per-program completion
barriers, offset-based short-IN continuation and owning immutable IN results.
Submission failures, short OUTs, zero-progress INs, transfer failures and event
errors drain active transfers before returning. Cancellation remains between
programs under the existing backend lock; numerical execution stays on Coral.

V3 and older modes keep their original runtime. The pinned vendor files are
unchanged; no process-global runtime function is replaced. `prepared_usb=False`
and `wide_quantization=False` are internal comparison options on the V4 backend.

## Alpha 8 versus original V4

Physical Qwen3.5-0.8B test, USB SuperSpeed, 500 MHz, same process and warm host
parameter/program caches. Both implementations received a complete warmup,
then A B B A was repeated twice (four measured samples per implementation).

| Measured work | Original V4 (Alpha 7) | Enhanced V4 (Alpha 8) | Time reduction |
|---|---:|---:|---:|
| 13-token prefill, including greedy selection | 11.565 s | 10.429 s | 9.82% |
| Ordinary one-token decode, including selection | 3.407 s | 3.228 s | 5.24% |
| Successful two-token speculative block | 4.853 s | 4.537 s | 6.52% |

One iteration contained those three measurements. TPU calls fell from 44091
to 38449 (12.8% fewer), including 810 uses of wide primary quantization. All
four full-vocabulary logit vectors and all caches, including MTP, matched bit
for bit in every iteration. The accepted output IDs were unchanged.

The two-token block is 2.269 s/token; it is not the whole-request average,
which also includes calibration, rejections and final single steps. Initial
loading and first program/parameter generation are excluded from this A/B.
These timings compare the enhanced mode with original V4, not with V3.
The bounded plan cache and OUT pinning do not retain another full set of
framed model weights. Weight payload size is unchanged; activation/result
traffic and launches decrease because the quantization chain is fused.

An earlier dispatch-only experiment (without wide quantization) measured
4.88% less prefill time, 3.81% less ordinary decode time and 3.25% less
two-token-block time in the same balanced order. This separates transport
improvements from the quantization fusion; speedup percentages are not additive.

Eight-output-token tests checked each full logit vector and the final trunk
caches against ordinary original-V4 decoding:

| Input | Prompt tokens | Enhanced V4 warm prefill | Whole decode / 7 subsequent tokens | Accepted drafts |
|---|---:|---:|---:|---:|
| Greeting | 13 | 10.392 s | 16.491 s / 7 = 2.356 s/token | 3/3 |
| Arithmetic question | 18 | 15.396 s | 17.038 s / 7 = 2.434 s/token | 3/3 |
| English explanation | 17 | 14.537 s | 19.951 s / 7 = 2.850 s/token | 2/3 |

Every vector and state matched bit for bit. The greeting additionally forced
both acceptance and rejection, checking exact committed prefix restoration.
These whole-request decode results include calibration and rejected proposals;
the prompt lengths and acceptance rates differ between rows.

The physical GUI/worker test completed fresh V4, reset/repeated V4, V4-to-V3
reload and V3-to-V4 reload in one worker, then shut down cleanly. Its first
eight-output-token V4 request measured 13.083 s prefill and 16.980 s decode
(2.426 s/token), 30.114 s total. The warm request measured 10.472 s prefill
and 16.444 s decode (2.349 s/token), 26.956 s total. Both used 55421 TPU calls
and identical output IDs. Live token/s, phase progress, chart samples, prompt
IDs and the dispatch/quantization flags were checked across all four requests.

## Original Alpha 7 readback optimization

The pinned USB driver used `bytes(buffer[:received])` on a ctypes byte array.
Its slice first built a Python integer list. V4 wraps that device instance's
buffer pool in `usb_buffers.DirectReadBuffers`. Contiguous reads use
`ctypes.string_at()` and produce immutable bytes directly. The result still
owns its storage and remains valid when the next transfer reuses the buffer.

This is a buffer adapter. It uses the existing driver's submission order,
short-packet continuation, status checks, cancellation and event handling.
ctypes casts, writes and `byref(buffer, offset)` keep their normal semantics.
The view pins the original allocation with `from_buffer`; newly allocated
transfer buffers receive the same adapter. Installation is allowed only while
the batch is idle and applies only to the V4 backend's device instance.

V4 requires asynchronous USB. The vendor source is not edited and runtime
functions are not generated or replaced. Quantization precision and FP32
formulas remain unchanged. Model arithmetic continues to execute on the Coral.

## Historical Alpha 7 V3/V4 A/B

Same process, prepared Qwen3.5-0.8B Q4_K_M INT8 cache, USB SuperSpeed, 500 MHz.
After a complete warmup, the order was V3, V4, V4, V3; values below are the
medians of the two samples per mode. Host parameter/program caches were warm.

| Measured work | V3 | V4 | Time reduction |
|---|---:|---:|---:|
| 13-token prefill | 12.904 s | 11.687 s | 9.43% |
| Ordinary one-token decode | 3.693 s | 3.459 s | 6.33% |
| Successful two-token speculative block | 5.466 s | 4.957 s | 9.32% |

Each measured iteration used the same 44091 TPU calls. All full-vocabulary
logits and all caches, including the MTP cache, matched bit for bit. The
two-token block includes draft computation, full verification and selection.
Its 2.478 s/token is a block result; whole-request decode also includes
ordinary calibration, rejected proposals, final single steps and EOS work.

Initial model loading and first program/parameter generation are excluded from
this warm A/B. Fresh-worker timings are recorded separately by the GUI test.
V4 reduces host conversion overhead; it does not reduce the number of bytes
that the existing model graph transfers over USB.

Eight-token generation tests compared every emitted full logit vector and the
final trunk caches with ordinary V3 decoding. All matched bit for bit; the
greeting test also forced both acceptance and rejection-prefix restoration.

| Input | Prompt tokens | V4 whole decode / 7 visible tokens | Accepted drafts |
|---|---:|---:|---:|
| Greeting | 13 | 18.005 s / 7 = 2.572 s/token | 3/3 |
| Arithmetic question | 18 | 18.188 s / 7 = 2.598 s/token | 3/3 |
| English explanation | 17 | 21.698 s / 7 = 3.100 s/token | 2/3 |

These timings include the ordinary-step calibration and all proposal work.
They are not a guarantee of equal speedup at other acceptance rates.

The physical GUI test completed four requests in one worker: fresh V4,
reset/repeated V4, switch to V3, then switch back to V4. The first V4 request
measured 14.587 s prefill and 18.189 s decode for seven subsequent tokens
(2.598 s/token); request total was 32.832 s. With warm host parameter/program
caches, prefill was 11.637 s and decode 17.644 s (2.521 s/token), total 29.321 s.
Both eight-token V4 requests used 62483 TPU calls and the same output IDs.
The test also checked real token/s, phase progress, chart sampling, exact
prompt reporting, backend identity across reloads and clean device shutdown.

## Settings and compatibility

The existing speculation checkbox applies to V3/V4; automatic cost-based
fallback is unchanged. The shared V2--V4 frequency selector offers 250/500 MHz.
The higher setting increases power and enclosure temperature; retain adequate
cooling as described in the [Coral frequency guidance](https://coral.ai/docs/accelerator/get-started/#install-with-maximum-operating-frequency-optional).
Model parameters remain in Settings; the main window contains the mode switch.

The worker also fixes a reporting bug: speculative output IDs no longer
overwrite the full prompt's `input_tokens` in the final event. This only
corrects metadata; it does not affect inference.

## Reproduce

Run physical tests sequentially with other Coral owners closed:

```powershell
.venv-isa/Scripts/python.exe -m pytest tests/test_usb_buffers.py -q
.venv-isa/Scripts/python.exe -m pytest tests/test_usb_dispatch.py tests/test_isa_v4_fusions.py -q
.venv-isa/Scripts/python.exe scripts/probe-v3-native-batch.py --v4
.venv-isa/Scripts/python.exe scripts/verify-v4-fusions.py
.venv-isa/Scripts/python.exe scripts/benchmark-v4-dispatch.py --output reports/turbo-v4-enhanced-ab.json
.venv-isa/Scripts/python.exe scripts/benchmark-usb-output-copy.py
.venv-isa/Scripts/python.exe scripts/benchmark-v3-speculative.py --v4 --adaptive
.venv-isa/Scripts/python.exe scripts/benchmark-qwen-chat.py --profile turbo-v4 --new-tokens 8
.venv-isa/Scripts/python.exe scripts/run-qwen-tpu.py --matrix-profile turbo-v4 --no-speculative
```

Offline tests cover receive ownership, buffer reuse, offset writes for short
packets, slice bounds and rejection of installation during pending transfers.
Native physical probes include a 57344-byte input and 98304-byte output. The
full-model test verifies accepted proposals, forced rejection rollback and
complete output/state equivalence to the reference implementation.
