# Turbo V4 Alpha 9: shared quantization, fused selection and verified long blocks

The existing **Turbo V4** mode includes the Alpha 9 compute optimizations.
Quantization remains one weight/activation component and two readout digits.
All neural-network arithmetic runs on the Coral; host work is storage, copies,
tokenization, token-ID lookup and scheduling.

## Implemented compute path

- Projection groups take a bounded, read-only input snapshot. QKV/gate/beta/
  alpha and FFN gate/up reuse that group's quantized bytes and FP32 scale.
  Groups are protected by the backend lock and cleared on scope exit.
- Weighted 1024-column normalization can also produce primary quantization in
  the same scalar program. Its normalized FP32 values are retained. Reuse
  checks object identity and a byte snapshot, so changed arrays cannot use
  an old quantization result. Cache entries are bounded and weakly owned.
- SiLU and its following multiplication share one in-place program, keeping
  the two FP32 multiplies and the original sigmoid/SFU/Newton operation order.
- The full vocabulary head reconstructs FP32 logits and computes each original
  2048-value block's stable top four in one program. All logits remain available
  for precision checks. Greedy selection reuses those summaries after a byte
  validity check; changed arrays use the original selector.
- Native 16-position projections correct/upload bias banks only for active
  positions. Physical tuning selected native batches from three positions for
  smaller matrices and eight for larger matrices. The vocabulary head keeps
  its twelve-position threshold; FC was faster for two to four positions.

No instruction is omitted from the numerical reconstruction. Top-k uses the
same exclusion passes and tie order as Alpha 8. The vendor source is unchanged.

## Balanced comparison with Alpha 8

Physical Qwen3.5-0.8B, prepared Q4_K_M primary INT8 cache, USB SuperSpeed,
500 MHz. Frozen Alpha 8 engine control flow and backend paths were compared
with the new implementation, after a full warmup for both. A B B A was repeated
twice: four measured samples per path, medians below.

| Work | Alpha 8 | Alpha 9 | Elapsed-time reduction |
|---|---:|---:|---:|
| 13-token prefill plus selection | 10.600 s | 9.924 s | 6.38% |
| Ordinary one-token decode plus selection | 3.224 s | 3.178 s | 1.42% |
| Successful two-token speculative block | 4.536 s | 4.441 s | 2.11% |

Each iteration performs those three measurements. TPU calls fell from 38449
to 33962. Every full-vocabulary logit vector and every cache, including MTP,
matched bit for bit. These are warm timings; loading and first-time code/
parameter generation are excluded. The two-token block is not a request average.

## Three/four-position verification

`verify_block` evaluates the full trunk and vocabulary head for two to four
positions. Temporal kernels capture every intermediate recurrent state;
convolution history, attention KV lists and hidden history are retained for
every possible committed prefix. The verifier stops at the first mismatch.
Rejected suffixes are discarded, and cancellation can restore the zero-input
prefix before any result is committed.

All full logits and every commit prefix for blocks of length 2, 3 and 4 were
physically compared with frozen Alpha 8 ordinary decoding. Additional recurrent
probes checked three/four steps, including sixteen 128-by-128 state heads.

Two proposal strategies were implemented and measured:

1. Fixed long chains use the exact first MTP input, then estimated hidden states
   from the MTP module to propose further tokens. Estimated MTP states are
   discarded before verification. On a twelve-output-token greeting this was
   slower: 2-token adaptive decoding averaged 2.688 s/token, fixed 3 averaged
   3.231, and fixed 4 averaged 3.640. These experiments do not reduce output
   precision, but their later proposals had poor acceptance.
2. The GUI's optional long-block strategy extends a proposal only when a known
   history suffix of at least three token IDs supplies the continuation. It
   excludes stop markers, uses complete TPU verification, and falls back to
   two-position blocks when matches are absent or measured yield is worse.
   Token-ID lookup does not perform neural-network arithmetic on the host.

The default remains the stable **2-token** policy. Settings offers optional
history-matched limits of three/four tokens. Fixed estimated chains are CLI
experiments, not the GUI default.

### Actual long-block generation

Input: `请原样输出：1+1=2，1+1=2，1+1=2。` (35 prompt tokens), 16 output
tokens. Final history matching measured 27.940 s warm prefill and 27.930 s decode
for fifteen subsequent visible tokens: **1.862 s/token**. Ten draft predictions
were accepted, with no rejected predictions. Three blocks produced four visible
tokens each; the fastest took 6.325 s, or **1.581 s/token**.

Every emitted full logit vector and the committed trunk state after each block
matched frozen Alpha 8 ordinary decoding bit for bit. This repeated-content
result requires suitable history matches and is not a general-chat speed claim.

The final eight-output-token English explanation measured 13.980 s warm prefill
and 19.622 s decode for seven subsequent tokens (**2.803 s/token**), with two of
three drafts accepted. Full logits and every committed state matched bitwise.

The physical GUI/worker test completed five requests: fresh V4, warm/reset V4,
V3 reload, V4 reload, and optional four-position history matching. Fresh V4
measured 12.479 s prefill and 16.568 s decode for seven subsequent tokens
(2.367 s/token), 29.053 s total. Warm V4 measured 9.822 s prefill and 16.169 s
decode (2.310 s/token), 25.992 s total. Both used 49558 TPU calls and identical
output IDs. The repeated-content GUI request measured 27.294 s prefill and
28.001 s decode for fifteen subsequent tokens (1.867 s/token), with 116033
TPU calls. Mode flags, exact IDs, prompt counts, token/s, phase progress,
chart sampling and clean device shutdown were checked across all five requests.

## I/O and host operating system

Weight payload size is unchanged. These fusions reduce program launches and
activation/result transfers. See [Windows I/O measurements](v4-io-windows.md)
for throughput, process CPU and the limits of host-side attribution.

## Reproduce

Run hardware jobs sequentially with other Coral owners closed. The comparison
scripts need an unpacked Alpha 8 source tree at
`release/torch-coral-0.3.0a8-final` (or `--baseline-root` for the compute A/B).

```powershell
.venv-isa/Scripts/python.exe -m pytest tests/test_v4_compute.py tests/test_speculative_blocks.py -q
.venv-isa/Scripts/python.exe scripts/verify-v4-compute.py
.venv-isa/Scripts/python.exe scripts/verify-v4-prefixes.py
.venv-isa/Scripts/python.exe scripts/verify-v4-blocks.py
.venv-isa/Scripts/python.exe scripts/tune-v4-native.py
.venv-isa/Scripts/python.exe scripts/benchmark-v4-compute.py
.venv-isa/Scripts/python.exe scripts/benchmark-v4-blocks.py --modes 2,3,4,auto4
.venv-isa/Scripts/python.exe scripts/probe-v4-io.py
.venv-isa/Scripts/python.exe scripts/benchmark-qwen-chat.py --profile turbo-v4 --verification-tokens 4 --new-tokens 16
```

`run-qwen-tpu.py --matrix-profile turbo-v4 --verification-tokens 4` selects
history matching. Adding `--fixed-blocks` selects the experimental estimated
chain. Backend `reuse_quantization`, `compute_fusions`, `head_fusion` and
`active_bias` flags are available for isolated comparisons. Recent ordinary-step
calibration reuse is implemented as an experimental backend option and remains
disabled by default; it can change block alignment and is not an unconditional
speed improvement.
