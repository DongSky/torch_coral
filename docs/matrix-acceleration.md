# Experimental matrix acceleration (0.3.0a2)

This page records alpha 2. Alpha 3 adds the [turbo profile and full-chain
measurements](chain-optimization.md); the GUI now initially selects turbo.

The Qwen executor now sends large static projections to the Coral's physical
INT8 matrix array (MXU). Activation conversion, output reconstruction and all
remaining numerical model operations also execute on the physical TPU.
The production worker has no CPU/CUDA model-arithmetic fallback. Host NumPy
weight conversion is one-time offline preparation; host matrix products appear
only in independent validation and benchmark references.

## Implemented strategy

`isa_mxu.MatrixTPU` extends `ScalarTPU`. GGUF weights are first decompressed and
then represented by two row-wise INT8 components and FP32 scales. This is an
additional approximate representation of the values in the GGUF, not direct
execution of Q4_K_M blocks. The cache uses source SHA256 and format-version
guards, and requires additional disk space. The existing FP32 cache is retained.

The TPU computes the activation maximum, scale, quantized bytes and residual.
Compensated profiles use both weight components and both activation components
(four INT8 products). The MXU output is eight-bit: the executor reads high
digits, corrects INT32 biases on the TPU, uploads only those biases, and reads
successive residual digits. Reconstruction, scaling and summation use TPU FP32.
Three readout digits recover exact integer products in the tested accumulator
range; this does not make the approximate weight/activation representation exact.
If the first readout saturates, the executor increases its range and retries.

Parameter caching reuses weights across prompt positions. Two components occupy
separate parameter regions when both fit the verified SRAM limit; otherwise
they stream through one region. Only bias bytes change between residual
readouts. The default output chunk is 6,144 rows. On the three tested matrices,
2,048- and 4,096-row alternatives did not give a consistent overall advantage;
smaller chunks force extra uploads for a matrix that could otherwise remain
resident. The whole model cannot fit in SRAM and still streams weights over USB.

The FP32 Gated DeltaNet update also fuses decay, two matrix-vector products and
a rank-one update into one program per column tile. Three random physical cases
matched the old physical TPU path bit for bit, and full-model logits before and
after fusion matched bit for bit. One 128x128 head took about 6.0–6.2 ms warm,
versus 19.8–20.9 ms for the separate old operations.

## Measured results

Measurements below used this Windows host, a direct USB 3 SuperSpeed connection,
250 MHz clock, synchronous USB and the pinned tinycoral source. They establish
the fastest configurations tested here, not a global hardware limit. Timing
varies between runs. Baselines are the previous physical scalar-FP32 executor,
not optimized CPU/GPU libraries or Google's compiled vision examples.

Full one-token graph: input ID 9707, selected ID 13, no MTP. Timings include
device initialization and model execution with prepared weight files; they
exclude model download and initial weight preparation. All profiles selected
the same next token in this one case.

| Profile | Seconds | Speedup vs old FP32 | TPU calls | Logits relative L2 vs old FP32 |
|---|---:|---:|---:|---:|
| Original FP32 | 140.249 | 1x | 519,121 | — |
| Fast: one component, two readouts | 9.114 | 15.39x | 15,596 | 32.94% |
| Balanced: two components, two readouts | 15.395 | 9.11x | 24,942 | 1.803% |
| Precise: two components, three readouts | 16.764 | 8.37x | 29,124 | 0.2082% |

Precise is the GUI default. Its independently computed HF reference comparison
used identical GGUF weights: top token matched, relative L2 was 0.00208139,
cosine 0.999998. The quantized test explicitly used a 0.005 relative-L2 limit;
it does not meet the older FP32 reference's stricter 0.001 default. These are
narrow numerical checks, not perplexity or broad language-quality evaluations.
Fast has substantial logit distortion and is an explicit experimental option.

A full precise chat processed 13 prompt tokens and generated two tokens,
replying `你好！` to `你好`, matching independent HF greedy token IDs. It took
125.87 seconds and 343,010 calls; a final repeat with the completed SRAM placement
took 122.38 seconds with identical token IDs and call counts. The old FP32 path
took 708.44 seconds and 1,546,214 calls (5.63–5.79x speedup). MTP is disabled in
that comparison. A separate precise
single-position trunk plus MTP execution took 20.96 seconds, 34,579 calls, with
physical coverage of the draft normalization, projection, attention, FFN and
shared head. MTP is still diagnostic; speculative acceptance is not implemented.

Matrix-vector timings below include physical activation conversion, MXU
execution, residual readout and reconstruction. Warm medians are ten repeated
calls with the same matrix and input. Resident fast weights are reused; this
benefit cannot be assumed for every layer of a streaming LLM.

| Weight shape [N,K] | Old FP32 ms | Fast warm ms | Fast first call ms | Precise warm ms | Precise first call ms |
|---|---:|---:|---:|---:|---:|
| [3584,1024] | 663.99 | 4.92 | 52.40 | 59.54 | 87.18 |
| [1024,3584] | 636.85 | 4.49 | 36.22 | 16.90 | 56.42 |
| [6144,1024] | 1132.68 | 6.86 | 42.24 | 94.18 | 101.49 |

Fast warm speedups were 135–165x; precise warm speedups were 11–38x. Matrix
relative-L2 errors were respectively 1.12–1.30% and 0.00459–0.00537%. Fast's
largest effective useful throughput in this run was about 1.83 GOP/s, counting
one multiply plus one add as two operations. This includes conversion/readout
and USB overhead; it is not a measurement of the chip's raw INT8 peak. Extra
compensation products are not counted as useful model operations.

## Use and reproduce

Open `Qwen Coral Chat.cmd`, choose a matrix profile and load. First load prepares
the INT8 cache beside the prepared FP32 folder. Changing profile requires reload.
MTP is optional and off by default because computing unaccepted drafts adds work.

From the repository root, after the documented ISA setup:

```powershell
.venv-isa/Scripts/python.exe scripts/run-qwen-tpu.py --matrix-profile precise --output reports/precise.json
.venv-isa/Scripts/python.exe scripts/benchmark-mxu.py
.venv-isa/Scripts/python.exe scripts/benchmark-qwen-chat.py
.venv-isa/Scripts/python.exe scripts/verify-mxu-readout.py
.venv-isa/Scripts/python.exe scripts/verify-mxu-recurrent.py
.venv-isa/Scripts/python.exe -m pytest tests/test_isa_scalar.py tests/test_isa_mxu.py tests/test_qwen_control.py -q
# Separate independent CPU validation, not a production execution backend:
.venv-torch/Scripts/python.exe scripts/validate-qwen-reference.py --report reports/precise.json --relative-l2-limit 0.005
.venv-torch/Scripts/python.exe scripts/validate-qwen-chat-reference.py
```

`verify-mxu-readout.py` needs the synthetic manifest created by `probe-mxu.py`;
run the probe first. Close all other Coral applications before hardware tests.
Do not run the raw ISA worker and TFLite delegate concurrently. Tests of the
offline ISA interpreter prove instruction semantics only, not physical support.

Further improvements can target USB command batching, keeping more intermediate
buffers on device, compact weight streaming and efficient batched MXU programs.
None of those unmeasured improvements is included in the speedups above.
