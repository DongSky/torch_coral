# Turbo V3 experiment

Select **Turbo V3** in Coral Studio. Turbo V2 remains the default for new
settings. V3 uses the same one-component INT8 representation and two readout
digits as Turbo/V2; full model arithmetic remains on the physical Edge TPU.

## Expected and observed performance

The initial V3 target was 12--14 s prefill for the 13-token greeting and
2.5--3.0 s per subsequent token when draft acceptance is high. These were
estimates, not acceptance criteria for every prompt.

With the final kernels, an initial V3 prefill in a new backend measured 15.62 s.
Repeated prefill in the same process measured 12.86 s after parameter packing
and program generation were cached. The first result has not reached the
12--14 s target; the cached result has. Model loading is excluded from these
prefill timings and is reported separately by the GUI workflow.

The strict physical Turbo V2 comparison used the original scalar schedule and
no V3 temporal convolution/native projection or top-k changes. Both backends
used 500 MHz. Decode means total wall time after the first visible token,
including draft computation, verification, selection and the ordinary-step
calibration. Timing is per visible output token, not per proposed token.

| Input | Prompt tokens | V2 median ordinary decode | V3 total decode / visible tokens | Accepted drafts |
|---|---:|---:|---:|---:|
| Greeting | 13 | 3.681 s/token | 19.432 s / 7 = 2.776 s/token | 3/3 |
| Arithmetic question | 18 | 3.740 s/token | 20.097 s / 7 = 2.871 s/token | 3/3 |
| English explanation | 17 | 3.708 s/token | 23.500 s / 7 = 3.357 s/token | 2/3 |

The English case did not reach the 2.5--3.0 s target. It rejected one draft and
still improved total decode time from 25.945 to 23.500 seconds. It must not be
represented by the higher-acceptance greeting result.

Every emitted full-vocabulary logit vector, output ID and final trunk cache
matched V2 bit for bit. The greeting test also forced one correct proposal and
one deliberately incorrect proposal, checking both logit rows and the exact
one-position cache after rejection. These are short tests, not a quality or
long-context benchmark. The existing Turbo approximation error remains.

## Native batch execution

`isa_batched.BatchProgram` maps up to 16 positions onto the 16 physical tiles.
Weights are distributed across parameter regions and broadcast for native
matrix execution. Each position has a reserved independent bias bank. This
preserves the second residual readout instead of using a shared correction
that would change quantization results.

The full projection sweep checks both digits and reconstructed FP32 bits.
Two/four-position batches can be slower than V2 because the 16-position program
pads unused positions. The scheduler uses native execution from eight
positions, or twelve for outputs wider than 4096 rows; smaller batches use V2.
The serialized FC/native weight blob is shared, avoiding a second host repack
when prefill switches to single-token decode.

Temporal convolution shares history, new projections and kernel taps across
time; it reproduces V2's four-tap sum and SiLU operation order. A scalar
scheduler shortens SRAM load waits only after an on-device canary matches the
original path. It relocates loops and branches and leaves SFU waits intact;
failure of the numerical canary retains the conservative schedule.

Native batch probes verified a single 57344-byte input and a single 98304-byte
output on real hardware. These are tested payload sizes, not claimed USB
limits. The 16384-byte scalar-SRAM bound in V2 applies to a different path.

## Verified speculative decoding

The first visible token comes from the ordinary full model. V3 measures one
ordinary decode step, then proposes one candidate with the MTP layer and
verifies `[previous token, candidate]` through all 24 trunk layers and the full
vocabulary head. A match emits the candidate plus the verified bonus token.
A mismatch emits the full model's first choice and restores the one-position
prefix. Intermediate recurrent state is returned by an extra output DMA;
rejection does not recompute the complete trunk.

The proposal head uses roughly 500 candidate rows: four winners from each
2048-logit block, recent IDs and stop IDs. Top-k is computed on the TPU; the host
only retains/maps IDs and packs the selected static weight rows. This limits
the auxiliary proposal head, not the full vocabulary verifier. A poor proposal
cannot bypass verification or alter the greedy result.

The measured cost controller includes the whole draft/verification block. After
two speculative blocks, it disables speculation if its observed per-token cost
is no better than the ordinary-step measurement. Initial exploration can still
make short or low-acceptance requests slower. An early one-winner shortlist
trial took 3.81 s/token and disabled speculation; it was replaced by four
winners. There is no universal speedup guarantee.

Settings contains a separate V3 speculation checkbox and the shared V2/V3
250/500 MHz choice. Maximum frequency increases power and enclosure temperature;
see the [official frequency guidance](https://coral.ai/docs/accelerator/get-started/#install-with-maximum-operating-frequency-optional).
The per-token chart uses block duration divided by the number of verified
visible tokens. MTP cost is included in V3 decode time.

A fresh GUI worker processed the greeting with eight visible output tokens in
35.94 s: prefill 15.778 s, TTFT 15.835 s, decode 20.102 s for seven subsequent
tokens (2.872 s/token, 0.348 token/s). It made 62483 TPU calls. Four chart
samples corresponded to one ordinary step and three verified pairs. Live
metrics, phase progress, emitted IDs and clean worker shutdown were checked.

## Reproduce

```powershell
.venv-isa/Scripts/python.exe scripts/probe-v3-native-batch.py
.venv-isa/Scripts/python.exe scripts/benchmark-v3-projections.py
.venv-isa/Scripts/python.exe scripts/verify-v3-convolution.py
.venv-isa/Scripts/python.exe scripts/verify-v3-state-capture.py
.venv-isa/Scripts/python.exe scripts/verify-v3-schedule.py
.venv-isa/Scripts/python.exe scripts/benchmark-v3-speculative.py --adaptive
.venv-isa/Scripts/python.exe scripts/benchmark-v3-speculative.py --adaptive --skip-forced --prompt "1+1等于多少？"
.venv-isa/Scripts/python.exe scripts/benchmark-qwen-chat.py --profile turbo-v3 --new-tokens 8
```

The strict benchmark reports V2 initial prefill and V3 cached prefill separately;
its V3 parameter cache is warmed by the reference run. A fresh GUI worker
provides the separate first-V3-prefill measurement. Close other Coral
owners and run physical benchmarks sequentially.
