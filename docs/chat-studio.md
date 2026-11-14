# Coral Studio chat interface (0.3.0a9)

Alpha 9 extends [Turbo V4](turbo-v4-a9.md) with shared quantization and compute
fusion. Settings adds an optional V4 history-matched three/four-token limit;
two-token verification remains the default. Every long-block proposal still
passes the full TPU trunk and vocabulary head, with exact prefix restoration.

Alpha 8 accelerates the existing [Turbo V4](turbo-v4.md) mode with prepared USB
dispatch, pinned OUT packets and wider in-place activation quantization. Its
quantization precision and verified MTP decoding are unchanged. No additional
mode switch or parameter is required.

Alpha 7 adds [Turbo V4](turbo-v4.md), an independent mode using direct USB
readback with V3's verified decoding. Speculation and frequency settings apply
to both V3 and V4. Existing saved mode choices are retained.

Alpha 6 adds the independent [Turbo V3 experiment](turbo-v3.md): native batch
projections, verified MTP proposals and automatic fallback when proposal blocks
cost more per output token. Settings controls V3 speculation separately from
the optional diagnostic MTP execution used by other profiles.

Alpha 5 adds the separate [Turbo V2 mode](turbo-v2.md), initially selected for
new preferences. Existing saved mode choices are retained. Its 250/500 MHz
selection lives in Settings; changing frequency releases the worker and takes
effect on reconnect. Other modes retain their existing frequency.

Launch `Qwen Coral Chat.cmd`, or run `coral-qwen-chat` from the frontend
environment. The isolated ISA worker still owns the Coral and executes the
numerical model operations. The existing inference profiles and device-release
behavior remain available.

The main window contains the mode selector, connection controls, conversation,
composer and a live inference dashboard. Model paths, Python environment,
prepared-weight location, context limit, output limit, MTP and animation controls
are in the separate Settings dialog. Settings are stored locally in
`.gui/qwen-chat-settings.json` under the configured workspace. Conversation text
is not written to that preferences file. Cancel discards unsaved changes.

Switching modes while connected reloads the backend and retains the displayed
conversation. The new backend reconstructs its state from the conversation on
the next request. Changing model/device paths or context releases the worker;
connect again to apply those changes. The connection button also lets the user
disconnect and release the USB device without closing the window.

## Live measurements

The worker emits structured monotonic timings and completed-layer events, rather
than estimating progress from a decorative animation. Metrics distinguish:

- **TTFT:** worker receipt of the chat request to its first visible output token;
  this includes prompt preparation and prefill, but not initial model loading.
- **Prefill rate:** newly processed prompt tokens divided by measured prefill
  time. Already cached prompt tokens are reported separately and not counted as
  newly processed tokens. The rate appears after prefill finishes.
- **Decode rate:** visible tokens after the first, divided by measured decode
  work. The first output token comes from prefill and is excluded. MTP diagnostic
  time is recorded separately and excluded from this rate.
- **Overall throughput:** all visible output tokens divided by complete request
  time, including prompt processing and any optional MTP work.
- **Rate chart:** one measured point per completed decode token, using that
  token's decode interval. No synthetic samples are added between arrivals.

For Turbo V3, required MTP draft and full-verifier work are both included in
decode time. A verified two-token block contributes one chart point at its
actual average rate (two visible tokens divided by the whole block duration).
Optional diagnostic MTP in the other modes retains the separate timing above.

The main decode number refreshes every 100 ms. While a decode step is running,
its elapsed time increases without inventing completed tokens. With no completed
decode token the display shows a dash, including one-token-only responses.
An EOS is not counted as visible output, while its execution time remains part
of the request/stage timing. Stop and error states freeze the measurements.

Prefill progress is based on completed layers across prompt chunks. Decode
progress tracks the current token's layers and output projection. These are
execution-progress indicators, not predictions of remaining time. The displayed
TPU call count is scoped to the current request.

One physical GUI/QProcess check used 13 prompt tokens and two output tokens for
`你好`, returning `你好！`. It measured 38.21 s prefill, 38.25 s TTFT, 4.91 s
decode and 43.15 s total: approximately 0.204 decode token/s and 0.0463 overall
token/s. It produced the same token IDs and 102,674 hardware calls as the prior
turbo path. Timings vary by machine and workload; this is a functional telemetry
check, not a broad performance or language-quality benchmark.

## Layout and motion

The warm light palette uses ivory surfaces, dark graphite/green text, coral
accents and teal telemetry. The resizable chat/dashboard layout was inspected at
1320×900 and 980×720; the dashboard scrolls vertically at compact sizes.
Short chat bubbles fit their contents, while long messages wrap. Scrolling up
keeps the user in place instead of forcing the view back to new text.

Motion includes message fades, streamed-text reveal, a rotating orbital mark,
typing dots, eased stage bars, flow highlights and a breathing chart marker.
These are decorative; they do not change counts or add measurements. Settings →
Generation & Appearance can disable motion while keeping live metrics active.
On Windows the client explicitly registers locally installed system fonts for
consistent native/headless rendering; fonts are not bundled in the package.

Use Ctrl+Enter to send. Stop requests are honored between completed TPU
programs. Closing the window waits for the worker to stop and release the device.

## Checks

`tests/test_chat_metrics.py` verifies timing boundaries, first-token exclusion,
MTP exclusion, cancellation and zero-duration behavior with a deterministic
clock. `tests/test_qwen_gui.py` checks settings separation and cancellation,
local preference storage, mode changes, event-driven rates, escaping and worker
exit recovery. The physical GUI check separately validates actual worker events,
layer progression, token IDs, rates and orderly device release.
