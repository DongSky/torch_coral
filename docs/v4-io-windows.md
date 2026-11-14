# V4 USB and Windows I/O measurements

These measurements concern the connected Coral on this Windows host. They do
not establish USB wire utilization or isolate the kernel's latency by themselves.

## Physical application-level probe

The probe repeatedly uploads one already-packed 6.316 MB parameter tile, then
runs 20000 small scalar programs using prebuilt inputs. Both transport paths
are warmed, with an A B B A order. The prepared path uses immutable pinned OUT
packets; the legacy path stages each OUT in a ctypes allocation.

Prepared-path observations:

- Parameter uploads sustained about **419 MB/s** of application payload. Each
  call also distributes parameters into SRAM and waits for completion. Process
  CPU was about **10% of one logical core** during this phase.
- A 1024-element scalar program took about **179 microseconds per call**;
  process CPU was about **36% of one logical core**. Remaining elapsed time
  includes USB, device execution, driver waits and synchronization.
- A passive 9.2-second window on this 24-logical-processor host observed process
  CPU around 26% of one core, system CPU at most 9%, and sampled per-core DPC
  and interrupt utilization at most 6% each. System page reads peaked at 3/s.
  These are short aggregate samples, not a kernel USB trace.

There is no sustained CPU/DPC/paging saturation in that sampled window. This
does not exclude per-request Windows/WinUSB latency. Large parameter transfers
and tiny program transfers have different bottlenecks; the link is not known
to be continuously full throughout model inference.

The earlier model payload census found about 754 MB of packed primary weights
per complete trunk/head pass, of which the vocabulary head is about 255 MB.
Alpha 9 leaves that weight payload unchanged. Its smaller activation traffic
and fewer program launches do not eliminate the dominant repeated weight stream.

## If the host OS is limiting throughput

First distinguish data transfer from waiting. Windows supports overlapped USB
operations, and the current runtime already queues asynchronous transfers and
reuses buffers. Microsoft documents [WinUSB asynchronous I/O](https://learn.microsoft.com/en-us/windows-hardware/drivers/usbcon/using-winusb-api-to-communicate-with-a-usb-device)
and [overlapped I/O](https://learn.microsoft.com/en-us/windows/win32/fileio/synchronous-and-asynchronous-i-o).

For a kernel-level diagnosis, correlate CPU scheduling, DPC/ISR activity and USB
request submission/completion on an ETW timeline. Microsoft's
[USB performance tracing workflow](https://learn.microsoft.com/en-us/windows-hardware/drivers/usbcon/analyzing-usb-performance-issues-by-using-xperf-and-netmon)
describes combining USB traces with CPU/scheduling traces. No kernel USB trace
was collected in this run, so Windows-specific delay has not been quantified.

The remedy depends on that evidence:

| Observed limit | Next experiment |
|---|---|
| Python/ctypes preparation consumes the critical path | Move descriptor preparation/event callbacks into native code; A/B the full request |
| Many short USB requests and completion gaps | Fuse more dependent device work and enlarge effective batches; measure request rate and gaps |
| Host-controller/DPC contention | Compare a direct port on another controller and identify competing driver work in the trace |
| Large transfers themselves reach a repeatable ceiling | Amortize weight uploads across verified tokens; host CPU tuning does not reduce those bytes |
| Suspected Windows driver-stack overhead | Run an otherwise identical native-Linux comparison on the same hardware before assigning an OS speedup |

The pinned single-endpoint protocol requires completion of earlier IN segments
before later OUT segments. Its source documents a firmware deadlock if this
boundary is crossed. More concurrency must preserve that device dependency;
removing the completion boundary is not a valid throughput optimization.

The repository probe uses process CPU time and application payload counts. It
does not attribute all non-CPU time to USB, and the passive samples do not prove
that switching operating systems would improve performance.
