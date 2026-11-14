# Windows USB compatibility build

The legacy Google Edge TPU runtime selects libusb's UsbDk backend. Some Windows
11 installations encounter redirector creation errors or initialization hangs.
The optional WinUSB compatibility build changes that selection only when the
worker explicitly sets `TORCH_CORAL_FORCE_WINUSB=1`. Device computation and USB
transfers still execute through the real runtime and hardware.

`../scripts/build-winusb.ps1` builds upstream **libusb 1.0.29**, with the small
change in `winusb-option.txt` inserted in `windows_set_option`. Only when the
worker sets `TORCH_CORAL_FORCE_WINUSB=1`, the legacy request to switch to UsbDk
returns success while retaining the default WinUSB backend. USB transfers and
actual model execution remain entirely real. Without that variable libusb's
original selection behavior is preserved.

The worker preloads these project-local DLLs before importing PyCoral:

- `.tools/coral-native/libusb-1.0.dll`: built from source, with the opt-in change.
- `.tools/coral-native/edgetpu.dll`: unmodified Google 20221024 standard-clock DLL.

Nothing is copied into System32 by this build. Existing driver installations and
other applications keep their own library behavior. The WinUSB driver installed
by Google is still needed. Visual Studio C++ tools (v143) and a Windows SDK are
required to reproduce the native build.

libusb is LGPL-2.1-or-later. Its COPYING file is included with the built DLL.
The complete matching source is in `.tools/libusb-1.0.29`, with the local change
applied. The upstream source archive SHA256 is in `libusb-source-sha256.txt`;
the exact local change and rebuild script are retained here. Upstream source:
https://github.com/libusb/libusb/tree/v1.0.29

The build writes `torch-coral-runtime.json` with relative filenames and hashes
beside the DLLs. Local absolute-path hash reports and build logs are private
output and must not be committed. No runtime DLLs are distributed in this
repository.

## Cold initialization and port reset

WinUSB's `libusb_reset_device` implementation resets endpoint pipes but does not
issue a full device/port reset. The Google USB boot flow expects a real reset
after DFU. A failure to open the device occurs before model execution and does
not establish that a model or operator is unsupported.

If initialization fails again, `scripts/restart-coral-device.ps1` requires an
administrator PowerShell and restarts only the single matching Coral device.
It does not reboot Windows. Normal inference stays unprivileged. Keep one
device-owning process at a time and close other clients before diagnostics.

## Experimental ISA path

`tinycoral-windows.patch` adds Windows locking and explicit device release to
the pinned tinycoral dependency. Its MIT notice is in `TINYCORAL-LICENSE`.
The ISA worker and the legacy delegate are separate device owners; do not run
them simultaneously on the same accelerator. See [Qwen setup](../docs/qwen-full-tpu.md).
