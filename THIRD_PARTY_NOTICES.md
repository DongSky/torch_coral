# Third-party software

The Python code in this distribution is MIT licensed. Dependencies retain their
own licenses. No Google runtime, compiler binary, libusb DLL, model weights,
datasets, or third-party wheels are bundled in the wheel or source distribution.

- PyTorch: BSD-3-Clause; https://github.com/pytorch/pytorch
- ONNX and ONNX Runtime: Apache-2.0 and MIT respectively;
  https://github.com/onnx/onnx and https://github.com/microsoft/onnxruntime
- onnx2tf: MIT; https://github.com/PINTO0309/onnx2tf
- LiteRT, TensorFlow Lite, PyCoral and libedgetpu: Apache-2.0;
  https://github.com/google-ai-edge/LiteRT and https://github.com/google-coral
- PySide6 is optional and has LGPL/GPL/commercial licensing options:
  https://doc.qt.io/qtforpython-6/licenses.html
- The optional experimental ISA path uses tinycoral (MIT, Frederik Rogalski,
  2026) and tinygrad (MIT). tinycoral is pinned to commit
  60794943cca3ddc63a4438cfff91b520b12bcfef. The Windows compatibility patch
  preserves its MIT notice in native/TINYCORAL-LICENSE. Upstream sources are
  external dependencies, not included in the wheel.
- GGUF reading/decompression uses llama.cpp's gguf Python package (MIT).
  Qwen/Unsloth model weights are separately licensed and are not redistributed.

The optional source-repository libusb patch in `native/winusb-option.txt` is
provided under LGPL-2.1-or-later, with the license in `native/COPYING-LIBUSB`.
Those optional native-build files are excluded from PyPI distributions. When
building or redistributing libusb, preserve its matching source and notices.
Compiler download helpers preserve upstream notices; using a mirror does not
make the proprietary compiler part of this project's source license.

Coral and PyTorch are names of their respective projects. This is an independent
community integration, not an official Google or PyTorch release.
