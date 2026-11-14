# Development

Use Python 3.12 for the PyTorch frontend and experimental ISA worker. Keep the
legacy PyCoral worker, converter and ISA dependencies in separate environments;
they have different compatibility requirements. USB drivers, runtime libraries,
compiler binaries, model weights and tokenizers are supplied separately.

## Frontend

From the repository root on Windows:

```powershell
py -3.12 -m venv .venv-torch
.venv-torch/Scripts/python.exe -m pip install -e '.[chat,gui,test,release]'
```

Choose the appropriate PyTorch CPU/CUDA build before installing if needed. For
legacy graph conversion and device execution, use the requirement files under
`torch_coral/dependencies/` and the configuration commands in the [README](../README.md).

## Experimental Qwen worker

```powershell
.venv-torch/Scripts/python.exe -m pip install uv
.venv-torch/Scripts/python.exe -m uv venv .venv-isa --python .venv-torch/Scripts/python.exe
.venv-torch/Scripts/python.exe -m uv pip install --python .venv-isa/Scripts/python.exe -r torch_coral/dependencies/isa-py312.txt
.venv-torch/Scripts/python.exe -m uv pip install --python .venv-isa/Scripts/python.exe --no-deps -e .
.venv-torch/Scripts/Activate.ps1
./scripts/prepare-qwen-environment.ps1
```

The preparation script uses uv from the activated frontend environment. It
checks out the pinned tinycoral source and applies its Windows patch. Supply
the runtime and model paths in Coral Studio Settings before connecting. See
[model setup](qwen-full-tpu.md) for revisions and licenses.

## Validation

Offline frontend tests do not access the USB device:

```powershell
.venv-torch/Scripts/python.exe -m pytest tests -q
```

Instruction-emulator tests require the pinned ISA dependencies. Install pytest
in that environment and run the `test_isa_*`, `test_v4_compute.py` and
`test_speculative_blocks.py` suites there. Offline checks validate emitted
programs and host scheduling; hardware validation is separate. Scripts named
`verify-*`, `probe-*` and `benchmark-*` may claim the physical device and must
run sequentially with other Coral owners closed.

Historical A/B scripts sometimes require a separately supplied older source
checkout through `--baseline-root` or the documented `release/` location.
Older snapshots, private measurements and model fixtures are not bundled.

## Public source

The public tree is created from a strict allowlist:

```powershell
python scripts/prepare-release.py --destination release/public-source --report reports/public-audit.json
```

Use an empty destination. The script refuses private file types, credentials
and user paths. Review every newly allowlisted file and scan the complete
staged Git tree before publishing. Local settings, conversations, reports,
environments, downloads, weights, native binaries and credentials are excluded.
Use a project contributor identity for commits when personal author metadata
should not be public. Git hosting still displays the publishing account.

Project Python code is MIT licensed. Preserve the separate upstream MIT and
LGPL notices described in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
