# Preparing a release

The distribution is `torch-coral`; the Python module is `torch_coral`. PyPI
normalizes runs of `-`, `_` and `.` to `-` and lowercases names. Those spellings
are the same project name, not alternatives. Check the public JSON and Simple
APIs immediately before publishing. A 404 means no visible project was found;
it does not reserve a name or guarantee PyPI will accept registration.

Use the source-repository release preparation tool to create a separate,
allowlisted public tree. It includes Python code, generic documentation, tests,
small synthetic examples and selected build helpers. It excludes experiment
reports, user paths, GUI sessions, virtual environments, downloaded software,
model weights, calibration tensors, logs, credentials and native binaries.
The original workspace is not an upload directory.

Build a wheel and sdist, check metadata/readme with `twine check`, inspect every
archive member, then install the wheel into a clean environment outside the
source directory. Verify imports and CLI entry points there; configure external
worker/converter/compiler paths and run a real compile/load inference smoke test.
Rebuild the wheel from the sdist as a separate check. Do not include PyPI tokens
or a developer-specific `.pypirc` in the project.

The release preparation workflow does not upload or publish anything. Actual
publication requires the maintainer's account, project ownership and credentials
or a separately configured PyPI Trusted Publisher. No repository/account URL is
invented in package metadata. Local checks do not prove ownership of the name.

The project code uses MIT. Optional libusb patch files are licensed separately
and excluded from PyPI archives. Runtime/compiler/SDK binaries are external;
their upstream notices remain with their distributions. This repository does
not redistribute model weights.
