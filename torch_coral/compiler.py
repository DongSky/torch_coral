import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from .runtime import ROOT, CoralModule
from .settings import compiler_path, python_executable, read_config

def flatten_output(value, leaves):
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32:
            raise TypeError("Only float32 tensor outputs are supported")
        index = len(leaves)
        leaves.append(value)
        return {"kind": "tensor", "index": index}
    if type(value) in (tuple, list):
        return {"kind": "tuple" if type(value) is tuple else "list",
                "children": [flatten_output(v, leaves) for v in value]}
    if type(value) is dict and all(isinstance(k, str) for k in value):
        return {"kind": "dict", "keys": list(value),
                "children": [flatten_output(v, leaves) for v in value.values()]}
    raise TypeError("Outputs must be tensors or tuple/list/string-keyed dict of tensors")

class _FlatOutput(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, *args):
        leaves = []
        flatten_output(self.module(*args), leaves)
        return tuple(leaves)

def compile(model, example_inputs, calibration_data, *, artifact_dir,
            converter_python=None, compiler=None, wsl_distro=None,
            allow_partial=False, max_calibration_samples=256,
            operator_strategy=None, approximations=None,
            quantization_atol=0.15, quantization_rtol=0.15, **runtime_options):
    """Export -> calibrated INT8 -> Edge TPU -> torch.nn.Module.

    calibration_data yields positional-input tuples (or a Tensor for one input),
    already preprocessed exactly like inference, each with the static example
    shape. Labels must be removed. No synthetic calibration is substituted.
    By default every compiled operation must map to the TPU.
    operator_strategy='coral' enables bounded static rewrites and the v16
    quantizer fixes. Nonlinear approximations require explicit configurations.
    Their combined rewrite/INT8 error is checked before publishing the artifact.
    """
    if model.training:
        raise ValueError("Call model.eval() before compiling")
    if operator_strategy not in (None, 'coral'):
        raise ValueError("operator_strategy must be None or 'coral'")
    if approximations is not None and not isinstance(approximations, dict):
        raise TypeError('approximations must be a dictionary of function names and bounded configurations')
    if approximations and any(name not in {'Sin', 'Cos', 'Exp', 'Softplus', 'Erf'} for name in approximations):
        raise ValueError('Supported bounded approximations: Sin, Cos, Exp, Softplus, Erf')
    if approximations and operator_strategy != 'coral':
        raise ValueError("approximations require operator_strategy='coral'")
    if quantization_atol < 0 or quantization_rtol < 0:
        raise ValueError('Quantization error tolerances must be nonnegative')
    if isinstance(example_inputs, torch.Tensor): example_inputs = (example_inputs,)
    examples = tuple(example_inputs)
    if not examples or any(not isinstance(x, torch.Tensor) or x.dtype != torch.float32 for x in examples):
        raise TypeError("Example inputs must be float32 tensors")
    if any(x.ndim == 0 or any(d <= 0 for d in x.shape) for x in examples):
        raise ValueError("Static, nonempty tensor dimensions required")
    if any(x.shape[0] != 1 for x in examples):
        raise ValueError("Coral compilation requires static batch size 1")
    if max_calibration_samples < 1: raise ValueError("max_calibration_samples must be positive")
    examples = tuple(x.detach().cpu().clone() for x in examples)
    samples = []
    for item in itertools.islice(iter(calibration_data), max_calibration_samples):
        values = (item,) if isinstance(item, torch.Tensor) else tuple(item)
        if len(values) != len(examples): raise ValueError("Calibration input count differs from examples")
        row = []
        for value, example in zip(values, examples):
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.shape != example.shape:
                raise ValueError("Every calibration tensor must have the example shape and float32 dtype")
            array = value.detach().cpu().numpy().copy()
            if not np.isfinite(array).all(): raise ValueError("Calibration contains NaN or infinity")
            row.append(array)
        samples.append(row)
    if not samples: raise ValueError("Nonempty representative calibration_data is required")
    compiler = compiler_path(compiler, operator_strategy)
    python = python_executable('converter', converter_python)
    folder = Path(artifact_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "manifest.json").exists():
        raise FileExistsError("Artifact already exists. Use torch_coral.load() or a new artifact_dir.")
    source = copy.deepcopy(model).cpu().eval()
    with torch.no_grad():
        leaves = []
        output_tree = flatten_output(source(*examples), leaves)
    if not leaves: raise ValueError("Model has no tensor outputs")
    input_specs = [{"name": "input_%d" % i, "shape": list(x.shape)} for i, x in enumerate(examples)]
    output_specs = [{"name": "output_%d" % i, "shape": list(x.shape)} for i, x in enumerate(leaves)]
    # Modern Dynamo/torch.export ONNX path; source module and its weights are not mutated.
    torch.onnx.export(_FlatOutput(source).eval(), examples, str(folder / "model.onnx"),
                      input_names=[s["name"] for s in input_specs],
                      output_names=[s["name"] for s in output_specs],
                      opset_version=18, dynamo=True, external_data=False, verbose=False)
    for i in range(len(examples)):
        np.save(folder / ("calibration_%d.npy" % i), np.stack([row[i] for row in samples]))
    job = {"format_version": 1, "inputs": input_specs, "outputs": output_specs,
           "output_tree": output_tree, "allow_partial": allow_partial,
           "wsl_distro": wsl_distro or os.environ.get('CORAL_WSL_DISTRO') or read_config().get('wsl_distro', 'Ubuntu'), "compiler": str(compiler) if compiler else None,
           "torch_version": torch.__version__, "calibration_samples": len(samples)}
    job.update(operator_strategy=operator_strategy, approximations=approximations or {},
               quantization_atol=quantization_atol, quantization_rtol=quantization_rtol,
               operator_strategy_revision=1 if operator_strategy else None)
    (folder / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    env = dict(os.environ, PYTHONUTF8="1")
    log = folder / "build.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run([python, str(Path(__file__).with_name("convert.py")), str(folder)],
                                stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=900)
    if result.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError("Coral compilation failed; see %s\n%s" % (log, tail))
    return CoralModule(folder, **runtime_options)
