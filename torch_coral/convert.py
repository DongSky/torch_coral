"""Isolated modern ONNX -> INT8 -> legacy compiler. No torch dependency."""
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import numpy as np
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter
from optimize_tflite import fold_flatten_transpose
from settings import compiler_path

def layouts(shape, target):
    rank = len(shape)
    candidates = [list(range(rank))]
    if rank in (3, 4, 5):
        candidates.insert(0, [0] + list(range(2, rank)) + [1])
    return [p for p in candidates if [shape[i] for i in p] == list(target)]

def _inverse(perm): return np.argsort(perm).tolist()

def determine_io(folder, job, float_model):
    """Discover public layouts by numerically checking against ONNX, including
    equal-sized dimensions where shapes alone cannot identify a permutation.
    """
    interp = Interpreter(model_path=str(float_model), num_threads=1)
    interp.allocate_tensors()
    ins, outs = interp.get_input_details(), interp.get_output_details()
    onnx = ort.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])
    input_map, output_map = {d["name"]: d for d in ins}, {d["name"]: d for d in outs}
    data = [np.load(folder / ("calibration_%d.npy" % i), allow_pickle=False) for i in range(len(ins))]
    for specs, details in [(job["inputs"], input_map), (job["outputs"], output_map)]:
        for spec in specs:
            if spec["name"] not in details:
                raise ValueError("Converter changed/removed public tensor " + spec["name"])
            spec["tflite_name"] = spec["name"]
    choices = [layouts(s["shape"], input_map[s["name"]]["shape"]) for s in job["inputs"]]
    if not all(choices): raise ValueError("Unknown converter input layout")
    if np.prod([len(c) for c in choices]) > 64:
        raise ValueError("Too many ambiguous input layouts; wrap the model with fewer inputs")
    # Structured boundary samples can all be symmetric and cannot disambiguate
    # layouts with equal-sized dimensions. Check spread-out real samples too.
    sample_indices = np.linspace(0, len(data[0]) - 1, min(len(data[0]), 8), dtype=int).tolist()
    refs = [onnx.run([s["name"] for s in job["outputs"]],
                     {s["name"]: a[k] for s, a in zip(job["inputs"], data)}) for k in sample_indices]
    for perms in itertools.product(*choices):
        output_choices = [layouts(s["shape"], output_map[s["name"]]["shape"]) for s in job["outputs"]]
        for reference_index, k in enumerate(sample_indices):
            for spec, array, perm in zip(job["inputs"], data, perms):
                interp.set_tensor(input_map[spec["name"]]["index"], np.ascontiguousarray(array[k].transpose(perm)))
            interp.invoke()
            for i, spec in enumerate(job["outputs"]):
                actual = interp.get_tensor(output_map[spec["name"]]["index"])
                output_choices[i] = [p for p in output_choices[i] if
                    np.allclose(actual.transpose(_inverse(p)), refs[reference_index][i], atol=1e-4, rtol=1e-3)]
        if all(output_choices):
            for spec, perm in zip(job["inputs"], perms): spec["perm"] = list(perm)
            for spec, options in zip(job["outputs"], output_choices): spec["inverse_perm"] = _inverse(options[0])
            return data
    raise ValueError("Float ONNX -> TFLite numerical parity failed. Inspect converter output before quantizing.")

def run_converter(folder, *args):
    command = [sys.executable, str(Path(__file__).with_name("onnx2tf_entry.py")), "-i", str(folder / "model.onnx"),
               "-o", str(folder / "converted"), "--non_verbose", *args]
    subprocess.run(command, check=True, timeout=600)

def compile_tflite(folder, job, model):
    compiler = compiler_path(job.get('compiler'), job.get('operator_strategy'))
    args = [compiler, "-s", "-o", str(folder), str(model)]
    if os.name == "nt":
        def linux(path):
            if str(path).startswith('/'):
                return str(path)
            return subprocess.check_output(["wsl.exe", "-d", job["wsl_distro"], "--exec", "wslpath", "-a", Path(path).as_posix()],
                                           text=True).strip()
        # No shell interpolation: paths and compiler flags remain distinct arguments.
        command = ["wsl.exe", "-d", job["wsl_distro"], "--exec", linux(compiler),
                   "-s", "-o", linux(folder), linux(model)]
    else:
        command = args
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding="utf-8", errors="replace", timeout=240)
    (folder / "compiler.log").write_text(result.stdout, encoding="utf-8")
    print(result.stdout, flush=True)
    if result.returncode: raise RuntimeError("Edge TPU compiler failed; see compiler.log")
    rows = re.findall(r"^\s*([A-Z][A-Z_0-9]*)\s+(\d+)\s+(.+?)\s*$", result.stdout, re.M)
    operations = [{"op": op, "count": int(count), "status": status} for op, count, status in rows]
    mapped = sum(r["count"] for r in operations if r["status"] == "Mapped to Edge TPU")
    cpu = sum(r["count"] for r in operations if r["status"] != "Mapped to Edge TPU")
    if mapped == 0: raise RuntimeError("Compiler mapped no operations to Edge TPU")
    if cpu and not job["allow_partial"]:
        raise RuntimeError("%d operations remain on CPU. Inspect compiler.log, change the model, or explicitly set allow_partial=True." % cpu)
    compiled = folder / (model.stem + "_edgetpu.tflite")
    if not compiled.is_file(): raise RuntimeError("Compiler produced no model")
    shutil.copyfile(compiled, folder / "model_edgetpu.tflite")
    version = re.search(r"Edge TPU Compiler version ([^\n]+)", result.stdout)
    return {"version": version.group(1).strip() if version else "unknown",
            "mapped_operations": mapped, "cpu_operations": cpu, "operations": operations}

def validate_quantized(folder, job, model, data, report_name='quantization-validation.json'):
    interp = Interpreter(model_path=str(model), num_threads=1)
    interp.allocate_tensors()
    source = folder / 'model.original.onnx'
    if not source.exists(): source = folder / 'model.onnx'
    reference = ort.InferenceSession(str(source), providers=['CPUExecutionProvider'])
    inputs = {d['name']: d for d in interp.get_input_details()}
    outputs = {d['name']: d for d in interp.get_output_details()}
    maximum = [0.] * len(job['outputs'])
    failures = 0
    count = min(32, len(data[0]))
    for k in range(count):
        expected = reference.run([s['name'] for s in job['outputs']],
            {s['name']: a[k] for s, a in zip(job['inputs'], data)})
        for spec, array in zip(job['inputs'], data):
            detail = inputs[spec['name']]
            scale, zero = detail['quantization']
            limits = np.iinfo(detail['dtype'])
            value = np.clip(np.rint(array[k].transpose(spec['perm']) / scale) + zero,
                            limits.min, limits.max).astype(detail['dtype'])
            interp.set_tensor(detail['index'], value)
        interp.invoke()
        for j, (spec, value) in enumerate(zip(job['outputs'], expected)):
            detail = outputs[spec['name']]
            scale, zero = detail['quantization']
            actual = (interp.get_tensor(detail['index']).astype(np.float32) - zero) * scale
            actual = actual.transpose(spec['inverse_perm'])
            maximum[j] = max(maximum[j], float(np.max(np.abs(actual - value))))
            if not np.all(np.isfinite(actual)) or not np.allclose(actual, value,
                    atol=job['quantization_atol'], rtol=job['quantization_rtol']):
                failures += 1
    result = {'status': 'PASS' if failures == 0 else 'FAILED', 'samples': count,
              'max_abs_errors': maximum, 'failed_sample_outputs': failures,
              'atol': job['quantization_atol'], 'rtol': job['quantization_rtol'],
              'reference': source.name, 'scope': 'Representative inputs; full-model task quality is not measured.'}
    (folder / report_name).write_text(json.dumps(result, indent=2), encoding='utf-8')
    if failures:
        raise ValueError('INT8 operator strategy exceeds numerical error tolerance; see quantization-validation.json')
    return result

def main():
    folder = Path(sys.argv[1]).resolve()
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    strategy_report = None
    if job.get('operator_strategy') == 'coral':
        os.environ['TORCH_CORAL_OPERATOR_STRATEGY'] = 'coral'
        from rewrite_onnx import rewrite
        strategy_report = rewrite(folder, job)
        (folder / 'operator-rewrites.json').write_text(json.dumps(strategy_report, indent=2), encoding='utf-8')
    run_converter(folder)
    data = determine_io(folder, job, folder / "converted" / "model_float32.tflite")
    calibration_args = []
    for i, (spec, array) in enumerate(zip(job["inputs"], data)):
        perm = [0] + [axis + 1 for axis in spec["perm"]]
        # Converter calibration data uses one leading sample dimension, while
        # onnx2tf consumes batch=1 per calibration record.
        if spec["shape"][0] != 1:
            raise ValueError("INT8 calibration currently requires static batch size 1")
        array = array.transpose(perm)[:, 0]
        path = folder / ("converter_calibration_%d.npy" % i)
        np.save(path, array)
        calibration_args += ["-cind", spec["name"], str(path), "0", "1"]
    run_converter(folder, "-oiqt", *calibration_args)
    quant = folder / "converted" / "model_full_integer_quant.tflite"
    optimized, rewrites = fold_flatten_transpose(quant.read_bytes())
    if rewrites:
        # Prove the rewrite on actual representative tensors using raw quantized
        # results before asking the hardware compiler to accept the new graph.
        before = Interpreter(model_content=quant.read_bytes(), num_threads=1)
        after = Interpreter(model_content=optimized, num_threads=1)
        before.allocate_tensors(); after.allocate_tensors()
        for k in range(min(8, len(data[0]))):
            for interp in (before, after):
                inputs = {d["name"]: d for d in interp.get_input_details()}
                for spec, array in zip(job["inputs"], data):
                    detail = inputs[spec["name"]]
                    scale, zero = detail["quantization"]
                    limits = np.iinfo(detail["dtype"])
                    value = np.clip(np.rint(array[k].transpose(spec["perm"]) / scale) + zero, limits.min, limits.max).astype(detail["dtype"])
                    interp.set_tensor(detail["index"], value)
                interp.invoke()
            for a, b in zip(before.get_output_details(), after.get_output_details()):
                np.testing.assert_array_equal(before.get_tensor(a["index"]), after.get_tensor(b["index"]))
        quant = folder / "converted" / "model_optimized.tflite"
        quant.write_bytes(optimized)
    validation = None
    if strategy_report is not None:
        from quantization_strategy import align_domains
        aligned, domain_records = align_domains(quant.read_bytes())
        strategy_report['quantization_rewrites'] = domain_records
        quant = folder / 'converted/model_coral_strategy.tflite'
        quant.write_bytes(aligned)
        validation = validate_quantized(folder, job, quant, data)
    report = compile_tflite(folder, job, quant)
    manifest = {**job, "compiler_report": report, "model_file": "model_edgetpu.tflite",
                "quantized_model_file": quant.relative_to(folder).as_posix(),
                "sha256": hashlib.sha256((folder / "model_edgetpu.tflite").read_bytes()).hexdigest(),
                "onnx_sha256": hashlib.sha256((folder / "model.onnx").read_bytes()).hexdigest(),
                "converter_versions": {n: importlib.metadata.version(n) for n in ["onnx2tf", "onnx", "ai-edge-litert"]},
                "float_conversion_parity": "PASS", "compatibility_rewrites": rewrites,
                "converter_fixes": ["onnx2tf-2.6.9: depthwise per-channel bias quantization axis=0",
                                    "INT8-only export; skip unrelated INT16 activation variants"]}
    if strategy_report is not None:
        manifest.update(operator_rewrites=strategy_report, quantization_validation=validation,
            original_onnx_sha256=hashlib.sha256((folder / 'model.original.onnx').read_bytes()).hexdigest())
        manifest['converter_fixes'] += ['per-tensor BATCH_MATMUL, including dynamic RHS',
            'native RSQRT fusion and INT8 opcode versions',
            'pre-quantization alias-domain union; explicit reduction requantization']
        (folder / 'operator-rewrites.json').write_text(json.dumps(strategy_report, indent=2), encoding='utf-8')
    # Publish only when conversion, numerical validation and compiler mapping all passed.
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
