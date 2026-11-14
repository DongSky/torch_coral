"""Compare host call paths on identical artifacts. Run one SDK owner at a time."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def stats(values):
    return {key: float(np.percentile(values, percentile))
            for key, percentile in [('p50_ms', 50), ('p95_ms', 95)]}


def fingerprint(arrays):
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def measure(name, function, timing, repeats, warmup):
    for _ in range(warmup):
        function()
    wall, invoke = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = function()
        wall.append((time.perf_counter() - start) * 1000)
        invoke.append(timing()['invoke_ms'])
    arrays = [v.detach().cpu().numpy() if hasattr(v, 'detach') else v for v in outputs]
    return dict(path=name, end_to_end=stats(wall), delegate=stats(invoke),
                output_sha256=fingerprint(arrays), repeats=repeats, warmup=warmup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', nargs='+')
    parser.add_argument('--mode', choices=['legacy', 'frontend', 'direct'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-version')
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--source-root', help='Only for an uninstalled development checkout')
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error('repeats and warmup must be positive')
    if args.source_root:
        sys.path.insert(0, str(Path(args.source_root).resolve()))
    import torch_coral
    if args.expected_version and torch_coral.__version__ != args.expected_version:
        raise RuntimeError('Wrong installed version: ' + torch_coral.__version__)
    report = dict(version=torch_coral.__version__, python=sys.version.split()[0], mode=args.mode, models=[])
    for item in args.artifacts:
        folder = Path(item).resolve()
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        rng = np.random.default_rng(12345)
        floats = [rng.normal(0, .5, spec['shape']).astype(np.float32) for spec in manifest['inputs']]
        row = dict(name=folder.name, model_sha256=manifest['sha256'], input_sha256=fingerprint(floats), paths=[])
        def raw_inputs(description):
            by_name = {d['name']: d for d in description['inputs']}
            by_input = {s['tflite_name']: (a, s) for a, s in zip(floats, manifest['inputs'])}
            values = []
            for d in description['inputs']:
                array, spec = by_input[d['name']]
                scale, zero = d['quantization']
                dtype = np.dtype(d['dtype'])
                limits = np.iinfo(dtype)
                values.append(np.clip(np.rint(array.transpose(spec['perm']) / scale) + zero,
                                      limits.min, limits.max).astype(dtype))
            return values
        if args.mode == 'direct':
            from torch_coral_runtime import DirectSession
            with DirectSession(folder) as session:
                q = raw_inputs(session.describe())
                out = [np.empty(d['shape'], d['dtype']) for d in session.describe()['outputs']]
                def classic():
                    for value, detail in zip(q, session.inputs):
                        session.interpreter.set_tensor(detail['index'], value)
                    session.invoke()
                    return [session.interpreter.get_tensor(d['index']) for d in session.outputs]
                def buffered():
                    for i, value in enumerate(q): session.input_buffer(i).copy_from(value)
                    session.invoke()
                    for i, dest in enumerate(out): session.output_buffer(i).read_into(dest)
                    return out
                for name, fn in [('sdk_set_get_raw', classic), ('direct_preallocated_raw', buffered),
                                 ('direct_float', lambda: session.run_float(floats))]:
                    row['paths'].append(measure(name, fn, lambda: session.last_timing, args.repeats, args.warmup))
                row['hardware'] = session.hardware
        else:
            import torch
            tensors = [torch.from_numpy(a) for a in floats]
            with torch_coral.load(folder) as pipe, torch.no_grad():
                row['paths'].append(measure('v1_npz_float' if args.mode == 'legacy' else 'v2_pipe_float',
                    lambda: [pipe(*tensors)], lambda: pipe.last_timing, args.repeats, args.warmup))
                if args.mode == 'frontend':
                    with torch_coral.load(folder, transport='shared_memory') as shared:
                        row['paths'].append(measure('v2_shared_float', lambda: [shared(*tensors)],
                            lambda: shared.last_timing, args.repeats, args.warmup))
                        with shared.prepare(raw=True) as prepared:
                            q = raw_inputs(prepared.description)
                            out = [np.empty(b.shape, b.dtype) for b in prepared.outputs]
                            row['paths'].append(measure('v2_prepared_raw', lambda: prepared.run(q, out=out),
                                lambda: prepared.last_timing, args.repeats, args.warmup))
                            row['paths'].append(measure('v2_pipe_raw', lambda: pipe.run_raw(*q),
                                lambda: pipe.last_timing, args.repeats, args.warmup))
                row['hardware'] = pipe.hardware
        report['models'].append(row)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(row['name'], [(r['path'], round(r['end_to_end']['p50_ms'], 4)) for r in row['paths']], flush=True)


if __name__ == '__main__': main()
