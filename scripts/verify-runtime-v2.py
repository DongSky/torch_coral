"""Physical-device checks for IPC buffers, concurrency, options and reopen."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from multiprocessing import shared_memory
from pathlib import Path
import numpy as np
import torch
import torch_coral


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact')
    parser.add_argument('--output', default='reports/runtime-v2-hardware.json')
    args = parser.parse_args()
    report = {'checks': {}}
    with torch_coral.load(args.artifact) as pipe, torch_coral.load(args.artifact, transport='shared_memory') as shared:
        shapes = [s['shape'] for s in pipe.manifest['inputs']]
        inputs = [torch.zeros(s, dtype=torch.float32) for s in shapes]
        with torch.no_grad():
            expected, actual = pipe(*inputs), shared(*inputs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert pipe._session is shared._session
        report['checks']['float_and_single_worker'] = 'PASS'
        bindings = [shared.prepare(raw=True) for _ in range(4)]
        description = bindings[0].description
        rng = np.random.default_rng(19)
        arrays = [[rng.integers(-64, 64, size=d['shape']).astype(d['dtype'])
                   for d in description['inputs']] for _ in range(40)]
        reference = [pipe.run_raw(*values) for values in arrays]
        def task(index):
            binding = bindings[index % len(bindings)]
            out = [np.empty(d['shape'], d['dtype']) for d in description['outputs']]
            results = binding.run(arrays[index], out=out)
            assert all(a is b for a, b in zip(results, out))
            return results
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(task, range(40)))
        for expected, actual in zip(reference, results):
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a, b)
        report['checks']['preallocated_concurrent_raw_40'] = 'PASS'
        handle = bindings[0].inputs[0]
        name = bindings[0]._memory.memory.name
        bindings[0].close()
        try:
            handle.numpy()
            raise AssertionError('Closed buffer remained usable')
        except RuntimeError:
            pass
        try:
            leaked = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            pass
        else:
            leaked.close()
            raise AssertionError('Shared memory leaked')
        report['checks']['buffer_invalidation_and_unlink'] = 'PASS'
        conflict = torch_coral.load(args.artifact, delegate_options={'Performance': 'High'})
        try:
            conflict.run_raw(*arrays[0])
            raise AssertionError('Conflicting options silently reused a session')
        except ValueError as exc:
            assert 'different delegate options' in str(exc)
        finally:
            conflict.close()
        report['checks']['option_conflict_rejected'] = 'PASS'
    # Explicit close invalidates bindings even when user handles remain alive.
    for binding in bindings:
        assert binding.closed
    report['checks']['module_close_closes_bindings'] = 'PASS'
    # Drop binding references so their worker process can release the USB device.
    bindings.clear()
    del binding, handle
    for options in ({'Performance': 'High', 'Usb.MaxBulkInQueueLength': 32}, {}):
        with torch_coral.load(args.artifact, delegate_options=options, num_threads=2) as module:
            module.run_raw(*arrays[0])
    report['checks']['options_and_release_reopen'] = 'PASS'
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__': main()
