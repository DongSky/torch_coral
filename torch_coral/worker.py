"""Private Python 3.9 process. Original delegate ABI, one owner per device."""
import importlib.util
import json
import sys
import traceback
from collections import OrderedDict
from pathlib import Path
from protocol import encode, decode, read_packet, write_packet
from shared import SharedArrays

# Load only our packages; never add the parent's site-packages to sys.path.
# That would shadow the worker's NumPy with a binary for another Python ABI.
if 'torch_coral' not in sys.modules:
    root = Path(__file__).resolve().parent
    for name, folder in [('torch_coral', root), ('torch_coral_runtime', root.parent / 'torch_coral_runtime')]:
        spec = importlib.util.spec_from_file_location(name, folder / '__init__.py', submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
from torch_coral_runtime import DirectSession, normalize_options
from torch_coral_runtime.direct import _sdk


def main():
    _, _, hardware = _sdk()
    if not hardware['devices']:
        raise RuntimeError('No real Edge TPU found; CPU fallback is disabled')
    models, bindings = OrderedDict(), {}
    options = normalize_options(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
    write_packet(sys.stdout.buffer, encode(ready=True, **hardware))
    try:
        while True:
            try:
                payload = read_packet(sys.stdin.buffer)
            except EOFError:
                return
            try:
                arrays, request = decode(payload)
                action = request.get('action', 'infer')
                if action == 'release_shared':
                    item = bindings.pop(request['token'], None)
                    if item:
                        item['memory'].close()
                    write_packet(sys.stdout.buffer, encode(released=True))
                    continue
                if action == 'invoke_shared':
                    binding = bindings[request['token']]
                    bound = binding['session']
                    memory = binding['memory']
                    run = bound.run_raw if binding['raw'] else bound.run_float
                    run(memory.arrays['inputs'], out=memory.arrays['outputs'])
                    write_packet(sys.stdout.buffer, encode(**bound.last_timing))
                    continue
                folder = Path(request['artifact_dir'])
                # A recompiled artifact must not reuse a stale interpreter.
                fingerprint = (folder / 'manifest.json').read_bytes()
                key = (str(folder), fingerprint, request.get('num_threads', 1))
                if key not in models:
                    models[key] = DirectSession(folder, device=sys.argv[1], delegate_options=options,
                                                num_threads=key[2])
                    if len(models) > 16:
                        models.popitem(last=False)
                models.move_to_end(key)
                session = models[key]
                raw = request.get('raw', False)
                if action == 'describe':
                    write_packet(sys.stdout.buffer, encode(description=session.describe(raw), hardware=session.hardware))
                elif action == 'register_shared':
                    token = request['token']
                    if token in bindings:
                        raise ValueError('Shared binding already registered')
                    memory = SharedArrays(session.describe(raw), name=request['shared_name'], untrack=True)
                    bindings[token] = dict(memory=memory, session=session, raw=raw)
                    write_packet(sys.stdout.buffer, encode(registered=True))
                elif action == 'infer':
                    run = session.run_raw if raw else session.run_float
                    results = run(arrays)
                    write_packet(sys.stdout.buffer, encode(results, **session.last_timing))
                else:
                    raise ValueError('Unknown worker action: ' + str(action))
            except Exception:
                write_packet(sys.stdout.buffer, encode(error=traceback.format_exc()))
    finally:
        for binding in bindings.values():
            binding['memory'].close()
        bindings.clear()
        models.clear()
        session = bound = binding = run = item = None


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write_packet(sys.stdout.buffer, encode(error=traceback.format_exc()))
        sys.exit(1)
