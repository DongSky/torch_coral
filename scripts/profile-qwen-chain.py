"""Profile physical Qwen execution with exclusive host/API wall time.

USB API time includes waiting for device work and is not chip compute time.
No extra model arithmetic is introduced by the profiler.
"""
import argparse
from collections import defaultdict
from functools import wraps
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from torch_coral.isa_mxu import MatrixTPU
from torch_coral.qwen_tpu import QwenTPU


class Trace:
    def __init__(self):
        self.rows = defaultdict(lambda: dict(calls=0, total_ms=0., exclusive_ms=0., bytes=0))
        self.stack = []
    def wrap(self, obj, name, label=None, byte_count=None):
        original = getattr(obj, name)
        @wraps(original)
        def call(*args, **kwargs):
            key = label(*args, **kwargs) if callable(label) else label or name
            frame = [time.perf_counter(), 0.]
            self.stack.append(frame)
            try: return original(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - frame[0]) * 1000
                self.stack.pop()
                if self.stack: self.stack[-1][1] += elapsed
                row = self.rows[key]; row['calls'] += 1
                row['total_ms'] += elapsed; row['exclusive_ms'] += elapsed - frame[1]
                if byte_count: row['bytes'] += byte_count(*args, **kwargs)
        setattr(obj, name, call)
    def report(self):
        return dict(sorted(self.rows.items(), key=lambda item: -item[1]['exclusive_ms']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--optimized', action='store_true')
    parser.add_argument('--v2', action='store_true')
    parser.add_argument('--clock',choices=['high','max'],default='high')
    parser.add_argument('--async-usb', action='store_true')
    parser.add_argument('--tokens', default='9707')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--output', default=str(ROOT / 'reports/qwen-chain-baseline.json'))
    args = parser.parse_args()
    backend = MatrixTPU
    if args.optimized:
        from torch_coral.isa_optimized import OptimizedMatrixTPU
        backend = OptimizedMatrixTPU
    if args.v2:
        from torch_coral.isa_streamed import StreamedMatrixTPU
        backend = StreamedMatrixTPU
    tokens = [int(v) for v in args.tokens.split(',')]
    start = time.perf_counter()
    with backend(ROOT / '.tools/models/qwen35-gguf/int8-matrices', components=1, output_digits=2, async_usb=args.async_usb,clock=args.clock) as device:
        trace = Trace()
        for name in ('project', 'quantize_input', '_upload_bias', '_correct', '_reconstruct', 'recurrent', 'matvec', 'matmul', 'argmax', 'reduce'):
            trace.wrap(device, name)
        for name in ('normalize', 'activation', 'convolution_silu', 'recurrent_many', 'softmax', 'rope'):
            if hasattr(device, name): trace.wrap(device, name)
        trace.wrap(device, 'op', lambda kind, *a, **kw: 'scalar.' + kind)
        trace.wrap(device, '_run', 'scalar.runtime')
        trace.wrap(device, '_execute', lambda exe, **kw: 'mxu.runtime.' + exe.type)
        trace.wrap(device.device, 'send', lambda tag, *a, **kw: 'usb.send.' + str(tag), lambda tag, data, **kw: len(data))
        trace.wrap(device.device, 'read_output', 'usb.output', lambda n, **kw: n)
        trace.wrap(device.device, 'read_event', 'usb.event', lambda **kw: 16)
        trace.wrap(device.device.batch, 'run', 'usb.async.batch',
                   lambda ops, **kw: sum(value if isinstance(value, int) else len(value) for _, value in ops))
        from coral.programs import Matmul
        trace.wrap(Matmul, 'params', 'host.pack_weights', lambda self, w, *a, **kw: w.nbytes)
        trace.wrap(Matmul, 'programs', 'host.programs')
        engine = QwenTPU(ROOT / '.tools/models/qwen35-gguf/fp32-prepared', device)
        steps = []
        for iteration in range(args.steps):
            engine.reset()
            t = time.perf_counter(); logits = engine.prefill(tokens); selected = engine.greedy(logits)
            elapsed = time.perf_counter() - t
            row = dict(iteration=iteration, seconds=elapsed, selected=selected, tpu_calls=device.calls)
            steps.append(row); print(row, flush=True)
            np.save(Path(args.output).with_suffix('.step%d.npy' % iteration), logits)
        from coral.runtime import TRAFFIC
        report = dict(tokens=tokens, optimized=args.optimized, turbo_v2=args.v2,clock=args.clock,async_usb=args.async_usb, usb_speed=device.device.usb.speed(),
                      initialization_and_runs_seconds=time.perf_counter() - start, steps=steps,
                      async_usb_traffic=dict(TRAFFIC) if args.async_usb else None,
                      profile=trace.report(), execution=engine.report())
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(list(trace.report().items())[:15]), indent=2), flush=True)


if __name__ == '__main__': main()
