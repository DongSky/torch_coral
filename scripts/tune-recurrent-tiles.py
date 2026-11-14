"""Measure Turbo V2 tile thresholds on the physical TPU, with bitwise checks."""
import json
import argparse
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch_coral.isa_streamed import StreamedMatrixTPU, time_chunks
from torch_coral.isa_mxu import MatrixTPU


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clock',choices=['high','max'],default='max')
    parser.add_argument('--output',default='reports/recurrent-v2-final-tiles.json')
    args=parser.parse_args()
    rng = np.random.default_rng(517)
    rows = []
    with StreamedMatrixTPU('.tools/models/qwen35-gguf/int8-matrices', components=1, output_digits=2,clock=args.clock) as b:
        for steps in (1, 13):
            state = rng.normal(size=(16,128,128)).astype(np.float32)*np.float32(.1)
            k,q,v = rng.normal(size=(3,steps,16,128)).astype(np.float32)*np.float32(.1)
            beta = np.full((steps,16),.3,np.float32); decay = np.full((steps,16),.9,np.float32)
            expected_state = state.copy(); expected = []
            start = time.perf_counter(); before = b.calls
            for t in range(steps):
                updated, output = [], []
                for h in range(16):
                    s,y = MatrixTPU.recurrent(b,expected_state[h],k[t,h],q[t,h],v[t,h],beta[t,h],decay[t,h])
                    updated.append(s); output.append(y)
                expected_state = np.stack(updated); expected.append(np.stack(output))
            print(dict(steps=steps, baseline_ms=(time.perf_counter()-start)*1000, baseline_calls=b.calls-before), flush=True)
            expected = np.stack(expected)
            for cap in (8, 10, 12, 13, 16, 18, 20, 22, 24, 26, 28, 29):
                b.stream_columns = cap; samples = []
                for repeat in range(7):
                    before=b.calls; start=time.perf_counter()
                    actual_state, actual = b.recurrent_many(state,k,q,v,beta,decay)
                    elapsed=(time.perf_counter()-start)*1000
                    np.testing.assert_array_equal(actual_state.view(np.uint32),expected_state.view(np.uint32))
                    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
                    if repeat >= 2: samples.append(elapsed)
                row=dict(steps=steps,clock=args.clock,heads=16, column_cap=cap, time_chunks=time_chunks(128,128,steps,cap),
                         calls=b.calls-before, median_ms=statistics.median(samples), min_ms=min(samples), bitwise_equal=True)
                rows.append(row); print(row,flush=True)
                Path(args.output).write_text(json.dumps(rows,indent=2))
