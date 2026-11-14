"""Probe larger scalar transfers without changing the public verified limits."""
import sys
import json
import statistics
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch_coral.isa_scalar import _parts
from torch_coral.isa_optimized import OptimizedMatrixTPU


def program(size):
    Emitter, E, SC, _ = _parts()
    base = E.SMEM_TOP - size
    e = Emitter(); e.scalar(SC.exe_prologue(), seqs=1)
    tiles = E._sm_single_row(e, size, base)
    e.scalar(E.S_RING_OUTFEED_RESET)
    e.scalar(SC.output_wait(e.seq, tiles), seqs=2)
    e.scalar(E.smem_to_host(base // 4, size // 4, size))
    e.sync(SC.epilogue)
    return E.executable(e.program(), size, size)


if __name__ == '__main__':
    rows = []
    with OptimizedMatrixTPU('.tools/models/qwen35-gguf/int8-matrices', components=1, output_digits=2) as b:
        for size in (1024, 2048, 4096, 8192, 12288, 16000, 16384):
            payload = np.arange(size // 4, dtype=np.uint32).tobytes()
            print('START', size, flush=True)
            exe = program(size)
            samples = []
            for repeat in range(105):
                start = time.perf_counter()
                actual = b._execute(exe, inputs=payload)
                elapsed = (time.perf_counter() - start) * 1000
                assert actual == payload
                if repeat >= 5: samples.append(elapsed)
            row = dict(bytes_per_direction=size, repeats=len(samples), bitwise_equal=True,
                       median_ms=statistics.median(samples), min_ms=min(samples),
                       p95_ms=float(np.percentile(samples, 95)),
                       combined_payload_MB_s=2*size/statistics.median(samples)/1000)
            rows.append(row); print(row, flush=True)
    Path('reports/scalar-sram-probe.json').write_text(json.dumps(rows, indent=2))
