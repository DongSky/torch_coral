"""Physical temporal convolution against V2's per-position implementation."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch_coral.isa_batched import BatchedMatrixTPU


if __name__=='__main__':
    rng=np.random.default_rng(912); rows=[]
    with BatchedMatrixTPU('.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2,clock='max') as b:
        for steps,channels in ((2,128),(13,6144),(16,6144)):
            history=rng.normal(size=(3,channels)).astype(np.float32)
            projections=rng.normal(size=(steps,channels)).astype(np.float32)
            kernel=rng.normal(size=(channels,4)).astype(np.float32)
            timeline=np.concatenate([history,projections])
            expected=np.stack([b.convolution_silu(timeline[t:t+4],kernel) for t in range(steps)])
            actual=b.convolution_many(history,projections,kernel)
            np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
            rows.append(dict(steps=steps,channels=channels,status='bitwise PASS')); print(rows[-1],flush=True)
    Path('reports/v3-convolution.json').write_text(json.dumps(rows,indent=2))
