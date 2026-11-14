"""Compare whole native batch projections with V2, including both readout digits."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch_coral.gguf_weights import WeightStore
from torch_coral.isa_batched import BatchedMatrixTPU
from torch_coral.isa_streamed import StreamedMatrixTPU


if __name__=='__main__':
    rng=np.random.default_rng(321); rows=[]
    store=WeightStore('.tools/models/qwen35-gguf/fp32-prepared')
    with BatchedMatrixTPU('.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2,clock='max',native_batch_min=2) as b:
        for name in ('blk.0.ffn_up.weight','blk.0.ffn_down.weight','blk.0.attn_qkv.weight'):
            for count in (2,4,8,13,16):
                x=rng.normal(size=(count,store[name].shape[1])).astype(np.float32)
                print('START',name,count,flush=True)
                # Warm every program/parameter layout before recording medians.
                times=[[],[]]
                for repeat in range(4):
                    b._resident=None; b._allocation=None; b._batch_resident=None
                    start=time.perf_counter(); old=StreamedMatrixTPU._project(b,name,store[name],x)
                    t_old=time.perf_counter()-start
                    b._resident=None; b._allocation=None; b._batch_resident=None
                    start=time.perf_counter(); new=b._project(name,store[name],x)
                    t_new=time.perf_counter()-start
                    np.testing.assert_array_equal(old.view(np.uint32),new.view(np.uint32))
                    if repeat: times[0].append(t_old); times[1].append(t_new)
                row=dict(name=name,batch=count,v2_ms=float(np.median(times[0]))*1000,v3_ms=float(np.median(times[1]))*1000,bitwise_equal=True)
                rows.append(row); print(row,flush=True)
                Path('reports/v3-projections.json').write_text(json.dumps(rows,indent=2))
