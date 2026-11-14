"""Tune FC versus native batches with active-only bias correction on Coral."""
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from torch_coral.gguf_weights import WeightStore
from torch_coral.isa_io import IOOptimizedMatrixTPU

def main():
    store=WeightStore(ROOT/'.tools/models/qwen35-gguf/fp32-prepared')
    rng=np.random.default_rng(649);rows=[]
    with IOOptimizedMatrixTPU(ROOT/'.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2,clock='max') as b:
        for name in ('blk.0.ffn_up.weight','blk.0.ffn_down.weight','blk.0.attn_qkv.weight','token_embd.weight'):
            for count in ((2,3,4) if name=='token_embd.weight' else (2,3,4,8,13)):
                x=rng.normal(size=(count,store[name].shape[1])).astype(np.float32)
                golden=None;times={mode:[] for mode in ('fc','native16','active')}
                for repeat in range(4):
                    for mode in times:
                        b.native_batch_min=b.native_large_min=17 if mode=='fc' else 2
                        b.native_active_bias=mode=='active'
                        b._resident=None;b._allocation=None;b._batch_resident=None
                        start=time.perf_counter();actual=b.project(name,store[name],x)
                        if name=='token_embd.weight':
                            selected=[b.argmax(row) for row in actual]
                        elapsed=time.perf_counter()-start
                        if golden is None:golden=actual.copy()
                        else:np.testing.assert_array_equal(actual.view(np.uint32),golden.view(np.uint32))
                        if repeat:times[mode].append(elapsed)
                row=dict(name=name,positions=count,median_ms={mode:statistics.median(values)*1000 for mode,values in times.items()},
                         full_logits_bitwise=True,includes_greedy=name=='token_embd.weight')
                rows.append(row);print(json.dumps(row),flush=True)
                (ROOT/'reports/turbo-v4-a9-native-tuning.json').write_text(json.dumps(dict(status='RUNNING',rows=rows),indent=2))
    (ROOT/'reports/turbo-v4-a9-native-tuning.json').write_text(json.dumps(dict(status='bitwise PASS',rows=rows),indent=2))

if __name__=='__main__':main()
