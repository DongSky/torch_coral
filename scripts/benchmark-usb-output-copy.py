"""A/B of the production V4 buffer adapter and the original V3 buffer pool."""
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from torch_coral.isa_io import IOOptimizedMatrixTPU
from torch_coral.qwen_tpu import QwenTPU
from torch_coral.speculative import SpeculativeGreedy


def main():
    prompt=[248045,846,198,109266,248046,198,248045,74455,198,248068,271,248069,271]
    rows=[]; golden=None
    with IOOptimizedMatrixTPU(ROOT/'.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2,clock='max',prepared_usb=False,
                             wide_quantization=False,reuse_quantization=False,compute_fusions=False,head_fusion=False,active_bias=False,
                             native_batch_min=8) as b:
        b.native_large_min=12
        original=b._legacy_readback_buffers
        direct=b.device.batch.bufs
        e=QwenTPU(ROOT/'.tools/models/qwen35-gguf/fp32-prepared',b)
        try:
            for index,mode in enumerate(('warmup','original','direct','direct','original')):
                b.device.batch.bufs=original if mode in ('warmup','original') else direct
                e.reset(); before=b.calls
                start=time.perf_counter(); scores=e.prefill(prompt); first=e.greedy(scores)
                prefill=time.perf_counter()-start; logits=[scores.copy()]
                start=time.perf_counter(); scores=e.step(first); anchor=e.greedy(scores)
                ordinary=time.perf_counter()-start; logits.append(scores.copy())
                decoder=SpeculativeGreedy(e,prompt+[first],adaptive=False)
                start=time.perf_counter(); emitted,verified,stop=decoder.advance(anchor,4,set())
                speculative=time.perf_counter()-start; logits.extend(x.copy() for x in verified)
                state={layer:tuple(np.array(x,copy=True) for x in cache) for layer,cache in e.caches.items()}
                if golden is None: golden=(logits,state)
                else:
                    for a,expected in zip(logits,golden[0]): np.testing.assert_array_equal(a.view(np.uint32),expected.view(np.uint32))
                    for layer,cache in state.items():
                        for a,expected in zip(cache,golden[1][layer]): np.testing.assert_array_equal(a.view(np.uint32),expected.view(np.uint32))
                assert first==109266 and anchor==6115 and emitted==[116577,108888] and not stop
                row=dict(iteration=index,mode=mode,prefill_s=prefill,ordinary_decode_s=ordinary,
                         speculative_pair_s=speculative,tpu_calls=b.calls-before,logits_and_all_states_bitwise_equal=True)
                print(json.dumps(row),flush=True)
                if mode!='warmup': rows.append(row)
                report=dict(status='bitwise PASS',experiment='Turbo V4 production buffer adapter versus V3 buffer pool',
                            clock='max',order='A B B A after full warmup',rows=rows)
                Path('reports/turbo-v4-copy-ab.json').write_text(json.dumps(report,indent=2))
        finally: b.device.batch.bufs=direct
    summary={}
    for field in ('prefill_s','ordinary_decode_s','speculative_pair_s'):
        old_time=statistics.median(row[field] for row in rows if row['mode']=='original')
        new_time=statistics.median(row[field] for row in rows if row['mode']=='direct')
        summary[field]=dict(original_s=old_time,direct_s=new_time,saved_s=old_time-new_time,
                            elapsed_reduction_fraction=1-new_time/old_time)
    report['summary']=summary
    Path('reports/turbo-v4-copy-ab.json').write_text(json.dumps(report,indent=2)); print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
