"""Physical A/B against frozen Alpha 8 control flow, full logits and caches."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from torch_coral.isa_io import IOOptimizedMatrixTPU
from torch_coral.qwen_tpu import QwenTPU
from torch_coral.speculative import SpeculativeGreedy

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycles',type=int,default=2)
    parser.add_argument('--baseline-root',type=Path,default=ROOT/'release/torch-coral-0.3.0a8-final')
    parser.add_argument('--output',default='reports/turbo-v4-a9-compute-ab.json')
    args=parser.parse_args()
    old=load('torch_coral._a8_benchmark_engine',args.baseline_root/'torch_coral/qwen_tpu.py')
    old_spec=load('torch_coral._a8_benchmark_spec',args.baseline_root/'torch_coral/speculative.py')
    prompt=[248045,846,198,109266,248046,198,248045,74455,198,248068,271,248069,271]
    rows=[]; golden=None
    with IOOptimizedMatrixTPU(ROOT/'.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2,clock='max') as b:
        for index,mode in enumerate(['warmup-baseline','warmup-fused']+['baseline','fused','fused','baseline']*args.cycles):
            active=mode.endswith('fused')
            b.reuse_projection_quantization=b.compute_fusions=b.fused_head_selection=active
            b.native_active_bias=active
            b.native_batch_min=3 if active else 8
            b.native_large_min=8 if active else 12
            b.decode_calibration.clear()
            b._normalized_quantization.clear(); b._head_summaries.clear()
            engine=(QwenTPU if active else old.QwenTPU)(ROOT/'.tools/models/qwen35-gguf/fp32-prepared',b)
            start=time.perf_counter(); before=b.calls
            scores=engine.prefill(prompt); first=engine.greedy(scores)
            prefill=time.perf_counter()-start; logits=[scores.copy()]
            start=time.perf_counter(); scores=engine.step(first); anchor=engine.greedy(scores)
            ordinary=time.perf_counter()-start; logits.append(scores.copy())
            decoder=old_spec.SpeculativeGreedy(engine,prompt+[first],adaptive=False)
            start=time.perf_counter(); emitted,verified,stop=decoder.advance(anchor,4,set())
            pair=time.perf_counter()-start; logits.extend(v.copy() for v in verified)
            states={layer:tuple(np.array(v,copy=True) for v in state) for layer,state in engine.caches.items()}
            if golden is None: golden=logits,states
            else:
                assert len(logits)==len(golden[0]) and set(states)==set(golden[1])
                for a,e in zip(logits,golden[0]): np.testing.assert_array_equal(a.view(np.uint32),e.view(np.uint32))
                for layer,state in states.items():
                    for a,e in zip(state,golden[1][layer]): np.testing.assert_array_equal(a.view(np.uint32),e.view(np.uint32))
            assert first==109266 and anchor==6115 and emitted==[116577,108888] and not stop
            row=dict(iteration=index,mode=mode,prefill_s=prefill,ordinary_decode_s=ordinary,speculative_pair_s=pair,
                     tpu_calls=b.calls-before,all_logits_and_caches_bitwise_equal=True,matrix_stats=dict(b.matrix_stats))
            print(json.dumps(row),flush=True)
            if not mode.startswith('warmup'): rows.append(row)
            result=dict(status='RUNNING',baseline='Frozen Alpha 8 engine, Alpha 8 backend paths',clock='max',
                        order='A B B A after both full warmups',rows=rows)
            Path(args.output).write_text(json.dumps(result,indent=2))
        result['summary']={}
        for field in ('prefill_s','ordinary_decode_s','speculative_pair_s'):
            a=statistics.median(r[field] for r in rows if r['mode']=='baseline')
            new=statistics.median(r[field] for r in rows if r['mode']=='fused')
            result['summary'][field]=dict(baseline_s=a,fused_s=new,elapsed_reduction_fraction=1-new/a)
        result['status']='bitwise PASS'
        Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result['summary']),flush=True)

if __name__=='__main__': main()
