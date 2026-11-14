"""Compare real Turbo / Turbo V2 chat graphs and every final recurrent state."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from torch_coral.isa_optimized import OptimizedMatrixTPU
from torch_coral.isa_streamed import StreamedMatrixTPU
from torch_coral.qwen_tpu import QwenTPU

GREETING=[248045,846,198,109266,248046,198,248045,74455,198,248068,271,248069,271]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extended',action='store_true')
    parser.add_argument('--include-max',action='store_true')
    parser.add_argument('--v3',action='store_true',help='Compare V2 max and experimental V3 max')
    parser.add_argument('--output',default='reports/turbo-v2-chat.json')
    args=parser.parse_args()
    cases={'greeting':GREETING}
    if args.extended: cases.update(raw=[9707],boundary=GREETING+[109266,6115,198,9707])
    results={}; golden={}
    profiles=[('turbo',OptimizedMatrixTPU,'high'),('turbo-v2',StreamedMatrixTPU,'high')]
    if args.include_max: profiles.append(('turbo-v2-max',StreamedMatrixTPU,'max'))
    if args.v3:
        from torch_coral.isa_batched import BatchedMatrixTPU
        profiles=[('turbo-v2',StreamedMatrixTPU,'max'),('turbo-v3',BatchedMatrixTPU,'max')]
    for profile,backend,clock in profiles:
        with backend(ROOT/'.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2,clock=clock) as b:
            engine=QwenTPU(ROOT/'.tools/models/qwen35-gguf/fp32-prepared',b)
            rows=[]
            for name,tokens in cases.items():
                print('START',profile,name,flush=True); engine.reset()
                start=time.perf_counter(); before=b.calls
                logits=engine.prefill(tokens); prefill=time.perf_counter()-start
                selected=engine.greedy(logits)
                print('PREFILL',profile,name,prefill,selected,flush=True)
                row=dict(case=name,clock=clock,prompt_tokens=len(tokens),prefill_s=prefill,prefill_calls=b.calls-before,
                         selected=[selected],decode_s=[],decode_calls=[],cpu_model_fallback=False)
                saved_logits=[logits.copy()]
                for _ in range(3):
                    start=time.perf_counter(); before=b.calls
                    logits=engine.step(selected); selected=engine.greedy(logits)
                    row['decode_s'].append(time.perf_counter()-start); row['decode_calls'].append(b.calls-before)
                    row['selected'].append(selected); saved_logits.append(logits.copy())
                    print('DECODE',profile,name,row['decode_s'][-1],selected,flush=True)
                states={str(layer): tuple(np.array(x,copy=True) for x in cache) for layer,cache in engine.caches.items()}
                # Attention cache lists become dense arrays. All comparisons are raw FP32 bits.
                if profile==profiles[0][0]: golden[name]=(saved_logits,states)
                else:
                    old_logits,old_states=golden[name]
                    for a,e in zip(saved_logits,old_logits):
                        np.testing.assert_array_equal(a.view(np.uint32),e.view(np.uint32))
                    for layer,cache in states.items():
                        for a,e in zip(cache,old_states[layer]):
                            np.testing.assert_array_equal(a.view(np.uint32),e.view(np.uint32))
                    row['logits_and_all_states_bitwise_equal']=True
                row['decode_median_s']=statistics.median(row['decode_s'])
                row['decode_tokens_s']=1/row['decode_median_s']
                rows.append(row); results[profile]=rows
                Path(args.output).write_text(json.dumps(results,indent=2))
                print('PASS',row,flush=True)


if __name__=='__main__': main()
