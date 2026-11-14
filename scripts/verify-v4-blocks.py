"""Full-model 2/3/4-position logits and every rollback prefix versus Alpha 8."""
import importlib.util
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from torch_coral.isa_io import IOOptimizedMatrixTPU
from torch_coral.qwen_tpu import QwenTPU

def bits(a,b):np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint32),np.ascontiguousarray(b).view(np.uint32))
def clone(caches):return {layer:tuple([v.copy() for v in item] if isinstance(item,list) else item.copy() for item in cache) for layer,cache in caches.items()}
def check(actual,expected):
    assert set(actual)==set(expected)
    for layer,cache in actual.items():
        for a,b in zip(cache,expected[layer]):bits(np.asarray(a),np.asarray(b))

def main():
    spec=importlib.util.spec_from_file_location('torch_coral._block_a8_reference',ROOT/'release/torch-coral-0.3.0a8-final/torch_coral/qwen_tpu.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    prompt=[248045,846,198,109266,248046,198,248045,74455,198,248068,271,248069,271]
    results=[]
    with IOOptimizedMatrixTPU(ROOT/'.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2,clock='max') as b:
        b.reuse_projection_quantization=b.compute_fusions=b.fused_head_selection=False
        ref=old.QwenTPU(ROOT/'.tools/models/qwen35-gguf/fp32-prepared',b)
        scores=ref.prefill(prompt);tokens=[ref.greedy(scores)];initial=clone(ref.caches)
        initial_hidden=[v.copy() for v in ref.hidden_history]
        logits=[];states=[];hiddens=[]
        for _ in range(4):
            scores=ref.step(tokens[-1]);tokens.append(ref.greedy(scores));logits.append(scores.copy())
            states.append(clone(ref.caches));hiddens.append(ref.last_hidden.copy())
        print('reference complete',tokens,flush=True)
        b.reuse_projection_quantization=b.compute_fusions=b.fused_head_selection=True
        for count in (2,3,4):
            engine=QwenTPU(ROOT/'.tools/models/qwen35-gguf/fp32-prepared',b)
            engine.caches=clone(initial);engine.position=len(prompt)
            engine.hidden_history=[v.copy() for v in initial_hidden];engine.last_hidden=initial_hidden[-1].copy()
            started=time.perf_counter();before=b.calls
            actual=engine.verify_block(tokens[0],tokens[1:count])
            seconds=time.perf_counter()-started
            for index in range(count):bits(actual[index],logits[index])
            prefixes=engine.verified_prefix_caches
            for index in range(count-1):check(prefixes[index],states[index])
            check(engine.caches,states[count-1])
            final=clone(engine.caches);history=engine.hidden_history.copy()
            for committed in range(1,count+1):
                engine.caches=clone(final);engine.position=len(prompt)+count
                engine.hidden_history=history.copy();engine.last_hidden=history[-1].copy()
                engine.verified_prefix_caches=prefixes
                engine.finish_block(committed)
                check(engine.caches,states[committed-1]);bits(engine.last_hidden,hiddens[committed-1])
                assert engine.position==len(prompt)+committed==len(engine.hidden_history)
            row=dict(block_tokens=count,verification_seconds=seconds,tpu_calls=b.calls-before,
                     all_full_logits_bitwise=True,every_committed_prefix_bitwise=True)
            results.append(row);print(json.dumps(row),flush=True)
    (ROOT/'reports/turbo-v4-a9-block-prefixes.json').write_text(json.dumps(dict(status='bitwise PASS',cases=results),indent=2))

if __name__=='__main__':main()
