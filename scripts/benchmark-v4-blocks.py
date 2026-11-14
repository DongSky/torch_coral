"""Benchmark actual full generation with 2/3/4-token and adaptive verification."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from torch_coral.isa_io import IOOptimizedMatrixTPU
from torch_coral.qwen_tpu import QwenTPU
from torch_coral.speculative import SpeculativeGreedy

def bits(a,b):np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint32),np.ascontiguousarray(b).view(np.uint32))
def caches(engine):return {layer:tuple(np.array(v,copy=True) for v in cache) for layer,cache in engine.caches.items() if layer<engine.layers}
def check(engine,expected):
    assert set(caches(engine))==set(expected)
    for layer,cache in caches(engine).items():
        for a,b in zip(cache,expected[layer]):bits(a,b)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt',default='你好')
    parser.add_argument('--new-tokens',type=int,default=12)
    parser.add_argument('--modes',default='2,3,4,auto4')
    parser.add_argument('--output',default='reports/turbo-v4-a9-block-generation.json')
    args=parser.parse_args()
    from tokenizers import Tokenizer
    from jinja2.sandbox import SandboxedEnvironment
    tokenizer=Tokenizer.from_file(str(ROOT/'.tools/models/qwen35-0.8b/tokenizer.json'))
    template=SandboxedEnvironment(trim_blocks=True,lstrip_blocks=True).from_string((ROOT/'.tools/models/qwen35-0.8b/chat_template.jinja').read_text(encoding='utf-8'))
    tokens=tokenizer.encode(template.render(messages=[dict(role='user',content=args.prompt)],add_generation_prompt=True,
                                           enable_thinking=False,tools=None,add_vision_id=False),add_special_tokens=False).ids
    stops={tokenizer.token_to_id('<|im_end|>'),tokenizer.token_to_id('<|endoftext|>')}
    spec=importlib.util.spec_from_file_location('torch_coral._generation_a8_reference',ROOT/'release/torch-coral-0.3.0a8-final/torch_coral/qwen_tpu.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    rows=[]
    with IOOptimizedMatrixTPU(ROOT/'.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2,clock='max') as b:
        b.reuse_projection_quantization=b.compute_fusions=b.fused_head_selection=b.native_active_bias=False
        b.native_batch_min=8;b.native_large_min=12
        ref=old.QwenTPU(ROOT/'.tools/models/qwen35-gguf/fp32-prepared',b)
        scores=ref.prefill(tokens);first=ref.greedy(scores)
        expected=[first];golden=[scores.copy()];states=[caches(ref)]
        while len(expected)<args.new_tokens:
            scores=ref.step(expected[-1]);selected=ref.greedy(scores)
            if selected in stops:break
            expected.append(selected);golden.append(scores.copy());states.append(caches(ref))
        print('REFERENCE',expected,flush=True)
        b.reuse_projection_quantization=b.compute_fusions=b.fused_head_selection=b.native_active_bias=True
        b.native_batch_min=3;b.native_large_min=8
        for mode in args.modes.split(','):
            maximum=int(mode.removeprefix('auto'))
            adaptive=mode.startswith('auto') or mode=='2'
            b.decode_calibration.clear();b._normalized_quantization.clear();b._head_summaries.clear()
            engine=QwenTPU(ROOT/'.tools/models/qwen35-gguf/fp32-prepared',b)
            before=b.calls;begin=time.perf_counter();scores=engine.prefill(tokens)
            prefill=time.perf_counter()-begin;bits(scores,golden[0]);actual=[engine.greedy(scores)]
            decoder=SpeculativeGreedy(engine,tokens,adaptive=adaptive,max_block_tokens=maximum)
            blocks=[];started=time.perf_counter()
            while len(actual)<len(expected):
                begin=time.perf_counter();emitted,logits,stop=decoder.advance(actual[-1],len(expected)-len(actual),stops)
                elapsed=time.perf_counter()-begin
                for index,selected in enumerate(emitted):
                    assert selected==expected[len(actual)]
                    bits(logits[index],golden[len(actual)]);actual.append(selected)
                check(engine,states[len(actual)-1])
                blocks.append(dict(seconds=elapsed,tokens=emitted,committed_position=engine.position))
                print(mode,'BLOCK',blocks[-1],flush=True)
                if stop:break
            decode=time.perf_counter()-started
            assert actual==expected
            row=dict(mode=mode,adaptive=adaptive,maximum_block_tokens=maximum,prefill_seconds=prefill,
                     decode_seconds=decode,decode_tokens=len(actual)-1,seconds_per_token=decode/(len(actual)-1),
                     generated_tokens=actual,tpu_calls=b.calls-before,stats=decoder.stats,blocks=blocks,
                     all_logits_and_every_committed_state_bitwise_equal=True)
            rows.append(row);print('RESULT',json.dumps(row),flush=True)
            report=dict(status='RUNNING',prompt=args.prompt,prompt_tokens=len(tokens),clock='max',rows=rows,
                        baseline='Frozen Alpha 8 ordinary decode',notes=['All modes start with an empty calibration cache.',
                            'Fixed 3/4 modes skip the ordinary calibration step; adaptive modes include it.'])
            Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    report['status']='bitwise PASS'
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
