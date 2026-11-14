"""Physical V3 acceptance, exact prefix rejection and MTP decode benchmark."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch_coral.isa_batched import BatchedMatrixTPU
from torch_coral.qwen_tpu import QwenTPU
from torch_coral.speculative import SpeculativeGreedy
from torch_coral.isa_scalar import ScalarTPU
from tokenizers import Tokenizer


def bits(a,b): np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint32),np.ascontiguousarray(b).view(np.uint32))
def caches(e): return {layer:tuple(np.array(v,copy=True) for v in cache) for layer,cache in e.caches.items() if layer<e.layers}
def check_cache(e,expected):
    for layer,cache in caches(e).items():
        for a,b in zip(cache,expected[layer]): bits(a,b)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens',default='248045,846,198,109266,248046,198,248045,74455,198,248068,271,248069,271')
    parser.add_argument('--new-tokens',type=int,default=8)
    parser.add_argument('--adaptive',action='store_true')
    parser.add_argument('--skip-forced',action='store_true')
    parser.add_argument('--prompt',default=None)
    parser.add_argument('--v4',action='store_true',help='Compare original V4 ordinary decode with prepared V4 speculation')
    parser.add_argument('--output',default='reports/turbo-v3-speculative.json')
    args=parser.parse_args(); tokens=[int(v) for v in args.tokens.split(',')]
    tokenizer=Tokenizer.from_file('.tools/models/qwen35-0.8b/tokenizer.json')
    if args.prompt is not None:
        from jinja2.sandbox import SandboxedEnvironment
        env=SandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
        template=env.from_string(Path('.tools/models/qwen35-0.8b/chat_template.jinja').read_text(encoding='utf-8'))
        tokens=tokenizer.encode(template.render(messages=[dict(role='user',content=args.prompt)],add_generation_prompt=True,
            enable_thinking=False,tools=None,add_vision_id=False),add_special_tokens=False).ids
    stops={tokenizer.token_to_id('<|im_end|>'),tokenizer.token_to_id('<|endoftext|>')}
    backend=BatchedMatrixTPU
    if args.v4:
        from torch_coral.isa_io import IOOptimizedMatrixTPU
        backend=IOOptimizedMatrixTPU
    with backend('.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2,clock='max') as b:
        class V2:
            def __getattr__(self,name):
                if name in ('convolution_many','project_indexed','draft_candidate_ids'): raise AttributeError(name)
                return getattr(b,name)
            def argmax(self,x): return ScalarTPU.argmax(b,x)
        if args.v4:
            prepared_batch,prepared_runtime=b.device.batch,b._prepared_runtime
            b.device.batch,b._prepared_runtime=b._legacy_v4_batch,None
            b.wide_projection_quantization=False
            b.reuse_projection_quantization=b.compute_fusions=b.fused_head_selection=b.native_active_bias=False
            b.native_batch_min=8;b.native_large_min=12
        else:
            b.native_batch_min=17; b.compact_loads_enabled=False
        e=QwenTPU('.tools/models/qwen35-gguf/fp32-prepared',b if args.v4 else V2())
        begin=time.perf_counter(); scores=e.prefill(tokens); prefill=time.perf_counter()-begin
        first=e.greedy(scores); baseline=[first]; baseline_scores=[scores.copy()]; times=[]; states=[]
        print('BASELINE PREFILL',prefill,first,flush=True)
        for index in range(args.new_tokens-1):
            begin=time.perf_counter(); scores=e.step(baseline[-1]); token=e.greedy(scores)
            elapsed=time.perf_counter()-begin; times.append(elapsed)
            baseline_scores.append(scores.copy()); states.append(caches(e))
            if token in stops: break
            baseline.append(token); print('BASELINE',len(baseline),token,elapsed,flush=True)
        baseline_final=caches(e)
        if args.v4:
            b.device.batch,b._prepared_runtime=prepared_batch,prepared_runtime
            b.wide_projection_quantization=True
            b.reuse_projection_quantization=b.compute_fusions=b.fused_head_selection=b.native_active_bias=True
            b.native_batch_min=3;b.native_large_min=8
        b.native_batch_min=8; b.compact_loads_enabled=True
        e=QwenTPU('.tools/models/qwen35-gguf/fp32-prepared',b)
        # Force a rejection and validate the committed one-position cache.
        if not args.skip_forced:
            e.reset(); e.prefill(tokens)
            wrong=(baseline[1]+1)%len(baseline_scores[0])
            pair=e.verify_pair(first,wrong); bits(pair[0],baseline_scores[1]); e.finish_pair(False)
            check_cache(e,states[0]); print('FORCED REJECTION PREFIX bitwise PASS',flush=True)
        # Force acceptance and validate both full logit rows and the two-position cache.
        if not args.skip_forced:
            e.reset(); e.prefill(tokens)
            pair=e.verify_pair(first,baseline[1]); bits(pair[0],baseline_scores[1]); bits(pair[1],baseline_scores[2]); e.finish_pair(True)
            check_cache(e,states[1]); print('FORCED ACCEPTANCE bitwise PASS',flush=True)
        e.reset(); begin=time.perf_counter(); scores=e.prefill(tokens); speculative_prefill=time.perf_counter()-begin
        selected=e.greedy(scores); actual=[selected]; decoder=SpeculativeGreedy(e,tokens,adaptive=args.adaptive)
        blocks=[]; begin_decode=time.perf_counter()
        while len(actual)<args.new_tokens:
            begin=time.perf_counter()
            emitted,logits,stop=decoder.advance(actual[-1],args.new_tokens-len(actual),stops)
            elapsed=time.perf_counter()-begin
            for i,token in enumerate(emitted):
                bits(logits[i],baseline_scores[len(actual)])
                assert token==baseline[len(actual)],(token,baseline[len(actual)])
                actual.append(token)
            row=dict(seconds=elapsed,tokens=emitted,accepted=decoder.stats['accepted'],rejected=decoder.stats['rejected'])
            blocks.append(row); print('SPEC',row,flush=True)
            if stop: break
        decode=time.perf_counter()-begin_decode
        assert actual==baseline
        check_cache(e,baseline_final)
        result=dict(status='bitwise PASS',prefill_seconds=prefill,speculative_prefill_seconds=speculative_prefill,
            baseline_prefill_seconds=prefill,cached_optimized_prefill_seconds=speculative_prefill,
            optimized_cache_warmed_by_reference=True,
            baseline_tokens=baseline,actual_tokens=actual,baseline_decode_seconds=sum(times),
            baseline_decode_s_per_token=statistics.median(times),speculative_decode_seconds=decode,
            speculative_decode_s_per_token=decode/max(1,len(actual)-1),stats=decoder.stats,blocks=blocks,
            forced_acceptance_and_rejection_prefix=not args.skip_forced,all_logits_and_final_trunk_states_equal=True,
            baseline='Original Turbo V4 0.3.0a7, 500 MHz, direct USB readback' if args.v4 else 'Turbo V2, 500 MHz, strict scalar schedule',
            optimized_profile='turbo-v4' if args.v4 else 'turbo-v3',input_tokens=tokens)
        Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
