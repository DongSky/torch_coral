"""Compare shorter SRAM load waits with the original physical instruction path."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch_coral.isa_batched import BatchedMatrixTPU
from torch_coral.isa_scalar import ScalarTPU


def compare(b,label,call):
    b.compact_loads_enabled=False; expected=call()
    b.compact_loads_enabled=True; actual=call()
    if not isinstance(actual,tuple): actual,expected=(actual,),(expected,)
    for a,e in zip(actual,expected):
        if a is None: assert e is None
        else: np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(e).view(np.uint8))
    print('PASS',label,flush=True)


if __name__=='__main__':
    rng=np.random.default_rng(412)
    with BatchedMatrixTPU('.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2,clock='max') as b:
        assert b.compact_loads_enabled
        x=rng.uniform(.5,2,size=1024).astype(np.float32)
        for op in ('copy','exp','rsqrt','recip','log_unit'):
            compare(b,op,lambda:b.op(op,x))
        for op in ('add','sub','mul','max','min'):
            compare(b,op,lambda:b.op(op,x,x[::-1]))
        compare(b,'activation',lambda:b.activation('silu',x))
        compare(b,'normalization',lambda:b.normalize(x.reshape(8,128),x[:128],False,1e-6))
        compare(b,'quantization',lambda:b.quantize_projection_input(x))
        q=rng.integers(0,256,size=6144,dtype=np.uint8); bias=np.zeros(6144,np.int32)
        compare(b,'correction',lambda:b._correct(q,bias,4096.))
        compare(b,'reconstruction',lambda:b._reconstruct([q,q[::-1]],(4096.,32.),x[:1].repeat(6144),x[:1]))
        history=rng.normal(size=(3,6144)).astype(np.float32)
        projections=rng.normal(size=(13,6144)).astype(np.float32)
        kernel=rng.normal(size=(6144,4)).astype(np.float32)
        compare(b,'temporal convolution',lambda:b.convolution_many(history,projections,kernel))
        state=rng.normal(size=(16,128,128)).astype(np.float32)*np.float32(.1)
        k,q,v=rng.normal(size=(3,2,16,128)).astype(np.float32)*np.float32(.1)
        beta=np.full((2,16),.3,np.float32); decay=np.full((2,16),.9,np.float32)
        b.capture_first_state=True
        compare(b,'recurrent + intermediate DMA',lambda:b.recurrent_many(state,k,q,v,beta,decay))
        b.capture_first_state=False
        values=rng.normal(size=248320).astype(np.float32)
        expected=ScalarTPU.argmax(b,values); actual=b.argmax(values)
        assert actual==expected
        assert len(b.draft_candidate_ids)==488
        print('PASS argmax and proposal top-four per block',flush=True)
    Path('reports/v3-schedule.json').write_text(json.dumps(dict(status='bitwise PASS',load_latency_canary=True,cases=18)))
