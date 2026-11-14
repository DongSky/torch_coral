"""Bitwise physical verification of V4 compute fusions and device top-k."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from torch_coral.isa_io import IOOptimizedMatrixTPU
from torch_coral.isa_streamed import StreamedMatrixTPU
from torch_coral.isa_v2_fusions import reconstruct_inplace_program
from torch_coral.isa_batched import block_topk_program

def bits(a,b): np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(b).view(np.uint8))

def main():
    rng=np.random.default_rng(28); rows=[]
    with IOOptimizedMatrixTPU(ROOT/'.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2,clock='max') as b:
        for length in (1,128,1024,2000,3584):
            a=rng.normal(size=length).astype(np.float32)*np.float32(20)
            other=rng.normal(size=length).astype(np.float32)
            bits(b.silu_multiply(a,other),b.op('mul',b.activation('silu',a),other))
            rows.append(dict(op='silu_multiply',length=length,bitwise=True))
        for count in (1,3,13):
            x=rng.normal(size=(count,1024)).astype(np.float32); x[0]=0
            w=rng.normal(size=1024).astype(np.float32)
            expected=b.normalize(x,w,False,1e-6)
            actual=b.normalize_project_input(x,w,False,1e-6); bits(actual,expected)
            with b.projection_scope(actual) as frozen:
                for index,row in enumerate(frozen):
                    q,s,_=b.quantize_projection_input(row)
                    old=StreamedMatrixTPU.quantize_projection_input(b,expected[index])
                    bits(q,old[0]); bits(s,old[1])
            rows.append(dict(op='normalize_quantize',rows=count,bitwise=True))
        for length in (4,512,2048):
            for tied in (False,True):
                q=[rng.integers(0,256,size=length,dtype=np.uint8) for _ in range(2)]
                scales=rng.uniform(.001,.01,size=length).astype(np.float32)
                if tied: scales[:]=0
                payload=b''.join(v.tobytes() for v in q)+scales.tobytes()+np.float32(.03).tobytes()
                expected=b._inplace_result(reconstruct_inplace_program(length,(4096.,32.)),payload,length)
                top=b._run(block_topk_program(length,4),expected.tobytes(),8)
                actual=b._inplace_result(reconstruct_inplace_program(length,(4096.,32.),4),payload+b'\0'*32,length+8)
                bits(actual[:length],expected); bits(actual[length:],top)
                rows.append(dict(op='reconstruct_topk',length=length,tied=tied,bitwise=True))
    result=dict(status='bitwise PASS',clock='max',cases=rows)
    (ROOT/'reports/turbo-v4-a9-physical-fusions.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
