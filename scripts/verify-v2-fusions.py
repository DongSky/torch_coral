"""Verify every new Turbo V2 fusion against the physical Turbo chain."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch_coral.isa_streamed import StreamedMatrixTPU
from torch_coral.isa_optimized import OptimizedMatrixTPU
from torch_coral.isa_mxu import MatrixTPU
from torch_coral.qwen_tpu import QwenTPU


def bits(a,b):
    np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint32),np.ascontiguousarray(b).view(np.uint32))


if __name__ == '__main__':
    rng=np.random.default_rng(514); rows=[]
    with StreamedMatrixTPU('.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2) as b:
        class Original:
            op=b.op
            reduce=b.reduce
        old=object.__new__(QwenTPU); old.backend,old.cancel=Original(),None
        for length in (4,128,1024,3584):
            for x in (np.zeros(length,np.float32),rng.normal(size=length).astype(np.float32)):
                actual=b.quantize_projection_input(x); expected=b.quantize_input(x)
                bits(actual[0],expected[0]) if length%16==0 else np.testing.assert_array_equal(actual[0],expected[0])
                bits(actual[1],expected[1])
            rows.append(dict(op='primary_quantization',length=length))
        for length in (128,256,1024):
            for weighted in (False,True):
                for l2 in (False,True):
                    x=rng.normal(size=(33,length)).astype(np.float32); x[0]=0
                    w=rng.normal(size=length).astype(np.float32) if weighted else None
                    bits(b.normalize(x,w,l2,1e-6),OptimizedMatrixTPU.normalize(b,x,w,l2,1e-6))
                    rows.append(dict(op='norm',length=length,weighted=weighted,l2=l2))
        for length in (1,2,13,128,1024,2048):
            for x in (np.zeros(length,np.float32),rng.normal(size=length).astype(np.float32)*np.float32(10)):
                bits(b.softmax(x),old.softmax(x))
                rows.append(dict(op='softmax',length=length))
        old.rope_dim=64; old.cos=rng.normal(size=(2,32)).astype(np.float32); old.sin=rng.normal(size=(2,32)).astype(np.float32)
        for shape in ((2,256),(8,256),(33,256)):
            x=rng.normal(size=shape).astype(np.float32)
            bits(b.rope(x,old.cos[1],old.sin[1]),old.rope(x,1))
            rows.append(dict(op='rope',shape=shape))
        for length in (4,1024,2560,3072,6144):
            q=rng.integers(0,256,size=length,dtype=np.uint8)
            bias=rng.integers(-10000000,10000000,size=length,dtype=np.int32)
            for coefficient in (4096.,16384.,32.):
                bits(b._correct(q,bias,coefficient),MatrixTPU._correct(b,q,bias,coefficient))
            scales=rng.uniform(.001,.02,size=length).astype(np.float32)
            for coefficients in ((4096.,32.),(4096.,32.,.25)):
                outputs=[rng.integers(0,256,size=length,dtype=np.uint8) for _ in coefficients]
                scale=np.asarray([.037],np.float32)
                bits(b._reconstruct(outputs,coefficients,scales,scale),MatrixTPU._reconstruct(b,outputs,coefficients,scales,scale))
            rows.append(dict(op='correction_and_reconstruction',length=length))
    Path('reports/v2-fusions-physical.json').write_text(json.dumps(dict(status='bitwise PASS',cases=rows),indent=2))
    print('PASS',len(rows),'physical cases',flush=True)
