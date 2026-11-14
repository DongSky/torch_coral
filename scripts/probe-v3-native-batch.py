"""Physical native batch outputs versus integer references and independent FCs."""
import json
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch_coral.isa_streamed import StreamedMatrixTPU
from torch_coral.isa_mxu import bias_executable
from torch_coral.isa_batched import batch_program
from coral.programs import Quant,fc_program


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--v4',action='store_true'); args=parser.parse_args()
    if args.v4:
        from torch_coral.isa_io import IOOptimizedMatrixTPU
        StreamedMatrixTPU=IOOptimizedMatrixTPU
    rng=np.random.default_rng(831); rows=[]
    with StreamedMatrixTPU('.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2,clock='max') as b:
        for n,k in ((128,256),(768,1024),(6144,1024),(1024,3584)):
            print('START',n,k,flush=True)
            x=rng.integers(120,137,size=(16,k),dtype=np.uint8)
            w=rng.integers(120,137,size=(n,k),dtype=np.uint8)
            bias=rng.integers(-10000,10001,size=(16,n),dtype=np.int32)
            prog=batch_program(n,k); q=Quant(128,128,128,1/256)
            pc,eo=prog.programs(q)
            b._execute(pc,parameters=prog.params(w))
            b._execute(bias_executable(16*n,k,prog.bias_base),parameters=bias.tobytes())
            start=time.perf_counter(); raw=b._execute(eo,inputs=prog.input_bytes(x))
            batch_ms=(time.perf_counter()-start)*1000
            actual=prog.output_array(raw)
            expected=np.clip(np.rint(((x.astype(np.int32)-128)@(w.astype(np.int32)-128).T+bias)/256)+128,0,255).astype(np.uint8)
            np.testing.assert_array_equal(actual,expected)
            ref=fc_program(n,k); pc,_=ref.programs(0,0,q)
            b._execute(pc,parameters=ref.params(w,np.zeros(n,np.int32),128))
            for row in range(16):
                b._upload_bias(n,k,bias[row])
                _,exe=ref.programs(0,0,q)
                old=ref.output_array(b._execute(exe,inputs=ref.input_bytes(x[row:row+1],128)))[0]
                np.testing.assert_array_equal(actual[row],old)
            record=dict(shape=[16,n,k],batch_ms=batch_ms,per_position_bias=True,bitwise_fc=True)
            rows.append(record); print(record,flush=True)
            Path('reports/'+('v4' if args.v4 else 'v3')+'-native-batch-probe.json').write_text(json.dumps(rows,indent=2))
