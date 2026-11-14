"""Physical retained-state sequence versus per-token physical FP32 kernels."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch_coral.isa_streamed import StreamedMatrixTPU
from torch_coral.isa_mxu import MatrixTPU


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--full',action='store_true'); args=parser.parse_args()
    rng=np.random.default_rng(413); rows=[]
    cases=[(1,4,2,128),(1,128,1,128),(1,128,13,128)]
    if args.full: cases += [(16,128,13,128),(16,128,16,128)]
    with StreamedMatrixTPU('.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2) as b:
        for heads,length,steps,columns in cases:
            print('START',heads,length,steps,columns,flush=True); b.stream_columns=columns
            state=rng.normal(size=(heads,length,length)).astype(np.float32)*np.float32(.1)
            k,q,v=rng.normal(size=(3,steps,heads,length)).astype(np.float32)*np.float32(.1)
            beta=np.full((steps,heads),.3,np.float32); decay=np.full((steps,heads),.9,np.float32)
            current=state.copy(); expected=[]; before=b.calls; begin=time.perf_counter()
            for t in range(steps):
                updated=[]; outputs=[]
                for h in range(heads):
                    s,y=MatrixTPU.recurrent(b,current[h],k[t,h],q[t,h],v[t,h],beta[t,h],decay[t,h]); updated.append(s); outputs.append(y)
                current=np.stack(updated); expected.append(np.stack(outputs))
            original_ms=(time.perf_counter()-begin)*1000; original_calls=b.calls-before
            before=b.calls; begin=time.perf_counter()
            actual_state,actual=b.recurrent_many(state,k,q,v,beta,decay)
            stream_ms=(time.perf_counter()-begin)*1000; stream_calls=b.calls-before
            np.testing.assert_array_equal(actual_state.view(np.uint32),current.view(np.uint32))
            np.testing.assert_array_equal(actual.view(np.uint32),np.stack(expected).view(np.uint32))
            row=dict(heads=heads,length=length,steps=steps,columns=columns,status='bitwise PASS',original_ms=original_ms,stream_ms=stream_ms,original_calls=original_calls,stream_calls=stream_calls)
            rows.append(row); print(row,flush=True)
            Path('reports/streamed-recurrent-physical.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
