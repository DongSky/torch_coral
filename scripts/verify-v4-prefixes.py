"""Physical state capture at every prefix of a three/four-position block."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from torch_coral.isa_io import IOOptimizedMatrixTPU

def bits(a,b):np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint32),np.ascontiguousarray(b).view(np.uint32))

def main():
    rng=np.random.default_rng(712);rows=[]
    with IOOptimizedMatrixTPU(ROOT/'.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2,clock='max') as b:
        for steps,heads,length in ((3,1,4),(4,1,4),(3,2,128),(4,16,128)):
            state=rng.normal(size=(heads,length,length)).astype(np.float32)*np.float32(.1)
            k,q,v=rng.normal(size=(3,steps,heads,length)).astype(np.float32)*np.float32(.1)
            beta=np.full((steps,heads),.3,np.float32);decay=np.full((steps,heads),.9,np.float32)
            current=state.copy();expected=[];outputs=[]
            for step in range(steps):
                current,y=b.recurrent_many(current,k[step:step+1],q[step:step+1],v[step:step+1],beta[step:step+1],decay[step:step+1])
                expected.append(current.copy());outputs.append(y[0].copy())
            b.capture_state_prefixes=steps-1
            actual,y=b.recurrent_many(state,k,q,v,beta,decay)
            b.capture_state_prefixes=0
            bits(actual,expected[-1]);bits(y,np.stack(outputs))
            for i in range(steps-1):bits(b.prefix_recurrent_states[i],expected[i])
            row=dict(steps=steps,heads=heads,length=length,every_prefix_bitwise_equal=True)
            rows.append(row);print(json.dumps(row),flush=True)
    (ROOT/'reports/turbo-v4-a9-recurrent-prefixes.json').write_text(json.dumps(dict(status='bitwise PASS',cases=rows),indent=2))

if __name__=='__main__':main()
