"""Physical intermediate-state DMA for exact speculative rejection rollback."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch_coral.isa_streamed import StreamedMatrixTPU
from torch_coral.isa_mxu import MatrixTPU


if __name__=='__main__':
    rng=np.random.default_rng(719); rows=[]
    with StreamedMatrixTPU('.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2,clock='max') as b:
        for length,heads in ((4,1),(128,1),(128,16)):
            print('START',length,heads,flush=True)
            state=rng.normal(size=(heads,length,length)).astype(np.float32)*np.float32(.1)
            k,q,v=rng.normal(size=(3,2,heads,length)).astype(np.float32)*np.float32(.1)
            beta=np.full((2,heads),.3,np.float32); decay=np.full((2,heads),.9,np.float32)
            initial=state.copy()
            old=initial.copy(); expected=[]; outputs=[]
            for t in range(2):
                s=[]; y=[]
                for h in range(heads):
                    sh,yh=MatrixTPU.recurrent(b,old[h],k[t,h],q[t,h],v[t,h],beta[t,h],decay[t,h]); s.append(sh); y.append(yh)
                old=np.stack(s); expected.append(old.copy()); outputs.append(np.stack(y))
            b.capture_first_state=True
            actual,y=b.recurrent_many(initial,k,q,v,beta,decay)
            np.testing.assert_array_equal(b.first_recurrent_state.view(np.uint32),expected[0].view(np.uint32))
            np.testing.assert_array_equal(actual.view(np.uint32),expected[1].view(np.uint32))
            np.testing.assert_array_equal(y.view(np.uint32),np.stack(outputs).view(np.uint32))
            b.capture_first_state=False
            print('PASS',length,heads,flush=True)
    Path('reports/v3-state-capture.json').write_text(json.dumps(dict(status='bitwise PASS',shapes=[[4,1],[128,1],[128,16]])))
