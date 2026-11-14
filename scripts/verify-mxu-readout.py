from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch_coral.isa_mxu import MatrixTPU
from coral.programs import fc_program,Quant
rng=np.random.default_rng(29);rows=[]
with MatrixTPU('artifacts/mxu-probe',output_digits=3) as b:
 for n,k in [(3584,1024),(6144,1024),(2560,1024),(1024,3584)]:
  w=rng.integers(96,161,(n,k),dtype=np.uint8);x=rng.integers(48,209,(1,k),dtype=np.uint8)
  true=(x.astype(np.int64)-128)@(w.astype(np.int64)-128).T
  p=fc_program(n,k);pc,_=p.programs(0,0,Quant(128,128,128,1/4096));b._execute(pc,parameters=p.params(w,np.zeros(n,np.int32),128))
  bias=np.zeros(n,'<i4');outs=[];scales=[];scale=4096.;t=time.perf_counter()
  for digit in range(3):
   _,exe=p.programs(0,0,Quant(128,128,128,1/scale))
   q=p.output_array(b._execute(exe,inputs=p.input_bytes(x,128)))[0].copy();outs.append(q);scales.append(scale)
   if digit<2:bias=b._correct(q,bias,scale);b._upload_bias(n,k,bias)
   scale/=128
  y=b._reconstruct(outs,scales,np.ones(n,np.float32),np.ones(1,np.float32))
  np.testing.assert_array_equal(y,true[0]);r=dict(n=n,k=k,status='PASS',readout_ms=(time.perf_counter()-t)*1000)
  rows.append(r);print(r,flush=True)
Path('reports/mxu-exact-readout.json').write_text(json.dumps(rows,indent=2))