import os,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch_coral.isa_mxu import MatrixTPU,prepare_matrix_weights
from torch_coral.isa_scalar import ScalarTPU
from coral.programs import fc_program,Quant
rng=np.random.default_rng(23)
folder=Path('artifacts/mxu-probe');folder.mkdir(exist_ok=True)
(folder/'matrices.json').write_text(json.dumps(dict(format_version=2,source_sha256='test',matrices={})))
with MatrixTPU(folder) as b:
 print('USB speed',b.device.usb.speed(),flush=True)
 x=rng.uniform(-2,2,128).astype(np.float32)
 q,s,res=b.quantize_input(x)
 expected=np.clip(np.rint(x/float(s[0])),-127,127).astype(np.int16)+128
 print('quant error bytes',int(np.max(abs(q.astype(np.int16)-expected))),flush=True)
 np.testing.assert_array_equal(q,expected.astype(np.uint8))
 np.testing.assert_allclose(res,x-(q.astype(np.float32)-128)*s,atol=1e-7)
 n,k=256,128
 w=rng.integers(108,149,size=(n,k),dtype=np.uint8)
 xq=rng.integers(60,200,size=(1,k),dtype=np.uint8)
 true=(xq.astype(np.int64)-128)@(w.astype(np.int64)-128).T
 prog=fc_program(n,k)
 pc,exe=prog.programs(0,0,Quant(128,128,128,1/128))
 b._execute(pc,parameters=prog.params(w,np.zeros(n,np.int32),128))
 raw=b._execute(exe,inputs=prog.input_bytes(xq,128));q0=prog.output_array(raw)[0].copy()
 np.testing.assert_array_equal(q0,np.clip(np.rint(true[0]/128)+128,0,255).astype(np.uint8))
 bias=b._correct(q0,np.zeros(n,'<i4'),128.)
 np.testing.assert_array_equal(bias,-(q0.astype(np.int32)-128)*128)
 b._upload_bias(n,k,bias)
 _,exe=prog.programs(0,0,Quant(128,128,128,1.))
 q1=prog.output_array(b._execute(exe,inputs=prog.input_bytes(xq,128)))[0].copy()
 expected1=np.clip(np.rint(true[0]+bias)+128,0,255).astype(np.uint8)
 print('bias update maxerror',int(abs(q1.astype(np.int16)-expected1.astype(np.int16)).max()),flush=True)
 np.testing.assert_array_equal(q1,expected1)
 recovered=b._reconstruct([q0,q1],[128.,1.],np.ones(n,np.float32),np.ones(1,np.float32))
 np.testing.assert_array_equal(recovered,true[0])
 print('EXACT MXU RECONSTRUCTION PASS',flush=True)
