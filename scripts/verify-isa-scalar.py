from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from torch_coral.isa_scalar import ScalarTPU
report={}
with ScalarTPU() as tpu:
    x=np.linspace(.5,2,32,dtype=np.float32)
    for name,expected in [('copy',x),('exp',np.exp(x)),('recip',1/x),('rsqrt',1/np.sqrt(x))]:
        y=tpu.op(name,x)
        print(name,y[:4],flush=True)
        np.testing.assert_allclose(y,expected,atol=2e-5,rtol=2e-5)
        report[name]='PASS'
    for name,expected in [('add',x+x),('sub',x-x),('mul',x*x),('max',np.maximum(x,.7)),('min',np.minimum(x,.7))]:
        b=x if name in ('add','sub','mul') else np.float32(.7)
        y=tpu.op(name,x,b)
        np.testing.assert_allclose(y,expected,atol=2e-5,rtol=2e-5)
        report[name]='PASS'
    for name,expected in [('sum',x.sum()),('max',x.max())]:
        y=tpu.reduce(name,x)
        np.testing.assert_allclose(y,expected,atol=2e-5,rtol=2e-5)
        report['reduce_'+name]='PASS'
    rng=np.random.default_rng(14)
    w=rng.normal(size=(9,517)).astype(np.float32);x=rng.normal(size=517).astype(np.float32)
    y=tpu.matvec(w,x)
    np.testing.assert_allclose(y,w@x,atol=2e-4,rtol=2e-4)
    report['tiled_matvec']='PASS'
    x=rng.normal(size=(17,65)).astype(np.float32);w=rng.normal(size=(41,65)).astype(np.float32)
    y=tpu.matmul(w,x)
    np.testing.assert_allclose(y,x@w.T,atol=2e-4,rtol=2e-4)
    report['batched_projection']='PASS'
    values=np.linspace(1,2,23,dtype=np.float32)
    np.testing.assert_allclose(tpu.op('log_unit',values),np.log(values),atol=3e-6,rtol=2e-5)
    report['log_unit']='PASS'
    assert tpu.argmax(np.array([-2,4,4,-10,3],np.float32))==1
    report['argmax']='PASS'
    report['calls']=tpu.calls
Path('reports/isa-scalar-hardware.json').write_text(json.dumps(report,indent=2))
print(report)
