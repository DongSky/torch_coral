"""Full TPU projection latency, including activation conversion and readout.

The host matrix multiplication below is an independent validation reference.
Warm latency can reuse weights and must not be read as full-model throughput.
"""
from pathlib import Path
import argparse,sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch_coral.gguf_weights import WeightStore
from torch_coral.isa_mxu import MatrixTPU
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--chunk-rows',type=int,default=6144)
parser.add_argument('--iterations',type=int,default=10)
parser.add_argument('--output',default='reports/mxu-matrix-benchmark-final.json')
args=parser.parse_args()
if args.iterations<1:parser.error('iterations must be positive')
rng=np.random.default_rng(31);store=WeightStore('.tools/models/qwen35-gguf/fp32-prepared');report=[]
with MatrixTPU('.tools/models/qwen35-gguf/int8-matrices',chunk_rows=args.chunk_rows) as b:
 print('USB speed',b.device.usb.speed(),flush=True)
 for name in ['blk.0.ffn_up.weight','blk.0.ffn_down.weight','blk.0.attn_qkv.weight']:
  w=store[name];x=rng.normal(size=w.shape[1]).astype(np.float32);expected=w@x
  t=time.perf_counter();original=b.matvec(w,x);baseline=(time.perf_counter()-t)*1000
  np.testing.assert_allclose(original,expected,atol=3e-4,rtol=3e-4)
  for profile,components,digits in [('fast',1,2),('balanced',2,2),('precise',2,3)]:
   b.components=components;b.output_digits=digits
   b._resident=None  # require a real weight upload for the first call
   t=time.perf_counter();out=b.project(name,w,x);cold=(time.perf_counter()-t)*1000
   timings=[]
   for _ in range(args.iterations):
    t=time.perf_counter();out=b.project(name,w,x);timings.append((time.perf_counter()-t)*1000)
   error=float(np.linalg.norm(out-expected)/np.linalg.norm(expected))
   assert error < dict(fast=.025,balanced=.001,precise=.00015)[profile], (profile,error)
   row=dict(matrix=name,shape=list(w.shape),profile=profile,chunk_rows=args.chunk_rows,iterations=args.iterations,baseline_ms=baseline,first_call_ms=cold,p50_ms=float(np.median(timings)),p95_ms=float(np.percentile(timings,95)),relative_l2=error,speedup=baseline/float(np.median(timings)),effective_gops=2*w.size/(float(np.median(timings))*1e6),status='PASS')
   report.append(row);print(row,flush=True)
Path(args.output).write_text(json.dumps(report,indent=2))
