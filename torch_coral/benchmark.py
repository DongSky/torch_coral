"""Repeatable workload-effective TOPS measurement on the real Edge TPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from .runtime import ROOT, load
from .compiler import compile
from .settings import python_executable

def measure_artifact(folder, *, warmup=20, repeats=100, progress=None):
    folder=Path(folder).resolve()
    if warmup<0 or not 1<=repeats<=5000: raise ValueError('repeats must be 1..5000; warmup >= 0')
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest['compiler_report']['cpu_operations']:
        raise ValueError('Mixed CPU/TPU graph: cannot attribute whole-model MAC work to TPU')
    model=folder/manifest.get('quantized_model_file','converted/model_coral_strategy.tflite')
    if not model.exists(): model=folder/'converted/model_optimized.tflite'
    if not model.exists(): model=folder/'converted/model_full_integer_quant.tflite'
    python=python_executable('converter')
    work=json.loads(subprocess.check_output([python,str(Path(__file__).with_name('count_ops.py')),str(model)],text=True))
    inputs=[]
    for i in range(len(manifest['inputs'])):
        values=np.load(folder/('calibration_%d.npy'%i),allow_pickle=False)
        inputs.append(torch.from_numpy(values[0].copy()))
    delegate=[]; total=[]
    with torch.no_grad(),load(folder) as coral:
        for i in range(warmup+repeats):
            start=time.perf_counter(); coral(*inputs); elapsed=(time.perf_counter()-start)*1000
            if i>=warmup:
                total.append(elapsed);delegate.append(coral.last_timing['invoke_ms'])
            if progress and i%20==0: progress('measuring',i,warmup+repeats)
        hardware=coral.hardware
    p50=float(np.median(delegate));e2e=float(np.median(total))
    return {'name':folder.name,'artifact_dir':str(folder),'work':work,'repeats':repeats,'warmup':warmup,
            'delegate_p50_ms':p50,'delegate_p95_ms':float(np.percentile(delegate,95)),
            'end_to_end_p50_ms':e2e,'end_to_end_p95_ms':float(np.percentile(total,95)),
            'delegate_effective_tops':work['ops_2_per_mac']/(p50*1e9),
            'end_to_end_effective_tops':work['ops_2_per_mac']/(e2e*1e9),
            'delegate_samples_ms':delegate,'end_to_end_samples_ms':total,'hardware':hardware,
            'cpu_operations':0,'timing_scope':'delegate includes USB/driver/device; end-to-end also includes host quantization, layout and IPC'}

def run_suite(*, repeats=100, progress=None):
    torch.set_num_threads(4)
    folders=[ROOT/'artifacts/mobilenet-v2']
    configs=[('conv32-s32-d4',32,32,4),('conv64-s64-d8',64,64,8),
             ('conv128-s32-d8',128,32,8),('conv128-s64-d8',128,64,8)]
    for name,channels,size,depth in configs:
        folder=ROOT/'artifacts/compute-bench'/name
        if not (folder/'manifest.json').exists():
            if progress: progress('compiling '+name,0,1)
            torch.manual_seed(2026)
            layers=[]
            for i in range(depth):
                conv=torch.nn.Conv2d(3 if i==0 else channels,channels,3,padding=1)
                torch.nn.init.kaiming_normal_(conv.weight,nonlinearity='relu');torch.nn.init.zeros_(conv.bias)
                layers.extend([conv,torch.nn.ReLU()])
            model=torch.nn.Sequential(*layers,torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten()).eval()
            calibration=[torch.randn(1,3,size,size) for _ in range(8)]
            compile(model,calibration[0],calibration,artifact_dir=folder).close()
        folders.append(folder)
    rows=[]
    for folder in folders:
        if not (folder/'manifest.json').exists(): continue
        if progress: progress('benchmark '+folder.name,0,1)
        rows.append(measure_artifact(folder,repeats=repeats,progress=progress))
    report={'status':'PASS','created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'formula':'effective_TOPS = (2 * nominal_INT8_MACs) / (milliseconds * 1e9)',
            'interpretation':'Measured workload-effective throughput, not silicon peak. Delegate timing includes USB transfers. Standard-clock runtime; no power measurement.',
            'results':rows,'best_delegate_effective_tops':max(r['delegate_effective_tops'] for r in rows),
            'best_end_to_end_effective_tops':max(r['end_to_end_effective_tops'] for r in rows)}
    (ROOT/'reports/edge-tpu-compute.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=100)
    args=parser.parse_args()
    result=run_suite(repeats=args.repeats,progress=lambda stage,i,n:print(stage,i,n,flush=True))
    print(json.dumps([{k:r[k] for k in ['name','delegate_p50_ms','end_to_end_p50_ms','delegate_effective_tops','end_to_end_effective_tops']} for r in result['results']],indent=2))
