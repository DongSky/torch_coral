"""Measure application payload throughput and process CPU, not USB wire rate."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from torch_coral.isa_io import IOOptimizedMatrixTPU
from torch_coral.isa_scalar import elementwise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight-repeats',type=int,default=200)
    parser.add_argument('--scalar-repeats',type=int,default=20000)
    args=parser.parse_args();rows=[]
    with IOOptimizedMatrixTPU(ROOT/'.tools/models/qwen35-gguf/int8-matrices',components=1,output_digits=2,clock='max') as b:
        from coral.programs import fc_program,Quant
        name='blk.0.attn_qkv.weight'
        shape=b.matrix_manifest['matrices'][name]['shape'];n,k=min(6144,shape[0]),shape[1]
        weights=np.load(b.matrix_directory/(name+'.q0.npy'),mmap_mode='r',allow_pickle=False)[:n]
        program=fc_program(n,k);blob=b.parameter_blob(program,('io-probe',n,k),np.ascontiguousarray(weights))
        caching,_=program.programs(0,0,Quant(128,128,128,1/4096))
        scalar=elementwise('add',1024);payload=np.ones(2048,np.float32).tobytes()
        prepared,legacy,runtime=b.device.batch,b._legacy_v4_batch,b._prepared_runtime
        for _ in range(5):b._execute(caching,parameters=blob);b._run(scalar,payload,1024)
        for mode in ('legacy','prepared','prepared','legacy'):
            b.device.batch=prepared if mode=='prepared' else legacy
            b._prepared_runtime=runtime if mode=='prepared' else None
            for category,repeats in (('parameter_upload',args.weight_repeats),('small_scalar',args.scalar_repeats)):
                before=os.times();cpu=time.process_time();start=time.perf_counter();epoch=time.time()
                for _ in range(repeats):
                    if category=='parameter_upload':b._execute(caching,parameters=blob)
                    else:b._run(scalar,payload,1024)
                wall=time.perf_counter()-start;process=time.process_time()-cpu;after=os.times()
                row=dict(mode=mode,category=category,start_epoch_seconds=epoch,wall_seconds=wall,calls=repeats,
                         process_cpu_seconds=process,process_one_core_cpu_percent=100*process/wall,
                         process_user_seconds=after.user-before.user,process_kernel_seconds=after.system-before.system,
                         parameter_payload_bytes=len(blob)*repeats if category=='parameter_upload' else 0,
                         application_parameter_MB_per_second=(len(blob)*repeats/wall/1e6) if category=='parameter_upload' else None,
                         microseconds_per_call=wall/repeats*1e6)
                rows.append(row);print(json.dumps(row),flush=True)
        b.device.batch,b._prepared_runtime=prepared,runtime
        result=dict(status='PASS',usb_speed_enum=b.device.usb.speed(),clock='max',rows=rows,
                    notes=['CPU percent is relative to one logical core and includes all process threads.',
                           'Parameter upload includes framing, host/runtime work, SRAM distribution and completion.',
                           'MB/s is application payload throughput, not a measured USB wire utilization.',
                           'This probe does not separate USB host-controller/DPC time from device wait.'])
        (ROOT/'reports/turbo-v4-a9-io-probe.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
