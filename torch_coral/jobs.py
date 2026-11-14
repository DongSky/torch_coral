"""Subprocess job entry point used by the GUI and headless integration tests."""
import argparse
import json
from pathlib import Path
import sys
import traceback
from .runtime import ROOT

def emit(kind,**data): print(json.dumps({'event':kind,**data},ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
    args=parser.parse_args();path=Path(args.request)
    request=json.loads(path.read_text(encoding='utf-8'))
    try:
        kind=request['kind'];emit('progress',message='开始任务：'+kind)
        if kind=='inspect':
            from .hub import inspect_model
            result=inspect_model(request['source'],request.get('revision','main'),request.get('task','auto'))
        elif kind=='convert':
            from .hub import convert_hub
            result=convert_hub(request,progress=lambda msg:emit('progress',message=msg))
        elif kind=='infer':
            from .hub import infer_hub
            result=infer_hub(request,progress=lambda msg:emit('progress',message=msg))
        elif kind=='benchmark':
            from .benchmark import run_suite,measure_artifact
            progress=lambda stage,i,n:emit('progress',message=stage,current=i,total=n)
            result=measure_artifact(request['artifact_dir'],repeats=int(request.get('repeats',100)),progress=progress) if request.get('artifact_only') else run_suite(repeats=int(request.get('repeats',100)),progress=progress)
        else: raise ValueError('未知任务：'+kind)
        output=path.with_suffix('.result.json');output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        emit('result',result=result,result_path=str(output))
    except Exception as error:
        message=f'{type(error).__name__}: {error}'
        detail=traceback.format_exc();path.with_suffix('.error.txt').write_text(detail,encoding='utf-8')
        emit('error',message=message,detail_path=str(path.with_suffix('.error.txt')))
        return 1
    return 0

if __name__=='__main__': sys.exit(main())
