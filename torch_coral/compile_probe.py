"""Isolated converter-environment compiler comparison; no PyTorch import."""
import hashlib
import json
from pathlib import Path
import sys
from ai_edge_litert.interpreter import Interpreter
from convert import determine_io,compile_tflite

def main():
    source=Path(sys.argv[1]).resolve();dest=Path(sys.argv[2]).resolve();dest.mkdir(parents=True,exist_ok=True)
    job=json.loads((source/'job.json').read_text(encoding='utf-8'))
    determine_io(source,job,source/'converted/model_float32.tflite')
    job['compiler']=str(Path(sys.argv[3]).resolve());job['allow_partial']=True
    quant=source/'converted/model_optimized.tflite'
    if not quant.exists():quant=source/'converted/model_full_integer_quant.tflite'
    interp=Interpreter(model_path=str(quant));interp.allocate_tensors()
    report=compile_tflite(dest,job,quant)
    manifest={**job,'model_file':'model_edgetpu.tflite','compiler_report':report,
              'sha256':hashlib.sha256((dest/'model_edgetpu.tflite').read_bytes()).hexdigest(),
              'probe_source':str(source)}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':main()
