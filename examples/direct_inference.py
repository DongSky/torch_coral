"""Run in the compatible SDK environment, e.g. Python 3.9 on Windows."""
import argparse
import numpy as np
from torch_coral_runtime import DirectSession

parser = argparse.ArgumentParser()
parser.add_argument('artifact', help='Compiled artifact directory or Edge TPU TFLite model')
args = parser.parse_args()
with DirectSession(args.artifact, num_threads=1) as session:
    print(session.describe(), session.hardware)
    # Replace zeros with correctly quantized/preprocessed application inputs.
    for i, spec in enumerate(session.describe()['inputs']):
        session.input_buffer(i).copy_from(np.zeros(spec['shape'], dtype=spec['dtype']))
    session.invoke()
    print([session.output_buffer(i).numpy() for i in range(len(session.outputs))])
    print(session.last_timing)
