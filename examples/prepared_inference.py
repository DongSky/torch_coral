"""Preallocated native buffers across the modern frontend / original SDK boundary."""
import argparse
import numpy as np
import torch_coral

parser = argparse.ArgumentParser()
parser.add_argument('artifact')
args = parser.parse_args()
with torch_coral.load(args.artifact, transport='shared_memory', num_threads=1) as model:
    with model.prepare(raw=True) as prepared:
        # Buffer handles do not expose unsafe NumPy views into shared memory.
        inputs = [np.zeros(b.shape, b.dtype) for b in prepared.inputs]
        outputs = [np.empty(b.shape, b.dtype) for b in prepared.outputs]
        for _ in range(10):
            prepared.run(inputs, out=outputs)
        print(outputs, prepared.last_timing)
        # Group manual writes/invoke/reads for atomic use across threads.
        with prepared.transaction():
            prepared.inputs[0].copy_from(inputs[0])
            prepared.invoke()
            prepared.outputs[0].read_into(outputs[0])
