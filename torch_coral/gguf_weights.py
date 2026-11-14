"""One-time GGUF decompression; inference reads immutable FP32 memory maps.

Decompression is weight preparation, not a CPU model execution backend.
No model weights are bundled in torch-coral distributions.
"""
import hashlib
import json
from pathlib import Path
import re
import numpy as np

QWEN_GGUF_SHA256 = 'ac7c9d7a1b3e3695bb3bd50f8ceaa97f9c93e99ccc3d3d1a620301b6dd6d3d86'


def prepare_weights(source, destination, *, progress=print):
    from gguf import GGUFReader
    from gguf.quants import dequantize
    source, destination = Path(source).resolve(), Path(destination).resolve()
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if (destination / 'weights.json').exists():
        store = WeightStore(destination)
        if store.manifest['source_sha256'] != digest:
            raise ValueError('Prepared weights belong to a different GGUF')
        return store
    reader = GGUFReader(str(source))
    if reader.get_field('general.architecture').contents() != 'qwen35':
        raise ValueError('Only dense Qwen3.5 GGUF is supported by this experiment')
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        if not re.fullmatch(r'[A-Za-z0-9_.]+', name):
            raise ValueError('Unsafe tensor name')
        shape = tuple(int(d) for d in reversed(tensor.shape))
        path = destination / (name + '.npy')
        target = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
        if len(shape) == 1:
            target[:] = dequantize(tensor.data, tensor.tensor_type).reshape(shape)
        else:
            for start in range(0, shape[0], 256):
                block = dequantize(tensor.data[start:start + 256], tensor.tensor_type)
                target[start:start + len(block)] = block.reshape((-1,) + shape[1:])
        target.flush()
        del target
        records[name] = dict(file=path.name, shape=list(shape), source_type=tensor.tensor_type.name,
                             bytes=path.stat().st_size)
        progress('%d/%d %s' % (i + 1, len(reader.tensors), name))
    metadata = {name: field.contents() for name, field in reader.fields.items()
                if name.startswith(('qwen35.', 'general.'))}
    manifest = dict(format_version=1, architecture='qwen35', source_sha256=digest,
                    known_source=digest == QWEN_GGUF_SHA256, metadata=metadata, tensors=records)
    temp = destination / 'weights.json.tmp'
    temp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    temp.replace(destination / 'weights.json')
    return WeightStore(destination)


class WeightStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / 'weights.json').read_text(encoding='utf-8'))
        if self.manifest.get('format_version') != 1 or self.manifest.get('architecture') != 'qwen35':
            raise ValueError('Unsupported prepared weight manifest')
        self.metadata = self.manifest['metadata']
        self._arrays = {}

    def __getitem__(self, name):
        if name not in self._arrays:
            record = self.manifest['tensors'][name]
            path = (self.directory / record['file']).resolve()
            if not path.is_relative_to(self.directory):
                raise ValueError('Weight path escapes prepared directory')
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.dtype != np.float32 or list(array.shape) != record['shape']:
                raise ValueError('Prepared weight dtype/shape mismatch: ' + name)
            self._arrays[name] = array
        return self._arrays[name]
