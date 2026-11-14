"""Inspect GGUF metadata/tensor types without loading weights or running a model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct

TYPES = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1',
         8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K',
         13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 30: 'BF16'}
SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f',
           7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def inspect(path):
    with Path(path).open('rb') as stream:
        def number(fmt):
            size = struct.calcsize('<' + fmt)
            data = stream.read(size)
            if len(data) != size:
                raise ValueError('Incomplete GGUF header; obtain a larger file prefix')
            return struct.unpack('<' + fmt, data)[0]
        def string():
            size = number('Q')
            if size > 64 * 1024 * 1024:
                raise ValueError('GGUF string exceeds inspection limit')
            data = stream.read(size)
            if len(data) != size:
                raise ValueError('Incomplete GGUF string')
            return data.decode('utf-8', errors='replace')
        def value(kind, keep):
            if kind in SCALARS:
                return number(SCALARS[kind])
            if kind == 8:
                return string()
            if kind == 9:
                subtype, count = number('I'), number('Q')
                if subtype == 9 or count > 10_000_000:
                    raise ValueError('Invalid GGUF metadata array')
                values = []
                for _ in range(count):
                    item = value(subtype, keep)
                    if keep:
                        values.append(item)
                return values if keep else {'element_type': subtype, 'count': count}
            raise ValueError('Unknown GGUF metadata type: %s' % kind)
        if stream.read(4) != b'GGUF':
            raise ValueError('Not a GGUF file')
        version = number('I')
        if version not in (2, 3):
            raise ValueError('Unsupported GGUF version')
        tensors, entries = number('Q'), number('Q')
        if tensors > 100_000 or entries > 100_000:
            raise ValueError('GGUF header count exceeds inspection limit')
        metadata = {}
        for _ in range(entries):
            key, kind = string(), number('I')
            # Tokenizer vocabularies can be large; retain counts, not token data.
            metadata[key] = value(kind, not key.startswith('tokenizer.'))
        records = []
        for _ in range(tensors):
            name, dims = string(), number('I')
            if dims > 8:
                raise ValueError('Invalid GGUF tensor dimensions')
            shape = [number('Q') for _ in range(dims)]
            kind, offset = number('I'), number('Q')
            records.append(dict(name=name, shape=shape, type=TYPES.get(kind, 'GGML_TYPE_%d' % kind), offset=offset))
        return dict(version=version, tensor_count=tensors, header_bytes=stream.tell(),
                    tensor_types=dict(Counter(r['type'] for r in records)), metadata=metadata, tensors=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', help='GGUF file, or a downloaded prefix containing its complete header')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = inspect(args.file)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('version', 'tensor_count', 'header_bytes', 'tensor_types')}))


if __name__ == '__main__': main()
