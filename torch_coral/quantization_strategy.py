"""Repair legacy same-domain constraints using real INT8 requantize operators.

Existing tensor scales and kernel biases are never relabelled. Internal tensors
share a union domain; Quantize nodes bridge to the original calibrated domains.
"""
import copy

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s


def align_domains(blob):
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(blob, 0))
    supported = {getattr(s.BuiltinOperator, n) for n in (
        'SLICE', 'STRIDED_SLICE', 'TRANSPOSE', 'RESHAPE', 'SQUEEZE', 'EXPAND_DIMS',
        'SPLIT', 'CONCATENATION', 'MAXIMUM', 'MINIMUM', 'MEAN', 'REDUCE_MAX', 'REDUCE_MIN')}
    quant_code = next((i for i, c in enumerate(model.operatorCodes)
                       if c.builtinCode == s.BuiltinOperator.QUANTIZE and c.version == 2), None)
    if quant_code is None:
        code = s.OperatorCodeT()
        code.builtinCode = code.deprecatedBuiltinCode = s.BuiltinOperator.QUANTIZE
        code.version = 2
        quant_code = len(model.operatorCodes)
        model.operatorCodes.append(code)
    records = []
    for graph in model.subgraphs:
        rewritten = []

        def active(index):
            t = graph.tensors[index]
            return t.type == s.TensorType.INT8 and t.quantization is not None and len(t.quantization.scale) == 1

        def domain(index):
            q = graph.tensors[index].quantization
            return float(q.scale[0]), int(q.zeroPoint[0])

        def internal(index, scale, zero):
            t = copy.deepcopy(graph.tensors[index])
            t.name = f'coral_domain_{len(graph.tensors)}'.encode()
            t.buffer = 0
            t.quantization.scale, t.quantization.zeroPoint = [scale], [zero]
            t.quantization.min = t.quantization.max = None
            graph.tensors.append(t)
            return len(graph.tensors) - 1

        def bridge(source, target):
            op = s.OperatorT()
            op.opcodeIndex, op.inputs, op.outputs = quant_code, [source], [target]
            return op

        for op in graph.operators:
            opcode = model.operatorCodes[op.opcodeIndex].builtinCode
            if opcode not in supported:
                rewritten.append(op)
                continue
            indices = [int(i) for i in [*op.inputs, *op.outputs] if i >= 0 and active(int(i))]
            if not indices or len({domain(i) for i in indices}) == 1:
                rewritten.append(op)
                continue
            lo = min(((-128 - domain(i)[1]) * domain(i)[0]) for i in indices)
            hi = max(((127 - domain(i)[1]) * domain(i)[0]) for i in indices)
            scale = np.float32(max((hi - lo) / 255, 1e-12)).item()
            zero = int(np.clip(np.rint(-128 - lo / scale), -128, 127))
            inputs, outputs, post = list(map(int, op.inputs)), list(map(int, op.outputs)), []
            for j, index in enumerate(inputs):
                if index >= 0 and active(index) and domain(index) != (scale, zero):
                    new = internal(index, scale, zero)
                    rewritten.append(bridge(index, new))
                    inputs[j] = new
            for j, index in enumerate(outputs):
                if index >= 0 and active(index) and domain(index) != (scale, zero):
                    new = internal(index, scale, zero)
                    outputs[j] = new
                    post.append(bridge(new, index))
            op.inputs, op.outputs = inputs, outputs
            rewritten += [op, *post]
            records.append({'strategy': 'explicit_same_domain_requantization',
                            'opcode': int(opcode), 'scale': scale, 'zero_point': zero})
        graph.operators = rewritten
    builder = flatbuffers.Builder(len(blob) * 2)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    return bytes(builder.Output()), records
