"""Legacy TPU compatibility rewrites with numerical verification by the caller."""
import copy
import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as s

def fold_flatten_transpose(blob):
    """Fold NHWC->NCHW transpose before flatten/linear into FC constant weights.

    W * flatten(transpose(x)) == W[:, inverse_permutation] * flatten(x).
    This avoids an unsupported activation transpose on the old Edge TPU, while
    retaining the original PyTorch model and its learned feature ordering.
    """
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(blob, 0))
    changes = []
    types = {s.TensorType.INT8: np.int8, s.TensorType.UINT8: np.uint8, s.TensorType.FLOAT32: np.float32}
    for graph in model.subgraphs:
        def code(op): return model.operatorCodes[op.opcodeIndex].builtinCode
        def consumers(index): return [op for op in graph.operators if index in op.inputs]
        def values(index):
            tensor = graph.tensors[index]
            return np.frombuffer(bytes(model.buffers[tensor.buffer].data), dtype=types[tensor.type]).reshape(tensor.shape)
        remove = []
        for transpose in graph.operators:
            if code(transpose) != s.BuiltinOperator.TRANSPOSE: continue
            trans_out = int(transpose.outputs[0])
            uses = consumers(trans_out)
            if trans_out in graph.outputs or len(uses) != 1 or code(uses[0]) != s.BuiltinOperator.RESHAPE: continue
            reshape = uses[0]
            flat_out = int(reshape.outputs[0])
            uses = consumers(flat_out)
            if flat_out in graph.outputs or len(uses) != 1 or code(uses[0]) != s.BuiltinOperator.FULLY_CONNECTED: continue
            fc = uses[0]
            source = graph.tensors[int(transpose.inputs[0])]
            output = graph.tensors[flat_out]
            weight = graph.tensors[int(fc.inputs[1])]
            if source.type not in types or weight.type not in types: continue
            if len(output.shape) != 2 or output.shape[0] != 1 or source.shape[0] != 1: continue
            if source.quantization is not None and output.quantization is not None:
                if not np.array_equal(source.quantization.scale, output.quantization.scale): continue
                if not np.array_equal(source.quantization.zeroPoint, output.quantization.zeroPoint): continue
            perm_tensor = graph.tensors[int(transpose.inputs[1])]
            if perm_tensor.type != s.TensorType.INT32: continue
            buffer = model.buffers[perm_tensor.buffer].data
            if buffer is None or weight.buffer == 0: continue
            perm = np.frombuffer(bytes(buffer), dtype=np.int32)
            if perm[0] != 0 or sorted(perm.tolist()) != list(range(len(source.shape))): continue
            weights = values(int(fc.inputs[1]))
            if weights.ndim != 2 or weights.shape[1] != int(np.prod(source.shape)): continue
            index_map = np.arange(int(np.prod(source.shape))).reshape(source.shape).transpose(perm).reshape(-1)
            reordered = np.ascontiguousarray(weights[:, np.argsort(index_map)])
            new_buffer = s.BufferT()
            new_buffer.data = np.frombuffer(reordered.tobytes(), dtype=np.uint8).copy()
            model.buffers.append(new_buffer)
            new_weight = copy.deepcopy(weight)
            new_weight.name = weight.name + b"_transpose_folded"
            new_weight.buffer = len(model.buffers)-1
            graph.tensors.append(new_weight)
            fc.inputs = fc.inputs.copy()
            fc.inputs[1] = len(graph.tensors)-1
            reshape.inputs = reshape.inputs.copy()
            reshape.inputs[0] = transpose.inputs[0]
            remove.append(transpose)
            changes.append({"rewrite": "transpose_flatten_fully_connected",
                            "weight_shape": weights.shape, "activation_shape": source.shape.tolist(),
                            "permutation": perm.tolist()})
        graph.operators = [op for op in graph.operators if all(op is not item for item in remove)]
    if not changes: return blob, []
    builder = flatbuffers.Builder(len(blob))
    offset = model.Pack(builder)
    builder.Finish(offset, file_identifier=b"TFL3")
    for change in changes: change["weight_shape"] = list(change["weight_shape"])
    return bytes(builder.Output()), changes
