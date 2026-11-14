"""Process-local fixes for the pinned converter's Coral operator route.

No site-packages are edited. Standard CNN conversion does not enable this module.
"""
import contextvars
import copy
import numpy as np


def install():
    import onnx2tf.tflite_builder as builder
    import onnx2tf.tflite_builder.quantization as q
    import onnx2tf.tflite_builder.model_writer as writer

    current_op = contextvars.ContextVar('coral_weight_operator', default=None)
    original_axis = q._kernel_weight_quant_axis_for_strict
    original_weight = q._quantize_weight_tensor

    def axis(op_type, tensor):
        current_op.set(op_type)
        return original_axis(op_type, tensor)

    def weight(**kwargs):
        tensor = kwargs['tensor']
        if current_op.get() == 'BATCH_MATMUL':
            if tensor.data is None:
                if tensor.dtype != 'INT8' or not isinstance(tensor.quantization, q.QuantParamIR):
                    raise ValueError('Dynamic BATCH_MATMUL RHS requires calibrated per-tensor INT8')
                return tensor.quantization
            kwargs['quant_type'] = 'per-tensor'
        return original_weight(**kwargs)

    q._kernel_weight_quant_axis_for_strict = axis
    q._quantize_weight_tensor = weight
    q._STRICT_FULL_INTEGER_SUPPORTED_OPS.add('RSQRT')
    # Upstream forces an already-produced input tensor to the output's narrower
    # range, silently clipping e.g. squares before a normalization MEAN. Give
    # every activation its own calibration first, then insert real requantizes.
    q._STRICT_FULL_INTEGER_SAME_QPARAM_INDICES = {}
    original_options = writer._build_builtin_options

    def options(schema_tflite, op):
        if op.op_type == 'RSQRT':
            return writer._enum(schema_tflite, 'BuiltinOptions', 'NONE'), None
        return original_options(schema_tflite, op)

    writer._build_builtin_options = options
    original_lower = builder.lower_onnx_to_ir

    def lower(*args, **kwargs):
        model = original_lower(*args, **kwargs)
        producers = {name: op for op in model.operators for name in op.outputs}
        uses = {name: sum(name in op.inputs for op in model.operators) for name in model.tensors}
        remove = []
        for op in model.operators:
            if op.op_type != 'DIV' or len(op.inputs) != 2:
                continue
            numerator = model.tensors[op.inputs[0]]
            sqrt = producers.get(op.inputs[1])
            if (numerator.data is not None and np.all(np.asarray(numerator.data) == 1)
                    and sqrt is not None and sqrt.op_type == 'SQRT'
                    and uses[sqrt.outputs[0]] == 1 and sqrt.outputs[0] not in model.outputs):
                op.op_type, op.inputs, op.options = 'RSQRT', [sqrt.inputs[0]], {}
                op.version = 1
                remove.append(sqrt)
        model.operators = [op for op in model.operators if all(op is not x for x in remove)]
        return model

    builder.lower_onnx_to_ir = lower
    original_integer = q._build_strict_full_integer_model_ir

    def integer(*args, **kwargs):
        source_model = args[0] if args else kwargs['model_ir']
        kwargs['calibration_ranges'] = union_alias_ranges(source_model, kwargs['calibration_ranges'])
        result = original_integer(*args, **kwargs)
        align_ir_domains(result.model_ir)
        for op in result.model_ir.operators:
            if op.op_type in ('RSQRT', 'BATCH_MATMUL'):
                op.version = 2
        return result

    q._build_strict_full_integer_model_ir = integer


def union_alias_ranges(model, ranges):
    """Keep value-preserving selection networks in the input's full domain.

    A Slice's per-position observations are not a valid bound for later inputs.
    Unify before quantization so no previously encoded bias is invalidated.
    MEAN is excluded: it needs an explicit bridge for useful variance precision.
    """
    from onnx2tf.tflite_builder.quantization import TensorCalibrationRange
    same = {'SLICE', 'STRIDED_SLICE', 'TRANSPOSE', 'RESHAPE', 'SQUEEZE', 'EXPAND_DIMS',
            'SPLIT', 'CONCATENATION', 'MAXIMUM', 'MINIMUM', 'REDUCE_MAX', 'REDUCE_MIN'}
    parent = {}

    def find(n):
        parent.setdefault(n, n)
        if parent[n] != n:
            parent[n] = find(parent[n])
        return parent[n]

    for op in model.operators:
        if op.op_type not in same:
            continue
        names = [n for n in [*op.inputs, *op.outputs] if n in model.tensors and model.tensors[n].dtype == 'FLOAT32']
        if names:
            root = find(names[0])
            for n in names[1:]:
                parent[find(n)] = root
    groups = {}
    for n in parent:
        groups.setdefault(find(n), []).append(n)
    result = dict(ranges)
    for names in groups.values():
        bounds = []
        for n in names:
            if n in ranges:
                bounds.append((ranges[n].min_value, ranges[n].max_value))
            data = model.tensors[n].data
            if data is not None:
                bounds.append((float(np.min(data)), float(np.max(data))))
        if bounds:
            lo, hi = min(v[0] for v in bounds), max(v[1] for v in bounds)
            for n in names:
                result[n] = TensorCalibrationRange(min_value=lo, max_value=hi, num_samples=1)
    return result


def align_ir_domains(model):
    from onnx2tf.tflite_builder.ir import OperatorIR, QuantParamIR
    same = {'SLICE', 'STRIDED_SLICE', 'TRANSPOSE', 'RESHAPE', 'SQUEEZE', 'EXPAND_DIMS',
            'SPLIT', 'CONCATENATION', 'MAXIMUM', 'MINIMUM', 'MEAN', 'REDUCE_MAX', 'REDUCE_MIN'}
    ops = []

    def active(name):
        tensor = model.tensors.get(name)
        return tensor is not None and tensor.dtype == 'INT8' and isinstance(tensor.quantization, QuantParamIR) and len(tensor.quantization.scale) == 1

    def domain(name):
        q = model.tensors[name].quantization
        return float(q.scale[0]), int(q.zero_point[0])

    def internal(name, scale, zero):
        key = f'coral_requantize_{len(model.tensors)}'
        if key in model.tensors:
            raise ValueError('Internal quantization tensor collision')
        tensor = copy.deepcopy(model.tensors[name])
        tensor.name, tensor.data = key, None
        tensor.quantization = QuantParamIR(scale=[scale], zero_point=[zero])
        model.tensors[key] = tensor
        return key

    for op in model.operators:
        names = [n for n in [*op.inputs, *op.outputs] if active(n)]
        if op.op_type not in same or len({domain(n) for n in names}) <= 1:
            ops.append(op)
            continue
        lo = min((-128 - domain(n)[1]) * domain(n)[0] for n in names)
        hi = max((127 - domain(n)[1]) * domain(n)[0] for n in names)
        scale = np.float32(max((hi - lo) / 255, 1e-12)).item()
        zero = int(np.clip(np.rint(-128 - lo / scale), -128, 127))
        post = []
        for j, name in enumerate(op.inputs):
            if active(name) and domain(name) != (scale, zero):
                new = internal(name, scale, zero)
                ops.append(OperatorIR('QUANTIZE', [name], [new], version=2))
                op.inputs[j] = new
        for j, name in enumerate(op.outputs):
            if active(name) and domain(name) != (scale, zero):
                new = internal(name, scale, zero)
                op.outputs[j] = new
                post.append(OperatorIR('QUANTIZE', [new], [name], version=2))
        ops += [op, *post]
    model.operators = ops
