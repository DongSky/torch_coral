"""Explicit, bounded operator lowering for static Coral inference graphs."""
import copy
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
import onnxruntime as ort


def rewrite(folder, job):
    path = Path(folder) / 'model.onnx'
    original = onnx.load(path)
    model = onnx.shape_inference.infer_shapes(copy.deepcopy(original))
    constants = {t.name: nh.to_array(t) for t in model.graph.initializer}
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in [*model.graph.input, *model.graph.output, *model.graph.value_info]}
    producers = {name: n for n in model.graph.node for name in n.output}
    used = {name for n in model.graph.node for name in n.input} | {v.name for v in model.graph.output}
    for n in model.graph.node:
        if n.op_type == 'Constant':
            value = next((a.t for a in n.attribute if a.name == 'value'), None)
            if value is not None:
                constants[n.output[0]] = nh.to_array(value)
    nodes, records, constraints = [], [], []
    counter = 0

    def name(label):
        nonlocal counter
        counter += 1
        value = f'coral_strategy_{counter}_{label}'
        if value in shapes or value in constants or value in producers:
            raise ValueError('Generated ONNX name collision')
        return value

    def const(value, label='constant'):
        key = name(label)
        array = np.asarray(value)
        model.graph.initializer.append(nh.from_array(array, key))
        constants[key] = array
        return key

    def emit(op, inputs, output=None, **attrs):
        output = output or name(op)
        nodes.append(h.make_node(op, inputs, [output], name=name(op + '_node'), **attrs))
        return output

    def attrs(n):
        return {a.name: h.get_attribute_value(a) for a in n.attribute}

    def static_shape(key):
        shape = shapes.get(key, list(constants[key].shape) if key in constants else None)
        if not shape or any(d <= 0 for d in shape):
            raise ValueError('Coral strategy requires a known static shape: ' + key)
        return shape

    def abs_value(x):
        return emit('Max', [x, emit('Mul', [x, const(np.array(-1, np.float32))])])

    def hats(x, knots, step):
        expanded = emit('Unsqueeze', [x, const(np.array([-1], np.int64))])
        delta = emit('Sub', [expanded, const(np.asarray(knots, np.float32))])
        distance = emit('Mul', [abs_value(delta), const(np.array(1 / step, np.float32))])
        return emit('Relu', [emit('Sub', [const(np.array(1, np.float32)), distance])])

    def pwl(node, options):
        lo, hi = map(float, options['range'])
        segments = int(options.get('segments', 16))
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi and 2 <= segments <= 64):
            raise ValueError('PWL requires a finite increasing range and 2..64 segments')
        functions = {'Sin': np.sin, 'Cos': np.cos, 'Exp': np.exp,
                     'Softplus': lambda x: np.logaddexp(0, x),
                     'Erf': np.vectorize(math.erf)}
        fn = functions[node.op_type]
        knots = np.linspace(lo, hi, segments + 1, dtype=np.float32)
        weights = np.asarray(fn(knots), np.float32)
        if not np.isfinite(weights).all():
            raise ValueError('PWL output range is not finite')
        x = emit('Max', [node.input[0], const(np.array(lo, np.float32))])
        x = emit('Min', [x, const(np.array(hi, np.float32))])
        method = options.get('method', 'soft_lookup')
        step = (hi - lo) / segments
        if method == 'soft_lookup':
            # Normalized weights avoid the amplitude drift of quantized hat
            # bases. Linear ghost samples remove the one-sided endpoint bias.
            knots = np.concatenate(([lo - step], knots, [hi + step])).astype(np.float32)
            weights = np.concatenate(([2 * weights[0] - weights[1]], weights,
                                      [2 * weights[-1] - weights[-2]])).astype(np.float32)
            expanded = emit('Unsqueeze', [x, const(np.array([-1], np.int64))])
            delta = emit('Sub', [expanded, const(knots)])
            logits = emit('Mul', [abs_value(delta), const(np.array(-1.5 / step, np.float32))])
            basis = emit('Softmax', [logits], axis=-1)
        elif method == 'pwl':
            basis = hats(x, knots, step)
        else:
            raise ValueError('Approximation method must be soft_lookup or pwl')
        result = emit('MatMul', [basis, const(weights[:, None])])
        emit('Squeeze', [result, const(np.array([-1], np.int64))], node.output[0])
        dense = np.linspace(lo, hi, 10001)
        if method == 'soft_lookup':
            logits = -1.5 / step * np.abs(dense[:, None] - knots)
            probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
            estimate = probabilities @ weights / probabilities.sum(axis=-1)
        else:
            estimate = np.interp(dense, knots, weights)
        error = np.abs(estimate - fn(dense)).max()
        records.append({'op': node.op_type, 'strategy': 'normalized_soft_lookup' if method == 'soft_lookup' else 'piecewise_linear_hat_basis',
                        'approximate': True,
                        'domain': [lo, hi], 'segments': segments, 'saturates_outside_domain': True,
                        'sampled_float_max_abs_error': float(error)})

    for node in model.graph.node:
        a = attrs(node)
        op = node.op_type
        if op == 'CumSum':
            shape = static_shape(node.input[0])
            if node.input[1] not in constants:
                raise ValueError('Dynamic Cumsum axis cannot be statically lowered')
            axis = int(constants[node.input[1]].item()) % len(shape)
            length = shape[axis]
            if length > 256:
                raise ValueError('Dense Cumsum lowering is limited to 256 elements per axis')
            weights = np.tril(np.ones((length, length), np.float32)) if a.get('reverse', 0) else np.triu(np.ones((length, length), np.float32))
            if a.get('exclusive', 0):
                np.fill_diagonal(weights, 0)
            perm = [i for i in range(len(shape)) if i != axis] + [axis]
            x = node.input[0]
            if axis != len(shape) - 1:
                x = emit('Transpose', [x], perm=perm)
            y = emit('MatMul', [x, const(weights)])
            emit('Transpose', [y], node.output[0], perm=np.argsort(perm).tolist())
            records.append({'op': op, 'strategy': 'static_triangular_matmul', 'length': length,
                            'constant_elements': length * length})
        elif op == 'Gather' and node.input[1] in constants:
            shape = static_shape(node.input[0])
            axis = int(a.get('axis', 0)) % len(shape)
            indices = np.asarray(constants[node.input[1]], np.int64)
            if indices.size > 256 or shape[axis] * indices.size > 65536:
                raise ValueError('Static Gather selector exceeds the bounded lowering budget')
            if np.any(indices < -shape[axis]) or np.any(indices >= shape[axis]):
                raise ValueError('Gather index out of range')
            indices = indices % shape[axis]
            selector = np.eye(shape[axis], dtype=np.float32)[:, indices.reshape(-1)]
            perm = [i for i in range(len(shape)) if i != axis] + [axis]
            x = emit('Transpose', [node.input[0]], perm=perm)
            y = emit('MatMul', [x, const(selector)])
            outer = [shape[i] for i in perm[:-1]]
            y = emit('Reshape', [y, const(np.array(outer + list(indices.shape), np.int64))])
            q = indices.ndim
            out_perm = list(range(axis)) + list(range(len(shape) - 1, len(shape) - 1 + q)) + list(range(axis, len(shape) - 1))
            emit('Transpose', [y], node.output[0], perm=out_perm)
            records.append({'op': op, 'strategy': 'constant_selector_matmul', 'indices': indices.tolist()})
        elif op == 'Gather' and node.input[0] in constants and int(a.get('axis', 0)) == 0:
            table = np.asarray(constants[node.input[0]], np.float32)
            if table.ndim != 2 or not (2 <= table.shape[0] <= 32):
                raise ValueError('Bounded integer embedding strategy requires a constant 2D table with 2..32 rows')
            index = node.input[1]
            while index in producers and producers[index].op_type in ('Cast', 'Identity'):
                index = producers[index].input[0]
            if index not in {v.name for v in model.graph.input}:
                raise ValueError('Bounded integer embedding indices must originate from a public input')
            values = np.load(Path(folder) / f"calibration_{next(i for i, spec in enumerate(job['inputs']) if spec['name'] == index)}.npy")
            if np.any(values != np.rint(values)) or values.min() < 0 or values.max() >= table.shape[0]:
                raise ValueError('Embedding calibration inputs must be integer-valued IDs in the table range')
            expanded = emit('Unsqueeze', [index, const(np.array([-1], np.int64))])
            delta = emit('Sub', [expanded, const(np.arange(table.shape[0], dtype=np.float32))])
            score = emit('Sub', [const(np.array(.5, np.float32)), abs_value(delta)])
            score = emit('Mul', [score, const(np.array(4., np.float32))])
            gate = emit('Min', [emit('Relu', [score]), const(np.array(1., np.float32))])
            emit('MatMul', [gate, const(table)], node.output[0])
            constraints.append({'name': index, 'integer_values': True, 'min': 0, 'max': table.shape[0] - 1})
            records.append({'op': op, 'strategy': 'bounded_integer_indicator_embedding',
                            'rows': table.shape[0], 'integer_input_required': True,
                            'float_semantics': 'exact for valid integer-valued IDs; INT8 error checked separately'})
        elif op == 'GatherND' and node.input[1] in constants:
            shape = static_shape(node.input[0])
            indices = np.asarray(constants[node.input[1]], np.int64)
            if a.get('batch_dims', 0) != 0 or indices.shape[-1] != 1 or indices.size > 256:
                raise ValueError('GatherND selector strategy currently requires batch_dims=0 and index depth 1')
            selected = indices[..., 0]
            if np.any(selected < -shape[0]) or np.any(selected >= shape[0]):
                raise ValueError('GatherND index out of range')
            size = int(np.prod(shape[1:]))
            if selected.size * shape[0] > 65536:
                raise ValueError('GatherND selector exceeds the static matrix budget')
            x = emit('Reshape', [node.input[0], const(np.array([shape[0], size], np.int64))])
            selector = np.eye(shape[0], dtype=np.float32)[selected.reshape(-1) % shape[0]]
            x = emit('Transpose', [x], perm=[1, 0])
            y = emit('MatMul', [x, const(selector.T.copy())])
            y = emit('Transpose', [y], perm=[1, 0])
            emit('Reshape', [y, const(np.array(list(selected.shape) + shape[1:], np.int64))], node.output[0])
            records.append({'op': op, 'strategy': 'constant_selector_matmul', 'indices': selected.tolist()})
        elif op == 'Resize':
            shape = static_shape(node.input[0])
            target = static_shape(node.output[0])
            mode = a.get('mode', b'nearest')
            mode = mode.decode() if isinstance(mode, bytes) else mode
            coordinate = a.get('coordinate_transformation_mode', b'half_pixel')
            coordinate = coordinate.decode() if isinstance(coordinate, bytes) else coordinate
            nearest = a.get('nearest_mode', b'round_prefer_floor')
            nearest = nearest.decode() if isinstance(nearest, bytes) else nearest
            if mode != 'nearest' or len(shape) != 4 or shape[:2] != target[:2]:
                raise ValueError('Resize selector strategy supports static nearest NCHW spatial resize')
            scales = {}
            if len(node.input) > 2 and node.input[2] in constants and constants[node.input[2]].size:
                values = np.asarray(constants[node.input[2]]).reshape(-1)
                axes = a.get('axes', list(range(4)))
                if len(values) != len(axes) or np.any(values <= 0) or not np.isfinite(values).all():
                    raise ValueError('Invalid constant Resize scales')
                scales = {int(axis) % 4: float(value) for axis, value in zip(axes, values)}
            elif not (len(node.input) > 3 and node.input[3] in constants):
                raise ValueError('Resize strategy requires constant scales or sizes')
            x = node.input[0]
            for axis in (2, 3):
                n, m = shape[axis], target[axis]
                if n == m:
                    continue
                if n * m > 65536:
                    raise ValueError('Resize selector exceeds the 65536-element matrix budget per axis')
                position = np.arange(m, dtype=np.float64)
                inverse_scale = 1 / scales[axis] if axis in scales else n / m
                if coordinate == 'asymmetric': position *= inverse_scale
                elif coordinate in ('half_pixel', 'pytorch_half_pixel'):
                    position = (position + .5) * inverse_scale - .5 if m > 1 or coordinate == 'half_pixel' else position * 0
                elif coordinate == 'align_corners': position *= (n - 1) / (m - 1) if m > 1 else 0
                else: raise ValueError('Unsupported nearest Resize coordinate mode: ' + coordinate)
                if nearest == 'floor': indices = np.floor(position)
                elif nearest == 'ceil': indices = np.ceil(position)
                elif nearest == 'round_prefer_floor': indices = np.ceil(position - .5)
                elif nearest == 'round_prefer_ceil': indices = np.floor(position + .5)
                else: raise ValueError('Unsupported nearest Resize rounding mode: ' + nearest)
                indices = np.clip(indices.astype(int), 0, n - 1)
                selector = np.eye(n, dtype=np.float32)[:, indices]
                perm = [i for i in range(4) if i != axis] + [axis]
                x = emit('Transpose', [x], perm=perm)
                x = emit('MatMul', [x, const(selector)])
                x = emit('Transpose', [x], perm=np.argsort(perm).tolist())
                shape = list(shape)
                shape[axis] = m
            emit('Identity', [x], node.output[0])
            records.append({'op': op, 'strategy': 'separable_nearest_selector_matmul', 'target_shape': target})
        elif op == 'TopK':
            if len(node.output) > 1 and node.output[1] in used:
                raise ValueError('TopK sorting strategy currently supports values only; index output requires a separate routing strategy')
            shape = static_shape(node.input[0])
            axis = int(a.get('axis', -1)) % len(shape)
            if axis != len(shape) - 1 or node.input[1] not in constants:
                raise ValueError('TopK strategy requires a static k and the last axis')
            n, k = shape[-1], int(constants[node.input[1]].item())
            if n > 32 or n & (n - 1) or not (1 <= k <= n):
                raise ValueError('TopK sorting network requires power-of-two length <=32 and 1<=k<=length')
            lanes = [emit('Slice', [node.input[0], const(np.array([i], np.int64)),
                      const(np.array([i + 1], np.int64)), const(np.array([axis], np.int64))]) for i in range(n)]
            width = 2
            comparators = 0
            while width <= n:
                step = width // 2
                while step:
                    updated = list(lanes)
                    for i in range(n):
                        other = i ^ step
                        if other <= i:
                            continue
                        high = emit('Max', [lanes[i], lanes[other]])
                        low = emit('Min', [lanes[i], lanes[other]])
                        descending = (i & width) == 0
                        updated[i], updated[other] = (high, low) if descending else (low, high)
                        comparators += 1
                    lanes = updated
                    step //= 2
                width *= 2
            if not a.get('largest', 1):
                lanes.reverse()
            emit('Concat', lanes[:k], node.output[0], axis=axis)
            records.append({'op': op, 'strategy': 'bitonic_minmax_network', 'length': n,
                            'k': k, 'comparators': comparators, 'outputs': 'values_only'})
        elif op in ('Sin', 'Cos', 'Exp', 'Softplus', 'Erf'):
            options = job.get('approximations', {}).get(op)
            if options is None:
                raise ValueError(f'{op} needs an explicit approximations["{op}"] range and segment count')
            pwl(node, options)
        elif op == 'Where' and producers.get(node.input[0]) is not None:
            condition = producers[node.input[0]]
            value = producers.get(node.input[2])
            options = job.get('approximations', {}).get('Softplus')
            if (options and condition.op_type == 'Greater' and value is not None
                    and value.op_type == 'Softplus' and value.input[0] == node.input[1]
                    and condition.input[0] == node.input[1] and condition.input[1] in constants
                    and float(np.asarray(constants[condition.input[1]]).item()) >= options['range'][1]):
                emit('Identity', [node.input[2]], node.output[0])
                records.append({'op': op, 'strategy': 'softplus_threshold_branch_outside_declared_domain'})
            else:
                nodes.append(node)
        elif op == 'Abs':
            emit('Max', [node.input[0], emit('Mul', [node.input[0], const(np.array(-1, np.float32))])], node.output[0])
            records.append({'op': op, 'strategy': 'maximum_positive_negative'})
        elif op == 'Neg':
            emit('Mul', [node.input[0], const(np.array(-1, np.float32))], node.output[0])
            records.append({'op': op, 'strategy': 'multiply_negative_one'})
        else:
            nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # Remove unused TopK index outputs, stale Casts, and other dead subgraphs.
    needed = {v.name for v in model.graph.output}
    live = []
    for node in reversed(model.graph.node):
        if needed.intersection(node.output):
            live.append(node)
            needed.update(node.input)
    del model.graph.node[:]
    model.graph.node.extend(reversed(live))
    initializers = [t for t in model.graph.initializer if t.name in needed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    before = ort.InferenceSession(original.SerializeToString(), providers=['CPUExecutionProvider'])
    after = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    data = [np.load(Path(folder) / f'calibration_{i}.npy', allow_pickle=False) for i in range(len(job['inputs']))]
    maximum = 0.
    for sample in range(min(len(data[0]), 8)):
        feed = {spec['name']: array[sample] for spec, array in zip(job['inputs'], data)}
        for expected, actual in zip(before.run(None, feed), after.run(None, feed)):
            maximum = max(maximum, float(np.max(np.abs(expected - actual))))
            approximate = any(r.get('approximate') for r in records)
            np.testing.assert_allclose(actual, expected,
                atol=job['quantization_atol'] if approximate else 1e-5,
                rtol=job['quantization_rtol'] if approximate else 1e-4,
                err_msg='Operator rewrite failed comparison against the original ONNX model')
    onnx.save(original, Path(folder) / 'model.original.onnx')
    onnx.save(model, path)
    return {'rewrites': records, 'float_rewrite_max_abs_error': maximum,
            'calibration_samples_checked': min(len(data[0]), 8), 'input_constraints': constraints}
