import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
import pytest
import torch

from torch_coral.rewrite_onnx import rewrite


def fixture(tmp_path, op, x, *, constants=None, attributes=None, outputs=None, approximations=None):
    constants = constants or {}
    outputs = outputs or [('output_0', onnx.TensorProto.FLOAT, None)]
    node_outputs = [v[0] for v in outputs]
    if op == 'TopK' and len(node_outputs) == 1:
        node_outputs.append('unused_indices')
    node = h.make_node(op, ['input_0', *constants], node_outputs, **(attributes or {}))
    graph = h.make_graph([node], 'test', [h.make_tensor_value_info('input_0', onnx.TensorProto.FLOAT, list(x.shape))],
                        [h.make_tensor_value_info(*v) for v in outputs], [nh.from_array(v, k) for k, v in constants.items()])
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 18)])
    model.ir_version = 10
    onnx.save(model, tmp_path / 'model.onnx')
    np.save(tmp_path / 'calibration_0.npy', x[None])
    job = {'inputs': [{'name': 'input_0'}], 'quantization_atol': .05, 'quantization_rtol': .05,
           'approximations': approximations or {}}
    return rewrite(tmp_path, job)


@pytest.mark.parametrize('reverse,exclusive', [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_cumsum_reverse_and_exclusive(tmp_path, reverse, exclusive):
    report = fixture(tmp_path, 'CumSum', np.array([[1, -2, 3, 4]], np.float32),
                     constants={'axis': np.array(-1, np.int64)},
                     attributes={'reverse': reverse, 'exclusive': exclusive})
    assert report['float_rewrite_max_abs_error'] == 0


def test_topk_duplicates_negative_values_and_sorted_order(tmp_path):
    report = fixture(tmp_path, 'TopK', np.array([[-2, -2, 0, -1, 3, 3, 2, 1]], np.float32),
                     constants={'k': np.array([4], np.int64)},
                     outputs=[('output_0', onnx.TensorProto.FLOAT, [1, 4])])
    assert report['float_rewrite_max_abs_error'] == 0


def test_topk_index_output_is_explicitly_rejected(tmp_path):
    with pytest.raises(ValueError, match='values only'):
        fixture(tmp_path, 'TopK', np.arange(8, dtype=np.float32)[None],
                constants={'k': np.array([4], np.int64)},
                outputs=[('output_0', onnx.TensorProto.FLOAT, [1, 4]),
                         ('indices', onnx.TensorProto.INT64, [1, 4])])


@pytest.mark.parametrize('axis', [0, 1, -1])
def test_static_gather_with_negative_indices_and_axes(tmp_path, axis):
    x = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    indices = np.array([-1, 0], np.int64)
    report = fixture(tmp_path, 'Gather', x, constants={'indices': indices}, attributes={'axis': axis})
    assert report['float_rewrite_max_abs_error'] == 0


def test_approximation_requires_explicit_domain(tmp_path):
    with pytest.raises(ValueError, match='explicit approximations'):
        fixture(tmp_path, 'Sin', np.array([[-1, 0, 1]], np.float32))


def test_soft_lookup_has_small_float_error_and_records_saturation(tmp_path):
    report = fixture(tmp_path, 'Sin', np.linspace(-np.pi, np.pi, 129, dtype=np.float32)[None],
                     approximations={'Sin': {'range': [-np.pi, np.pi], 'segments': 32}})
    assert report['float_rewrite_max_abs_error'] < .02
    assert report['rewrites'][0]['saturates_outside_domain']


def test_bad_approximation_domain_is_not_published(tmp_path):
    with pytest.raises(AssertionError):
        fixture(tmp_path, 'Sin', np.array([[1]], np.float32),
                approximations={'Sin': {'range': [-.1, .1], 'segments': 16}})
    assert not (tmp_path / 'model.original.onnx').exists()


@pytest.mark.parametrize('coordinate,rounding', [('asymmetric', 'floor'),
    ('half_pixel', 'round_prefer_floor'), ('pytorch_half_pixel', 'ceil'), ('align_corners', 'round_prefer_ceil')])
def test_fractional_nearest_resize_preserves_coordinate_rules(tmp_path, coordinate, rounding):
    report = fixture(tmp_path, 'Resize', np.arange(30, dtype=np.float32).reshape(1, 2, 3, 5),
                     constants={'roi': np.array([], np.float32), 'scales': np.array([1, 1, 1.7, 2.4], np.float32)},
                     attributes={'mode': 'nearest', 'coordinate_transformation_mode': coordinate, 'nearest_mode': rounding})
    assert report['float_rewrite_max_abs_error'] == 0


@pytest.mark.parametrize('value', [.5, 16, -1])
def test_embedding_invalid_ids_fail_before_opening_usb(value):
    import torch_coral
    folder = Path(__file__).resolve().parents[1] / 'artifacts/operator-strategies/integer_embedding'
    if not (folder / 'manifest.json').exists():
        pytest.skip('Build operator strategy integration artifacts first')
    with torch_coral.load(folder) as model:
        with pytest.raises(ValueError, match='integer-valued|outside the bounded'):
            model(torch.full((1, 16), value, dtype=torch.float32))


def test_gui_strategy_is_explicit_and_invalid_json_starts_no_job(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from torch_coral.gui import Workbench
    app = QApplication.instance() or QApplication([])
    window = Workbench()
    requests = []
    window.start_job = requests.append
    assert not window.operator_strategy.isChecked()
    assert not window.approximations.isEnabled()
    window.operator_strategy.setChecked(True)
    window.approximations.setText('invalid json')
    window.convert()
    assert not requests
    window.approximations.setText('{"Sin":{"range":[-3.14,3.14],"segments":32}}')
    window.convert()
    assert requests[0]['operator_strategy'] == 'coral'
    assert requests[0]['approximations']['Sin']['segments'] == 32
    window.close()
