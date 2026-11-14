"""Pinned converter compatibility fixes, scoped to this process only."""
import copy
import os
import importlib.metadata
import onnx2tf.tflite_builder.quantization as quantization

# onnx2tf 2.6.9 copies the depthwise kernel's quantized_dimension=3 to its
# rank-one bias tensor. Bias scales are per output channel along bias axis 0.
# Keep all scales and rounded bias values unchanged; fix only the metadata axis.
if importlib.metadata.version("onnx2tf") != "2.6.9":
    raise RuntimeError("Revalidate converter compatibility fixes before changing the pinned onnx2tf version")
_original_bias = quantization._quantize_bias_tensor

def _bias_axis_fix(*, tensor, input_qparams, weight_qparams, reporter):
    bias_qparams = copy.copy(weight_qparams)
    bias_qparams.quantized_dimension = 0
    return _original_bias(tensor=tensor, input_qparams=input_qparams,
                          weight_qparams=bias_qparams, reporter=reporter)

quantization._quantize_bias_tensor = _bias_axis_fix

# Coral consumes INT8 only. The CLI otherwise also builds unrelated INT16
# activation variants, whose depthwise biases need INT64 and fail validation in
# this pinned release. Explicitly skip those variants; INT8 validation stays on.
import onnx2tf.tflite_builder as builder
builder.strict_int16_activation_skip_reasons = lambda model_ir: ["torch_coral requests INT8 only; INT16 variants are not applicable"]

if os.environ.get('TORCH_CORAL_OPERATOR_STRATEGY') == 'coral':
    from converter_strategy import install
    install()

if __name__ == "__main__":
    import onnx2tf
    onnx2tf.main()
