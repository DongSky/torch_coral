"""Experimental decompositions for *small static* operator investigations.

These are not automatically applied to user models and do not add ISA opcodes.
Quantized accuracy and performance must be measured on each target workload.
"""
import torch

class BroadcastBMM(torch.nn.Module):
    """A @ B via broadcast MUL + SUM; O(B*M*K*N) intermediate elements.

    Unlike a native GEMM, each product becomes a quantized activation before
    reduction. This changes error accumulation and is unsuitable as a blanket
    replacement for large attention matrices.
    """
    def forward(self,a,b):
        return (a.unsqueeze(-1)*b.unsqueeze(-3)).sum(dim=-2)

class TanhGELU(torch.nn.Module):
    """The common tanh approximation; not identical to erf-based GELU."""
    def forward(self,x):
        return .5*x*(1+torch.tanh(.7978845608028654*(x+.044715*x*x*x)))

class ExpandedRMSNorm(torch.nn.Module):
    """Math reference only: current ONNX quantizer still needs RSQRT lowering."""
    def __init__(self,eps=1e-5):
        super().__init__();self.eps=eps
    def forward(self,x):
        return x*torch.rsqrt((x*x).mean(dim=-1,keepdim=True)+self.eps)

class StaticCumsum(torch.nn.Module):
    """Last-axis prefix sum via a constant matrix; O(length**2) weights.

    This trades a scan for dense matrix multiplication. INT8 saturation and
    calibration of the growing output range need workload-specific validation.
    """
    def __init__(self, length):
        super().__init__()
        self.register_buffer('prefix_weights', torch.triu(torch.ones(length, length)))
        self.register_buffer('zero_bias', torch.zeros(length))

    def forward(self, x):
        # Explicit zero bias retains ONNX Gemm / TFLite FullyConnected lowering;
        # MatMul lowering currently chooses unsupported per-axis BatchMatMul.
        return torch.nn.functional.linear(x, self.prefix_weights.T, self.zero_bias)
