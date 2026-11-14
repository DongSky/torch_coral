"""Validate decomposition math separately from INT8 and hardware acceptance."""
import pytest
import torch
from torch_coral.experimental_ops import BroadcastBMM, TanhGELU, ExpandedRMSNorm, StaticCumsum


@pytest.mark.parametrize('shapes', [((1, 4, 16), (1, 16, 4)), ((2, 3, 5), (1, 5, 7))])
def test_broadcast_bmm_matches_dynamic_matmul(shapes):
    generator = torch.Generator().manual_seed(11)
    a, b = (torch.randn(shape, generator=generator) for shape in shapes)
    torch.testing.assert_close(BroadcastBMM()(a, b), a @ b)


def test_gelu_is_the_tanh_approximation():
    x = torch.linspace(-6, 6, 301)
    torch.testing.assert_close(TanhGELU()(x), torch.nn.functional.gelu(x, approximate='tanh'))


def test_rmsnorm_math_before_quantization():
    x = torch.randn(1, 4, 16)
    torch.testing.assert_close(ExpandedRMSNorm(1e-5)(x), torch.nn.functional.rms_norm(x, (16,), eps=1e-5))


def test_static_cumsum_math():
    x = torch.randn(1, 4, 16)
    torch.testing.assert_close(StaticCumsum(16)(x), x.cumsum(-1))
