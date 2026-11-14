"""Exact fusion arithmetic and scoped input reuse; no USB required."""
from collections import OrderedDict
import threading
import numpy as np
import pytest
pytest.importorskip('coral.isa.scalar_core')
from coral.isa.scalar_core import run
from torch_coral.isa_io import IOOptimizedMatrixTPU
from torch_coral.isa_streamed import StreamedMatrixTPU


class Emulated:
    def __init__(self):
        self.components = 1
        self.compute_fusions = self.reuse_projection_quantization = True
        self.wide_projection_quantization = True
        self._quant_scope = None
        self._normalized_quantization = OrderedDict()
        self.lock = threading.RLock()
        self.calls = 0
        self.matrix_stats = dict(reused_quantization_rows=0, wide_quantization_calls=0,
                                 fused_normalization_rows=0, fused_silu_multiply_calls=0)
    def _run(self, program, payload, count):
        _, words, base = program
        memory = bytearray(16384); memory[base:base+len(payload)] = payload
        run(words, memory, load_latency=5); self.calls += 1
        return np.frombuffer(memory, '<f4', count=count).copy()
    def _inplace_result(self, program, payload, count):
        _, words, base, output_base = program
        memory = bytearray(16384); memory[base:base+len(payload)] = payload
        run(words, memory, load_latency=5); self.calls += 1
        return np.frombuffer(memory, '<f4', count=count, offset=output_base).copy()


class Backend(Emulated, IOOptimizedMatrixTPU):
    pass


def bits(a,b):
    np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(b).view(np.uint8))


@pytest.mark.parametrize('length', [1, 128, 1024, 2000, 3584])
def test_silu_multiply_matches_separate_operations(length):
    b=Backend(); rng=np.random.default_rng(917)
    gate=rng.normal(size=length).astype(np.float32)*np.float32(10)
    other=rng.normal(size=length).astype(np.float32)
    expected=b.op('mul',b.activation('silu',gate),other)
    bits(b.silu_multiply(gate,other),expected)


@pytest.mark.parametrize('rows', [1, 3])
def test_norm_quantization_matches_both_original_outputs(rows):
    b=Backend(); rng=np.random.default_rng(123)
    x=rng.normal(size=(rows,1024)).astype(np.float32); x[0]=0
    weight=rng.normal(size=1024).astype(np.float32)
    expected=b.normalize(x,weight,False,1e-6)
    actual=b.normalize_project_input(x,weight,False,1e-6)
    bits(actual,expected)
    before=b.calls
    with b.projection_scope(actual) as frozen:
        for index,row in enumerate(frozen):
            q,scale,_=b.quantize_projection_input(row)
            ref=StreamedMatrixTPU.quantize_projection_input(b,expected[index])
            bits(q,ref[0]); bits(scale,ref[1])
    assert b.calls-before == rows  # Only the explicit reference quantizers ran.


def test_scoped_quantization_pins_snapshot_and_never_reuses_changed_input():
    b=Backend(); source=np.linspace(-2,2,1024,dtype=np.float32)
    with b.projection_scope(source) as frozen:
        q=b.quantize_projection_input(frozen); count=b.calls
        source[:]=0
        same=b.quantize_projection_input(frozen)
        assert b.calls == count and same is q
    assert b._quant_scope is None
    with b.projection_scope(source) as frozen:
        other=b.quantize_projection_input(frozen)
        assert b.calls == count+1
        assert not np.array_equal(other[0],q[0])
    normalized=b.normalize_project_input(source,np.ones(1024,np.float32),False,1e-6)
    normalized[:]=1
    before=b.calls
    with b.projection_scope(normalized) as frozen:
        b.quantize_projection_input(frozen)
    assert b.calls==before+1  # Modified normalized output cannot hit the fused cache.


@pytest.mark.parametrize('length', [4, 512, 2048])
@pytest.mark.parametrize('tied', [False, True])
def test_fused_reconstruction_topk_keeps_all_logits_and_stable_ties(length,tied):
    from torch_coral.isa_v2_fusions import reconstruct_inplace_program
    from torch_coral.isa_batched import block_topk_program
    b=Backend(); rng=np.random.default_rng(19)
    outputs=[rng.integers(0,256,size=length,dtype=np.uint8) for _ in range(2)]
    scales=rng.uniform(.001,.01,size=length).astype(np.float32)
    if tied: scales[:]=0
    input_scale=np.asarray([.03],np.float32); coefficients=(4096.,32.)
    payload=b''.join(v.tobytes() for v in outputs)+scales.tobytes()+input_scale.tobytes()
    expected=b._inplace_result(reconstruct_inplace_program(length,coefficients),payload,length)
    top=b._run(block_topk_program(length,4),expected.tobytes(),8)
    actual=b._inplace_result(reconstruct_inplace_program(length,coefficients,4),payload+b'\0'*32,length+8)
    bits(actual[:length],expected); bits(actual[length:],top)
