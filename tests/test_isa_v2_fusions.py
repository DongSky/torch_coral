"""FP32 rounding, SRAM aliasing and residual omission contracts for Turbo V2."""
import numpy as np
import pytest
pytest.importorskip('coral.isa.scalar_core')
from coral.isa.scalar_core import run
from torch_coral.isa_streamed import StreamedMatrixTPU, time_chunks, tile_columns
from torch_coral.isa_optimized import OptimizedMatrixTPU
from torch_coral.isa_mxu import MatrixTPU
from torch_coral.qwen_tpu import QwenTPU


class Offline:
    def __init__(self): self.components=1
    def _run(self, program, payload, count):
        _,words,base=program
        memory=bytearray(16384); memory[base:base+len(payload)]=payload
        run(words,memory,load_latency=5)
        return np.frombuffer(memory,'<f4',count=count).copy()
    def _inplace_result(self,program,payload,count):
        _,words,base,out_base=program
        memory=bytearray(16384); memory[base:base+len(payload)]=payload
        run(words,memory,load_latency=5)
        return np.frombuffer(memory,'<f4',count=count,offset=out_base).copy()


class Old(Offline,OptimizedMatrixTPU): pass
class New(Offline,StreamedMatrixTPU): pass


def bits(a,b):
    np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(b).view(np.uint8))


@pytest.mark.parametrize('length',[128,1024])
@pytest.mark.parametrize('weighted,l2',[(False,False),(True,False),(False,True),(True,True)])
def test_batch_norm_matches_turbo(length,weighted,l2):
    x=np.random.default_rng(81).normal(size=(17,length)).astype(np.float32); x[0]=0
    w=np.linspace(.1,2.,length,dtype=np.float32) if weighted else None
    bits(New().normalize(x,w,l2,1e-6),Old().normalize(x,w,l2,1e-6))


@pytest.mark.parametrize('length',[1,13,2048])
def test_softmax_matches_original_chain(length):
    old=object.__new__(QwenTPU); old.backend=Old(); old.cancel=None
    x=np.linspace(-50,50,length,dtype=np.float32)
    bits(New().softmax(x),old.softmax(x))


def test_rope_split_half_and_unchanged_tail():
    rng=np.random.default_rng(43)
    old=object.__new__(QwenTPU); old.backend=Old(); old.cancel=None; old.rope_dim=64
    old.cos=rng.normal(size=(1,32)).astype(np.float32); old.sin=rng.normal(size=(1,32)).astype(np.float32)
    x=rng.normal(size=(33,256)).astype(np.float32)
    bits(New().rope(x,old.cos[0],old.sin[0]),old.rope(x,0))


@pytest.mark.parametrize('length',[4,1024,2560,3072])
def test_larger_inplace_correction_and_readout_alias_safely(length):
    rng=np.random.default_rng(71)
    q=rng.integers(0,256,size=length,dtype=np.uint8); bias=rng.integers(-10000000,10000000,size=length,dtype=np.int32)
    bits(New()._correct(q,bias,4096.),MatrixTPU._correct(Old(),q,bias,4096.))
    scales=rng.uniform(.001,.02,size=length).astype(np.float32)
    scale=np.asarray([.037],np.float32); outputs=[q,q[::-1].copy()]
    bits(New()._reconstruct(outputs,(4096.,32.),scales,scale),Old()._reconstruct(outputs,(4096.,32.),scales,scale))


def test_only_projection_primary_path_omits_unused_residual():
    b=New(); x=np.linspace(-3,3,1024,dtype=np.float32)
    actual=b.quantize_projection_input(x); expected=b.quantize_input(x)
    bits(actual[0],expected[0]); bits(actual[1],expected[1]); assert actual[2] is None
    assert expected[2].shape==x.shape
    b.components=2
    for a,e in zip(b.quantize_projection_input(x),expected): bits(a,e)


def test_temporal_planner_respects_working_space_for_every_chunk():
    assert time_chunks(128,128,13)==(7,6)
    assert tile_columns(128,1)==29
    for steps in range(1,17):
        chunks=time_chunks(128,128,steps)
        assert sum(chunks)==steps
        for count in chunks:
            columns=tile_columns(128,count)
            size=((columns*128+count*(258+columns))*4+7)//8*8
            assert size<=16000 and 4*count*columns<=16384-size
