"""Shorter load waits must relocate nested loops and column/time branches."""
import numpy as np
import pytest
pytest.importorskip('coral.isa.scalar_core')
from coral.isa.scalar_core import run
from torch_coral.isa_schedule import compact_load_waits
from torch_coral.isa_scalar import elementwise,gemm
from torch_coral.isa_v2_fusions import norm_rows_program,primary_quantize_program
from torch_coral.isa_streamed import recurrent_stream_program
from torch_coral.isa_batched import convolution_many_program,block_topk_program


@pytest.mark.parametrize('factory',[
    lambda:elementwise('rsqrt',128),lambda:gemm(2,3,32),
    lambda:norm_rows_program(128,15,True,False,1e-6),
    lambda:primary_quantize_program(1024),lambda:recurrent_stream_program(16,128,7),
    lambda:convolution_many_program(124,13),lambda:block_topk_program(128,4)])
def test_latency_two_schedule_matches_latency_five_reference(factory):
    _,words,base=factory()
    rng=np.random.default_rng(62)
    payload=rng.uniform(.1,.2,size=(16384-base)//4).astype(np.float32).tobytes()
    old=bytearray(16384); old[base:base+len(payload)]=payload
    new=bytearray(old)
    run(words,old,load_latency=5)
    revised=compact_load_waits(tuple(words))
    run(revised,new,load_latency=2)
    assert new==old
    assert len(revised)<len(words)
