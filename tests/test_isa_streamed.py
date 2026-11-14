"""Offline semantic checks; physical DMA support is checked separately."""
import numpy as np
import pytest
pytest.importorskip('coral.isa.scalar_core')
from coral.isa.scalar_core import run
from torch_coral.isa_streamed import recurrent_stream_program
from torch_coral.isa_mxu import MatrixTPU


class Offline(MatrixTPU):
    recurrent_columns = 13
    def __init__(self): pass
    def _run(self, program, payload, count):
        _, words, base = program; memory = bytearray(16384)
        memory[base:base + len(payload)] = payload
        run(words, memory, load_latency=5)
        return np.frombuffer(memory, '<f4', count=count).copy()


@pytest.mark.parametrize('columns,length,steps', [(1,4,2), (29,128,1), (16,128,7), (4,128,13)])
def test_retained_state_and_all_positions_match_original_bits(columns, length, steps):
    rng = np.random.default_rng(411)
    state = rng.normal(size=(length, columns)).astype(np.float32)
    k, q = rng.normal(size=(2, steps, length)).astype(np.float32) * np.float32(.1)
    v = rng.normal(size=(steps, columns)).astype(np.float32)
    beta = np.full(steps, .3, np.float32); decay = np.full(steps, .9, np.float32)
    current = state.copy(); expected = []
    for index in range(steps):
        current, y = Offline().recurrent(current, k[index], q[index], v[index], beta[index], decay[index]); expected.append(y)
    exe, words, base = recurrent_stream_program(columns, length, steps)
    frames = [state.T.tobytes()]
    for t in range(steps): frames.append(k[t].tobytes() + q[t].tobytes() + v[t].tobytes() + beta[t].tobytes() + decay[t].tobytes())
    memory = bytearray(16384)
    payload = b''.join(frames)
    memory[base:base + len(payload)] = payload
    run(words, memory, load_latency=5)
    output_words = state.size + steps * columns
    result = np.frombuffer(memory, '<f4', count=output_words, offset=base)
    np.testing.assert_array_equal(result[:state.size].reshape(columns,length).T.view(np.uint32), current.view(np.uint32))
    np.testing.assert_array_equal(result[state.size:].reshape(steps,columns).view(np.uint32), np.stack(expected).view(np.uint32))
    # Use the corrected scalar-core splitter: the old generic splitter treats
    # the one-word conditional branch as a 29-word instruction.
    from coral.isa.scalar_core import dma_seqs
    assert dma_seqs(exe.bitstreams[0].data)==[(1,(len(payload)+7)//8*8),(3,(output_words*4+7)//8*8),(4,0)]


def test_stream_rejects_unverified_buffer_sizes():
    with pytest.raises(ValueError): recurrent_stream_program(30,128,1)
    with pytest.raises(ValueError): recurrent_stream_program(13,128,17)
