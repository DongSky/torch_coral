"""Stream time-dependent inputs while retaining a recurrent FP32 state tile."""
from functools import lru_cache
import numpy as np
from .isa_scalar import _parts
from .isa_optimized import OptimizedMatrixTPU


def recurrent_step_words(columns, length, frame_base, output_base, state_base=0):
    """Update a column-major state in place, preserving reference arithmetic order."""
    _, _, _, S = _parts()
    words = [S.movi(7, frame_base + 2 * length + columns), S.ld(15, 7, inc=True),
             S.ld(16, 7)] + [S.NOP] * 4
    # Keep the column loop in the device program rather than retransmitting
    # one copy of the instruction body for each state column.
    words += [S.movi(31,state_base), S.movi(29,frame_base + 2 * length), S.movi(30,output_base)]
    top = len(words)
    words += [S.mov(4, 31), S.movi(5, frame_base), S.movi(10, 0)]
    dot = [S.ld(8, 4, inc=True), S.ld(9, 5, inc=True)] + [S.NOP] * 4 + [
        S.alu(S.FMUL, 8, 8, reg=16), S.alu(S.FMUL, 11, 8, reg=9),
        S.NOP, S.alu(S.FADD, 10, 10, reg=11), S.NOP]
    unroll = 4 if length % 4 == 0 else 1
    if unroll == 4:
        pairs = ((8,9),(20,21),(22,23),(24,25))
        dot = [op for a,b in pairs for op in (S.ld(a,4,inc=True),S.ld(b,5,inc=True))] + [S.NOP]*4
        for a,b in pairs:
            dot += [S.alu(S.FMUL,a,a,reg=16),S.alu(S.FMUL,11,a,reg=b),
                    S.alu(S.FADD,10,10,reg=11)]
    words += [S.loop(length // unroll, len(dot))] + dot
    words += [S.mov(7, 29), S.ld(12, 7)] + [S.NOP] * 4 + [
        S.alu(S.FSUB, 12, 12, reg=10), S.alu(S.FMUL, 12, 12, reg=15),
        S.mov(4, 31), S.movi(5, frame_base), S.movi(7, frame_base + length),
        S.mov(6, 31), S.movi(10, 0)]
    update = [S.ld(8, 4, inc=True), S.ld(9, 5, inc=True), S.ld(13, 7, inc=True)] + [S.NOP] * 4 + [
        S.alu(S.FMUL, 8, 8, reg=16), S.alu(S.FMUL, 9, 9, reg=12),
        S.alu(S.FADD, 8, 8, reg=9), S.st(6, 8), S.alu(S.ADD, 6, 6, imm=1),
        S.alu(S.FMUL, 11, 8, reg=13), S.NOP, S.alu(S.FADD, 10, 10, reg=11), S.NOP]
    if unroll == 4:
        triples = ((8,9,13),(20,21,22),(23,24,25),(26,27,28))
        update = [op for a,b,c in triples for op in (S.ld(a,4,inc=True),S.ld(b,5,inc=True),S.ld(c,7,inc=True))] + [S.NOP]*4
        for a,b,c in triples:
            update += [S.alu(S.FMUL,a,a,reg=16),S.alu(S.FMUL,b,b,reg=12),
                S.alu(S.FADD,a,a,reg=b),S.st(6,a) | S.alu(S.FMUL,11,a,reg=c),
                S.alu(S.ADD,6,6,imm=1),S.alu(S.FADD,10,10,reg=11)]
    words += [S.loop(length // unroll, len(update))] + update
    words += [S.st(30, 10), S.alu(S.ADD,31,31,imm=length),
              S.alu(S.ADD,29,29,imm=1), S.alu(S.ADD,30,30,imm=1),
              S.alu(S.LT,0,29,imm=frame_base + 2 * length + columns)]
    words += [S.branch(top - len(words))]
    return words


@lru_cache(maxsize=128)
def recurrent_stream_program(columns, length, steps, capture_first=False,compact=False,capture_prefixes=0):
    """One DMA, in-place state and a protected low-SRAM projection buffer.

    The 16000-byte single-DMA limit is verified physically; no multi-input
    streaming is used. Frames are overwritten only after every time step.
    """
    Emitter, E, SC, S = _parts()
    if min(columns, length, steps) < 1 or steps > 16:
        raise ValueError('Invalid recurrent dimensions')
    if capture_first and steps!=2:
        raise ValueError('First-state capture is defined for two-position verification')
    prefixes = int(capture_prefixes) or int(capture_first)
    if not 0 <= prefixes < steps:
        raise ValueError('Invalid recurrent prefix count')
    state_words = columns * length
    first_bytes=(state_words*4+7)//8*8
    frame_words = 2 * length + columns + 2
    input_bytes = ((state_words + steps * frame_words) * 4 + 7) // 8 * 8
    output_words = state_words + steps * columns
    output_bytes = (output_words * 4 + 7) // 8 * 8
    base = E.SMEM_TOP - input_bytes
    if max(input_bytes, output_bytes) > 16000 or steps * columns * 4 > base:
        raise ValueError('Recurrent tile exceeds verified in-place SRAM limits')
    words = []
    for index in range(steps):
        words += recurrent_step_words(columns, length, base // 4 + state_words + index * frame_words,
                                      index * columns, state_base=base // 4)
        if index < prefixes:
            words += E.smem_to_host(base//4,state_words,first_bytes)
    copy = [S.ld(8, 4, inc=True)] + [S.NOP] * 4 + [S.st(6, 8), S.alu(S.ADD, 6, 6, imm=1)]
    words += [S.movi(4, 0), S.movi(6, base // 4 + state_words),
              S.loop(steps * columns, len(copy))] + copy
    if compact:
        from .isa_schedule import compact_load_waits
        words=list(compact_load_waits(tuple(words)))
    e = Emitter(); e.scalar(SC.exe_prologue(), seqs=1)
    tiles = E._sm_single_row(e, input_bytes, base)
    e.scalar(E.S_RING_OUTFEED_RESET)
    e.scalar(SC.output_wait(e.seq, tiles), seqs=2)
    e.scalar(words)
    e.scalar(E.smem_to_host(base // 4, output_words, output_bytes)); e.sync(SC.epilogue)
    if prefixes:
        from coral.programs import _exe
        from coral.executable import Hint
        exe=_exe('STAND_ALONE',e.program(),[Hint('dma','INFEED','INPUT','x',0,input_bytes)] +
            [Hint('dma','OUTFEED','OUTPUT','prefix%d'%index,0,first_bytes) for index in range(prefixes)] +
            [Hint('dma','OUTFEED','OUTPUT','final',0,output_bytes)])
    else: exe=E.executable(e.program(),input_bytes,output_bytes)
    return exe, words, base


def tile_columns(length, steps, limit=128):
    for columns in range(limit, 0, -1):
        size = ((columns * length + steps * (2 * length + columns + 2)) * 4 + 7) // 8 * 8
        if size <= 16000 and steps * columns * 4 <= 16384 - size:
            return columns
    return 0


@lru_cache(maxsize=128)
def time_chunks(length, width, steps, limit=128):
    """Minimize physical launches subject to the verified SRAM layout."""
    best = [(0, ())]
    for total in range(1, steps + 1):
        choices = []
        for count in range(1, min(16, total) + 1):
            columns = tile_columns(length, count, min(width, limit))
            if columns:
                cost, previous = best[total - count]
                choices.append((cost + (width + columns - 1) // columns, previous + (count,)))
        if not choices: raise ValueError('Recurrent state exceeds verified scalar SRAM tile')
        best.append(min(choices, key=lambda row: (row[0], len(row[1]))))
    return best[-1][1]


class StreamedMatrixTPU(OptimizedMatrixTPU):
    """Turbo V2: bounded single-DMA temporal tiles, all arithmetic on TPU."""
    def __init__(self, *args, stream_columns=128, **kwargs):
        if not 1 <= stream_columns <= 128: raise ValueError('stream_columns must be 1..128')
        self.stream_columns = stream_columns
        super().__init__(*args, **kwargs)
        self.matrix_stats.update(recurrent_stream_calls=0, recurrent_stream_input_bytes=0,
                                 recurrent_stream_output_bytes=0, recurrent_stream_positions=0)

    def normalize(self, array, weight, l2, epsilon):
        from .isa_v2_fusions import norm_rows_program
        x = np.asarray(array, np.float32)
        length = x.shape[-1]
        if length > 1024: raise ValueError('Fused normalization supports at most 1024 columns')
        values = x.reshape(-1, length)
        result = np.empty_like(values)
        batch = 2048 // length - int(weight is not None)
        for offset in range(0, len(values), batch):
            part = values[offset:offset + batch]
            payload = part.tobytes() + (weight.tobytes() if weight is not None else b'')
            result[offset:offset + len(part)] = self._run(norm_rows_program(length, len(part), weight is not None,
                l2, float(epsilon)), payload, part.size).reshape(part.shape)
        return result.reshape(x.shape)

    def quantize_projection_input(self, array):
        if self.components!=1 or array.size>1024:
            return self.quantize_input(array)
        from .isa_v2_fusions import primary_quantize_program
        x=np.ascontiguousarray(array,np.float32).reshape(-1)
        out=self._run(primary_quantize_program(len(x)),x.tobytes(),len(x)//4+1)
        return out[:-1].view(np.uint8).copy(),out[-1:].copy(),None

    def _inplace_result(self, program, payload, count):
        payload += b'\0' * (-len(payload) % 8)
        raw=self._execute(program[0],inputs=payload)
        if len(raw)<count*4:
            self.close(); raise RuntimeError('Short in-place scalar result')
        return np.frombuffer(raw,'<f4',count=count).copy()

    def _correct(self, q, bias, coefficient):
        from .isa_v2_fusions import correction_inplace_program
        parts=[]
        for offset in range(0,len(q),3072):
            part=q[offset:offset+3072]
            payload=part.tobytes()+bias[offset:offset+len(part)].tobytes()
            parts.append(self._inplace_result(correction_inplace_program(len(part),float(coefficient)),payload,len(part)).view('<i4'))
        return np.concatenate(parts)

    def _reconstruct(self, outputs, coefficients, row_scale, input_scale):
        from .isa_v2_fusions import reconstruct_inplace_program
        # Three-digit profiles leave more space for the extra packed readout.
        tile=2560 if len(outputs)<=2 else 2048
        result=[]
        for offset in range(0,len(row_scale),tile):
            scales=np.ascontiguousarray(row_scale[offset:offset+tile],np.float32)
            payload=b''.join(q[offset:offset+len(scales)].tobytes() for q in outputs)+scales.tobytes()+input_scale.tobytes()
            result.append(self._inplace_result(reconstruct_inplace_program(len(scales),tuple(coefficients)),payload,len(scales)))
        return np.concatenate(result)

    def softmax(self, array):
        from .isa_v2_fusions import softmax_program
        x = np.asarray(array, np.float32)
        if x.ndim != 1 or not 1 <= len(x) <= 2048:
            raise ValueError('Fused softmax supports 1..2048 elements')
        return self._run(softmax_program(len(x)), x.tobytes(), len(x))

    def rope(self, array, cosine, sine):
        from .isa_v2_fusions import rope_program
        half = len(cosine)
        rotated = np.ascontiguousarray(array[..., :2 * half]).reshape(-1, 2 * half)
        result = np.empty_like(rotated)
        batch = 2048 // (2 * half) - 1
        for offset in range(0, len(rotated), batch):
            part = rotated[offset:offset + batch]
            payload = part.tobytes() + cosine.tobytes() + sine.tobytes()
            result[offset:offset + len(part)] = self._run(rope_program(half, len(part)), payload, part.size).reshape(part.shape)
        return np.concatenate([result.reshape(array.shape[:-1] + (2 * half,)), array[..., 2 * half:]], axis=-1)

    def recurrent(self, state, k, q, v, beta, decay):
        updated, output = self.recurrent_many(state[None], k[None, None], q[None, None],
            v[None, None], np.asarray([[beta]], np.float32), np.asarray([[decay]], np.float32))
        return updated[0], output[0, 0]

    def recurrent_many(self, state, k, q, v, beta, decay):
        """Retain each state tile across time without FP32 reassociation."""
        steps, heads, length = k.shape
        width = state.shape[-1]
        if (steps < 1 or q.shape != k.shape or v.shape != (steps, heads, width)
                or state.shape != (heads, length, width) or beta.shape != (steps, heads)
                or decay.shape != beta.shape):
            raise ValueError('Invalid recurrent sequence shapes')
        updated = np.array(state, np.float32, copy=True)
        capture_first=bool(getattr(self,'capture_first_state',False))
        prefixes=int(getattr(self,'capture_state_prefixes',0)) or int(capture_first)
        if capture_first and steps!=2: raise ValueError('Capture requires two positions')
        temporal_chunks=time_chunks(length,width,steps,self.stream_columns)
        if prefixes and (not 0 < prefixes < steps or temporal_chunks!=(steps,)):
            raise ValueError('Prefix capture requires one complete temporal chunk')
        prefix_states=np.empty((prefixes,)+updated.shape,np.float32) if prefixes else None
        outputs = np.empty(v.shape, np.float32)
        start = 0
        for count_steps in temporal_chunks:
            columns = tile_columns(length, count_steps, min(width, self.stream_columns))
            for head in range(heads):
                for offset in range(0, width, columns):
                    count = min(columns, width - offset)
                    program, _, _ = recurrent_stream_program(count, length, count_steps,capture_first=capture_first,
                                                             compact=getattr(self,'compact_loads_enabled',False),
                                                             capture_prefixes=prefixes if not capture_first else 0)
                    chunks = [np.ascontiguousarray(updated[head, :, offset:offset + count].T).tobytes()]
                    for index in range(start, start + count_steps):
                        chunks.append(k[index, head].tobytes() + q[index, head].tobytes()
                            + v[index, head, offset:offset + count].tobytes()
                            + np.float32(beta[index, head]).tobytes() + np.float32(decay[index, head]).tobytes())
                    payload = b''.join(chunks)
                    payload += b'\0' * (-len(payload) % 8)
                    raw = self._execute(program, inputs=payload)
                    if prefixes:
                        first_bytes=(count*length*4+7)//8*8
                        if len(raw)<prefixes*first_bytes:
                            self.close(); raise RuntimeError('Short first-state capture')
                        for prefix in range(prefixes):
                            prefix_states[prefix,head,:,offset:offset+count]=np.frombuffer(
                                raw,'<f4',count=count*length,offset=prefix*first_bytes).reshape(count,length).T
                        raw=raw[prefixes*first_bytes:]
                    output_words = count * length + count_steps * count
                    if len(raw) < output_words * 4:
                        self.close(); raise RuntimeError('Short recurrent state result')
                    values = np.frombuffer(raw, '<f4', count=output_words)
                    updated[head, :, offset:offset + count] = values[:count * length].reshape(count, length).T
                    outputs[start:start + count_steps, head, offset:offset + count] = values[count * length:].reshape(count_steps, count)
                    self.matrix_stats['recurrent_stream_calls'] += 1
                    self.matrix_stats['recurrent_stream_input_bytes'] += len(payload)
                    self.matrix_stats['recurrent_stream_output_bytes'] += len(raw)
                    self.matrix_stats['recurrent_stream_positions'] += count_steps
            start += count_steps
        if prefixes:
            self.prefix_recurrent_states=prefix_states
            self.first_recurrent_state=prefix_states[0]
        return updated, outputs
