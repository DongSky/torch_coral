"""Experimental native 16-position MXU program with per-position bias banks.

Each physical tile owns one prompt position. Weights stream from distributed
parameter regions to all tiles; independent readout corrections live in a
reserved bias bank on every tile. No numerical host fallback is used.
"""
from functools import lru_cache
import numpy as np
from .isa_streamed import StreamedMatrixTPU
from .isa_mxu import bias_executable


@lru_cache(maxsize=32)
def block_topk_program(length,count=4):
    from .isa_scalar import _parts,_wrap
    _,_,_,S=_parts()
    def body(base):
        words=[]
        for rank in range(count):
            words += [S.movi(4,base),S.movi(7,0),S.movi(10,0xff800000),S.movi(11,0)]
            loop=[S.ld(8,4,inc=True)]+[S.NOP]*4
            for previous in range(rank):
                loop += [S.alu(S.NE,0,7,reg=20+previous),
                    S.alu(S.MOV,8,imm=0xff800000,gate=1,pred_reg=0,yes_pred=0)]
            loop += [S.alu(S.FMAX,12,10,reg=8),S.alu(S.NE,0,12,reg=10),S.mov(10,12),
                S.alu(S.MOV,11,reg=7,gate=1,pred_reg=0,yes_pred=1),S.alu(S.ADD,7,7,imm=1)]
            words += [S.loop(length,len(loop))]+loop+[S.mov(20+rank,11),S.i2f(11,11),
                S.movi(6,rank*2),S.st(6,10),S.movi(6,rank*2+1),S.st(6,11)]
        return words
    return _wrap(length,count*2,body)


@lru_cache(maxsize=64)
def convolution_many_program(columns,steps,compact=False):
    from .isa_scalar import _parts
    from .isa_optimized import _sigmoid
    Emitter,E,SC,S=_parts()
    size=((steps+7)*columns*4+7)//8*8
    out_size=(steps*columns*4+7)//8*8
    base=E.SMEM_TOP-size
    if max(size,out_size)>16000 or out_size>base:
        raise ValueError('Temporal convolution overlaps its input')
    words=[S.movi(25,0),S.movi(26,base//4),S.movi(27,0)]
    top=len(words)
    for tap in range(4):
        words += [S.alu(S.ADD,4+tap,26,imm=tap*columns),S.movi(21+tap,base//4+(steps+3+tap)*columns)]
    loop=[]
    for tap in range(4):
        loop += [S.ld(8,4+tap,inc=True),S.ld(9,21+tap,inc=True)]+[S.NOP]*4+[
            S.alu(S.FMUL,10 if tap==0 else 11,8,reg=9)]
        if tap: loop += [S.alu(S.FADD,10,10,reg=11)]
    loop += [S.mov(8,10)]+_sigmoid(S)+[S.alu(S.FMUL,10,8,reg=10),S.st(25,10),S.alu(S.ADD,25,25,imm=1)]
    words += [S.loop(columns,len(loop))]+loop+[S.alu(S.ADD,26,26,imm=columns),
        S.alu(S.ADD,27,27,imm=1),S.alu(S.LT,0,27,imm=steps)]
    words += [S.branch(top-len(words))]
    if compact:
        from .isa_schedule import compact_load_waits
        words=list(compact_load_waits(tuple(words)))
    e=Emitter(); e.scalar(SC.exe_prologue(),seqs=1)
    tiles=E._sm_single_row(e,size,base)
    e.scalar(E.S_RING_OUTFEED_RESET); e.scalar(SC.output_wait(e.seq,tiles),seqs=2)
    e.scalar(words); e.scalar(E.smem_to_host(0,steps*columns,out_size)); e.sync(SC.epilogue)
    return E.executable(e.program(),size,out_size),words,base


class BatchProgram:
    def __init__(self, n, k):
        from coral.codegen import conv as C
        from coral.programs import conv_program, WIDE_BYTES
        if n<64 or n%64 or k%4:
            raise ValueError('Native batch requires complete 64-output groups')
        self.n,self.k=n,k
        self.base=conv_program(16,n,k)
        self.g=C.ConvGeom(16,n,k)
        self.bias_base=(min(WIDE_BYTES,C.param_limit(self.g)*64)-n*4)//256*256
        if self.bias_base<=0: raise ValueError('No room for per-position bias banks')
        self.pieces=C.split_pieces(self.g.blocks,tuple(range(16)),0,self.bias_base//64)
        self.param_bytes=self.g.param_bytes
        self.in_size,self.out_size=self.base.in_size,self.base.out_size

    def params(self, weights, bias=None, w_zp=128):
        return self.base.params(weights,np.zeros(self.n,np.int32) if bias is None else bias,w_zp)

    def input_bytes(self, array): return self.base.input_bytes(array,128)
    def output_array(self, raw): return self.base.output_array(raw)

    @lru_cache(maxsize=128)
    def programs(self, quant):
        from coral.codegen import conv as C,Emitter,caching_program,piece_parts
        from coral.isa import scalar as SC,wide_narrow as WN
        from coral.programs import _exe
        from coral.executable import Hint
        g=self.g; nl=C.narrow_layout(g); wl=C.wide_layout(g)
        e=Emitter(); e.scalar(SC.exe_prologue(),seqs=1)
        C.input_stage(e,g,nl['Stg'],nl['X'],nl['C'],wl['in_fifo'],wl['const'])
        e.sync(SC.sync_reset17); e.scalar(SC.scsync_nop()); e.sync(SC.sync_drain)
        e.tile(C.rc_param(g,e.seq,wl))
        fields=WN.decode_wide_to_narrow(C.bias_load(e.seq,0xffff,self.bias_base//64,g.b,g.fifo,g.G,g.cg))
        fields.update(wide_circ=0,wide_rows=0,wide_lvl_mask=3 if g.G>1 else 1)
        fields.update(WN.ttu('w',[1,1],[1,g.G],4))
        e.tile(WN.encode_wide_to_narrow(**fields))
        e.tile(C.conv_op(g,e.seq,nl['X'],nl['Y'],wl,quant.fields()))
        first=0
        for tile,off,count in self.pieces:
            e.tile(C.rprod_param(e.seq,tile,off,count,first)); first+=count
        C.output_stage(e,g,nl['Y'],first)
        pc=_exe('PARAMETER_CACHING',caching_program(g.param_bytes,piece_parts([self.pieces])),
                [Hint('dma','INFEED','PARAMETER',None,0,g.param_bytes)])
        eo=_exe('EXECUTION_ONLY',e.program(),[Hint('dma','INFEED','INPUT','x',0,self.in_size),
                                           Hint('dma','OUTFEED','OUTPUT','y',0,self.out_size)])
        return pc,eo


@lru_cache(maxsize=64)
def batch_program(n,k): return BatchProgram(n,k)


@lru_cache(maxsize=128)
def batch_bias_executable(n, positions, parameter_offset):
    """Load whole per-position bias banks only on active physical tiles."""
    from coral.codegen import caching_program
    from coral.isa import ttu
    from coral.programs import _exe
    from coral.executable import Hint
    if not 1 <= positions <= 16 or n % 64:
        raise ValueError('Invalid active bias bank layout')
    groups=n//64
    parts=[]
    for tile in range(positions):
        rc=dict(tile_mask=1<<tile,addr=2+parameter_offset//64,sdims=int(groups>1),**ttu('',[1],[groups],4))
        parts.append((rc,tile*n*4,n*4,4))
    size=positions*n*4
    return _exe('PARAMETER_CACHING',caching_program(size,[parts]),[Hint('dma','INFEED','PARAMETER',None,0,size)])


class BatchedMatrixTPU(StreamedMatrixTPU):
    """Turbo V3 experiment. Native 16-position projections, V2 for short rows."""
    def __init__(self,*args,native_batch_min=8,**kwargs):
        self.native_batch_min=native_batch_min
        self._batch_resident=None
        self.compact_loads_enabled=False
        self._compact_programs={}
        super().__init__(*args,**kwargs)
        self.matrix_stats.update(native_batch_projections=0,native_batch_mxu_calls=0)
        values=np.arange(64,dtype=np.float32)
        expected=self.op('add',values,np.float32(3))
        self.compact_loads_enabled=True
        actual=self.op('add',values,np.float32(3))
        self.compact_loads_enabled=actual.tobytes()==expected.tobytes()
        self.matrix_stats['scalar_load_wait_bundles']=1 if self.compact_loads_enabled else 4

    def _scheduled_program(self,program,payload_size,count,inplace=False):
        from .isa_schedule import compact_load_waits
        from .isa_scalar import _wrap
        from .isa_v2_fusions import _inplace
        key=(program[0].bitstreams[0].data,payload_size,count,inplace)
        if key not in self._compact_programs:
            words=list(compact_load_waits(tuple(program[1])))
            if inplace:
                offset=(program[3]-program[2])//4
                revised=_inplace((payload_size+7)//8*2,count,offset,lambda base:words)
            else: revised=_wrap((payload_size+7)//8*2,count,lambda base:words)
            if len(self._compact_programs)>=256: self._compact_programs.clear()
            self._compact_programs[key]=revised
        return self._compact_programs[key]

    def _run(self,program,payload,count):
        if self.compact_loads_enabled:
            program=self._scheduled_program(program,len(payload),count)
        return super()._run(program,payload,count)

    def _inplace_result(self,program,payload,count):
        if self.compact_loads_enabled:
            program=self._scheduled_program(program,len(payload),count,inplace=True)
        return super()._inplace_result(program,payload,count)

    def convolution_many(self,history,projections,kernel):
        steps,channels=projections.shape
        timeline=np.concatenate([history,projections],axis=0)
        result=np.empty_like(projections)
        columns=(4096//(2*steps+7))//4*4
        for offset in range(0,channels,columns):
            count=min(columns,channels-offset)
            part=timeline[:,offset:offset+count]
            weights=kernel[offset:offset+count].T
            payload=part.tobytes()+weights.tobytes(); payload+=b'\0'*(-len(payload)%8)
            exe,_,_=convolution_many_program(count,steps,self.compact_loads_enabled)
            raw=self._execute(exe,inputs=payload)
            if len(raw)<steps*count*4:
                self.close(); raise RuntimeError('Short temporal convolution result')
            result[:,offset:offset+count]=np.frombuffer(raw,'<f4',count=steps*count).reshape(steps,count)
        return result

    def argmax(self,array):
        values=np.asarray(array,np.float32).reshape(-1)
        if not len(values): raise ValueError('Cannot select from an empty tensor')
        vocab=self.matrix_manifest['matrices']['token_embd.weight']['shape'][0]
        if len(values)!=vocab: return super().argmax(values)
        maxima,indices=[],[]
        candidates=[]
        for offset in range(0,len(values),2048):
            part=values[offset:offset+2048]
            pair=self._run(block_topk_program(len(part),4),part.tobytes(),8).reshape(4,2)
            maxima.append(pair[0,:1]); indices.append(offset+int(pair[0,1]))
            candidates.extend(offset+int(row[1]) for row in pair)
        winner=super().argmax(np.concatenate(maxima)) if len(maxima)>1 else 0
        self.draft_candidate_ids=tuple(candidates)
        return indices[winner]

    def project_indexed(self,name,indices,inputs):
        with self.lock:
            return self._project_indexed(name,indices,inputs)

    def _project_indexed(self,name,indices,inputs):
        """Evaluate a proposal-only head shortlist using the existing INT8 rows."""
        import hashlib
        from types import SimpleNamespace
        if name not in self._matrices: raise ValueError('Full head must be initialized before drafting')
        ids=list(indices)
        if not ids: raise ValueError('A proposal shortlist must not be empty')
        padded=ids+[ids[-1]]*(-len(ids)%64)
        index=np.asarray(padded,np.int64)
        key='__draft_rows_'+hashlib.sha256(index.tobytes()).hexdigest()
        arrays,scales=self._matrices[name]
        self._matrices[key]=([np.ascontiguousarray(a[index]) for a in arrays],np.ascontiguousarray(scales[:,index]))
        self.matrix_manifest['matrices'][key]=dict(shape=[len(padded),arrays[0].shape[1]])
        try:
            shape=SimpleNamespace(shape=(len(padded),arrays[0].shape[1]))
            return self.project(key,shape,inputs)[:len(ids)]
        finally:
            del self._matrices[key]; del self.matrix_manifest['matrices'][key]

    def close(self):
        super().close()
        self._compact_programs.clear()
        self._batch_resident=None

    def _project(self,name,weights,inputs):
        x=np.asarray(inputs,np.float32)
        if (self.components!=1 or self.output_digits!=2 or x.ndim!=2
                or not max(self.native_batch_min,getattr(self,'native_large_min',12) if weights.shape[0]>4096 else 1)<=len(x)<=16 or x.shape[1]>3840
                or name not in self.matrix_manifest['matrices']):
            self._batch_resident=None
            return super()._project(name,weights,inputs)
        if name not in self._matrices:
            self._matrices[name]=([np.load(self.matrix_directory/(name+'.q%d.npy'%i),mmap_mode='r',allow_pickle=False)
                                   for i in range(2)],np.load(self.matrix_directory/(name+'.scales.npy'),mmap_mode='r',allow_pickle=False))
        arrays,scales=self._matrices[name]
        if tuple(weights.shape)!=arrays[0].shape: raise ValueError('Matrix shape mismatch')
        n_total,k=weights.shape
        prepared=[self.quantize_projection_input(row)[:2] for row in x]
        packed=np.stack([row[0] for row in prepared])
        result=np.empty((len(x),n_total),np.float32)
        self._resident=None; self._allocation=None
        from coral.programs import Quant
        for offset in range(0,n_total,self.chunk_rows):
            n=min(self.chunk_rows,n_total-offset)
            if n%64:
                self._batch_resident=None
                return super()._project(name,weights,inputs)
            prog=batch_program(n,k)
            key=('native16',name,offset,n,k)
            if self._batch_resident!=key:
                # FC and native convolution use the same serialized weight blob.
                # Share it so switching to single-token decode needs no repack.
                blob_key=(name,offset,0,n,k)
                blob=self.parameter_blob(prog,blob_key,np.ascontiguousarray(arrays[0][offset:offset+n]))
                pc,_=prog.programs(Quant(128,128,128,1/4096))
                self._execute(pc,parameters=blob); self._batch_resident=key
                self.matrix_stats['weight_uploads']+=1; self.matrix_stats['weight_bytes']+=len(blob)
            payload=prog.input_bytes(packed)
            pending=list(range(len(x))); scale=4096.
            for attempt in range(8):
                active=len(x) if getattr(self,'native_active_bias',False) else 16
                bias=np.zeros((active,n),'<i4')
                bias_program=(batch_bias_executable(n,active,prog.bias_base) if active<16
                              else bias_executable(16*n,k,prog.bias_base))
                self._execute(bias_program,parameters=bias.tobytes())
                self.matrix_stats['bias_uploads']+=1
                _,exe=prog.programs(Quant(128,128,128,1/scale))
                high=prog.output_array(self._execute(exe,inputs=payload)).copy()
                self.matrix_stats['mxu_calls']+=1; self.matrix_stats['native_batch_mxu_calls']+=1
                bad=[row for row in pending if 0 in high[row] or 255 in high[row]]
                good=[row for row in pending if row not in bad]
                if good:
                    bias=self._correct(high[:active].reshape(-1),bias.reshape(-1),scale).reshape(active,n)
                    self._execute(bias_program,parameters=bias.tobytes())
                    self.matrix_stats['bias_uploads']+=1
                    _,exe=prog.programs(Quant(128,128,128,128/scale))
                    low=prog.output_array(self._execute(exe,inputs=payload)).copy()
                    self.matrix_stats['mxu_calls']+=1; self.matrix_stats['native_batch_mxu_calls']+=1
                    for row in good:
                        result[row,offset:offset+n]=self._reconstruct([high[row],low[row]],[scale,scale/128],
                            scales[0,offset:offset+n],prepared[row][1])
                if not bad: break
                self.matrix_stats['range_retries']+=len(bad)
                pending=bad; scale*=4
            else: raise RuntimeError('Native batch readout could not avoid clipping')
        self.matrix_stats['projections']+=1
        self.matrix_stats['native_batch_projections']+=1
        return result
