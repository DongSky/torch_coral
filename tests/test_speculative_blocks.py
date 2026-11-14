"""Exact multi-token control flow, bounded proposals, EOS and cancellation."""
import numpy as np
import pytest
from types import SimpleNamespace
from torch_coral.speculative import SpeculativeGreedy,ngram_tail
from torch_coral.qwen_tpu import QwenTPU


class Engine:
    def __init__(self,chains):
        self.inputs=[7];self.chains=iter(chains);self.backend=SimpleNamespace();self.context=100
        self.chain_calls=0
    @property
    def position(self):return len(self.inputs)
    def step(self,anchor):self.inputs.append(anchor);return dict(token=anchor+1)
    def greedy(self,scores):return scores['token']
    def mtp_draft_chain(self,tokens,count,**kwargs):
        assert len(tokens)==self.position+1
        self.chain_calls+=1;return next(self.chains)[:count]
    def verify_block(self,anchor,drafts):
        incoming=[anchor,*drafts]
        self.prefixes=[self.inputs+incoming[:n] for n in range(1,len(incoming)+1)]
        self.inputs+=incoming
        return [dict(token=token+1) for token in incoming]
    def finish_block(self,committed):self.inputs=self.prefixes[committed-1]
    def mtp_draft(self,tokens):return dict(token=next(self.chains)[0])
    def verify_pair(self,anchor,draft):return self.verify_block(anchor,[draft])
    def finish_pair(self,accepted):self.finish_block(2 if accepted else 1)


@pytest.mark.parametrize('size',[3,4])
def test_all_accepted_and_every_first_rejection_commit_only_verified_prefix(size):
    for failed in range(size):
        drafts=list(range(2,size+1))
        if failed<size-1:drafts[failed]=99
        e=Engine([drafts]);decoder=SpeculativeGreedy(e,[7],adaptive=False,max_block_tokens=size)
        tokens,logits,stop=decoder.advance(1,10,set())
        accepted=min(failed,size-1)
        assert tokens==list(range(2,accepted+3)) and not stop
        assert [value['token'] for value in logits]==tokens
        assert e.inputs==decoder.processed==[7,*range(1,accepted+2)]
        assert decoder.stats['accepted']==accepted
        assert decoder.stats['rejected']==int(failed<size-1)


@pytest.mark.parametrize('stop_id',[2,3,4,5])
def test_eos_never_commits_or_emits_unverified_drafts(stop_id):
    e=Engine([[2,3,4]]);decoder=SpeculativeGreedy(e,[7],adaptive=False,max_block_tokens=4)
    tokens,_,stop=decoder.advance(1,9,{stop_id})
    assert stop and tokens==list(range(2,stop_id))
    assert e.inputs==decoder.processed==[7,*range(1,stop_id)]


def test_remaining_and_context_bound_block_size():
    e=Engine([[2,3,4]]);decoder=SpeculativeGreedy(e,[7],adaptive=False,max_block_tokens=4)
    tokens,_,_=decoder.advance(1,3,set())
    assert tokens==[2,3,4] and e.position==4
    e.context=5
    tokens,_,_=decoder.advance(4,3,set())
    assert tokens==[5] and e.chain_calls==1


def test_ngram_lookup_requires_a_complete_known_tail():
    history=[1,2,3,4,5,6,1,2]
    assert ngram_tail(history,[9,1,2],3)==[3,4,5]
    assert ngram_tail(history,[9,8,2],3) is None
    assert ngram_tail(history,[1,2,3,4,5,6],4) is None
    assert ngram_tail(history,[9,1,2],3,minimum_width=3) is None
    assert ngram_tail([1,2,3,4,9,1,2,3,99,8],[1,2,3],2,excluded={99})==[4,9]


def test_optional_recent_device_calibration_expires_without_changing_tokens():
    import time
    backend=SimpleNamespace(reuse_decode_calibration=True,decode_calibration={})
    first=Engine([[2]]);first.backend=backend
    initial=SpeculativeGreedy(first,[7],adaptive=True)
    assert initial.advance(1,4,set())[0]==[2]
    assert initial.stats['baseline_source']=='fresh ordinary step'
    second=Engine([[2]]);second.backend=backend
    cached=SpeculativeGreedy(second,[7],adaptive=True)
    assert cached.advance(1,4,set())[0]==[2,3]
    assert cached.stats['baseline_source']=='recent same-device measurement'
    backend.decode_calibration[0]=(1.,time.perf_counter()-61)
    third=Engine([[2]]);third.backend=backend
    expired=SpeculativeGreedy(third,[7],adaptive=True)
    assert expired.advance(1,4,set())[0]==[2]
    assert expired.stats['baseline_source']=='fresh ordinary step'


def test_verification_exception_restores_preverification_host_state():
    e=object.__new__(QwenTPU)
    e.layers=24;e.context=100;e.position=1;e.hidden_history=[np.asarray([7],np.float32)]
    e.last_hidden=e.hidden_history[-1];e.caches={0:([np.asarray([1])],[np.asarray([2])])}
    e.verifying_pair=False;e.verification_prefixes=0;e.verified_prefix_caches=None
    e.backend=SimpleNamespace(recurrent_many=lambda:None)
    def fail(tokens,**kwargs):
        e.caches[0][0].append(np.asarray([99]));e.position=3;e.hidden_history.append(np.asarray([99]))
        raise InterruptedError('cancelled')
    e.prefill=fail
    with pytest.raises(InterruptedError):e.verify_block(1,[2,3])
    assert e.position==len(e.hidden_history)==1 and len(e.caches[0][0])==1
    assert e.verified_prefix_caches is None and not e.verification_prefixes and not e.backend.capture_state_prefixes


def test_completed_verification_can_abort_before_any_prefix_is_committed():
    e=object.__new__(QwenTPU)
    e.layers=24;e.context=100;e.position=1;e.hidden_history=[np.asarray([7],np.float32)]
    e.last_hidden=e.hidden_history[-1];e.caches={0:([np.asarray([1])],[np.asarray([2])])}
    e.verifying_pair=False;e.verification_prefixes=0;e.verified_prefix_caches=None
    e.backend=SimpleNamespace(recurrent_many=lambda:None)
    def complete(tokens,**kwargs):
        e.caches[0][0].append(np.asarray([99]));e.position+=len(tokens)
        e.hidden_history.extend(np.asarray([99],np.float32) for _ in tokens)
        return np.zeros((len(tokens),4),np.float32)
    e.prefill=complete
    e.verify_block(1,[2,3]);e.finish_block(0)
    assert e.position==len(e.hidden_history)==1 and len(e.caches[0][0])==1
    assert e.verified_prefix_caches is None and e._verification_start is None


@pytest.mark.parametrize('interrupt',[False,True])
def test_estimated_mtp_states_are_discarded_after_chain_or_failure(interrupt):
    e=object.__new__(QwenTPU);e.layers=24;e.position=2;e.mtp_ready=0
    e.hidden_history=[np.zeros(4,np.float32),np.ones(4,np.float32)]
    e.caches={24:([np.asarray([1])],[np.asarray([2])])};e.mtp_last_hidden=None
    calls=0
    def draft(tokens,**kwargs):
        nonlocal calls
        calls+=1
        assert len(tokens)==e.position+1==len(e.hidden_history)+1
        e.caches[24][0].append(np.asarray([calls]))
        e.caches[24][1].append(np.asarray([calls]))
        e.mtp_ready=e.position;e.mtp_last_hidden=np.full(4,calls,np.float32)
        if interrupt and calls==2:raise InterruptedError('cancelled')
        return 10+calls
    e.mtp_draft=draft;e.greedy=lambda value:value
    if interrupt:
        with pytest.raises(InterruptedError):e.mtp_draft_chain([7,8,9],3)
    else:assert e.mtp_draft_chain([7,8,9],3)==[11,12,13]
    assert e.position==len(e.hidden_history)==e.mtp_ready==2
    assert len(e.caches[24][0])==len(e.caches[24][1])==2
    np.testing.assert_array_equal(e.mtp_last_hidden,np.ones(4,np.float32))
