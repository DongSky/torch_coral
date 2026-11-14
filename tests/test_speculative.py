"""Control-flow contracts for verified proposals, exact rejection and fallback."""
from types import SimpleNamespace
import torch_coral.speculative as module
from torch_coral.speculative import SpeculativeGreedy


class Engine:
    def __init__(self,drafts,clock=None):
        self.inputs=[7]; self.drafts=iter(drafts); self.backend=SimpleNamespace(); self.context=100
        self.clock=clock; self.draft_calls=0
    @property
    def position(self): return len(self.inputs)
    def tick(self,value):
        if self.clock is not None: self.clock.now+=value
    def step(self,anchor):
        self.inputs.append(anchor); self.tick(4); return dict(token=anchor+1)
    def greedy(self,scores): return scores['token']
    def mtp_draft(self,tokens):
        assert len(tokens)==self.position+1
        self.tick(1); self.draft_calls+=1; return dict(token=next(self.drafts))
    def verify_pair(self,anchor,draft):
        self.prefix=self.inputs+[anchor]; self.inputs += [anchor,draft]
        self.tick(6); return [dict(token=anchor+1),dict(token=draft+1)]
    def finish_pair(self,accepted):
        if not accepted: self.inputs=self.prefix


def test_acceptance_and_rejection_leave_exact_consumed_prefix():
    e=Engine([2,99]); decoder=SpeculativeGreedy(e,[7],adaptive=False)
    tokens,_,stop=decoder.advance(1,4,set())
    assert tokens==[2,3] and not stop
    assert e.inputs==decoder.processed==[7,1,2]
    tokens,_,stop=decoder.advance(3,2,set())
    assert tokens==[4] and not stop
    assert e.inputs==decoder.processed==[7,1,2,3]
    tokens,_,_=decoder.advance(4,1,set())
    assert tokens==[5] and e.draft_calls==2
    assert e.inputs==decoder.processed==[7,1,2,3,4]
    assert decoder.advance(5,0,set())==([],[],False)


def test_stop_proposal_does_not_commit_a_second_position():
    e=Engine([2]); decoder=SpeculativeGreedy(e,[7],adaptive=False)
    tokens,_,stop=decoder.advance(1,4,{2})
    assert tokens==[] and stop
    assert e.inputs==decoder.processed==[7,1]


def test_low_yield_turns_off_speculation_using_measured_block_cost(monkeypatch):
    class Clock:
        now=0.
        def __call__(self): return self.now
    clock=Clock(); monkeypatch.setattr(module.time,'perf_counter',clock)
    e=Engine([99,99],clock); decoder=SpeculativeGreedy(e,[7],adaptive=True)
    decoder.advance(1,8,set())  # actual ordinary-step baseline, four seconds
    decoder.advance(2,7,set()); decoder.advance(3,6,set())
    assert decoder.stats['baseline_seconds']==4
    assert decoder.stats['disabled'] and not decoder.enabled
    assert decoder.stats['speculative_seconds']==14
    assert decoder.stats['speculative_tokens']==2
    decoder.advance(4,5,set())
    assert e.draft_calls==2 and e.inputs==decoder.processed==[7,1,2,3,4]
