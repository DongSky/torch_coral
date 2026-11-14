import pytest
from torch_coral.chat_metrics import InferenceMetrics


class Clock:
    now = 0.
    def __call__(self): return self.now
    def advance(self, seconds): self.now += seconds


def test_prefill_first_token_and_mtp_do_not_inflate_decode_speed():
    clock = Clock(); m = InferenceMetrics(13, 4, 16, clock=clock)
    clock.advance(20); m.begin('sampling'); clock.advance(.2)
    first = m.token()
    assert first['ttft_seconds'] == pytest.approx(20.2)
    assert first['decode_tokens_per_second'] is None
    assert first['prefill_tokens_per_second'] == pytest.approx(13 / 20)
    m.begin('mtp'); clock.advance(3); m.begin('decode'); clock.advance(5)
    second = m.token()
    assert second['decode_tokens'] == 1 and second['generated_tokens'] == 2
    assert second['decode_tokens_per_second'] == pytest.approx(.2)
    assert second['last_decode_tokens_per_second'] == pytest.approx(.2)
    m.begin('mtp'); clock.advance(2); m.begin('complete')
    final = m.snapshot()
    assert final['mtp_seconds'] == 5
    assert final['decode_seconds'] == 5
    assert final['end_to_end_tokens_per_second'] == pytest.approx(2 / 30.2)


def test_pending_decode_updates_denominator_without_inventing_tokens():
    clock = Clock(); m = InferenceMetrics(1, 0, 3, clock=clock)
    m.begin('sampling'); m.token(); m.begin('decode'); clock.advance(4); m.token()
    clock.advance(4)
    frame = m.snapshot()
    assert frame['generated_tokens'] == 2
    assert frame['decode_tokens_per_second'] == pytest.approx(1 / 8)
    assert frame['last_decode_tokens_per_second'] == pytest.approx(1 / 4)


def test_cancellation_is_not_a_completed_prefill_and_zero_times_are_safe():
    clock = Clock(); m = InferenceMetrics(8, 0, 1, clock=clock)
    assert m.snapshot()['end_to_end_tokens_per_second'] == 0
    clock.advance(2); m.begin('cancelled')
    frame = m.snapshot()
    assert not frame['prefill_complete']
    assert frame['prefill_tokens_per_second'] is None
    assert frame['ttft_seconds'] is None


def test_request_preparation_time_is_included_in_ttft():
    clock = Clock(); clock.advance(1)
    m = InferenceMetrics(1, 0, 1, clock=clock, started_at=0.)
    clock.advance(3); m.begin('sampling'); frame = m.token()
    assert frame['ttft_seconds'] == 4
    assert frame['prefill_seconds'] == 3


def test_verified_pair_uses_whole_block_cost_and_two_visible_tokens():
    clock=Clock(); m=InferenceMetrics(13,0,8,clock=clock)
    clock.advance(13); m.begin('sampling'); m.token()
    m.begin('decode'); clock.advance(5.2)  # includes MTP, verification and rollback work
    frame=m.token(count=2)
    assert frame['decode_tokens']==2 and frame['generated_tokens']==3
    assert frame['decode_seconds']==pytest.approx(5.2)
    assert frame['last_decode_tokens_per_second']==pytest.approx(2/5.2)
    assert frame['decode_tokens_per_second']==pytest.approx(2/5.2)
    with pytest.raises(ValueError): m.token(count=0)
