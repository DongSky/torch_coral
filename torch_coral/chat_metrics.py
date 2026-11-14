"""Monotonic inference timings; independent of Qt and numerical backends."""
import time


class InferenceMetrics:
    def __init__(self, prompt_tokens, reused_tokens, max_new_tokens, *, clock=time.perf_counter, started_at=None):
        self.clock = clock
        self.phase_started = clock()
        self.started = self.phase_started if started_at is None else started_at
        self.phase = 'prefill'
        self.times = dict(prefill=0., sampling=0., decode=0., mtp=0.)
        self.prompt_tokens = prompt_tokens
        self.reused_tokens = reused_tokens
        self.max_new_tokens = max_new_tokens
        self.generated = 0
        self.ttft = None
        self.last_decode_seconds = None
        self.last_sample_decode_time = 0.
        self.prefill_complete = False

    def phase_time(self, name, now=None):
        now = self.clock() if now is None else now
        return self.times.get(name, 0.) + (now - self.phase_started if self.phase == name else 0.)

    def begin(self, phase):
        now = self.clock()
        if self.phase in self.times:
            self.times[self.phase] += max(0., now - self.phase_started)
        if self.phase == 'prefill' and phase == 'sampling': self.prefill_complete = True
        self.phase, self.phase_started = phase, now

    def token(self, count=1):
        if not isinstance(count,int) or count<1:
            raise ValueError('Visible token count must be positive')
        if self.generated==0 and count!=1:
            raise ValueError('The first visible token must be measured separately')
        now = self.clock()
        if self.generated == 0:
            self.ttft = now - self.started
        else:
            total = self.phase_time('decode', now)
            self.last_decode_seconds = (total - self.last_sample_decode_time)/count
            self.last_sample_decode_time = total
        self.generated += count
        return self.snapshot(now)

    def snapshot(self, now=None):
        now = self.clock() if now is None else now
        elapsed = max(0., now - self.started)
        prefill = self.phase_time('prefill', now)
        decode = self.phase_time('decode', now)
        decoded = max(0, self.generated - 1)
        return dict(phase=self.phase, elapsed_seconds=elapsed, prefill_complete=self.prefill_complete,
            phase_elapsed_seconds=max(0., now - self.phase_started),
            prefill_seconds=prefill, decode_seconds=decode, mtp_seconds=self.phase_time('mtp', now),
            prompt_tokens=self.prompt_tokens, reused_tokens=self.reused_tokens,
            generated_tokens=self.generated, decode_tokens=decoded, max_new_tokens=self.max_new_tokens,
            ttft_seconds=self.ttft,
            prefill_tokens_per_second=self.prompt_tokens / prefill if self.prefill_complete and prefill > 0 else None,
            decode_tokens_per_second=decoded / decode if decoded and decode > 0 else None,
            last_decode_tokens_per_second=1 / self.last_decode_seconds if self.last_decode_seconds and self.last_decode_seconds > 0 else None,
            end_to_end_tokens_per_second=self.generated / elapsed if elapsed > 0 else 0.)
