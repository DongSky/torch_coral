"""Balanced physical A/B: original V4 versus prepared/pinned V4 dispatch."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from torch_coral.isa_io import IOOptimizedMatrixTPU
from torch_coral.qwen_tpu import QwenTPU
from torch_coral.speculative import SpeculativeGreedy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--dispatch-only', action='store_true', help='Isolate dispatch; retain original wide quantization')
    parser.add_argument('--output', default='reports/turbo-v4-dispatch-ab.json')
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error('cycles must be positive')
    prompt = [248045,846,198,109266,248046,198,248045,74455,198,248068,271,248069,271]
    rows, golden = [], None
    with IOOptimizedMatrixTPU(ROOT/'.tools/models/qwen35-gguf/int8-matrices', components=1, output_digits=2, clock='max') as b:
        b.reuse_projection_quantization=b.compute_fusions=b.fused_head_selection=b.native_active_bias=False
        b.native_batch_min=8;b.native_large_min=12
        prepared, legacy, runtime = b.device.batch, b._legacy_v4_batch, b._prepared_runtime
        engine = QwenTPU(ROOT/'.tools/models/qwen35-gguf/fp32-prepared', b)
        report = dict(status='RUNNING', baseline='V4 0.3.0a7: direct readback',
                      optimized='V4: prepared dispatch, pinned OUT packets and wide primary quantization',
                      wide_quantization=not args.dispatch_only, clock='max',
                      order='A B B A repeated after two full warmups', rows=rows)
        try:
            order = ['warmup-original', 'warmup-prepared'] + ['original', 'prepared', 'prepared', 'original'] * args.cycles
            for iteration, mode in enumerate(order):
                active = mode.endswith('prepared')
                with b.lock:
                    assert b.device.batch.pending.value == 0
                    b.device.batch = prepared if active else legacy
                    b._prepared_runtime = runtime if active else None
                    b.wide_projection_quantization = active and not args.dispatch_only
                engine.reset()
                before = b.calls
                quantize_before = b.matrix_stats['wide_quantization_calls']
                begin = time.perf_counter()
                scores = engine.prefill(prompt); first = engine.greedy(scores)
                prefill = time.perf_counter() - begin
                logits = [scores.copy()]
                begin = time.perf_counter()
                scores = engine.step(first); anchor = engine.greedy(scores)
                ordinary = time.perf_counter() - begin
                logits.append(scores.copy())
                decoder = SpeculativeGreedy(engine, prompt + [first], adaptive=False)
                begin = time.perf_counter()
                emitted, verified, stop = decoder.advance(anchor, 4, set())
                speculative = time.perf_counter() - begin
                logits.extend(value.copy() for value in verified)
                state = {layer: tuple(np.array(value, copy=True) for value in cache) for layer, cache in engine.caches.items()}
                if golden is None:
                    golden = logits, state
                else:
                    assert len(logits) == len(golden[0])
                    for value, expected in zip(logits, golden[0]):
                        np.testing.assert_array_equal(value.view(np.uint32), expected.view(np.uint32))
                    assert set(state) == set(golden[1])
                    for layer, cache in state.items():
                        for value, expected in zip(cache, golden[1][layer]):
                            np.testing.assert_array_equal(value.view(np.uint32), expected.view(np.uint32))
                assert first == 109266 and anchor == 6115 and emitted == [116577, 108888] and not stop
                row = dict(iteration=iteration, mode=mode, prefill_s=prefill, ordinary_decode_s=ordinary,
                           speculative_pair_s=speculative, tpu_calls=b.calls-before,
                           logits_and_all_states_bitwise_equal=True,
                           dispatch_cache_hits=runtime.hits, dispatch_cache_misses=runtime.misses)
                row['wide_quantization_calls'] = b.matrix_stats['wide_quantization_calls'] - quantize_before
                print(json.dumps(row), flush=True)
                if not mode.startswith('warmup'):
                    rows.append(row)
                Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
        finally:
            b.device.batch, b._prepared_runtime = prepared, runtime
    report['summary'] = {}
    for field in ('prefill_s', 'ordinary_decode_s', 'speculative_pair_s'):
        old = statistics.median(row[field] for row in rows if row['mode'] == 'original')
        new = statistics.median(row[field] for row in rows if row['mode'] == 'prepared')
        report['summary'][field] = dict(original_s=old, prepared_s=new, saved_s=old-new,
                                        elapsed_reduction_fraction=1-new/old)
    assert all(row['tpu_calls'] == 44091 for row in rows if row['mode'] == 'original')
    assert len({row['tpu_calls'] for row in rows if row['mode'] == 'prepared'}) == 1
    report['status'] = 'bitwise PASS'
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
