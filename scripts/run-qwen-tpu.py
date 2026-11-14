"""Strict full-text-model Edge TPU reference inference. No CPU/CUDA fallback."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from torch_coral.isa_scalar import ScalarTPU
from torch_coral.qwen_tpu import QwenTPU


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', default=str(ROOT / '.tools/models/qwen35-gguf/fp32-prepared'))
    parser.add_argument('--tokens', default='9707', help='Comma-separated token IDs; tokenization is separate')
    parser.add_argument('--new-tokens', type=int, default=1)
    parser.add_argument('--mtp', action='store_true')
    parser.add_argument('--matrix-profile', choices=['fp32', 'fast', 'turbo', 'turbo-v2', 'turbo-v3', 'turbo-v4', 'balanced', 'precise'], default='fp32')
    parser.add_argument('--no-speculative',action='store_true',help='Disable V3/V4 greedy speculative verification')
    parser.add_argument('--verification-tokens',type=int,choices=[2,3,4],default=2)
    parser.add_argument('--fixed-blocks',action='store_true',help='Experimental estimated MTP chains; may be slower than history matching')
    parser.add_argument('--matrix-dir', default=str(ROOT / '.tools/models/qwen35-gguf/int8-matrices'))
    parser.add_argument('--chunk-rows', type=int, default=6144)
    parser.add_argument('--clock',choices=['high','max'],default=None,help='Turbo V2/V3/V4 default to max; other modes to high')
    parser.add_argument('--output', default=str(ROOT / 'reports/qwen-tpu-full.json'))
    args = parser.parse_args()
    tokens = [int(v) for v in args.tokens.split(',')]
    generated = []
    start = time.perf_counter()
    if args.matrix_profile == 'fp32':
        device = ScalarTPU(clock=args.clock or 'high')
    else:
        from torch_coral.isa_mxu import MatrixTPU
        components, digits = {'fast':(1,2), 'turbo':(1,2), 'turbo-v2':(1,2), 'turbo-v3':(1,2), 'turbo-v4':(1,2), 'balanced':(2,2), 'precise':(2,3)}[args.matrix_profile]
        if args.matrix_profile == 'turbo':
            from torch_coral.isa_optimized import OptimizedMatrixTPU
            MatrixTPU = OptimizedMatrixTPU
        elif args.matrix_profile == 'turbo-v2':
            from torch_coral.isa_streamed import StreamedMatrixTPU
            MatrixTPU = StreamedMatrixTPU
        elif args.matrix_profile=='turbo-v3':
            from torch_coral.isa_batched import BatchedMatrixTPU
            MatrixTPU=BatchedMatrixTPU
        elif args.matrix_profile=='turbo-v4':
            from torch_coral.isa_io import IOOptimizedMatrixTPU
            MatrixTPU=IOOptimizedMatrixTPU
        device = MatrixTPU(args.matrix_dir, components=components, output_digits=digits, chunk_rows=args.chunk_rows,
                           clock=args.clock or ('max' if args.matrix_profile in ('turbo-v2','turbo-v3','turbo-v4') else 'high'))
    with device:
        engine = QwenTPU(args.weights, device, progress=lambda s: print(s, flush=True))
        scores = engine.prefill(tokens)
        if args.matrix_profile in ('turbo-v3','turbo-v4') and not args.no_speculative:
            from torch_coral.speculative import SpeculativeGreedy
            decoder=SpeculativeGreedy(engine,tokens,adaptive=not args.fixed_blocks,
                max_block_tokens=args.verification_tokens if args.matrix_profile=='turbo-v4' else 2)
            generated.append(engine.greedy(scores))
            while len(generated)<args.new_tokens:
                emitted,rows,stop=decoder.advance(generated[-1],args.new_tokens-len(generated),set())
                generated.extend(emitted)
                if emitted: scores=rows[len(emitted)-1]
                if stop: break
            print('GENERATED',generated,flush=True)
            if args.mtp:
                draft=engine.mtp_draft(decoder.processed+[generated[-1]])
                np.save(Path(args.output).with_suffix('.mtp.logits.npy'),draft)
        for index in range(0 if args.matrix_profile in ('turbo-v3','turbo-v4') and not args.no_speculative else args.new_tokens):
            assert np.isfinite(scores).all(), 'Model produced non-finite logits'
            token = engine.greedy(scores)
            generated.append(token)
            if args.mtp:
                draft = engine.mtp_draft(tokens + generated)
                assert np.isfinite(draft).all()
                np.save(Path(args.output).with_suffix('.mtp.logits.npy'), draft)
            print('GENERATED', token, flush=True)
            if index + 1 < args.new_tokens:
                scores = engine.step(token)
        report = engine.report()
        report.update(input_tokens=tokens, generated_tokens=generated,
                      elapsed_seconds=time.perf_counter() - start)
        np.save(str(Path(args.output).with_suffix('.logits.npy')), scores)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
