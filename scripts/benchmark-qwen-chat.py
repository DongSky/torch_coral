"""Benchmark the same persistent worker used by the GUI on a physical Coral."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['fp32', 'fast', 'turbo', 'turbo-v2', 'turbo-v3', 'turbo-v4', 'balanced', 'precise'], default='precise')
    parser.add_argument('--no-speculative',action='store_true')
    parser.add_argument('--verification-tokens',type=int,choices=[2,3,4],default=2)
    parser.add_argument('--prompt', default='你好')
    parser.add_argument('--new-tokens', type=int, default=2)
    parser.add_argument('--context',type=int,default=256)
    parser.add_argument('--mtp', action='store_true')
    parser.add_argument('--v2-clock',choices=['high','max'],default='max')
    parser.add_argument('--output', default=str(ROOT / 'reports/qwen-mxu-chat.json'))
    args = parser.parse_args()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    with output.with_suffix('.stderr.log').open('w', encoding='utf-8') as errors:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'torch_coral.qwen_chat_worker'],
            cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
            text=True, encoding='utf-8')
        def send(command):
            process.stdin.write(json.dumps(command, ensure_ascii=True) + '\n'); process.stdin.flush()
        start = time.perf_counter()
        done = False
        try:
            send(dict(action='load', gguf=str(ROOT / '.tools/models/qwen35-gguf/Qwen3.5-0.8B-Q4_K_M.gguf'),
                      prepared_dir=str(ROOT / '.tools/models/qwen35-gguf/fp32-prepared'),
                      tokenizer_dir=str(ROOT / '.tools/models/qwen35-0.8b'), context=args.context, matrix_profile=args.profile,v2_clock=args.v2_clock))
            for line in process.stdout:
                event = json.loads(line)
                print(json.dumps(event, ensure_ascii=True), flush=True)
                kind = event['event']
                if kind == 'error': raise RuntimeError(event['message'])
                if kind == 'loaded':
                    send(dict(action='chat', messages=[dict(role='user', content=args.prompt)],
                              max_new_tokens=args.new_tokens, mtp=args.mtp,speculative=not args.no_speculative,
                              speculative_block_tokens=args.verification_tokens))
                if kind == 'done':
                    assert event['tpu_calls'] > 0 and not event['report']['cpu_model_fallback']
                    result = dict(total_seconds=time.perf_counter() - start, profile=args.profile, mtp=args.mtp,
                                  context=args.context,max_new_tokens=args.new_tokens,prompt=args.prompt,result=event)
                    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                    done = True
                    send(dict(action='shutdown'))
                if kind == 'closed': break
            process.stdin.close(); process.wait(timeout=30)
            assert process.returncode == 0 and done, 'Worker exited without a completed benchmark'
        finally:
            if process.poll() is None:
                send(dict(action='cancel')); send(dict(action='shutdown')); process.wait(timeout=60)


if __name__ == '__main__': main()
