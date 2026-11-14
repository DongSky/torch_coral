"""Physical fast/turbo comparisons on identical prompts and two decoding steps."""
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tokenizers import Tokenizer
from jinja2.sandbox import SandboxedEnvironment
from torch_coral.isa_mxu import MatrixTPU
from torch_coral.isa_optimized import OptimizedMatrixTPU
from torch_coral.qwen_tpu import QwenTPU


def main():
    tokenizer = Tokenizer.from_file(str(ROOT / '.tools/models/qwen35-0.8b/tokenizer.json'))
    template = SandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
        (ROOT / '.tools/models/qwen35-0.8b/chat_template.jinja').read_text(encoding='utf-8'))
    cases = [('raw', [9707])]
    for name, text in [('greeting', '你好'), ('math', '1+1等于几？'), ('english', 'Explain matrix multiplication.')]:
        prompt = template.render(messages=[dict(role='user', content=text)], add_generation_prompt=True,
                                 enable_thinking=False, tools=None, add_vision_id=False)
        cases.append((name, tokenizer.encode(prompt, add_special_tokens=False).ids))
    report = dict(status='RUNNING', reference='Original fast physical TPU path; no new quantization', cases=[])
    golden = {}
    for mode, cls in [('fast', MatrixTPU), ('turbo', OptimizedMatrixTPU)]:
        with cls(ROOT / '.tools/models/qwen35-gguf/int8-matrices', components=1, output_digits=2) as backend:
            model = QwenTPU(ROOT / '.tools/models/qwen35-gguf/fp32-prepared', backend)
            for name, tokens in cases:
                model.reset(); begin = time.perf_counter(); before = backend.calls
                logits = model.prefill(tokens); selected = []
                for position in range(2):
                    assert np.isfinite(logits).all()
                    if mode == 'fast': golden[name, position] = logits.copy()
                    else: np.testing.assert_array_equal(logits.view(np.uint32), golden[name, position].view(np.uint32))
                    np.save(ROOT / ('reports/equivalence-%s-%s-%d.npy' % (mode, name, position)), logits)
                    token = model.greedy(logits); selected.append(token)
                    if position == 0: logits = model.step(token)
                row = dict(mode=mode, case=name, input_tokens=len(tokens), generated_tokens=selected,
                           text=tokenizer.decode(selected, skip_special_tokens=True),
                           seconds=time.perf_counter() - begin, tpu_calls=backend.calls - before,
                           bitwise_match=mode == 'turbo', async_usb=backend.async_usb)
                report['cases'].append(row)
                print(json.dumps(row, ensure_ascii=True), flush=True)
                (ROOT / 'reports/fast-turbo-equivalence.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    report['status'] = 'bitwise PASS'
    (ROOT / 'reports/fast-turbo-equivalence.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
