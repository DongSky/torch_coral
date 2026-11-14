"""Compare a physical chat result with independent HF greedy generation.

The CPU reference is used only by this validation script, never by inference.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoTokenizer
from torch_coral.gguf_weights import WeightStore

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default=str(ROOT / 'reports/qwen-mxu-chat.json'))
    args = parser.parse_args()
    path = Path(args.report)
    actual = json.loads(path.read_text(encoding='utf-8'))['result']
    spec = importlib.util.spec_from_file_location('qwen_reference', ROOT / 'scripts/validate-qwen-reference.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model = module.load_reference(WeightStore(ROOT / '.tools/models/qwen35-gguf/fp32-prepared'))
    tokenizer = AutoTokenizer.from_pretrained(ROOT / '.tools/models/qwen35-0.8b', local_files_only=True)
    inputs = torch.tensor([actual['input_tokens']])
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
            max_new_tokens=actual['new_tokens'], do_sample=False, eos_token_id=[248044, 248046], pad_token_id=248044)
    expected_ids = output[0, inputs.shape[1]:].tolist()
    expected = tokenizer.decode(expected_ids, skip_special_tokens=True)
    result = dict(reference='Independent HF CPU validation only, same GGUF weights',
                  expected_text=expected, actual_text=actual['text'], expected_tokens=expected_ids,
                  actual_tokens=actual['generated_tokens'],
                  status='PASS' if expected_ids == actual['generated_tokens'] else 'FAILED')
    path.with_name(path.stem + '-reference.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if result['status'] != 'PASS': raise SystemExit(1)


if __name__ == '__main__': main()
