"""Validation ONLY: compare physical TPU logits against HF PyTorch + same GGUF weights.

This reference is never imported by the inference worker or the chat GUI.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from torch_coral.gguf_weights import WeightStore


def load_reference(weights):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(ROOT / '.tools/models/qwen35-0.8b', local_files_only=True).text_config
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        model = Qwen3_5ForCausalLM(config)
    state = {}
    translations = {
        'input_layernorm.weight': ('attn_norm.weight', 'zero_norm'),
        'post_attention_layernorm.weight': ('post_attention_norm.weight', 'zero_norm'),
        'mlp.gate_proj.weight': ('ffn_gate.weight', None),
        'mlp.up_proj.weight': ('ffn_up.weight', None),
        'mlp.down_proj.weight': ('ffn_down.weight', None),
        'linear_attn.in_proj_qkv.weight': ('attn_qkv.weight', None),
        'linear_attn.in_proj_z.weight': ('attn_gate.weight', None),
        'linear_attn.in_proj_b.weight': ('ssm_beta.weight', None),
        'linear_attn.in_proj_a.weight': ('ssm_alpha.weight', None),
        'linear_attn.conv1d.weight': ('ssm_conv1d.weight', 'conv'),
        'linear_attn.A_log': ('ssm_a', 'negative_exp'),
        'linear_attn.dt_bias': ('ssm_dt.bias', None),
        'linear_attn.norm.weight': ('ssm_norm.weight', None),
        'linear_attn.out_proj.weight': ('ssm_out.weight', None),
        'self_attn.q_proj.weight': ('attn_q.weight', None),
        'self_attn.k_proj.weight': ('attn_k.weight', None),
        'self_attn.v_proj.weight': ('attn_v.weight', None),
        'self_attn.o_proj.weight': ('attn_output.weight', None),
        'self_attn.q_norm.weight': ('attn_q_norm.weight', 'zero_norm'),
        'self_attn.k_norm.weight': ('attn_k_norm.weight', 'zero_norm'),
    }
    for name, parameter in model.named_parameters():
        if name == 'model.embed_tokens.weight': source, transform = 'token_embd.weight', None
        elif name == 'model.norm.weight': source, transform = 'output_norm.weight', 'zero_norm'
        elif name.startswith('model.layers.'):
            parts = name.split('.')
            layer, suffix = int(parts[2]), '.'.join(parts[3:])
            key, transform = translations[suffix]
            source = 'blk.%d.%s' % (layer, key)
        else:
            raise ValueError('Unmapped reference parameter: ' + name)
        array = weights[source].copy()
        if transform == 'zero_norm': array -= 1
        elif transform == 'conv': array = array[:, None, :]
        elif transform == 'negative_exp': array = np.log(-array)
        tensor = torch.from_numpy(array)
        if tensor.shape != parameter.shape:
            raise ValueError('Reference weight shape mismatch: ' + name)
        state[name] = tensor
    state['lm_head.weight'] = state['model.embed_tokens.weight']
    model.load_state_dict(state, strict=True, assign=True)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default=str(ROOT / 'reports/qwen-tpu-full.json'))
    parser.add_argument('--relative-l2-limit', type=float, default=.001,
                        help='Explicit validation tolerance; FP32 default remains 0.001')
    args = parser.parse_args()
    path = Path(args.report)
    result = json.loads(path.read_text())
    device_logits = np.load(path.with_suffix('.logits.npy'), allow_pickle=False)
    weights = WeightStore(ROOT / '.tools/models/qwen35-gguf/fp32-prepared')
    model = load_reference(weights)
    tokens = result['input_tokens'] + result['generated_tokens'][:-1]
    with torch.inference_mode():
        expected = model(input_ids=torch.tensor([tokens]), logits_to_keep=1).logits[0, -1].numpy()
    error = device_logits - expected
    metrics = dict(scope='Independent HF PyTorch reference; same dequantized GGUF weights',
        reference_backend='CPU, validation only', tpu_backend=result['backend'],
        tpu_top1=int(np.argmax(device_logits)), reference_top1=int(np.argmax(expected)),
        max_abs_error=float(np.abs(error).max()), rmse=float(np.sqrt(np.mean(error**2))),
        relative_l2=float(np.linalg.norm(error)/np.linalg.norm(expected)),
        cosine=float(np.dot(device_logits,expected)/(np.linalg.norm(device_logits)*np.linalg.norm(expected))))
    metrics['relative_l2_limit'] = args.relative_l2_limit
    metrics['status'] = 'PASS' if metrics['tpu_top1'] == metrics['reference_top1'] and metrics['relative_l2'] < args.relative_l2_limit else 'FAILED'
    target = path.with_name(path.stem + '-reference.json')
    target.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))
    if metrics['status'] != 'PASS': raise SystemExit(1)


if __name__ == '__main__': main()
