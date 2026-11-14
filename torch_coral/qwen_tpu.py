"""Qwen3.5 text inference with every numerical model operation on a TPU backend.

The host performs tokenization, storage, views/copies, loop scheduling and I/O.
There is no Torch/NumPy model-arithmetic fallback in this module. FP32 scalar
tiling is a slow reference implementation, not a claim of practical LLM speed.
"""
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from contextlib import nullcontext
from functools import wraps
from .gguf_weights import WeightStore


def shared_projection_inputs(method):
    @wraps(method)
    def call(self, layer, inputs, *args, **kwargs):
        with self.projection_inputs(inputs) as frozen:
            return method(self, layer, frozen, *args, **kwargs)
    return call


class QwenTPU:
    def __init__(self, weights, backend, *, context=256, progress=lambda message: None, cancel=None, telemetry=None):
        self.weights = weights if isinstance(weights, WeightStore) else WeightStore(weights)
        if hasattr(backend, 'matrix_manifest') and backend.matrix_manifest['source_sha256'] != self.weights.manifest['source_sha256']:
            raise ValueError('INT8 cache belongs to a different model')
        self.backend, self.context = backend, context
        self.progress, self.cancel = progress, cancel
        self.telemetry = telemetry
        meta = self.weights.metadata
        self.dim = int(meta['qwen35.embedding_length'])
        self.layers = int(meta['qwen35.block_count']) - int(meta.get('qwen35.nextn_predict_layers', 0))
        self.heads = int(meta['qwen35.attention.head_count'])
        self.kv_heads = int(meta['qwen35.attention.head_count_kv'])
        self.head_dim = int(meta['qwen35.attention.key_length'])
        self.ssm_heads = int(meta['qwen35.ssm.time_step_rank'])
        self.ssm_dim = int(meta['qwen35.ssm.state_size'])
        self.eps = float(meta['qwen35.attention.layer_norm_rms_epsilon'])
        self.rope_dim = int(meta['qwen35.rope.dimension_count'])
        if self.dim != 1024 or self.layers != 24 or self.heads != 8 or self.ssm_dim != 128:
            raise ValueError('This experimental executor targets dense Qwen3.5-0.8B only')
        if not 1 <= context <= 4096:
            raise ValueError('Experimental context must be 1..4096')
        # RoPE positions are immutable model constants, prepared before inference.
        frequencies = 1.0 / float(meta['qwen35.rope.freq_base']) ** (
            np.arange(0, self.rope_dim, 2, dtype=np.float64) / self.rope_dim)
        angles = np.outer(np.arange(context, dtype=np.float64), frequencies)
        self.cos = np.cos(angles).astype(np.float32)
        self.sin = np.sin(angles).astype(np.float32)
        self.reset()

    def reset(self):
        self.position = 0
        self.caches = {}
        self.coverage = Counter()
        self.hidden_history = []
        self.mtp_ready = 0
        self.verifying_pair = False
        self.verified_first_caches = None
        self.verification_prefixes = 0
        self.verified_prefix_caches = None
        self.mtp_last_hidden = None
        self._verification_start = None
        self.speculative_stats = None

    def _check_cancel(self):
        if self.cancel is not None and self.cancel.is_set():
            raise InterruptedError('Generation cancelled between TPU programs')

    def op(self, kind, a, b=None):
        self._check_cancel()
        return self.backend.op(kind, a, b)

    def linear(self, name, x):
        self._check_cancel()
        self.coverage[name] += 1
        if hasattr(self.backend, 'project'):
            return self.backend.project(name, self.weights[name], x)
        if x.ndim == 2:
            return self.backend.matmul(self.weights[name], x)
        return self.backend.matvec(self.weights[name], x)

    def projection_inputs(self, inputs):
        scope = getattr(self.backend, 'projection_scope', None)
        return scope(inputs) if scope is not None else nullcontext(inputs)

    def ffn_input(self, prefix, normalized):
        with self.projection_inputs(normalized) as frozen:
            gate = self.linear(prefix + 'ffn_gate.weight', frozen)
            up = self.linear(prefix + 'ffn_up.weight', frozen)
        if hasattr(self.backend, 'silu_multiply'):
            return self.backend.silu_multiply(gate, up)
        return self.op('mul', self.silu(gate), up)

    def norm(self, x, weight=None, *, l2=False, projection=False):
        if projection and hasattr(self.backend, 'normalize_project_input'):
            self._check_cancel()
            return self.backend.normalize_project_input(x, weight, l2, self.eps)
        if hasattr(self.backend, 'normalize'):
            self._check_cancel()
            return self.backend.normalize(x, weight, l2, self.eps)
        shape = x.shape
        rows = x.reshape(-1, shape[-1])
        output = []
        for row in rows:
            squares = self.op('mul', row, row)
            total = self.backend.reduce('sum', squares)
            if not l2:
                total = self.op('mul', total, np.float32(1 / len(row)))
            inverse = self.op('rsqrt', self.op('add', total, np.float32(self.eps)))
            value = self.op('mul', row, inverse)
            if weight is not None:
                value = self.op('mul', value, weight)
            output.append(value)
        return np.stack(output).reshape(shape)

    def sigmoid(self, x):
        if hasattr(self.backend, 'activation'):
            return self.backend.activation('sigmoid', x)
        negative = self.op('mul', x, np.float32(-1))
        negative = self.op('min', self.op('max', negative, np.float32(-80)), np.float32(80))
        return self.op('recip', self.op('add', self.op('exp', negative), np.float32(1)))

    def silu(self, x):
        if hasattr(self.backend, 'activation'):
            return self.backend.activation('silu', x)
        return self.op('mul', x, self.sigmoid(x))

    def softplus(self, x):
        absolute = self.op('max', x, self.op('mul', x, np.float32(-1)))
        exponential = self.op('exp', self.op('mul', absolute, np.float32(-1)))
        logarithm = self.op('log_unit', self.op('add', exponential, np.float32(1)))
        return self.op('add', self.op('max', x, np.float32(0)), logarithm)

    def softmax(self, x):
        if hasattr(self.backend, 'softmax') and 1 <= x.size <= 2048:
            self._check_cancel()
            return self.backend.softmax(x)
        maximum = self.backend.reduce('max', x)
        exponential = self.op('exp', self.op('sub', x, maximum))
        inverse = self.op('recip', self.backend.reduce('sum', exponential))
        return self.op('mul', exponential, inverse)

    def rope(self, x, position):
        if hasattr(self.backend, 'rope'):
            self._check_cancel()
            return self.backend.rope(x, self.cos[position], self.sin[position])
        # Qwen3.5 uses the NeoX split-half layout for its rotary subspace.
        rotated, tail = x[..., :self.rope_dim], x[..., self.rope_dim:]
        a, b = np.split(rotated, 2, axis=-1)
        c = self.op('copy', self.cos[position])
        s = self.op('copy', self.sin[position])
        left = self.op('sub', self.op('mul', a, c), self.op('mul', b, s))
        right = self.op('add', self.op('mul', a, s), self.op('mul', b, c))
        result = np.concatenate([left, right], axis=-1)
        return np.concatenate([result, tail], axis=-1)

    @shared_projection_inputs
    def attention(self, layer, x, position):
        prefix = 'blk.%d.' % layer
        qg = self.linear(prefix + 'attn_q.weight', x).reshape(self.heads, 2, self.head_dim)
        q, gate = qg[:, 0, :], qg[:, 1, :]
        k = self.linear(prefix + 'attn_k.weight', x).reshape(self.kv_heads, self.head_dim)
        v = self.linear(prefix + 'attn_v.weight', x).reshape(self.kv_heads, self.head_dim)
        q = self.norm(q, self.weights[prefix + 'attn_q_norm.weight'])
        k = self.norm(k, self.weights[prefix + 'attn_k_norm.weight'])
        q, k = self.rope(q, position), self.rope(k, position)
        if layer not in self.caches:
            self.caches[layer] = ([], [])
        keys, values = self.caches[layer]
        keys.append(k.copy())
        values.append(v.copy())
        keys, values = np.stack(keys), np.stack(values)
        outputs = []
        for head in range(self.heads):
            kv = head // (self.heads // self.kv_heads)
            scores = self.backend.matvec(keys[:, kv, :], q[head])
            probabilities = self.softmax(self.op('mul', scores, np.float32(self.head_dim ** -.5)))
            outputs.append(self.backend.matvec(values[:, kv, :].T, probabilities))
        output = self.op('mul', np.stack(outputs), self.sigmoid(gate)).reshape(-1)
        self.coverage[prefix + 'full_attention'] += 1
        return self.linear(prefix + 'attn_output.weight', output)

    @shared_projection_inputs
    def delta(self, layer, x):
        prefix = 'blk.%d.' % layer
        projection = self.linear(prefix + 'attn_qkv.weight', x)
        channels = len(projection)
        if layer not in self.caches:
            self.caches[layer] = (np.zeros((3, channels), np.float32),
                np.zeros((self.ssm_heads, self.ssm_dim, self.ssm_dim), np.float32))
        history, state = self.caches[layer]
        window = np.concatenate([history, projection[None, :]], axis=0)
        kernel = self.weights[prefix + 'ssm_conv1d.weight']
        if hasattr(self.backend, 'convolution_silu'):
            convolved = self.backend.convolution_silu(window, kernel)
        else:
            convolved = self.op('mul', window[0], kernel[:, 0])
            for tap in range(1, 4):
                convolved = self.op('add', convolved, self.op('mul', window[tap], kernel[:, tap]))
            convolved = self.silu(convolved)
        q, k, v = np.split(convolved, [self.ssm_heads * self.ssm_dim, self.ssm_heads * self.ssm_dim * 2])
        q = self.norm(q.reshape(self.ssm_heads, self.ssm_dim), l2=True)
        k = self.norm(k.reshape(self.ssm_heads, self.ssm_dim), l2=True)
        q = self.op('mul', q, np.float32(self.ssm_dim ** -.5))
        v = v.reshape(self.ssm_heads, self.ssm_dim)
        beta = self.sigmoid(self.linear(prefix + 'ssm_beta.weight', x))
        alpha = self.linear(prefix + 'ssm_alpha.weight', x)
        alpha = self.op('add', alpha, self.weights[prefix + 'ssm_dt.bias'])
        # GGUF stores the already transformed negative A constants.
        decay = self.op('exp', self.op('mul', self.softplus(alpha), self.weights[prefix + 'ssm_a']))
        output, updated = [], []
        for head in range(self.ssm_heads):
            if hasattr(self.backend, 'recurrent'):
                current, projected = self.backend.recurrent(state[head], k[head], q[head], v[head], beta[head], decay[head])
                output.append(projected); updated.append(current)
                continue
            current = self.op('mul', state[head], decay[head])
            memory = self.backend.matvec(current.T, k[head])
            delta = self.op('mul', self.op('sub', v[head], memory), beta[head])
            current = self.op('add', current, self.op('mul', k[head, :, None], delta[None, :]))
            output.append(self.backend.matvec(current.T, q[head]))
            updated.append(current)
        self.caches[layer] = (window[1:].copy(), np.stack(updated))
        output = self.norm(np.stack(output), self.weights[prefix + 'ssm_norm.weight'])
        gate = self.linear(prefix + 'attn_gate.weight', x).reshape(self.ssm_heads, self.ssm_dim)
        output = self.op('mul', output, self.silu(gate)).reshape(-1)
        self.coverage[prefix + 'gated_delta_net'] += 1
        return self.linear(prefix + 'ssm_out.weight', output)

    def block(self, layer, x, position):
        prefix = 'blk.%d.' % layer
        normalized = self.norm(x, self.weights[prefix + 'attn_norm.weight'], projection=True)
        self.coverage[prefix + 'attn_norm'] += 1
        if prefix + 'attn_qkv.weight' in self.weights.manifest['tensors']:
            residual = self.delta(layer, normalized)
        else:
            residual = self.attention(layer, normalized, position)
        x = self.op('add', x, residual)
        normalized = self.norm(x, self.weights[prefix + 'post_attention_norm.weight'], projection=True)
        down = self.linear(prefix + 'ffn_down.weight', self.ffn_input(prefix, normalized))
        self.coverage[prefix + 'ffn'] += 1
        return self.op('add', x, down)

    @shared_projection_inputs
    def delta_many(self, layer, inputs):
        prefix = 'blk.%d.' % layer
        projections = self.linear(prefix + 'attn_qkv.weight', inputs)
        gates = self.linear(prefix + 'attn_gate.weight', inputs).reshape(-1, self.ssm_heads, self.ssm_dim)
        beta = self.sigmoid(self.linear(prefix + 'ssm_beta.weight', inputs))
        alpha = self.op('add', self.linear(prefix + 'ssm_alpha.weight', inputs), self.weights[prefix + 'ssm_dt.bias'])
        decay = self.op('exp', self.op('mul', self.softplus(alpha), self.weights[prefix + 'ssm_a']))
        channels = projections.shape[-1]
        history, state = self.caches.get(layer, (np.zeros((3, channels), np.float32),
            np.zeros((self.ssm_heads, self.ssm_dim, self.ssm_dim), np.float32)))
        kernel = self.weights[prefix + 'ssm_conv1d.weight']
        collected = []
        prefix_histories = []
        streamed = hasattr(self.backend, 'recurrent_many')
        query_rows, key_rows, value_rows = [], [], []
        convolved_rows=self.backend.convolution_many(history,projections,kernel) if hasattr(self.backend,'convolution_many') else None
        for index, projection in enumerate(projections):
            window = np.concatenate([history, projection[None, :]], axis=0)
            if convolved_rows is not None:
                convolved=convolved_rows[index]
            elif hasattr(self.backend, 'convolution_silu'):
                convolved = self.backend.convolution_silu(window, kernel)
            else:
                convolved = self.op('mul', window[0], kernel[:, 0])
                for tap in range(1, 4):
                    convolved = self.op('add', convolved, self.op('mul', window[tap], kernel[:, tap]))
                convolved = self.silu(convolved)
            q, k, v = np.split(convolved, [self.ssm_heads * self.ssm_dim, self.ssm_heads * self.ssm_dim * 2])
            q = self.op('mul', self.norm(q.reshape(self.ssm_heads, self.ssm_dim), l2=True), np.float32(self.ssm_dim ** -.5))
            k = self.norm(k.reshape(self.ssm_heads, self.ssm_dim), l2=True)
            v = v.reshape(self.ssm_heads, self.ssm_dim)
            if self.verifying_pair and index==0:
                first_history=window[1:].copy()
            if self.verification_prefixes and index < self.verification_prefixes:
                prefix_histories.append(window[1:].copy())
            if streamed:
                query_rows.append(q); key_rows.append(k); value_rows.append(v)
                history = window[1:].copy()
                continue
            output, updated = [], []
            for head in range(self.ssm_heads):
                if hasattr(self.backend, 'recurrent'):
                    current, projected = self.backend.recurrent(state[head], k[head], q[head], v[head], beta[index, head], decay[index, head])
                    output.append(projected); updated.append(current)
                    continue
                current = self.op('mul', state[head], decay[index, head])
                memory = self.backend.matvec(current.T, k[head])
                delta = self.op('mul', self.op('sub', v[head], memory), beta[index, head])
                current = self.op('add', current, self.op('mul', k[head, :, None], delta[None, :]))
                output.append(self.backend.matvec(current.T, q[head]))
                updated.append(current)
            state = np.stack(updated)
            history = window[1:].copy()
            collected.append(np.stack(output))
        if streamed:
            state, collected = self.backend.recurrent_many(state, np.stack(key_rows), np.stack(query_rows),
                                                          np.stack(value_rows), beta, decay)
            if self.verifying_pair:
                self.verified_first_caches[layer]=(first_history,self.backend.first_recurrent_state.copy())
            for index, prefix_history in enumerate(prefix_histories):
                self.verified_prefix_caches[index][layer]=(prefix_history,self.backend.prefix_recurrent_states[index].copy())
        self.caches[layer] = history, state
        normalized = self.norm(np.stack(collected), self.weights[prefix + 'ssm_norm.weight'])
        output = (self.backend.silu_multiply(gates, normalized) if hasattr(self.backend, 'silu_multiply')
                  else self.op('mul', normalized, self.silu(gates))).reshape(len(inputs), -1)
        self.coverage[prefix + 'gated_delta_net'] += len(inputs)
        return self.linear(prefix + 'ssm_out.weight', output)

    @shared_projection_inputs
    def attention_many(self, layer, inputs, position):
        prefix = 'blk.%d.' % layer
        length = len(inputs)
        qg = self.linear(prefix + 'attn_q.weight', inputs).reshape(length, self.heads, 2, self.head_dim)
        q, gate = qg[:, :, 0, :], qg[:, :, 1, :]
        k = self.linear(prefix + 'attn_k.weight', inputs).reshape(length, self.kv_heads, self.head_dim)
        v = self.linear(prefix + 'attn_v.weight', inputs).reshape(length, self.kv_heads, self.head_dim)
        q = self.norm(q, self.weights[prefix + 'attn_q_norm.weight'])
        k = self.norm(k, self.weights[prefix + 'attn_k_norm.weight'])
        keys, values = self.caches.get(layer, ([], []))
        collected = []
        for index in range(length):
            rotated_q = self.rope(q[index], position + index)
            keys.append(self.rope(k[index], position + index).copy())
            values.append(v[index].copy())
            if self.verifying_pair and index==0:
                self.verified_first_caches[layer]=(keys.copy(),values.copy())
            if self.verification_prefixes and index < self.verification_prefixes:
                self.verified_prefix_caches[index][layer]=(keys.copy(),values.copy())
            key_array, value_array = np.stack(keys), np.stack(values)
            heads = []
            for head in range(self.heads):
                kv = head // (self.heads // self.kv_heads)
                scores = self.backend.matvec(key_array[:, kv, :], rotated_q[head])
                probabilities = self.softmax(self.op('mul', scores, np.float32(self.head_dim ** -.5)))
                heads.append(self.backend.matvec(value_array[:, kv, :].T, probabilities))
            collected.append(np.stack(heads))
        self.caches[layer] = keys, values
        output = self.op('mul', np.stack(collected), self.sigmoid(gate)).reshape(length, -1)
        self.coverage[prefix + 'full_attention'] += length
        return self.linear(prefix + 'attn_output.weight', output)

    def prefill(self, tokens, *, all_logits=False):
        """Execute prompt positions as shared-weight tiles; only the last needs logits."""
        if not tokens or self.position + len(tokens) > self.context:
            raise ValueError('Prompt exceeds experimental context limit')
        if len(tokens) == 1:
            value=self.step(tokens[0])
            return value[None] if all_logits else value
        for start in range(0, len(tokens), 16):
            chunk = tokens[start:start + 16]
            x = np.stack([self.op('copy', self.weights['token_embd.weight'][int(t)]) for t in chunk])
            self.coverage['token_embedding'] += len(chunk)
            for layer in range(self.layers):
                self.progress('prefill %d..%d · layer %d/%d · TPU calls %d' %
                    (self.position + 1, self.position + len(chunk), layer + 1, self.layers, self.backend.calls))
                prefix = 'blk.%d.' % layer
                normalized = self.norm(x, self.weights[prefix + 'attn_norm.weight'], projection=True)
                self.coverage[prefix + 'attn_norm'] += len(chunk)
                if prefix + 'attn_qkv.weight' in self.weights.manifest['tensors']:
                    residual = self.delta_many(layer, normalized)
                else:
                    residual = self.attention_many(layer, normalized, self.position)
                x = self.op('add', x, residual)
                normalized = self.norm(x, self.weights[prefix + 'post_attention_norm.weight'], projection=True)
                down = self.linear(prefix + 'ffn_down.weight', self.ffn_input(prefix, normalized))
                x = self.op('add', x, down)
                self.coverage[prefix + 'ffn'] += len(chunk)
                if self.telemetry is not None:
                    self.telemetry(dict(kind='layer', position=self.position, batch_tokens=len(chunk), layer=layer + 1, total_layers=self.layers))
            self.position += len(chunk)
            self.last_hidden = x[-1].copy()
            self.hidden_history.extend(row.copy() for row in x)
        head_hidden=np.stack(self.hidden_history[-len(tokens):]) if all_logits else self.last_hidden
        normalized = self.norm(head_hidden, self.weights['output_norm.weight'], projection=True)
        self.coverage['output_norm'] += len(tokens) if all_logits else 1
        self.coverage['lm_head'] += len(tokens) if all_logits else 1
        if self.telemetry is not None:
            self.telemetry(dict(kind='head', position=self.position, batch_tokens=1, layer=self.layers, total_layers=self.layers))
        return self.linear('token_embd.weight', normalized)

    def verify_pair(self, anchor, draft):
        """Verify two inputs and retain the exact one-input prefix for rejection."""
        if not hasattr(self.backend,'recurrent_many'):
            raise ValueError('Pair verification requires temporal state kernels')
        if self.verifying_pair: raise RuntimeError('A pair is already running')
        self.verified_first_caches={layer:cache for layer,cache in self.caches.items() if layer>=self.layers}
        self.verifying_pair=True; self.backend.capture_first_state=True
        try:
            return self.prefill([anchor,draft],all_logits=True)
        finally:
            self.verifying_pair=False; self.backend.capture_first_state=False

    def finish_pair(self, accepted):
        if self.verified_first_caches is None:
            raise RuntimeError('No verified pair to finish')
        if not accepted:
            self.caches=self.verified_first_caches
            self.position-=1
            self.hidden_history.pop()
            self.last_hidden=self.hidden_history[-1].copy()
        self.verified_first_caches=None

    def verify_block(self, anchor, drafts):
        """Full-trunk verification with an exact cache for every commit prefix."""
        tokens=[int(anchor),*[int(token) for token in drafts]]
        if not hasattr(self.backend,'recurrent_many'):
            raise ValueError('Block verification requires temporal state kernels')
        if not 2 <= len(tokens) <= 4 or self.position+len(tokens)>self.context:
            raise ValueError('Verification block must contain 2..4 available positions')
        if self.verifying_pair or self.verification_prefixes or self.verified_prefix_caches is not None:
            raise RuntimeError('A verification is already pending')
        self.verification_prefixes=len(tokens)-1
        self.verified_prefix_caches=[{layer:cache for layer,cache in self.caches.items() if layer>=self.layers}
                                     for _ in range(self.verification_prefixes)]
        self.backend.capture_state_prefixes=self.verification_prefixes
        before={layer:tuple(part.copy() if isinstance(part,list) else part for part in cache)
                for layer,cache in self.caches.items()}
        position,history_length,last_hidden=self.position,len(self.hidden_history),self.last_hidden
        self._verification_start=(before,position,history_length,last_hidden)
        try:
            return self.prefill(tokens,all_logits=True)
        except BaseException:
            self.caches=before;self.position=position;self.last_hidden=last_hidden
            del self.hidden_history[history_length:]
            self.verified_prefix_caches=None
            self._verification_start=None
            raise
        finally:
            self.verification_prefixes=0
            self.backend.capture_state_prefixes=0

    def finish_block(self, committed):
        if self.verified_prefix_caches is None:
            raise RuntimeError('No verified block to finish')
        total=len(self.verified_prefix_caches)+1
        if not 0 <= committed <= total:
            raise ValueError('Invalid committed verification prefix')
        if committed==0:
            before,position,history_length,last_hidden=self._verification_start
            self.caches=before;self.position=position;self.last_hidden=last_hidden
            del self.hidden_history[history_length:]
        elif committed < total:
            self.caches=self.verified_prefix_caches[committed-1]
            self.position-=total-committed
            del self.hidden_history[self.position:]
            self.last_hidden=self.hidden_history[-1].copy()
        self.verified_prefix_caches=None
        self._verification_start=None

    def mtp_draft(self, next_tokens, *, candidate_ids=None):
        """Execute the optional NextN module on TPU; never accept a draft unchecked.

        next_tokens contains all trunk input IDs followed by the trunk's selected
        next ID. Every new shifted pair is processed, preserving the MTP KV cache.
        The caller may verify a draft through verify_pair. candidate_ids only
        limits the proposal head; the verifier always evaluates the full head.
        """
        if len(next_tokens) != self.position + 1 or len(self.hidden_history) != self.position:
            raise ValueError('MTP requires aligned trunk hidden states and shifted token IDs')
        if self.position >= self.context:
            raise ValueError('No context space remains for the MTP draft position')
        layer = self.layers
        prefix = 'blk.%d.' % layer
        result = None
        for start in range(self.mtp_ready, self.position, 16):
            stop = min(start + 16, self.position)
            embeddings = np.stack([self.op('copy', self.weights['token_embd.weight'][int(t)])
                                   for t in next_tokens[start + 1:stop + 1]])
            hidden = np.stack(self.hidden_history[start:stop])
            hidden = self.norm(hidden, self.weights['output_norm.weight'])
            h = self.norm(hidden, self.weights[prefix + 'nextn.hnorm.weight'])
            e = self.norm(embeddings, self.weights[prefix + 'nextn.enorm.weight'])
            self.coverage[prefix + 'nextn.hnorm'] += stop - start
            self.coverage[prefix + 'nextn.enorm'] += stop - start
            x = self.linear(prefix + 'nextn.eh_proj.weight', np.concatenate([e, h], axis=-1))
            normalized = self.norm(x, self.weights[prefix + 'attn_norm.weight'], projection=True)
            residual = self.attention_many(layer, normalized, start + 1)
            x = self.op('add', x, residual)
            normalized = self.norm(x, self.weights[prefix + 'post_attention_norm.weight'], projection=True)
            x = self.op('add', x, self.linear(prefix + 'ffn_down.weight', self.ffn_input(prefix, normalized)))
            self.coverage[prefix + 'ffn'] += stop - start
            self.mtp_last_hidden=x[-1].copy()
            result = self.norm(x[-1], self.weights[prefix + 'nextn.shared_head_norm.weight'], projection=True)
            self.coverage[prefix + 'nextn.shared_head_norm'] += 1
        self.mtp_ready = self.position
        if result is None:
            raise ValueError('MTP draft for this trunk position was already computed')
        self.coverage['mtp_lm_head'] += 1
        if candidate_ids is not None:
            if not hasattr(self.backend,'project_indexed'):
                raise ValueError('Backend does not support proposal head shortlists')
            self.coverage['mtp_shortlist_rows']+=len(candidate_ids)
            return self.backend.project_indexed('token_embd.weight',candidate_ids,result)
        return self.linear('token_embd.weight', result)

    def mtp_draft_chain(self, next_tokens, count, *, candidate_ids=None, stop_ids=(),allow_estimates=True):
        """Use exact first MTP input; later drafts may use estimated hidden states.

        Estimated MTP states are discarded before full-trunk verification.
        They affect proposals only, never committed numerical model results.
        """
        if not 1 <= count <= 3:
            raise ValueError('A draft chain contains 1..3 proposals')
        tokens=list(next_tokens)
        reference=tokens.copy()
        self.last_ngram_drafts=0
        scores=self.mtp_draft(tokens,candidate_ids=candidate_ids)
        def select(value):
            chosen=self.greedy(value)
            return candidate_ids[chosen] if candidate_ids is not None else chosen
        drafts=[select(scores)]
        position,history_length,ready=self.position,len(self.hidden_history),self.mtp_ready
        saved_hidden=self.mtp_last_hidden
        saved_cache={layer:tuple(part.copy() if isinstance(part,list) else part for part in cache)
                     for layer,cache in self.caches.items() if layer>=self.layers}
        try:
            while len(drafts)<count and drafts[-1] not in stop_ids:
                from .speculative import ngram_tail
                tail=ngram_tail(reference,tokens+[drafts[-1]],count-len(drafts),
                                minimum_width=2 if allow_estimates else 3,excluded=stop_ids)
                if tail is not None:
                    drafts.extend(tail);self.last_ngram_drafts+=len(tail)
                    break
                if not allow_estimates:
                    break
                self.hidden_history.append(self.mtp_last_hidden.copy())
                self.position+=1
                tokens.append(drafts[-1])
                scores=self.mtp_draft(tokens,candidate_ids=candidate_ids)
                drafts.append(select(scores))
        finally:
            self.position=position
            del self.hidden_history[history_length:]
            self.mtp_ready=ready
            self.mtp_last_hidden=saved_hidden
            for layer in list(self.caches):
                if layer>=self.layers:
                    del self.caches[layer]
            self.caches.update(saved_cache)
        return drafts

    def step(self, token, *, logits=True):
        if not 0 <= self.position < self.context:
            raise ValueError('Experimental context limit reached; start a new conversation')
        if not 0 <= int(token) < len(self.weights['token_embd.weight']):
            raise ValueError('Token ID out of vocabulary')
        start = time.perf_counter()
        # Host selects a row for DMA; the embedding copy itself executes on TPU.
        x = self.op('copy', self.weights['token_embd.weight'][int(token)])
        self.coverage['token_embedding'] += 1
        for layer in range(self.layers):
            self._check_cancel()
            self.progress('token %d · layer %d/%d · TPU calls %d' %
                (self.position + 1, layer + 1, self.layers, self.backend.calls))
            x = self.block(layer, x, self.position)
            if self.telemetry is not None:
                self.telemetry(dict(kind='layer', position=self.position, batch_tokens=1, layer=layer + 1, total_layers=self.layers))
        self.last_hidden = x.copy()
        self.hidden_history.append(x.copy())
        self.position += 1
        if not logits:
            return None
        x = self.norm(x, self.weights['output_norm.weight'], projection=True)
        self.coverage['output_norm'] += 1
        if self.telemetry is not None:
            self.telemetry(dict(kind='head', position=self.position, batch_tokens=1, layer=self.layers, total_layers=self.layers))
        scores = self.linear('token_embd.weight', x)
        self.coverage['lm_head'] += 1
        self.last_step_seconds = time.perf_counter() - start
        return scores

    def greedy(self, scores):
        self.coverage['greedy_argmax'] += 1
        return self.backend.argmax(scores)

    def report(self):
        accelerated = hasattr(self.backend, 'matrix_stats')
        return dict(backend='Edge TPU INT8 MXU + FP32 ISA' if accelerated else 'Edge TPU custom FP32 ISA', cpu_model_fallback=False,
            position=self.position, tpu_calls=self.backend.calls, device_ms=self.backend.device_ms,
            coverage=dict(self.coverage), precision='INT8 tensor-array projections + FP32 scalar arithmetic' if accelerated else 'Decompressed GGUF weights + FP32 scalar arithmetic',
            scope='24 text decoder layers; optional MTP draft on TPU; selected GGUF contains no vision projector',
            mtp_positions=self.mtp_ready, speculative_acceptance=self.speculative_stats is not None,
            speculation=dict(self.speculative_stats) if self.speculative_stats is not None else None,
            matrix_stats=dict(self.backend.matrix_stats) if accelerated else None,
            matrix_profile=dict(components=self.backend.components, output_digits=self.backend.output_digits,
                                chunk_rows=self.backend.chunk_rows) if accelerated else None,
            optimization=dict(fused_fp32=hasattr(self.backend, 'normalize'),
                              direct_usb_readback=getattr(self.backend,'direct_usb_readback',False),
                              prepared_usb_dispatch=getattr(self.backend,'prepared_usb_dispatch',False),
                              wide_projection_quantization=getattr(self.backend,'wide_projection_quantization',False),
                              reused_projection_quantization=getattr(self.backend,'reuse_projection_quantization',False),
                              compute_fusions=getattr(self.backend,'compute_fusions',False),
                              fused_head_selection=getattr(self.backend,'fused_head_selection',False),
                              active_native_bias=getattr(self.backend,'native_active_bias',False),
                              native_batch=hasattr(self.backend,'convolution_many'),
                              compact_scalar_loads=getattr(self.backend,'compact_loads_enabled',False),
                              turbo_v2=hasattr(self.backend, 'recurrent_many'),
                              clock=getattr(getattr(self.backend,'device',None),'clock',None),
                              async_usb=getattr(self.backend, 'async_usb', False),
                              host_parameter_cache_limit_bytes=getattr(self.backend, 'parameter_cache_limit', 0)))
