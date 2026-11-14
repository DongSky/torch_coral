"""Persistent isolated full-TPU chat process. stdout is JSON events only."""
import contextlib
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback


def main():
    # Load numerical extensions before starting a blocking input thread. Some
    # Windows CRT extension initialization paths acquire standard-stream locks.
    from .isa_scalar import ScalarTPU
    from .gguf_weights import prepare_weights
    from .qwen_tpu import QwenTPU
    from .isa_mxu import MatrixTPU, prepare_matrix_weights
    from .isa_optimized import OptimizedMatrixTPU
    from .isa_streamed import StreamedMatrixTPU
    from .isa_batched import BatchedMatrixTPU
    from .isa_io import IOOptimizedMatrixTPU
    from .chat_metrics import InferenceMetrics
    output = sys.stdout
    write_lock = threading.Lock()
    def emit(kind, **values):
        with write_lock:
            output.write(json.dumps(dict(event=kind, **values), ensure_ascii=True) + '\n')
            output.flush()
    cancel = threading.Event()
    shutdown = threading.Event()
    commands = queue.Queue()
    def read_commands():
        buffer = b''
        while True:
            chunk = os.read(sys.stdin.fileno(), 65536)
            if not chunk: break
            buffer += chunk
            if len(buffer) > 1024 * 1024:
                emit('error', message='Command exceeds 1 MiB')
                break
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                if not line.strip(): continue
                try:
                    command = json.loads(line)
                    if command.get('action') == 'shutdown': shutdown.set()
                    if command.get('action') in ('cancel', 'shutdown'):
                        cancel.set()
                    else:
                        cancel.clear()
                    if command.get('action') != 'cancel':
                        commands.put(command)
                except Exception as exc:
                    emit('error', message=str(exc))
        cancel.set()
        shutdown.set()
        commands.put(dict(action='shutdown'))
    threading.Thread(target=read_commands, daemon=True).start()
    engine = device = tokenizer = template = None
    processed = []
    metrics = None
    prefill_origin = prefill_count = 0
    request_calls_start = 0
    layer_state = dict(progress=0., layer=0, total_layers=24, detail='')
    def snapshot(**details):
        frame = metrics.snapshot()
        frame.update(layer_state)
        frame.update(details)
        frame['tpu_calls'] = device.calls - request_calls_start if device is not None else 0
        return frame
    def publish(**details):
        if metrics is not None: emit('telemetry', **snapshot(**details))
    def on_model(event):
        if metrics is None or metrics.phase not in ('prefill', 'decode'): return
        layer, layers = event['layer'], event['total_layers']
        if event['kind'] == 'head':
            fraction, detail = .99, '输出投影 · LM head'
        elif metrics.phase == 'prefill':
            work = (event['position'] - prefill_origin) * layers + layer * event['batch_tokens']
            fraction = min(.99, work / max(1, prefill_count * layers + 1))
            detail = '第 %d/%d 层 · %d token' % (layer, layers, event['batch_tokens'])
        else:
            fraction, detail = layer / (layers + 1), '第 %d/%d 层' % (layer, layers)
        layer_state.update(progress=fraction, layer=layer, total_layers=layers, detail=detail)
        publish()
    def begin_phase(name, detail=''):
        metrics.begin(name)
        layer_state.update(progress=0., layer=0, detail=detail)
        publish()
    emit('ready')
    with contextlib.redirect_stdout(sys.stderr):
        try:
            while True:
                command = commands.get()
                action = command.get('action')
                if action == 'shutdown' or shutdown.is_set():
                    break
                try:
                    if action == 'load':
                        metrics = None
                        if device is not None:
                            device.close()
                            device = engine = None
                        from tokenizers import Tokenizer
                        from jinja2.sandbox import SandboxedEnvironment
                        tokenizer_dir = Path(command['tokenizer_dir']).resolve()
                        tokenizer = Tokenizer.from_file(str(tokenizer_dir / 'tokenizer.json'))
                        environment = SandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
                        def fail(message): raise ValueError(message)
                        environment.globals['raise_exception'] = fail
                        template = environment.from_string((tokenizer_dir / 'chat_template.jinja').read_text(encoding='utf-8'))
                        def progress(message):
                            if cancel.is_set(): raise InterruptedError('Cancelled')
                            emit('progress', message=message)
                        weights = prepare_weights(command['gguf'], command['prepared_dir'], progress=progress)
                        if cancel.is_set(): raise InterruptedError('Cancelled')
                        profile = command.get('matrix_profile', 'precise')
                        if profile == 'fp32':
                            device = ScalarTPU()
                        else:
                            if profile not in ('fast', 'turbo', 'turbo-v2', 'turbo-v3', 'turbo-v4', 'balanced', 'precise'):
                                raise ValueError('Unknown matrix precision profile')
                            matrix_dir = Path(command.get('matrix_dir') or Path(command['prepared_dir']).resolve().parent / 'int8-matrices')
                            prepare_matrix_weights(weights, matrix_dir, progress=progress)
                            if cancel.is_set(): raise InterruptedError('Cancelled')
                            emit('progress', message='连接 Edge TPU INT8 矩阵阵列与 FP32 指令后端')
                            components, digits = {'fast':(1,2), 'turbo':(1,2), 'turbo-v2':(1,2),
                                                  'turbo-v3':(1,2),'turbo-v4':(1,2),'balanced':(2,2), 'precise':(2,3)}[profile]
                            backend = {'turbo':OptimizedMatrixTPU, 'turbo-v2':StreamedMatrixTPU,
                                       'turbo-v3':BatchedMatrixTPU,'turbo-v4':IOOptimizedMatrixTPU}.get(profile, MatrixTPU)
                            device = backend(matrix_dir, components=components, output_digits=digits,
                                             clock=command.get('v2_clock','max') if profile in ('turbo-v2','turbo-v3','turbo-v4') else 'high')
                        device.cancel_event = cancel
                        engine = QwenTPU(weights, device, context=int(command.get('context', 256)),
                            progress=lambda s: emit('progress', message=s), cancel=cancel, telemetry=on_model)
                        if cancel.is_set(): raise InterruptedError('Cancelled')
                        processed = []
                        emit('loaded', source_sha256=weights.manifest['source_sha256'],
                             backend=engine.report()['backend'], matrix_profile=profile, cpu_model_fallback=False,
                             clock_mhz=500 if device.device.clock=='max' else 250)
                    elif action == 'reset':
                        metrics = None
                        if engine is not None: engine.reset()
                        processed = []
                        emit('reset')
                    elif action == 'chat':
                        request_started = time.perf_counter()
                        metrics = None
                        if engine is None:
                            raise RuntimeError('请先加载模型与 Edge TPU')
                        messages = command['messages']
                        if not isinstance(messages, list) or not messages:
                            raise ValueError('A conversation is required')
                        for m in messages:
                            if m.get('role') not in ('system', 'user', 'assistant') or not isinstance(m.get('content'), str):
                                raise ValueError('Only text conversation messages are supported')
                        prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=False,
                                                 tools=None, add_vision_id=False)
                        tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
                        limit = int(command.get('max_new_tokens', 32))
                        if not 1 <= limit <= 512 or len(tokens) + limit > engine.context:
                            raise ValueError('输入与生成长度超出上下文上限；请缩短输入或开启新对话')
                        if tokens[:len(processed)] != processed or len(tokens) <= len(processed):
                            engine.reset()
                            processed = []
                        started = time.perf_counter()
                        before = device.calls
                        request_calls_start = before
                        prefill_origin, prefill_count = engine.position, len(tokens) - len(processed)
                        metrics = InferenceMetrics(prefill_count, len(processed), limit, started_at=request_started)
                        layer_state.update(progress=0., layer=0, detail='准备输入')
                        publish()
                        emit('progress', message='输入 %d token，复用 %d token。正在由 Edge TPU 计算。' %
                             (len(tokens), len(processed)))
                        scores = engine.prefill(tokens[len(processed):])
                        begin_phase('sampling', '选择首个 token')
                        processed = tokens.copy()
                        generated = []
                        stop_ids = {v for v in (tokenizer.token_to_id('<|im_end|>'), tokenizer.token_to_id('<|endoftext|>'))
                                    if v is not None}
                        def visible(tokens):
                            generated.extend(tokens)
                            metrics.token(count=len(tokens))
                            frame = snapshot(progress=1.)
                            emit('text', text=tokenizer.decode(generated, skip_special_tokens=True), metrics=frame)
                            publish(progress=1., detail='token %d 已生成' % len(generated), sample=True)
                        if profile in ('turbo-v3','turbo-v4') and command.get('speculative',True):
                            from .speculative import SpeculativeGreedy
                            decoder=SpeculativeGreedy(engine,processed,adaptive=True,
                                max_block_tokens=int(command.get('speculative_block_tokens',2)) if profile=='turbo-v4' else 2)
                            first=engine.greedy(scores)
                            if first not in stop_ids:
                                visible([first])
                                while len(generated)<limit:
                                    begin_phase('decode','MTP 草稿与完整 TPU 校验')
                                    emitted,_,stop=decoder.advance(generated[-1],limit-len(generated),stop_ids)
                                    processed=decoder.processed.copy()
                                    if emitted: visible(emitted)
                                    publish(speculation=dict(decoder.stats))
                                    if stop: break
                        else:
                            for i in range(limit):
                                token = engine.greedy(scores)
                                if token in stop_ids: break
                                visible([token])
                                if command.get('mtp', True):
                                    begin_phase('mtp', 'MTP 草稿验证')
                                    emit('progress', message='在 TPU 执行 MTP 草稿头（参考验证，不做推测加速）')
                                    engine.mtp_draft(processed + [token])
                                if i + 1 < limit:
                                    begin_phase('decode', '计算下一个 token')
                                    scores = engine.step(token)
                                    processed.append(token)
                        elapsed = time.perf_counter() - started
                        begin_phase('complete', '生成完成')
                        publish(progress=1.)
                        emit('done', text=tokenizer.decode(generated, skip_special_tokens=True),
                             new_tokens=len(generated), elapsed_seconds=elapsed,
                             tpu_calls=device.calls - before, report=engine.report(),
                             input_tokens=tokens, generated_tokens=generated, metrics=snapshot(progress=1.))
                    else:
                        raise ValueError('Unknown command: ' + str(action))
                except InterruptedError:
                    if action == 'chat' and metrics is not None:
                        begin_phase('cancelled', '已停止')
                    if action == 'load' and device is not None:
                        device.close()
                        device = engine = None
                    if engine is not None: engine.reset()
                    processed = []
                    emit('cancelled')
                except Exception as exc:
                    if action == 'chat' and metrics is not None:
                        begin_phase('error', '计算中断')
                    if action == 'load' and device is not None:
                        device.close()
                        device = engine = None
                    if engine is not None: engine.reset()
                    processed = []
                    traceback.print_exc(file=sys.stderr)
                    emit('error', message=str(exc), requires_reload=device is None or device.closed)
        finally:
            if device is not None:
                device.close()
            emit('closed')


if __name__ == '__main__': main()
