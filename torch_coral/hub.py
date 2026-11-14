"""Hugging Face model inspection, real conversion and explicitly selected inference."""
import json
import os
from pathlib import Path
import re
import time
import numpy as np
import torch
from .compiler import compile
from .runtime import ROOT, load

TASKS=['image-classification','object-detection','image-segmentation','feature-extraction',
       'text-classification','token-classification','question-answering','text-generation',
       'text2text-generation','summarization','translation','audio-classification',
       'automatic-speech-recognition','zero-shot-image-classification','zero-shot-classification']

def hub_setup():
    import truststore
    truststore.inject_into_ssl()
    os.environ.setdefault('HF_HUB_DISABLE_XET','1')

def source_name(value):
    value=value.strip()
    if not value: raise ValueError('请填写 Hugging Face 模型 ID 或本地模型目录')
    if Path(value).is_dir(): return str(Path(value).resolve())
    if value.startswith('https://huggingface.co/'):
        value=value.removeprefix('https://huggingface.co/').strip('/').split('/tree/')[0].split('/resolve/')[0]
    if not re.fullmatch(r'[\w.-]+(?:/[\w.-]+)?',value):
        raise ValueError('模型来源应为 owner/model、Hugging Face 模型 URL 或本地目录')
    return value

def inspect_model(source,revision='main',task='auto'):
    hub_setup()
    from transformers import AutoConfig
    from huggingface_hub import model_info
    source=source_name(source)
    local=Path(source).is_dir()
    info=None if local else model_info(source,revision=revision,timeout=30)
    pinned=revision if local else info.sha
    config=AutoConfig.from_pretrained(source,revision=pinned,trust_remote_code=False)
    architectures=list(getattr(config,'architectures',None) or [])
    selected=task if task!='auto' else (getattr(info,'pipeline_tag',None) or infer_task(architectures))
    issues=[]
    if config.to_dict().get('auto_map'): issues.append('仓库声明自定义模型代码；本工具不执行远程 Python 代码')
    if selected in ('text-generation','text2text-generation','summarization','translation','automatic-speech-recognition'):
        issues.append('自回归生成/动态解码不能整体部署到旧 Edge TPU；可选择 CPU/CUDA 推理')
    if selected in ('text-classification','token-classification','question-answering','feature-extraction','zero-shot-classification'):
        issues.append('若输入含整数 token ID 或 attention mask，当前 Coral 浮点边界转换器会明确拒绝；可选 CPU/CUDA 或使用浮点 NPZ 输入适配')
    return {'source':source,'revision':pinned,'requested_revision':revision,'task':selected,
            'model_type':config.model_type,'architectures':architectures,'issues':issues,
            'coral_status':'需要实际导出/量化/编译验证；此检查不保证 TPU 兼容',
            'config':config.to_dict(),'local':local}

def infer_task(architectures):
    text=' '.join(architectures)
    for key,task in [('ForImageClassification','image-classification'),('ForObjectDetection','object-detection'),
                     ('ForSemanticSegmentation','image-segmentation'),('ForSequenceClassification','text-classification'),
                     ('ForTokenClassification','token-classification'),('ForQuestionAnswering','question-answering'),
                     ('ForCausalLM','text-generation'),('ForConditionalGeneration','text2text-generation')]:
        if key in text: return task
    return 'feature-extraction'

def _model_class(task):
    import transformers as t
    mapping={'image-classification':t.AutoModelForImageClassification,'object-detection':t.AutoModelForObjectDetection,
             'image-segmentation':t.AutoModelForSemanticSegmentation,'text-classification':t.AutoModelForSequenceClassification,
             'token-classification':t.AutoModelForTokenClassification,'question-answering':t.AutoModelForQuestionAnswering,
             'text-generation':t.AutoModelForCausalLM,'text2text-generation':t.AutoModelForSeq2SeqLM,
             'summarization':t.AutoModelForSeq2SeqLM,'translation':t.AutoModelForSeq2SeqLM,
             'audio-classification':t.AutoModelForAudioClassification,
             'automatic-speech-recognition':t.AutoModelForSpeechSeq2Seq}
    return mapping.get(task,t.AutoModel)

def image_files(folder,maximum):
    path=Path(folder)
    if not path.is_dir(): raise ValueError('校准图片目录不存在')
    files=sorted(p for p in path.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp'))[:maximum]
    if not files: raise ValueError('目录中没有可用校准图片；不会自动替换成随机数据')
    return files

class HubWrapper(torch.nn.Module):
    def __init__(self,model,names):
        super().__init__();self.model=model;self.names=names
    def forward(self,*values):
        out=self.model(**dict(zip(self.names,values)),return_dict=True)
        if hasattr(out,'logits'): return out.logits
        result={}
        for key,value in out.items():
            if isinstance(value,torch.Tensor) and value.is_floating_point(): result[key]=value.float()
        if not result: raise ValueError('模型没有可导出的浮点 Tensor 输出')
        return result

def convert_hub(request,progress=lambda message:None):
    spec=inspect_model(request['source'],request.get('revision','main'),request.get('task','auto'))
    source=spec['source'];revision=spec['revision'];task=spec['task']
    maximum=int(request.get('max_calibration',64))
    if not 1<=maximum<=256: raise ValueError('校准数量须在 1..256 之间')
    folder=Path(request['artifact_dir']).resolve()
    if (folder/'manifest.json').exists(): raise FileExistsError('产物已存在；请加载它或选择新目录')
    if task in ('text-generation','text2text-generation','summarization','translation','automatic-speech-recognition'):
        raise ValueError('旧 Edge TPU 无法整体执行动态生成/解码。请在推理页明确选择 CPU 或 CUDA；不会把 CPU 推理标作 TPU。')
    progress('下载并加载权重（固定 revision：'+revision+'）')
    model=_model_class(task).from_pretrained(source,revision=revision,trust_remote_code=False).float().cpu().eval()
    mode=request.get('calibration_mode','images')
    processor=None; samples=[]; names=[]
    if mode=='images':
        from PIL import Image
        from transformers import AutoImageProcessor
        processor=AutoImageProcessor.from_pretrained(source,revision=revision,trust_remote_code=False)
        for path in image_files(request['calibration_path'],maximum):
            with Image.open(path) as image: batch=processor(images=image.convert('RGB'),return_tensors='pt')
            current=[k for k,v in batch.items() if isinstance(v,torch.Tensor)]
            if not names: names=current
            if names!=current: raise ValueError('预处理器输入键发生变化')
            samples.append(tuple(batch[k] for k in names))
    elif mode=='npz':
        with np.load(request['calibration_path'],allow_pickle=False) as data:
            names=sorted(data.files)
            if not names: raise ValueError('NPZ 没有输入数组')
            arrays=[data[n] for n in names]
            count=min(len(a) for a in arrays)
            if any(len(a)!=count for a in arrays): raise ValueError('NPZ 各输入校准组数不同')
            for i in range(min(count,maximum)): samples.append(tuple(torch.from_numpy(a[i].copy()) for a in arrays))
    else: raise ValueError('选择真实校准图片目录或 NPZ（每个键为模型输入名，第一维为校准组）')
    if not samples: raise ValueError('校准数据为空')
    for key,value in zip(names,samples[0]):
        if value.dtype!=torch.float32:
            raise ValueError(f'输入 {key} 的类型为 {value.dtype}；Coral 转换接口要求 float32。整数 token 不会被伪装成 INT8 图片输入，请选 CPU/CUDA。')
    wrapper=HubWrapper(model,names).eval()
    progress('导出 ONNX → 数值校验 → INT8 校准 → Edge TPU 编译；详细日志在产物目录')
    compiled=compile(wrapper,samples[0],samples,artifact_dir=folder,max_calibration_samples=maximum,
                     allow_partial=bool(request.get('allow_partial',False)),
                     operator_strategy=request.get('operator_strategy'),approximations=request.get('approximations'))
    manifest=compiled.manifest;compiled.close()
    if processor is not None: processor.save_pretrained(folder/'processor')
    model.config.save_pretrained(folder/'config')
    meta={'source':source,'revision':revision,'task':task,'input_names':names,
          'id2label':getattr(model.config,'id2label',{}),'preprocessing':'images' if processor else 'npz',
          'calibration_count':len(samples)}
    (folder/'huggingface.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    return {'status':'PASS','artifact_dir':str(folder),'huggingface':{k:v for k,v in meta.items() if k!='id2label'},
            'label_count':len(meta['id2label']),'compiler':manifest['compiler_report'],
            'operator_rewrites':manifest.get('operator_rewrites'),'numerical_validation':manifest.get('quantization_validation')}

def _jsonable(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {str(k):_jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [_jsonable(v) for v in value]
    try:
        from PIL import Image
        if isinstance(value,Image.Image): return {'image_size':list(value.size),'mode':value.mode}
    except ImportError: pass
    return value

def infer_hub(request,progress=lambda message:None):
    hub_setup()
    backend=request.get('backend','Coral')
    mode=request.get('input_mode','image');value=request['input']
    if backend=='Coral':
        folder=Path(request['artifact_dir']).resolve()
        meta=json.loads((folder/'huggingface.json').read_text(encoding='utf-8')) if (folder/'huggingface.json').exists() else None
        if mode=='image':
            if meta is None: raise ValueError('该产物没有 HF 预处理器；请选择 NPZ 输入')
            from PIL import Image
            from transformers import AutoImageProcessor
            processor=AutoImageProcessor.from_pretrained(folder/'processor',local_files_only=True)
            with Image.open(value) as image: batch=processor(images=image.convert('RGB'),return_tensors='pt')
            inputs=tuple(batch[n] for n in meta['input_names'])
        elif mode=='npz':
            with np.load(value,allow_pickle=False) as data:
                keys=meta['input_names'] if meta else ['input_%d'%i for i in range(len(json.loads((folder/'manifest.json').read_text())['inputs']))]
                inputs=tuple(torch.from_numpy(data[k].copy()) for k in keys)
        else: raise ValueError('Coral 产物使用图片或预处理后的 NPZ；文本/音频请选 CPU/CUDA 或自行提供浮点 NPZ')
        progress('连接真实 Coral，执行量化推理')
        with torch.no_grad(),load(folder, transport=request.get('transport','pipe'),
                delegate_options=request.get('delegate_options'), num_threads=request.get('num_threads',1)) as model:
            start=time.perf_counter();output=model(*inputs);elapsed=(time.perf_counter()-start)*1000
            report={'backend':'Coral Edge TPU','hardware':model.hardware,'timing':model.last_timing,
                    'first_call_total_ms':elapsed,'output':_jsonable(output),'artifact_dir':str(folder)}
        if isinstance(output,torch.Tensor) and output.ndim==2 and meta and meta['task']=='image-classification':
            scores=output.softmax(-1)[0];values,indices=scores.topk(min(5,len(scores)))
            report['predictions']=[{'label':meta['id2label'].get(str(int(i)),str(int(i))),'score':float(v)} for i,v in zip(indices,values)]
        return report
    if backend not in ('CPU','CUDA'): raise ValueError('选择 Coral、CPU 或 CUDA')
    from transformers import pipeline
    spec=inspect_model(request['source'],request.get('revision','main'),request.get('task','auto'))
    if backend=='CUDA' and not torch.cuda.is_available(): raise RuntimeError('CUDA 不可用')
    progress('加载 Hugging Face pipeline，后端：'+backend)
    pipe=pipeline(task=spec['task'],model=spec['source'],revision=spec['revision'],
                  trust_remote_code=False,device=0 if backend=='CUDA' else -1)
    if mode=='json': payload=json.loads(value)
    elif mode=='npz': raise ValueError('通用 HF pipeline 接收图片、文本、音频或 JSON；NPZ 用于 Coral Tensor 推理')
    else: payload=value
    options={}
    if spec['task'] in ('text-generation','text2text-generation','summarization','translation'):
        options['max_new_tokens']=min(max(int(request.get('max_new_tokens',64)),1),512)
    if backend=='CUDA': torch.cuda.synchronize()
    start=time.perf_counter();output=pipe(payload,**options)
    if backend=='CUDA': torch.cuda.synchronize()
    return {'backend':backend,'source':spec['source'],'revision':spec['revision'],'task':spec['task'],
            'inference_ms':(time.perf_counter()-start)*1000,'output':_jsonable(output),
            'note':'所选 CPU/CUDA 推理；未声称在 Coral 上执行'}
