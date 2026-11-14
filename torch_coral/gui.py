"""Coral Workbench: native Qt interface; all expensive work runs out of process."""
import json
from pathlib import Path
import sys
import uuid
import os
import subprocess
import codecs
from PySide6.QtCore import QProcess,QProcessEnvironment,QTimer,Qt,QUrl
from PySide6.QtGui import QDesktopServices,QFont
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,
    QLabel,QLineEdit,QPushButton,QComboBox,QSpinBox,QCheckBox,QPlainTextEdit,QTabWidget,
    QFormLayout,QGroupBox,QFileDialog,QTableWidget,QTableWidgetItem,QHeaderView,QProgressBar,QSplitter)
from .runtime import ROOT
from .hub import TASKS

STYLE='''
QWidget { background:#101722; color:#dce7f3; font-family:"Microsoft YaHei UI"; font-size:13px; }
QMainWindow { background:#101722; }
QLabel#Title { font-size:26px; font-weight:700; color:#f4f8ff; }
QLabel#Muted { color:#91a4bb; }
QLabel#Metric { font-size:29px; font-weight:700; color:#6ce4bf; }
QLabel#Status { color:#6ce4bf; font-weight:600; }
QLabel { background:transparent; }
QGroupBox { border:1px solid #2a394c; border-radius:10px; margin-top:15px; padding:17px 14px 12px; background:#151f2e; }
QGroupBox::title { subcontrol-origin:margin; left:14px; padding:0 7px; color:#c4d5e8; }
QLineEdit,QComboBox,QSpinBox,QPlainTextEdit { background:#0d1521; border:1px solid #31435a; border-radius:6px; padding:8px; selection-background-color:#247b74; }
QLineEdit:focus,QPlainTextEdit:focus { border-color:#59d9b8; }
QPushButton { background:#25354b; border:1px solid #3b506c; padding:9px 15px; border-radius:6px; }
QPushButton:hover { background:#334a65; }
QPushButton#Primary { background:#48ceaa; color:#09251f; font-weight:700; border:0; }
QPushButton:disabled { color:#62748a; background:#1b2737; border-color:#243448; }
QTabWidget::pane { border:1px solid #29394f; border-radius:8px; }
QTabBar::tab { background:#192536; padding:12px 23px; margin-right:3px; border-top-left-radius:6px; border-top-right-radius:6px; color:#9eb1c8; }
QTabBar::tab:selected { background:#283c51; color:#78e7c5; }
QProgressBar { background:#1b2b3f; border:0; border-radius:5px; height:7px; text-align:center; }
QProgressBar::chunk { background:#4bd4b0; border-radius:5px; }
QTableWidget { background:#111c2b; gridline-color:#28384d; border:0; }
QHeaderView::section { background:#26384d; color:#c3d6eb; padding:9px; border:0; }
QTableWidget::item { padding:7px; }
QCheckBox { background:transparent; }
QCheckBox::indicator { width:14px; height:14px; border:1px solid #54718b; border-radius:3px; background:#0d1521; }
QCheckBox::indicator:checked { background:#48ceaa; }
QToolTip { color:#111c2b; background:#dce7f3; }
'''

def label(text,name=None):
    item=QLabel(text);item.setWordWrap(True)
    if name:item.setObjectName(name)
    return item

def button(text,callback,primary=False):
    item=QPushButton(text);item.clicked.connect(callback)
    if primary:item.setObjectName('Primary')
    return item

def row(*widgets):
    item=QWidget();layout=QHBoxLayout(item);layout.setContentsMargins(0,0,0,0)
    for widget in widgets:layout.addWidget(widget)
    return item

class Workbench(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('Coral Workbench · Hugging Face / Edge TPU')
        self.resize(1260,900);self.setMinimumSize(1020,720)
        self.process=None;self.buffer='';self.active_request=None;self.last_result=None;self.current_kind=None
        self.actions=[];self.locked=[]
        self.job_dir=ROOT/'.gui/jobs';self.job_dir.mkdir(parents=True,exist_ok=True)
        central=QWidget();self.setCentralWidget(central);outer=QVBoxLayout(central);outer.setContentsMargins(26,20,26,18);outer.setSpacing(14)
        head=QHBoxLayout();heading=QVBoxLayout();heading.addWidget(label('Coral Workbench','Title'))
        heading.addWidget(label('Hugging Face 模型 · INT8 转换 · 真实设备推理','Muted'));head.addLayout(heading,1)
        head.addWidget(label('CORAL EDGE TPU\n运行库与设备由本机配置','Status'));outer.addLayout(head)
        self.tabs=QTabWidget();outer.addWidget(self.tabs,1)
        self._model_tab();self._infer_tab();self._bench_tab();self._jobs_tab()
        footer=QHBoxLayout();self.status=label('就绪 · 转换前请提供真实代表性校准数据','Status');footer.addWidget(self.status,1)
        self.cancel_button=button('取消任务',self.cancel_job);self.cancel_button.setEnabled(False);footer.addWidget(self.cancel_button)
        self.open_job=button('打开当前记录',self.open_current);footer.addWidget(self.open_job)
        outer.addLayout(footer);self.progress=QProgressBar();self.progress.setRange(0,100);self.progress.setValue(0);self.progress.setTextVisible(False);outer.addWidget(self.progress)
        self._restore();self.refresh_benchmark()

    def action(self,text,callback,primary=False):
        item=button(text,callback,primary);self.actions.append(item);return item

    def _model_tab(self):
        page=QWidget();layout=QHBoxLayout(page);layout.setContentsMargins(18,18,18,18);layout.setSpacing(18)
        left=QVBoxLayout();left.setSpacing(13);layout.addLayout(left,3)
        group=QGroupBox('01   模型来源');form=QFormLayout(group)
        self.source=QLineEdit('google/mobilenet_v2_1.0_224');self.source.setObjectName('modelSource');self.source.setPlaceholderText('owner/model、https://huggingface.co/… 或本地模型目录')
        form.addRow('模型 ID / 路径',row(self.source,button('本地目录',lambda:self.browse(self.source,True))))
        self.revision=QLineEdit('main');form.addRow('Revision',self.revision)
        self.task=QComboBox();self.task.setEditable(True);self.task.addItems(['auto',*TASKS]);form.addRow('任务',self.task)
        form.addRow(self.action('检查模型与兼容性',self.inspect,True));left.addWidget(group)
        group=QGroupBox('02   校准与转换');form=QFormLayout(group)
        self.cal_mode=QComboBox();self.cal_mode.addItem('真实图片目录','images');self.cal_mode.addItem('多输入 NPZ','npz');form.addRow('数据类型',self.cal_mode)
        self.cal_path=QLineEdit(str(ROOT/'examples/coral_data'));form.addRow('校准数据',row(self.cal_path,button('选择',self.browse_calibration)))
        self.max_cal=QSpinBox();self.max_cal.setRange(1,256);self.max_cal.setValue(64);form.addRow('最多校准组数',self.max_cal)
        self.artifact=QLineEdit(str(ROOT/'artifacts/hf-mobilenet-v2'));self.artifact.setObjectName('artifactDir')
        form.addRow('产物目录',row(self.artifact,button('选择',lambda:self.browse(self.artifact,True))))
        self.partial=QCheckBox('允许编译后部分操作留在 CPU（报告会标明）');form.addRow(self.partial)
        self.operator_strategy=QCheckBox('扩展算子转换（v16，检查 INT8 数值误差）');form.addRow(self.operator_strategy)
        self.approximations=QLineEdit('{}');self.approximations.setEnabled(False)
        self.approximations.setPlaceholderText('{"Sin":{"range":[-3.14,3.14],"segments":32}}')
        self.approximations.setToolTip('{} 只做精确图改写；显式指定函数和有效范围才启用近似。范围须覆盖实际运行数据，报告会记录误差。')
        self.operator_strategy.toggled.connect(self.approximations.setEnabled)
        form.addRow('有界近似 JSON',self.approximations)
        form.addRow(self.action('转换为 Edge TPU 模型',self.convert,True));left.addWidget(group)
        left.addWidget(label('任意模型均可提交检查；实际 TPU 支持由导出、量化与旧编译器共同决定。动态生成模型及整数 token 输入不能整体转换。CPU / CUDA 推理可在下一页明确选择。','Muted'))
        left.addStretch()
        right=QVBoxLayout();layout.addLayout(right,2)
        self.model_summary=label('先检查模型\n\n会显示任务、架构、固定版本和已知限制。转换成功前不会承诺 TPU 兼容。','Muted')
        right.addWidget(self.model_summary)
        self.model_info=QPlainTextEdit();self.model_info.setReadOnly(True);self.model_info.setPlaceholderText('兼容性报告与转换结果');right.addWidget(self.model_info,1)
        self.tabs.addTab(page,'模型与转换')
        self.locked.extend([self.source,self.revision,self.task,self.cal_mode,self.cal_path,self.max_cal,self.artifact,self.partial,self.operator_strategy,self.approximations])

    def _infer_tab(self):
        page=QWidget();layout=QHBoxLayout(page);layout.setContentsMargins(18,18,18,18);left=QVBoxLayout();layout.addLayout(left,2)
        group=QGroupBox('执行后端');form=QFormLayout(group)
        self.backend=QComboBox();self.backend.addItems(['Coral','CPU','CUDA']);form.addRow('后端',self.backend)
        self.infer_artifact=QLineEdit(str(ROOT/'artifacts/hf-mobilenet-v2'));form.addRow('Coral 产物',row(self.infer_artifact,button('选择',lambda:self.browse(self.infer_artifact,True))))
        self.coral_transport=QComboBox();self.coral_transport.addItem('共享缓冲区','shared_memory');self.coral_transport.addItem('管道传输','pipe');form.addRow('Coral 传输',self.coral_transport)
        self.coral_threads=QSpinBox();self.coral_threads.setRange(1,32);self.coral_threads.setValue(1);form.addRow('TFLite CPU 线程',self.coral_threads)
        self.delegate_options=QLineEdit('{}');self.delegate_options.setPlaceholderText('例如 {"Performance":"High"}');form.addRow('委托参数 JSON',self.delegate_options)
        form.addRow(label('CPU / CUDA 使用“模型与转换”页的 Hugging Face 来源和任务；不会自动切换后端。','Muted'));left.addWidget(group)
        group=QGroupBox('推理输入');form=QFormLayout(group)
        self.input_mode=QComboBox()
        for name,key in [('图片路径','image'),('文本','text'),('音频路径','audio'),('预处理 NPZ','npz'),('Pipeline JSON','json')]:self.input_mode.addItem(name,key)
        form.addRow('输入类型',self.input_mode)
        self.input_value=QPlainTextEdit(str(ROOT/'examples/coral_data/parrot.jpg'));self.input_value.setMaximumHeight(155);self.input_value.setObjectName('inferenceInput');form.addRow(self.input_value)
        form.addRow(button('选择输入文件',self.browse_input))
        self.tokens=QSpinBox();self.tokens.setRange(1,512);self.tokens.setValue(64);form.addRow('生成 token 上限',self.tokens)
        form.addRow(self.action('运行推理',self.infer,True));left.addWidget(group);left.addStretch()
        right=QVBoxLayout();layout.addLayout(right,3)
        self.result_heading=label('等待推理','Title');right.addWidget(self.result_heading)
        self.inference_summary=label('选择输入后运行；首次调用包含模型加载时间。','Muted');right.addWidget(self.inference_summary)
        self.prediction_table=QTableWidget(0,2);self.prediction_table.setHorizontalHeaderLabels(['预测类别','分数'])
        self.prediction_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.prediction_table.verticalHeader().setVisible(False);self.prediction_table.setMaximumHeight(230);self.prediction_table.hide();right.addWidget(self.prediction_table)
        self.result_text=QPlainTextEdit();self.result_text.setReadOnly(True);self.result_text.setObjectName('inferenceResult');right.addWidget(self.result_text)
        right.addWidget(button('显示完整结果 JSON',self.show_full_inference))
        self.tabs.addTab(page,'推理')
        self.locked.extend([self.backend,self.infer_artifact,self.input_mode,self.input_value,self.tokens])
        self.locked.extend([self.coral_transport,self.coral_threads,self.delegate_options])

    def _bench_tab(self):
        page=QWidget();layout=QVBoxLayout(page);layout.setContentsMargins(18,18,18,18)
        self.metric_label=label('— TOPS','Metric');layout.addWidget(self.metric_label)
        self.metric_detail=label('按真实已编译模型测量，不将厂商 4 TOPS 当作实测值。','Muted');layout.addWidget(self.metric_detail)
        layout.addWidget(label('有效 TOPS = 2 × 名义 INT8 MAC 数 ÷ (耗时 ms × 10⁹)\n委托调用时间包含 USB / 驱动 / 设备；端到端还包含主机布局、量化及 IPC。仅对零 CPU 操作的图计算。','Muted'))
        actions=QHBoxLayout();self.repeats=QSpinBox();self.repeats.setRange(10,5000);self.repeats.setValue(100)
        actions.addWidget(label('重复次数'));actions.addWidget(self.repeats);actions.addStretch()
        actions.addWidget(self.action('测量当前产物',lambda:self.benchmark(True)))
        actions.addWidget(self.action('运行完整算力基准',lambda:self.benchmark(False),True));layout.addLayout(actions)
        self.bench_table=QTableWidget(0,6);self.bench_table.setHorizontalHeaderLabels(['工作负载','GMAC / 次','委托 p50 ms','端到端 p50 ms','委托有效 TOPS','端到端 TOPS'])
        self.bench_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);self.bench_table.verticalHeader().setVisible(False);layout.addWidget(self.bench_table,1)
        layout.addWidget(button('打开完整基准 JSON',lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(ROOT/'reports/edge-tpu-compute.json')))))
        self.tabs.addTab(page,'算力基准');self.locked.append(self.repeats)

    def _jobs_tab(self):
        page=QWidget();layout=QVBoxLayout(page);layout.setContentsMargins(18,18,18,18)
        layout.addWidget(label('任务日志','Title'));layout.addWidget(label('下载、转换和推理均在后台进程执行。失败原因与完整堆栈保存在本地任务记录。','Muted'))
        self.logs=QPlainTextEdit();self.logs.setReadOnly(True);self.logs.setMaximumBlockCount(5000);layout.addWidget(self.logs)
        self.tabs.addTab(page,'任务记录')

    def browse(self,field,directory=False):
        value=QFileDialog.getExistingDirectory(self,'选择目录',field.text()) if directory else QFileDialog.getOpenFileName(self,'选择文件',field.text())[0]
        if value:field.setText(value)

    def browse_calibration(self): self.browse(self.cal_path,self.cal_mode.currentData()=='images')
    def browse_input(self):
        path=QFileDialog.getOpenFileName(self,'选择推理输入',str(ROOT/'examples'))[0]
        if path:self.input_value.setPlainText(path)
    def common(self):return {'source':self.source.text(),'revision':self.revision.text() or 'main','task':self.task.currentText()}
    def inspect(self):self.start_job({'kind':'inspect',**self.common()})
    def convert(self):
        try:
            options=json.loads(self.approximations.text()) if self.operator_strategy.isChecked() else {}
            if not isinstance(options,dict):raise ValueError('需要 JSON 对象')
        except (ValueError,TypeError) as exc:
            self.status.setText('有界近似配置无效：'+str(exc));return
        self.start_job({'kind':'convert',**self.common(),'calibration_mode':self.cal_mode.currentData(),
            'calibration_path':self.cal_path.text(),'max_calibration':self.max_cal.value(),'artifact_dir':self.artifact.text(),
            'allow_partial':self.partial.isChecked(),'operator_strategy':'coral' if self.operator_strategy.isChecked() else None,
            'approximations':options})
    def infer(self):
        try:
            options=json.loads(self.delegate_options.text()) if self.backend.currentText()=='Coral' else {}
            if not isinstance(options,dict):raise ValueError('委托参数必须是 JSON 对象')
            from torch_coral_runtime import normalize_options
            options=normalize_options(options)
        except (ValueError,TypeError) as exc:
            self.status.setText('运行参数错误：'+str(exc));return
        self.start_job({'kind':'infer',**self.common(),'backend':self.backend.currentText(),
            'artifact_dir':self.infer_artifact.text(),'input_mode':self.input_mode.currentData(),
            'input':self.input_value.toPlainText(),'max_new_tokens':self.tokens.value(),
            'transport':self.coral_transport.currentData(),'num_threads':self.coral_threads.value(),
            'delegate_options':options})
    def benchmark(self,only):self.start_job({'kind':'benchmark','artifact_only':only,'artifact_dir':self.infer_artifact.text(),'repeats':self.repeats.value()})

    def start_job(self,request):
        if self.process is not None:return
        self._save();self.active_request=self.job_dir/(uuid.uuid4().hex+'.json');self.active_request.write_text(json.dumps(request,ensure_ascii=False,indent=2),encoding='utf-8')
        self.current_kind=request['kind'];self.last_result=None;self.buffer='';self.had_error=False;self.cancelled=False
        self.decoder=codecs.getincrementaldecoder('utf-8')('replace')
        for item in self.actions+self.locked:item.setEnabled(False)
        self.progress.setRange(0,0);self.status.setText('后台任务执行中：'+self.current_kind)
        self.cancel_button.setEnabled(True)
        self.logs.appendPlainText('\n══ '+self.current_kind+' · '+str(self.active_request)+' ══')
        process=QProcess(self);self.process=process
        process.setWorkingDirectory(str(ROOT));process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env=QProcessEnvironment.systemEnvironment();env.insert('PYTHONUTF8','1');env.insert('HF_HUB_DISABLE_XET','1');process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self.read_output);process.finished.connect(self.finished)
        process.errorOccurred.connect(self.process_error)
        process.start(sys.executable,['-u','-X','utf8','-m','torch_coral.jobs','--request',str(self.active_request)])

    def read_output(self):
        if self.process is None:return
        self.buffer+=self.decoder.decode(bytes(self.process.readAllStandardOutput()))
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1);self.handle_line(line)

    def handle_line(self,line):
        try:event=json.loads(line)
        except (ValueError,TypeError):
            self.logs.appendPlainText(line);return
        if not isinstance(event,dict) or 'event' not in event:
            self.logs.appendPlainText(line);return
        kind=event['event']
        if kind=='progress':self.status.setText(event['message']);self.logs.appendPlainText(event['message'])
        elif kind=='error':
            self.had_error=True;self.status.setText('任务失败 · 查看具体原因');self.logs.appendPlainText(event['message'])
            self.show_result({'error':event['message'],'detail_path':event.get('detail_path')})
        elif kind=='result':
            self.last_result=event['result'];self.logs.appendPlainText('结果已保存：'+event['result_path']);self.show_result(self.last_result)

    def show_result(self,result):
        text=json.dumps(result,ensure_ascii=False,indent=2)
        if self.current_kind=='inspect':
            overview={k:v for k,v in result.items() if k!='config'}
            if result.get('config'):
                config=result['config']
                overview['input_hints']={k:config[k] for k in ['image_size','num_channels','hidden_size','vocab_size','max_position_embeddings'] if k in config}
                overview['label_count']=len(config.get('id2label',{}))
            self.model_info.setPlainText(json.dumps(overview,ensure_ascii=False,indent=2))
            if 'error' not in result:
                self.model_summary.setText(f"{result['model_type']}\n{result['task']}\nRevision：{result['revision'][:12]}\n\n"+'\n'.join(result['issues'] or ['尚无已知输入限制，仍需实际转换验证。']))
        elif self.current_kind=='convert':
            self.model_info.setPlainText(text)
            if 'artifact_dir' in result:
                self.infer_artifact.setText(result['artifact_dir']);self.result_heading.setText('模型已转换 · 可运行推理')
        elif self.current_kind=='infer':
            self.result_heading.setText('推理失败' if 'error' in result else '推理完成 · '+result.get('backend',''))
            self.inference_result=result
            predictions=result.get('predictions')
            if predictions is None and isinstance(result.get('output'),list) and result['output'] and isinstance(result['output'][0],dict) and 'label' in result['output'][0]:predictions=result['output']
            self.prediction_table.setVisible(bool(predictions))
            if predictions:
                self.prediction_table.setRowCount(len(predictions))
                for i,p in enumerate(predictions):
                    self.prediction_table.setItem(i,0,QTableWidgetItem(str(p['label'])))
                    self.prediction_table.setItem(i,1,QTableWidgetItem(f"{p['score']*100:.2f}%"))
                self.inference_summary.setText(f"{predictions[0]['label']} · {predictions[0]['score']*100:.2f}%\n后端：{result.get('backend','')} · 任务结果和详细 JSON 已保存在本地")
                self.result_text.setPlainText('预测分数为模型输出，经量化后可能变化。\n\n任务记录保留原始输出、后端和耗时；可点击下方按钮查看完整 JSON。')
            else:
                self.inference_summary.setText(result.get('error','后端：'+result.get('backend','')))
                output=result.get('output',result)
                if isinstance(output,list) and output and isinstance(output[0],dict) and 'generated_text' in output[0]:
                    self.result_text.setPlainText('\n\n'.join(str(p['generated_text']) for p in output))
                else:self.result_text.setPlainText(json.dumps(output,ensure_ascii=False,indent=2)[:250000])
        elif self.current_kind=='benchmark':
            if 'error' in result:self.logs.appendPlainText(text)
            elif 'results' in result:self.refresh_benchmark(result)
            else:self.refresh_benchmark({'results':[result]})

    def finished(self,code,status):
        self.read_output()
        self.buffer+=self.decoder.decode(b'',final=True)
        if self.buffer:self.handle_line(self.buffer);self.buffer=''
        process=self.process;self.process=None
        if process:process.deleteLater()
        for item in self.actions+self.locked:item.setEnabled(True)
        self.approximations.setEnabled(self.operator_strategy.isChecked())
        self.progress.setRange(0,100);self.progress.setValue(100 if code==0 else 0)
        self.cancel_button.setEnabled(False)
        self.status.setText('任务已取消 · 未完成文件保留' if self.cancelled else ('任务完成 · 结果已保存' if code==0 else '任务失败 · 查看结果或任务记录'))

    def process_error(self,error):
        self.logs.appendPlainText('进程错误：'+str(error))
        if error==QProcess.ProcessError.FailedToStart:self.finished(-1,None)

    def cancel_job(self):
        if self.process is None:return
        self.cancelled=True
        pid=self.process.processId()
        if pid and os.name=='nt':
            # Kill only the job process tree created by this window, including
            # converter children. Partial artifacts remain available for review.
            subprocess.run(['taskkill.exe','/PID',str(pid),'/T','/F'],capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW,timeout=10)
        else:self.process.kill()
        self.logs.appendPlainText('用户取消任务；未完成文件保留，重新转换可选择新产物目录。')

    def refresh_benchmark(self,report=None):
        path=ROOT/'reports/edge-tpu-compute.json'
        if report is None:
            if not path.exists():return
            report=json.loads(path.read_text(encoding='utf-8'))
        rows=report['results'];self.bench_table.setRowCount(len(rows))
        for i,r in enumerate(rows):
            values=[r['name'],f"{r['work']['macs']/1e9:.4f}",f"{r['delegate_p50_ms']:.3f}",f"{r['end_to_end_p50_ms']:.3f}",f"{r['delegate_effective_tops']:.3f}",f"{r['end_to_end_effective_tops']:.3f}"]
            for j,value in enumerate(values):self.bench_table.setItem(i,j,QTableWidgetItem(value))
        if rows:
            best=max(r['delegate_effective_tops'] for r in rows);end=max(r['end_to_end_effective_tops'] for r in rows)
            self.metric_label.setText(f'{best:.3f} TOPS')
            self.metric_detail.setText(f'本轮最高委托有效算力 · 端到端最高 {end:.3f} TOPS · 标准频率运行库')

    def open_current(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.active_request.parent if self.active_request else self.job_dir)))

    def show_full_inference(self):
        if hasattr(self,'inference_result'):
            self.result_text.setPlainText(json.dumps(self.inference_result,ensure_ascii=False,indent=2)[:250000])

    def _save(self):
        data={'source':self.source.text(),'revision':self.revision.text(),'task':self.task.currentText(),
              'artifact':self.artifact.text(),'infer_artifact':self.infer_artifact.text(),'cal_path':self.cal_path.text()}
        (ROOT/'.gui/session.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

    def _restore(self):
        path=ROOT/'.gui/session.json'
        if not path.exists():return
        try:
            data=json.loads(path.read_text(encoding='utf-8'))
            for key in ['source','revision','artifact','infer_artifact','cal_path']:
                if key in data:getattr(self,key).setText(data[key])
            if data.get('task'):self.task.setCurrentText(data['task'])
        except (ValueError,OSError):pass

    def closeEvent(self,event):
        if self.process is not None:
            self.cancel_job();self.process.waitForFinished(5000)
        self._save();super().closeEvent(event)

def main():
    app=QApplication(sys.argv);app.setStyle('Fusion');app.setStyleSheet(STYLE)
    window=Workbench();window.show();sys.exit(app.exec())

if __name__=='__main__':main()
