"""Coral Studio: standalone local chat and live physical inference telemetry."""
import json
from pathlib import Path
import sys
import time
from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer, Signal
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QPlainTextEdit, QComboBox, QScrollArea, QSplitter, QDialog)
from .settings import workspace
from .qwen_chat_settings import SettingsDialog, load_preferences, save_preferences
from .qwen_chat_widgets import Panel, AmbientOrb, RateChart, PipelineView, MetricTile, ConversationView, settings_icon, initialize_fonts


STYLE = '''
QWidget { color:#293d34; font-family:"Microsoft YaHei UI"; font-size:12px; }
QMainWindow,QDialog { background:#f6f5ef; }
QLabel { background:transparent; }
QLabel#Brand { font-size:18px; font-weight:700; letter-spacing:1.5px; }
QLabel#Subtitle,QLabel#Muted { color:#738075; font-size:11px; }
QLabel#EmptyTitle { font-size:23px; font-weight:600; }
QLabel#DialogTitle { font-size:23px; font-weight:600; }
QLabel#SectionTitle { font-size:12px; font-weight:600; }
QLabel#MetricValue { font-family:"Cascadia Mono","Consolas"; font-size:43px; font-weight:600; color:#20564d; }
QLabel#SmallMetric { font-family:"Cascadia Mono","Consolas"; font-size:22px; font-weight:500; }
QLabel#LiveTag { color:#147d78; font-size:10px; font-weight:700; }
QLabel#DevicePill { color:#768174; background:#eff1e9; border:1px solid #dce0d5; border-radius:13px; padding:6px 12px; font-size:11px; }
QLabel#DevicePill[state="connected"] { color:#337451; background:#e9f4e9; border-color:#cce1cb; }
QLabel#DevicePill[state="busy"] { color:#b85b37; background:#fff0e5; border-color:#efd4bf; }
QFrame#Panel { background:#fffefb; border:1px solid #e0e4db; border-radius:16px; }
QFrame#Composer { background:#fffefa; border:1px solid #cfd6ca; border-radius:16px; }
QFrame#UserBubble { background:#f8e8dd; border:1px solid #edcfbb; border-radius:14px; }
QFrame#AssistantBubble { background:#f0f5ee; border:1px solid #dce6d9; border-radius:14px; }
QLabel#BubbleRole { color:#795b45; font-size:10px; font-weight:700; letter-spacing:1px; }
QLabel#BubbleText { font-size:14px; color:#2d4236; }
QPushButton { background:#d86b4b; color:#ffffff; border:1px solid #d86b4b; border-radius:9px; padding:9px 15px; font-weight:600; }
QPushButton:hover { background:#c65b3d; border-color:#c65b3d; }
QPushButton:pressed { background:#ad4b30; }
QPushButton:disabled { background:#e9ece3; color:#a3ac9e; border-color:#dfe4d8; }
QPushButton#Ghost { background:transparent; color:#506551; border:1px solid #d8ded1; font-weight:400; }
QPushButton#Ghost:hover { background:#eef2e9; color:#2a4433; border-color:#acbba4; }
QPushButton#Danger { background:#fae6df; color:#a74e35; border-color:#ecc5b5; }
QPushButton#Danger:hover { background:#f5d8cb; }
QPushButton#Suggestion { background:#f5f5ee; border:1px solid #e0e3d5; color:#6d755e; padding:9px 10px; font-size:11px; font-weight:400; }
QPushButton#Suggestion:hover { background:#f9e7db; border-color:#dfb49b; color:#955334; }
QComboBox { background:#fffefa; border:1px solid #d7ddcf; border-radius:9px; padding:8px 13px; min-height:18px; }
QComboBox:hover { border-color:#c99576; }
QComboBox:disabled { color:#a0aa99; }
QComboBox::drop-down { border:0; width:24px; }
QComboBox QAbstractItemView { background:#fffefa; selection-background-color:#f4dfcd; selection-color:#62432c; border:1px solid #d7ddcf; padding:7px; }
QPlainTextEdit { background:transparent; border:0; color:#293d34; font-size:14px; selection-background-color:#f2c8b2; }
QLineEdit,QSpinBox { background:#ffffff; border:1px solid #d7ddcf; border-radius:8px; padding:10px; selection-background-color:#f2c8b2; }
QLineEdit:focus,QSpinBox:focus { border-color:#d48767; }
QCheckBox { spacing:10px; }
QCheckBox::indicator { width:16px; height:16px; border:1px solid #b7c2ae; border-radius:4px; background:#ffffff; }
QCheckBox::indicator:checked { background:#d86b4b; border-color:#d86b4b; }
QScrollArea { border:0; background:transparent; }
QWidget#MessageSurface,QWidget#TelemetrySurface { background:transparent; }
QScrollBar:vertical { background:transparent; width:6px; margin:4px 0; }
QScrollBar::handle:vertical { background:#d7ded0; border-radius:3px; min-height:32px; }
QScrollBar::handle:vertical:hover { background:#a6b59c; }
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }
QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical { background:transparent; }
QSplitter::handle { background:transparent; }
QTabWidget::pane { border-top:1px solid #dce2d3; }
QTabBar::tab { padding:11px 18px; color:#7c866f; border-bottom:2px solid transparent; }
QTabBar::tab:selected { color:#b65c3c; border-bottom-color:#d86b4b; }
QToolTip { background:#304d3e; color:#f7faf2; border:1px solid #4f6b55; padding:6px; }
'''


class ChatInput(QPlainTextEdit):
    submitted = Signal()
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            self.submitted.emit(); event.accept(); return
        super().keyPressEvent(event)


def apply_palette():
    app = QApplication.instance()
    if hasattr(app.styleHints(), 'setColorScheme'): app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    palette = app.palette()
    for role, value in ((QPalette.ColorRole.Window, '#f6f5ef'), (QPalette.ColorRole.WindowText, '#293d34'),
                        (QPalette.ColorRole.Base, '#ffffff'), (QPalette.ColorRole.AlternateBase, '#f0f5ee'),
                        (QPalette.ColorRole.Text, '#293d34'), (QPalette.ColorRole.ButtonText, '#293d34'),
                        (QPalette.ColorRole.Button, '#fffefa'), (QPalette.ColorRole.Highlight, '#d86b4b'),
                        (QPalette.ColorRole.HighlightedText, '#ffffff')):
        palette.setColor(role, QColor(value))
    app.setPalette(palette)


class QwenChat(QMainWindow):
    def __init__(self):
        super().__init__()
        apply_palette()
        initialize_fonts()
        self.setWindowTitle('Coral Studio · Qwen 本地对话'); self.resize(1320, 900); self.setMinimumSize(980, 720)
        self.messages = []; self.reply = ''; self.buffer = ''; self.errors = ''
        self.busy = self.loaded = self.closing = self.shutting_down = False
        self.intentional_stop = False; self.latest_metrics = None; self.frame_received = time.monotonic(); self.last_chart_token = 0
        self.root = workspace(); self.preferences = load_preferences(self.root)
        self.settings_dialog = SettingsDialog(self.root, self.preferences, self)
        for key in ('python', 'gguf', 'tokenizer', 'prepared', 'limit', 'mtp', 'context'):
            setattr(self, key, getattr(self.settings_dialog, key))
        central = QWidget(); self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central); self.main_layout.setContentsMargins(24, 20, 24, 16); self.main_layout.setSpacing(18)
        header = QHBoxLayout(); header.setSpacing(12); self.main_layout.addLayout(header)
        self.brand_orb = AmbientOrb(44); header.addWidget(self.brand_orb)
        brand = QVBoxLayout(); brand.setSpacing(3)
        label = QLabel('CORAL STUDIO'); label.setObjectName('Brand'); brand.addWidget(label)
        subtitle = QLabel('Qwen3.5 · 本地对话'); subtitle.setObjectName('Subtitle'); brand.addWidget(subtitle); header.addLayout(brand)
        header.addStretch()
        self.device_badge = QLabel('● 未连接'); self.device_badge.setObjectName('DevicePill'); header.addWidget(self.device_badge)
        self.profile = QComboBox(); self.profile.setMinimumWidth(173)
        for label, value in [('Turbo V4 · I/O 加速','turbo-v4'), ('Turbo V2 · 全链路优化', 'turbo-v2'), ('Turbo V3 · 试验模式','turbo-v3'),
                             ('Turbo · 快速优化', 'turbo'), ('Precise · 高精度', 'precise'), ('Balanced · 平衡', 'balanced'),
                             ('Fast · 基础快速', 'fast'), ('FP32 · 参考模式', 'fp32')]: self.profile.addItem(label, value)
        saved = self.profile.findData(self.preferences.get('matrix_profile', 'turbo-v2'))
        self.profile.setCurrentIndex(saved if saved>=0 else self.profile.findData('turbo-v2'))
        self.profile.setToolTip('V4 在 V3 基础上加速 USB 结果读回；V3/V4 使用校验式推测解码，低收益时关闭推测。均保持 Turbo 精度。')
        header.addWidget(self.profile)
        self.settings_button = QPushButton('设置'); self.settings_button.setIcon(settings_icon()); self.settings_button.setObjectName('Ghost'); self.settings_button.clicked.connect(self.open_settings); header.addWidget(self.settings_button)
        self.load = QPushButton('连接 Coral'); self.load.clicked.connect(self.toggle_connection); header.addWidget(self.load)
        splitter = QSplitter(Qt.Orientation.Horizontal); splitter.setHandleWidth(16); self.main_layout.addWidget(splitter, 1)
        left = QWidget(); left_layout = QVBoxLayout(left); left_layout.setContentsMargins(0, 0, 0, 0); left_layout.setSpacing(14); splitter.addWidget(left)
        chat_panel = Panel(); cp = QVBoxLayout(chat_panel); cp.setContentsMargins(0, 0, 0, 0); cp.setSpacing(0)
        chat_header = QHBoxLayout(); chat_header.setContentsMargins(20, 14, 18, 10)
        caption = QLabel('当前会话'); caption.setObjectName('SectionTitle'); chat_header.addWidget(caption); chat_header.addStretch()
        self.new = QPushButton('＋ 新对话'); self.new.setObjectName('Ghost'); self.new.clicked.connect(self.reset); chat_header.addWidget(self.new); cp.addLayout(chat_header)
        self.chat = ConversationView(); self.chat.suggested.connect(self.use_suggestion); cp.addWidget(self.chat, 1); left_layout.addWidget(chat_panel, 1)
        composer = Panel(); composer.setObjectName('Composer'); composer_layout = QVBoxLayout(composer); composer_layout.setContentsMargins(17, 12, 14, 12); composer_layout.setSpacing(8)
        self.input = ChatInput(); self.input.setPlaceholderText('输入消息，让对话开始…'); self.input.setFixedHeight(68)
        self.input.submitted.connect(self.submit); self.input.textChanged.connect(self.update_actions); composer_layout.addWidget(self.input)
        composer_footer = QHBoxLayout(); composer_layout.addLayout(composer_footer)
        key_hint = QLabel('Ctrl + Enter 发送'); key_hint.setObjectName('Muted'); composer_footer.addWidget(key_hint); composer_footer.addStretch()
        self.send = QPushButton('发送  ↗'); self.send.setMinimumWidth(98); self.send.clicked.connect(self.submit); composer_footer.addWidget(self.send)
        self.stop = QPushButton('停止计算  ■'); self.stop.setObjectName('Danger'); self.stop.clicked.connect(lambda: self.send_command(dict(action='cancel'))); composer_footer.addWidget(self.stop)
        left_layout.addWidget(composer)
        self.sidebar = QScrollArea(); self.sidebar.setWidgetResizable(True); self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar.setMinimumWidth(260); self.sidebar.setMaximumWidth(320); splitter.addWidget(self.sidebar)
        surface = QWidget(); surface.setObjectName('TelemetrySurface'); sv = QVBoxLayout(surface); sv.setContentsMargins(0, 0, 4, 0); sv.setSpacing(12)
        rate_panel = Panel(); rv = QVBoxLayout(rate_panel); rv.setContentsMargins(18, 17, 18, 12); rv.setSpacing(8)
        rh = QHBoxLayout(); rc = QLabel('实时 Decode'); rc.setObjectName('SectionTitle'); rh.addWidget(rc); rh.addStretch()
        self.live_tag = QLabel('● WAIT'); self.live_tag.setObjectName('LiveTag'); rh.addWidget(self.live_tag); rv.addLayout(rh)
        rate_row = QHBoxLayout(); rate_row.setSpacing(7)
        self.rate = QLabel('—'); self.rate.setObjectName('MetricValue'); rate_row.addWidget(self.rate)
        unit = QLabel('token/s'); unit.setObjectName('Muted'); rate_row.addWidget(unit, 0, Qt.AlignmentFlag.AlignBottom); rate_row.addStretch(); rv.addLayout(rate_row)
        self.rate_hint = QLabel('独立统计逐 token 生成速度'); self.rate_hint.setObjectName('Muted'); rv.addWidget(self.rate_hint)
        self.rate_chart = RateChart(); rv.addWidget(self.rate_chart)
        self.overall = QLabel('整体吞吐  — token/s'); self.overall.setObjectName('Muted'); rv.addWidget(self.overall); sv.addWidget(rate_panel)
        pipeline_panel = Panel(); pv = QVBoxLayout(pipeline_panel); pv.setContentsMargins(17, 16, 17, 12); pv.setSpacing(8)
        ph = QHBoxLayout(); title = QLabel('执行流水线'); title.setObjectName('SectionTitle'); ph.addWidget(title); ph.addStretch()
        chip = QLabel('EDGE TPU'); chip.setObjectName('Muted'); ph.addWidget(chip); pv.addLayout(ph)
        self.pipeline = PipelineView(); pv.addWidget(self.pipeline)
        self.pipeline.setToolTip('按实际完成的层更新进度，不是剩余时间预估。首个输出 token 来自 Prefill，Decode 只统计后续 token。')
        self.call_count = QLabel('0 次 TPU 调用'); self.call_count.setObjectName('Muted'); pv.addWidget(self.call_count); sv.addWidget(pipeline_panel)
        grid = QGridLayout(); grid.setSpacing(10)
        self.ttft_tile = MetricTile('首 token · TTFT'); self.count_tile = MetricTile('已生成 token', '0')
        self.ttft_tile.setToolTip('工作进程收到对话请求至首个 token；模型加载时间不计入。')
        self.prefill_tile = MetricTile('Prefill · token/s'); self.elapsed_tile = MetricTile('总耗时', '0.0 s')
        for index, tile in enumerate((self.ttft_tile, self.count_tile, self.prefill_tile, self.elapsed_tile)): grid.addWidget(tile, index // 2, index % 2)
        sv.addLayout(grid); sv.addStretch(); self.sidebar.setWidget(surface)
        splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 0); splitter.setSizes([930, 304])
        footer = QHBoxLayout(); self.main_layout.addLayout(footer)
        self.status = QLabel('选择模式并连接 Coral，开始对话'); self.status.setObjectName('Muted'); self.status.setWordWrap(True); footer.addWidget(self.status, 1)
        self.metrics = QLabel('本地执行'); self.metrics.setObjectName('Muted'); footer.addWidget(self.metrics)
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.read_output); self.process.readyReadStandardError.connect(self.read_error)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(lambda error: self.handle_error('运行环境启动失败：' + self.process.errorString()))
        self.profile.currentIndexChanged.connect(self.profile_changed)
        for button in self.findChildren(QPushButton): button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clock = QTimer(self); self.clock.setInterval(100); self.clock.timeout.connect(self.refresh_metrics); self.clock.start()
        self.motion_timer = QTimer(self); self.motion_timer.setInterval(33); self.motion_timer.timeout.connect(self.animate)
        self.apply_motion(); self.update_actions(); self.render()

    def persist_preferences(self):
        self.preferences.update(self.settings_dialog.values()); self.preferences['matrix_profile'] = self.profile.currentData()
        try: save_preferences(self.root, self.preferences)
        except OSError as exc: self.status.setText('设置已应用，本次未能保存：' + str(exc))

    def open_settings(self):
        if self.busy: return
        self.settings_dialog.set_values(self.preferences)
        if self.settings_dialog.exec() != QDialog.DialogCode.Accepted:
            self.settings_dialog.set_values(self.preferences); return
        old = dict(self.preferences); self.persist_preferences(); self.apply_motion()
        changed = any(old.get(key) != self.preferences[key] for key in ('python', 'gguf', 'tokenizer', 'prepared', 'context','v2_clock'))
        if changed:
            self.loaded = False
            if self.process.state() != QProcess.ProcessState.NotRunning: self.release_worker()
            self.status.setText('设置已保存；重新连接后应用模型与设备设置')
        else: self.status.setText('设置已保存')
        self.update_actions()

    def apply_motion(self):
        enabled = self.preferences.get('animations', True)
        for widget in (self.brand_orb, self.chat, self.chat.orb, self.rate_chart, self.pipeline): widget.motion = enabled
        for _, bubble, _ in self.chat.nodes:
            bubble.motion = enabled; bubble.dots.motion = enabled
            if not enabled: bubble.set_target(bubble.full_text, streaming=bubble.streaming)
        if enabled: self.motion_timer.start()
        else: self.motion_timer.stop(); self.chat.update(); self.rate_chart.update(); self.pipeline.update()

    def use_suggestion(self, text):
        if not self.busy: self.input.setPlainText(text); self.input.setFocus()

    def profile_changed(self):
        self.persist_preferences()
        if self.loaded: self.load_model()

    def toggle_connection(self):
        if self.loaded: self.release_worker()
        else: self.load_model()

    def release_worker(self):
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self.loaded = False; self.shutting_down = False; self.status.setText('设备已断开'); self.update_actions(); return
        self.loaded = False; self.shutting_down = self.intentional_stop = True
        self.send_command(dict(action='shutdown')); self.status.setText('正在释放 Coral…'); self.update_actions()

    def start_worker(self):
        if self.process.state() != QProcess.ProcessState.NotRunning: return
        python = Path(self.python.text())
        if not python.is_file(): raise FileNotFoundError('找不到 TPU 运行环境，请在设置中选择：' + str(python))
        self.buffer = ''; self.errors = ''; self.intentional_stop = False
        env = QProcessEnvironment.systemEnvironment(); env.insert('PYTHONUTF8', '1'); env.insert('PYTHONUNBUFFERED', '1')
        env.insert('PYTHONPATH', str(Path(__file__).resolve().parents[1])); self.process.setProcessEnvironment(env)
        self.process.start(str(python), ['-u', '-m', 'torch_coral.qwen_chat_worker'])
        if not self.process.waitForStarted(5000): raise RuntimeError(self.process.errorString())

    def send_command(self, command):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.write((json.dumps(command, ensure_ascii=True) + '\n').encode('utf-8'))

    def load_model(self):
        if self.shutting_down: return
        try:
            self.start_worker(); self.busy = True; self.loaded = False; self.reset_metrics(); self.update_actions()
            self.send_command(dict(action='load', gguf=self.gguf.text(), tokenizer_dir=self.tokenizer.text(),
                                   prepared_dir=self.prepared.text(), context=self.context.value(), matrix_profile=self.profile.currentData(),
                                   v2_clock=self.preferences.get('v2_clock','max')))
            self.status.setText('正在准备模型与设备…')
        except Exception as exc: self.handle_error(str(exc))

    def submit(self):
        text = self.input.toPlainText().strip()
        if not text or self.busy or not self.loaded: return
        self.messages.append(dict(role='user', content=text)); self.input.clear(); self.reply = ''
        self.busy = True; self.reset_metrics(); self.live_tag.setText('● LIVE'); self.update_actions(); self.render()
        self.status.setText('正在准备输入…')
        self.send_command(dict(action='chat', messages=self.messages, max_new_tokens=self.limit.value(), mtp=self.mtp.isChecked(),
                               speculative=self.preferences.get('v3_speculative',True),
                               speculative_block_tokens=self.preferences.get('v4_block_tokens',2)))

    def reset(self):
        if self.busy: return
        self.messages = []; self.reply = ''; self.render(); self.reset_metrics()
        self.send_command(dict(action='reset')); self.status.setText('新对话已就绪')

    def render(self):
        pending = self.reply if self.busy and self.messages and self.messages[-1]['role'] == 'user' else None
        self.chat.set_conversation(self.messages, pending)

    def reset_metrics(self):
        self.latest_metrics = None; self.last_chart_token = 0; self.pipeline.reset(); self.rate_chart.clear()
        self.rate.setText('—'); self.rate_hint.setText('独立统计逐 token 生成速度'); self.overall.setText('整体吞吐  — token/s')
        self.rate_chart.empty_text = '等待首个 Decode 样本'
        self.live_tag.setText('● WAIT'); self.call_count.setText('0 次 TPU 调用'); self.ttft_tile.set_value('—')
        self.count_tile.set_value('0'); self.prefill_tile.set_value('—'); self.elapsed_tile.set_value('0.0 s')

    def accept_metrics(self, frame):
        self.latest_metrics = dict(frame); self.frame_received = time.monotonic(); self.pipeline.set_frame(frame)
        count = frame.get('decode_tokens', 0); rate = frame.get('last_decode_tokens_per_second')
        if count > self.last_chart_token and rate is not None:
            self.rate_chart.add_sample(frame.get('elapsed_seconds', 0.), rate); self.last_chart_token = count
        self.refresh_metrics()

    def refresh_metrics(self):
        if self.latest_metrics is None: return
        frame = self.latest_metrics; phase = frame.get('phase', 'idle')
        running = self.busy and phase not in ('complete', 'cancelled', 'error', 'idle')
        extra = max(0., time.monotonic() - self.frame_received) if running else 0.
        elapsed = frame.get('elapsed_seconds', 0.) + extra
        decode = frame.get('decode_seconds', 0.) + (extra if phase == 'decode' else 0.)
        decoded, generated = frame.get('decode_tokens', 0), frame.get('generated_tokens', 0)
        rate = decoded / decode if decoded and decode > 0 else None
        self.rate.setText('%.2f' % rate if rate is not None else '—')
        self.rate_hint.setText('本阶段平均 · 实时更新' if rate is not None and running else ('本阶段平均' if rate is not None else ('Prefill 进行中' if phase == 'prefill' else '等待后续 Decode token')))
        if not decoded and phase == 'complete':
            self.rate_hint.setText('本次没有后续 Decode 步骤'); self.rate_chart.empty_text = '没有 Decode 采样'
        overall = generated / elapsed if elapsed > 0 else 0.
        self.overall.setText(('整体吞吐  %.3f token/s' if overall < .1 else '整体吞吐  %.2f token/s') % overall)
        first = frame.get('ttft_seconds'); self.ttft_tile.set_value('%.1f s' % first if first is not None else '—')
        self.count_tile.set_value(str(generated)); self.elapsed_tile.set_value('%.1f s' % elapsed)
        prefill = frame.get('prefill_tokens_per_second'); self.prefill_tile.set_value('%.2f' % prefill if prefill is not None else '—')
        self.live_tag.setText('● LIVE' if running else ('● STOP' if phase in ('cancelled', 'error') else '● DONE'))
        self.call_count.setText('%s 次 TPU 调用' % format(frame.get('tpu_calls', 0), ','))
        self.pipeline.update()

    def animate(self):
        if not self.isVisible(): return
        self.brand_orb.update(); self.chat.tick()
        if self.busy or self.rate_chart.samples: self.rate_chart.update(); self.pipeline.update()

    def read_output(self):
        self.buffer += bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace')
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if not line: continue
            try:
                event = json.loads(line)
                if not isinstance(event, dict): raise ValueError('Invalid event')
                self.handle_event(event)
            except (ValueError, TypeError, KeyError): self.handle_error('运行环境返回了不完整的事件')

    def handle_event(self, event):
        kind = event.get('event')
        if kind == 'progress':
            self.status.setText(event['message']); self.status.setToolTip(event['message'])
        elif kind == 'telemetry':
            self.accept_metrics(event)
            names = dict(prefill='Prefill · 理解输入', sampling='选择首个 token', decode='Decode · 生成中', mtp='MTP · 草稿验证', complete='生成完成', cancelled='已停止', error='计算中断')
            label = names.get(event.get('phase'), '正在计算')
            spec=event.get('speculation')
            if spec is not None:
                label+=' · 草稿 %d/%d 通过'%(spec.get('accepted',0),spec.get('verified_pairs',0))
                if spec.get('disabled'): label+=' · 已切回普通解码'
            self.status.setText(label + ('  /  ' + event['detail'] if event.get('detail') else ''))
        elif kind == 'loaded':
            self.busy = False; self.loaded = True; self.reply = ''; self.render()
            clock_note=(' · '+str(event['clock_mhz'])+' MHz') if event.get('clock_mhz') else ''
            self.status.setText('Coral 已连接 · ' + self.profile.currentText()+clock_note); self.status.setToolTip(event['backend'])
        elif kind == 'text':
            self.reply = event['text']
            if 'metrics' in event: self.accept_metrics(event['metrics'])
            self.render()
        elif kind == 'done':
            if 'metrics' in event: self.accept_metrics(event['metrics'])
            self.messages.append(dict(role='assistant', content=event['text']))
            self.busy = False; self.reply = ''; self.render(); self.refresh_metrics()
            self.status.setText('生成完成 · %d token · %.1f 秒' % (event['new_tokens'], event['elapsed_seconds']))
        elif kind == 'cancelled':
            self.rollback_pending(); self.busy = False; self.reply = ''; self.status.setText('已停止，当前对话状态已重置'); self.render(); self.refresh_metrics()
        elif kind == 'error':
            if event.get('requires_reload'): self.loaded = False
            self.handle_error(event['message'])
        self.update_actions()

    def read_error(self): self.errors = (self.errors + bytes(self.process.readAllStandardError()).decode('utf-8', errors='replace'))[-16000:]

    def handle_error(self, message):
        self.rollback_pending(); self.busy = False; self.reply = ''
        self.status.setText(message); self.status.setToolTip(message + ('\n' + self.errors if self.errors else ''))
        if self.latest_metrics is not None:
            self.latest_metrics['phase'] = 'error'; self.pipeline.set_frame(self.latest_metrics); self.refresh_metrics()
        self.update_actions(); self.render()

    def rollback_pending(self):
        if self.busy and self.messages and self.messages[-1]['role'] == 'user':
            text = self.messages.pop()['content']
            if not self.input.toPlainText().strip(): self.input.setPlainText(text)

    def update_actions(self):
        self.send.setVisible(not self.busy); self.stop.setVisible(self.busy)
        self.send.setEnabled(self.loaded and not self.busy and bool(self.input.toPlainText().strip()))
        self.stop.setEnabled(self.busy and not self.shutting_down)
        self.load.setEnabled(not self.busy and not self.shutting_down); self.new.setEnabled(not self.busy and not self.shutting_down)
        self.profile.setEnabled(not self.busy and not self.shutting_down); self.settings_button.setEnabled(not self.busy and not self.shutting_down)
        self.input.setReadOnly(self.busy or self.shutting_down)
        self.load.setText('连接中…' if self.busy and not self.loaded else ('断开设备' if self.loaded else '连接 Coral'))
        button_style = 'Ghost' if self.loaded else 'Connect'
        if self.load.objectName() != button_style:
            self.load.setObjectName(button_style); self.load.style().unpolish(self.load); self.load.style().polish(self.load)
        state = 'busy' if self.busy else ('connected' if self.loaded else 'idle')
        self.device_badge.setText('● 计算中' if self.busy and self.loaded else ('● 连接中' if self.busy else ('● 已连接' if self.loaded else '● 未连接')))
        if self.device_badge.property('state') != state:
            self.device_badge.setProperty('state', state); self.device_badge.style().unpolish(self.device_badge); self.device_badge.style().polish(self.device_badge)

    def finished(self, *args):
        if self.busy: self.rollback_pending()
        self.loaded = self.busy = self.shutting_down = False; self.update_actions()
        self.reply = ''; self.render()
        if self.latest_metrics is not None and self.latest_metrics.get('phase') not in ('complete', 'cancelled', 'error'):
            self.latest_metrics['phase'] = 'cancelled' if self.intentional_stop or self.closing else 'error'
            self.pipeline.set_frame(self.latest_metrics); self.refresh_metrics()
        if self.closing: QTimer.singleShot(0, self.close)
        elif self.intentional_stop: self.status.setText('设备已释放，可随时重新连接')
        else: self.status.setText('设备进程已退出，可重新连接')

    def closeEvent(self, event):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.closing = self.shutting_down = True; self.send_command(dict(action='cancel')); self.send_command(dict(action='shutdown'))
            self.status.setText('正在停止计算并释放 Coral…'); self.update_actions(); event.ignore(); return
        self.clock.stop(); self.motion_timer.stop(); event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'sidebar'): self.sidebar.setMaximumWidth(286 if self.width() < 1150 else 320)


def main():
    app = QApplication(sys.argv); app.setStyle('Fusion'); app.setStyleSheet(STYLE)
    if hasattr(app.styleHints(), 'setColorScheme'): app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    window = QwenChat(); window.show(); sys.exit(app.exec())


if __name__ == '__main__': main()
