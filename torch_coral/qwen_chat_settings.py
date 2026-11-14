"""Local UI preferences and a separate settings dialog; chat text is not saved."""
import json
import os
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QSpinBox, QCheckBox, QTabWidget, QDialogButtonBox, QMessageBox, QComboBox)


def defaults(root):
    suffix = 'Scripts/python.exe' if os.name == 'nt' else 'bin/python'
    return dict(version=1, python=os.environ.get('CORAL_ISA_PYTHON', str(root / '.venv-isa' / suffix)),
        gguf=str(root / '.tools/models/qwen35-gguf/Qwen3.5-0.8B-Q4_K_M.gguf'),
        tokenizer=str(root / '.tools/models/qwen35-0.8b'), prepared=str(root / '.tools/models/qwen35-gguf/fp32-prepared'),
        context=256, limit=16, mtp=False, animations=True, matrix_profile='turbo-v2', v2_clock='max',v3_speculative=True,
        v4_block_tokens=2)


def load_preferences(root):
    values = defaults(root)
    try:
        data = json.loads((root / '.gui/qwen-chat-settings.json').read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('version') != 1: return values
        for key in ('python', 'gguf', 'tokenizer', 'prepared', 'matrix_profile'):
            if isinstance(data.get(key), str) and data[key].strip(): values[key] = data[key]
        for key, low, high in (('context', 32, 4096), ('limit', 1, 512)):
            if type(data.get(key)) is int and low <= data[key] <= high: values[key] = data[key]
        for key in ('mtp', 'animations','v3_speculative'):
            if type(data.get(key)) is bool: values[key] = data[key]
        if data.get('v2_clock') in ('high','max'): values['v2_clock']=data['v2_clock']
        if type(data.get('v4_block_tokens')) is int and data['v4_block_tokens'] in (2,3,4):values['v4_block_tokens']=data['v4_block_tokens']
        if values['limit'] >= values['context']: values['limit'] = min(16, values['context'] - 1)
    except (OSError, ValueError): pass
    return values


def save_preferences(root, values):
    path = root / '.gui/qwen-chat-settings.json'; path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


class SettingsDialog(QDialog):
    def __init__(self, root, preferences, parent=None):
        super().__init__(parent); self.root = root
        self.setWindowTitle('Coral Studio · 设置'); self.resize(730, 580); self.setMinimumWidth(620)
        layout = QVBoxLayout(self); layout.setContentsMargins(24, 22, 24, 20); layout.setSpacing(18)
        title = QLabel('工作区设置'); title.setObjectName('DialogTitle'); layout.addWidget(title)
        hint = QLabel('模型、设备和生成参数集中在这里。'); hint.setObjectName('Muted'); layout.addWidget(hint)
        tabs = QTabWidget(); layout.addWidget(tabs, 1)
        model = QWidget(); form = QFormLayout(model); form.setContentsMargins(8, 22, 8, 18); form.setSpacing(18)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.python = QLineEdit(); self.gguf = QLineEdit(); self.tokenizer = QLineEdit(); self.prepared = QLineEdit()
        for title, control, kind in [('TPU Python', self.python, 'python'), ('GGUF 模型', self.gguf, 'gguf'),
                                     ('分词器目录', self.tokenizer, 'directory'), ('权重缓存目录', self.prepared, 'directory')]:
            row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(8); h.addWidget(control, 1)
            browse = QPushButton('浏览'); browse.setObjectName('Ghost'); browse.setFixedWidth(62)
            browse.clicked.connect(lambda checked=False, edit=control, mode=kind: self.browse(edit, mode)); h.addWidget(browse)
            form.addRow(title, row)
        note = QLabel('更改路径或上下文后会释放当前设备，重新连接时应用。'); note.setObjectName('Muted'); note.setWordWrap(True); form.addRow(note)
        self.v2_clock=QComboBox()
        self.v2_clock.addItem('500 MHz · 官方最高档','max'); self.v2_clock.addItem('250 MHz · 常规档','high')
        form.addRow('Turbo V2–V4 频率',self.v2_clock)
        thermal=QLabel('500 MHz 功耗与发热更高，需保持散热；可切回 250 MHz。其他模式保持原频率。')
        thermal.setObjectName('Muted'); thermal.setWordWrap(True); form.addRow(thermal)
        tabs.addTab(model, '模型与设备')
        generation = QWidget(); f = QFormLayout(generation); f.setContentsMargins(12, 24, 12, 18); f.setSpacing(20)
        self.context = QSpinBox(); self.context.setRange(32, 4096); self.context.setSingleStep(32)
        self.limit = QSpinBox(); self.limit.setRange(1, 512)
        self.mtp = QCheckBox('执行 MTP 草稿层'); self.mtp.setToolTip('其他模式的附加模块验证，会增加耗时；V3/V4 使用下方独立的校验式推测策略。')
        self.v3_speculative=QCheckBox('Turbo V3 / V4：启用校验式 MTP 推测解码')
        self.v3_speculative.setToolTip('完整 TPU 校验后才接受草稿；收益不足时自动切回逐 token 解码。')
        self.animations = QCheckBox('启用动态效果')
        f.addRow('上下文上限', self.context); f.addRow('最多生成 token', self.limit)
        f.addRow('附加验证', self.mtp); f.addRow('界面动画', self.animations)
        f.addRow('推测解码策略',self.v3_speculative)
        self.v4_blocks=QComboBox()
        for label,value in [('2 token · 稳定',2),('最多 3 token · 历史匹配',3),('最多 4 token · 历史匹配',4)]:self.v4_blocks.addItem(label,value)
        self.v4_blocks.setToolTip('V4 仅在已有历史片段可匹配时延长验证块，否则保持双 token；所有草稿均由完整 TPU 主干校验。')
        f.addRow('V4 长块策略',self.v4_blocks)
        detail = QLabel('动态效果包含消息淡入、逐字显现、流光和呼吸动画。关闭动画后，性能数据仍实时更新。')
        detail.setObjectName('Muted'); detail.setWordWrap(True); f.addRow(detail)
        tabs.addTab(generation, '生成与外观')
        footer = QHBoxLayout(); layout.addLayout(footer)
        reset = QPushButton('恢复默认'); reset.setObjectName('Ghost'); reset.clicked.connect(lambda: self.set_values(defaults(root))); footer.addWidget(reset)
        footer.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText('保存设置')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setObjectName('Ghost')
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); footer.addWidget(buttons)
        self.set_values(preferences)

    def browse(self, control, kind):
        if kind == 'directory': path = QFileDialog.getExistingDirectory(self, '选择目录', control.text())
        else:
            pattern = 'GGUF (*.gguf)' if kind == 'gguf' else 'Python (python.exe python);;所有文件 (*)'
            path, _ = QFileDialog.getOpenFileName(self, '选择文件', control.text(), pattern)
        if path: control.setText(path)

    def set_values(self, values):
        for key in ('python', 'gguf', 'tokenizer', 'prepared'): getattr(self, key).setText(values[key])
        self.context.setValue(values['context']); self.limit.setValue(values['limit'])
        self.mtp.setChecked(values['mtp']); self.animations.setChecked(values['animations'])
        self.v2_clock.setCurrentIndex(max(0,self.v2_clock.findData(values.get('v2_clock','max'))))
        self.v3_speculative.setChecked(values.get('v3_speculative',True))
        self.v4_blocks.setCurrentIndex(max(0,self.v4_blocks.findData(values.get('v4_block_tokens',2))))

    def values(self):
        return dict(python=self.python.text().strip(), gguf=self.gguf.text().strip(), tokenizer=self.tokenizer.text().strip(),
                    prepared=self.prepared.text().strip(), context=self.context.value(), limit=self.limit.value(),
                    mtp=self.mtp.isChecked(), animations=self.animations.isChecked(),v2_clock=self.v2_clock.currentData(),
                    v3_speculative=self.v3_speculative.isChecked(),v4_block_tokens=self.v4_blocks.currentData())

    def accept(self):
        values = self.values()
        if any(not values[key] for key in ('python', 'gguf', 'tokenizer', 'prepared')):
            QMessageBox.information(self, '设置未完成', '请填写模型、分词器、缓存和运行环境路径。'); return
        if values['limit'] >= values['context']:
            QMessageBox.information(self, '生成长度过大', '最多生成 token 需要小于上下文上限，为输入留出空间。'); return
        super().accept()
