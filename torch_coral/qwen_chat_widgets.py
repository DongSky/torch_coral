"""Painted, lightweight chat and inference telemetry widgets for the Qt client."""
from collections import deque
import math
import os
from pathlib import Path
import time
from PySide6.QtCore import Qt, QRectF, QPointF, QTimer, Signal, QPropertyAnimation, QEasingCurve, QVariantAnimation
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPainterPath, QPen, QLinearGradient, QRadialGradient, QIcon, QPixmap
from PySide6.QtWidgets import (QWidget, QFrame, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QScrollArea, QStackedLayout, QGraphicsOpacityEffect, QSizePolicy)

INK = '#293d34'
MUTED = '#738075'
ACCENT = '#d86b4b'
TEAL = '#147d78'
_fonts_initialized = False


def initialize_fonts():
    """Load local system fonts explicitly, including Windows headless Qt builds."""
    global _fonts_initialized
    if _fonts_initialized: return
    if os.name == 'nt':
        directory = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
        families = []
        for name in ('msyh.ttc', 'consola.ttf'):
            path = directory / name
            if path.is_file():
                identifier = QFontDatabase.addApplicationFont(str(path))
                if identifier >= 0: families.extend(QFontDatabase.applicationFontFamilies(identifier))
        if families:
            from PySide6.QtWidgets import QApplication
            QApplication.instance().setFont(QFont(families[0], 10))
    _fonts_initialized = True


def color(value, alpha=1.):
    result = QColor(value); result.setAlphaF(max(0., min(1., alpha))); return result


def font(size, *, mono=False, bold=False):
    result = QFont('Consolas' if mono else 'Microsoft YaHei UI')
    result.setPixelSize(size); result.setBold(bold); return result


def settings_icon():
    canvas = QPixmap(20, 20); canvas.fill(Qt.GlobalColor.transparent)
    p = QPainter(canvas); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor('#506551'), 1.5)); p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(10, 10), 5, 5); p.drawEllipse(QPointF(10, 10), 1.8, 1.8)
    for index in range(8):
        a = index * math.tau / 8
        p.drawLine(QPointF(10 + math.cos(a) * 5, 10 + math.sin(a) * 5), QPointF(10 + math.cos(a) * 7.5, 10 + math.sin(a) * 7.5))
    p.end(); return QIcon(canvas)


class Panel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent); self.setObjectName('Panel')


class AmbientOrb(QWidget):
    def __init__(self, size=132, parent=None):
        super().__init__(parent); self.setFixedSize(size, size); self.motion = True
        self.setAccessibleName('Coral 装饰图形')

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = QPointF(self.width() / 2, self.height() / 2)
        radius = self.width() * .38
        t = time.monotonic() if self.motion else 0.
        glow = QRadialGradient(center, radius * 1.3)
        glow.setColorAt(0, color(ACCENT, .15)); glow.setColorAt(.6, color(ACCENT, .04)); glow.setColorAt(1, color(ACCENT, 0))
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(glow); p.drawEllipse(center, radius * 1.3, radius * 1.3)
        for ring, alpha in ((1., .15), (.79, .25)):
            r = radius * ring
            p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(color(ACCENT, alpha), 1))
            p.drawEllipse(center, r, r)
            p.setPen(QPen(color(TEAL if ring == 1 else ACCENT, .8), 2))
            angle = t * (13 if ring == 1 else -19)
            p.drawArc(QRectF(center.x() - r, center.y() - r, r * 2, r * 2), int(angle * 16), 72 * 16)
        for index in range(3):
            angle = t * .15 + index * math.tau / 3
            point = center + QPointF(math.cos(angle) * radius, math.sin(angle) * radius)
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(color(TEAL if index == 0 else ACCENT, .8)); p.drawEllipse(point, 2, 2)
        p.setFont(font(int(self.width() * .25), bold=True)); p.setPen(QColor(INK))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'C')


class RateChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumHeight(100)
        self.samples = deque(maxlen=40); self.motion = True
        self.empty_text = '等待首个 Decode 样本'
        self.setAccessibleName('实际 Decode token 速率曲线')

    def clear(self): self.samples.clear(); self.update()

    def add_sample(self, elapsed, rate):
        if rate is not None and math.isfinite(rate) and rate >= 0:
            self.samples.append((float(elapsed), float(rate))); self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(4, 9, max(1, self.width() - 12), max(1, self.height() - 32))
        p.setFont(font(10, mono=True))
        for fraction in (0., .5, 1.):
            y = bounds.top() + bounds.height() * fraction
            p.setPen(QPen(color('#527360', .13), 1)); p.drawLine(QPointF(bounds.left(), y), QPointF(bounds.right(), y))
        if not self.samples:
            p.setPen(QColor(MUTED)); p.setFont(font(11))
            p.drawText(bounds, Qt.AlignmentFlag.AlignCenter, self.empty_text)
            return
        maximum = max(.05, max(v for _, v in self.samples) * 1.22)
        start, end = self.samples[0][0], self.samples[-1][0]
        points = [QPointF(bounds.left() + ((t - start) / (end - start) if end > start else .5) * bounds.width(),
                         bounds.bottom() - rate / maximum * bounds.height()) for t, rate in self.samples]
        path = QPainterPath(points[0])
        for point in points[1:]: path.lineTo(point)
        if len(points) > 1:
            fill = QPainterPath(path); fill.lineTo(points[-1].x(), bounds.bottom()); fill.lineTo(points[0].x(), bounds.bottom()); fill.closeSubpath()
            gradient = QLinearGradient(0, bounds.top(), 0, bounds.bottom())
            gradient.setColorAt(0, color(TEAL, .18)); gradient.setColorAt(1, color(TEAL, 0))
            p.fillPath(fill, gradient)
            line = QLinearGradient(bounds.left(), 0, bounds.right(), 0)
            line.setColorAt(0, QColor(ACCENT)); line.setColorAt(1, QColor(TEAL))
            p.setPen(QPen(line, 2)); p.drawPath(path)
        pulse = 5 + (math.sin(time.monotonic() * 3) + 1) * 2 if self.motion else 6
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(color(TEAL, .12)); p.drawEllipse(points[-1], pulse, pulse)
        p.setBrush(QColor(TEAL)); p.drawEllipse(points[-1], 3, 3)
        p.setFont(font(10, mono=True)); p.setPen(QColor(MUTED))
        p.drawText(QRectF(bounds.left(), bounds.bottom() + 8, bounds.width(), 18), Qt.AlignmentFlag.AlignLeft, '%.2f token/s' % self.samples[-1][1])
        p.drawText(QRectF(bounds.left(), bounds.bottom() + 8, bounds.width(), 18), Qt.AlignmentFlag.AlignRight, '%d samples' % len(self.samples))


class PipelineView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumHeight(182)
        self.frame = {}; self.received = time.monotonic(); self.prefill_progress = self.decode_progress = 0.
        self.motion = True
        self.drawn = dict(prefill=0., decode=0.)
        self.transitions = {}
        for name in ('prefill', 'decode'):
            animation = QVariantAnimation(self); animation.setDuration(220); animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            animation.valueChanged.connect(lambda value, lane=name: self._draw_value(lane, value)); self.transitions[name] = animation
        self.setAccessibleName('Prefill 与 Decode 实际执行进度')

    def reset(self):
        self.frame = {}; self.prefill_progress = self.decode_progress = 0.
        for name, animation in self.transitions.items(): animation.stop(); self.drawn[name] = 0.
        self.update()

    def _draw_value(self, name, value): self.drawn[name] = float(value); self.update()

    def set_frame(self, frame):
        self.frame = frame; self.received = time.monotonic()
        phase = frame.get('phase')
        if phase == 'prefill': self.prefill_progress = frame.get('progress', 0.)
        elif frame.get('prefill_complete'): self.prefill_progress = 1.
        if phase == 'decode': self.decode_progress = frame.get('progress', 0.)
        elif phase == 'complete': self.decode_progress = 1.
        for name, value in (('prefill', self.prefill_progress), ('decode', self.decode_progress)):
            animation = self.transitions[name]; animation.stop()
            if self.motion and value > self.drawn[name]:
                animation.setStartValue(self.drawn[name]); animation.setEndValue(value); animation.start()
            else: self.drawn[name] = value
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        phase = self.frame.get('phase', 'idle')
        active_time = max(0., time.monotonic() - self.received) if phase not in ('idle', 'complete', 'cancelled', 'error') else 0.
        for index, (name, title, value, accent) in enumerate([
                ('prefill', 'Prefill · 输入理解', self.prefill_progress, ACCENT),
                ('decode', 'Decode · 逐字生成', self.decode_progress, TEAL)]):
            y = 5 + index * 83
            active = phase == name or (name == 'decode' and phase in ('sampling', 'mtp'))
            done = value >= 1. and (name == 'prefill' or phase == 'complete')
            point = QPointF(8, y + 12)
            if index == 0:
                p.setPen(QPen(color('#a5b39d', .4), 1)); p.drawLine(QPointF(8, y + 24), QPointF(8, y + 79))
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(color(accent, .18 if active else .06)); p.drawEllipse(point, 8, 8)
            p.setBrush(color(accent, 1 if active or done else .3)); p.drawEllipse(point, 3.5, 3.5)
            p.setFont(font(12, bold=True)); p.setPen(QColor(INK if active or done else MUTED))
            p.drawText(QRectF(28, y, self.width() - 40, 24), Qt.AlignmentFlag.AlignVCenter, title)
            text = '完成' if done else ('%d%%' % (value * 100) if active and phase != 'mtp' else ('草稿验证' if active else '等待'))
            if not done and value > 0 and phase in ('cancelled', 'error'): text = '已停止' if phase == 'cancelled' else '已中断'
            p.setFont(font(10, mono=False)); p.setPen(QColor(accent if active or done else MUTED))
            p.drawText(QRectF(28, y, self.width() - 32, 24), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, text)
            bar = QRectF(28, y + 32, self.width() - 34, 5)
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(color('#8b9d82', .17)); p.drawRoundedRect(bar, 2.5, 2.5)
            visible = self.drawn[name]
            if visible > 0:
                filled = QRectF(bar.left(), bar.top(), bar.width() * min(1., visible), bar.height())
                gradient = QLinearGradient(filled.topLeft(), filled.topRight())
                gradient.setColorAt(0, color(accent, .55)); gradient.setColorAt(1, QColor(accent))
                p.setBrush(gradient); p.drawRoundedRect(filled, 2.5, 2.5)
                if active and self.motion:
                    p.save(); clip = QPainterPath(); clip.addRoundedRect(filled, 2.5, 2.5); p.setClipPath(clip)
                    x = filled.left() + (time.monotonic() * .4 % 1) * (filled.width() + 50) - 25
                    shine = QLinearGradient(x - 20, 0, x + 20, 0)
                    shine.setColorAt(0, color('#ffffff', 0)); shine.setColorAt(.5, color('#ffffff', .45)); shine.setColorAt(1, color('#ffffff', 0))
                    p.fillRect(filled, shine); p.restore()
            seconds = self.frame.get(name + '_seconds', 0.) + (active_time if phase == name else 0.)
            detail = self.frame.get('detail', '') if active else ('%.1f 秒' % seconds if seconds else '尚未开始')
            if name == 'prefill' and done:
                detail = '%d token · %.1f 秒' % (self.frame.get('prompt_tokens', 0), seconds)
            elif name == 'decode' and phase == 'complete':
                detail = '%d token · %.1f 秒' % (self.frame.get('decode_tokens', 0), seconds)
            p.setPen(QColor(MUTED)); p.setFont(font(10))
            p.drawText(QRectF(28, y + 45, self.width() - 34, 22), Qt.AlignmentFlag.AlignVCenter, p.fontMetrics().elidedText(detail, Qt.TextElideMode.ElideRight, self.width() - 38))


class MetricTile(Panel):
    def __init__(self, title, value='—', parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self); layout.setContentsMargins(14, 12, 14, 12); layout.setSpacing(5)
        label = QLabel(title); label.setObjectName('Muted'); layout.addWidget(label)
        self.value = QLabel(value); self.value.setObjectName('SmallMetric'); layout.addWidget(self.value)

    def set_value(self, value): self.value.setText(value)


class TypingDots(QWidget):
    def __init__(self, parent=None): super().__init__(parent); self.setFixedSize(42, 15); self.motion = True
    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing); p.setPen(Qt.PenStyle.NoPen)
        for index in range(3):
            wave = (math.sin(time.monotonic() * 4 - index * .9) + 1) / 2 if self.motion else .5
            p.setBrush(color(ACCENT, .3 + wave * .6)); p.drawEllipse(QPointF(6 + index * 12, 7 - wave * 2), 2.5, 2.5)


class ChatBubble(QFrame):
    def __init__(self, role, text, *, motion=True, streaming=False, parent=None):
        super().__init__(parent)
        self.role, self.full_text, self.visible_text = role, '', ''
        self.motion, self.streaming = motion, streaming
        self.setObjectName('UserBubble' if role == 'user' else 'AssistantBubble')
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(self); layout.setContentsMargins(18, 14, 18, 14); layout.setSpacing(9)
        head = QLabel('你' if role == 'user' else 'QWEN 3.5'); head.setObjectName('BubbleRole'); layout.addWidget(head)
        self.body = QLabel(); self.body.setWordWrap(True); self.body.setTextFormat(Qt.TextFormat.PlainText)
        self.body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.body.setObjectName('BubbleText'); layout.addWidget(self.body)
        self.dots = TypingDots(); layout.addWidget(self.dots)
        self.reveal = QTimer(self); self.reveal.setInterval(16); self.reveal.timeout.connect(self._reveal)
        self.reveal_started = 0.; self.reveal_credit = 0.
        self.set_target(text, streaming=streaming)
        if motion:
            effect = QGraphicsOpacityEffect(self); self.setGraphicsEffect(effect)
            self.fade = QPropertyAnimation(effect, b'opacity', self); self.fade.setDuration(240)
            self.fade.setStartValue(0.); self.fade.setEndValue(1.); self.fade.setEasingCurve(QEasingCurve.Type.OutCubic); self.fade.start()

    def set_target(self, text, *, streaming=False):
        self.streaming = streaming
        if text != self.full_text:
            if not text.startswith(self.visible_text): self.visible_text = ''
            self.full_text = text
            if self.motion and streaming and text:
                self.reveal_started = time.monotonic(); self.reveal_credit = 0.; self.reveal.start()
            else: self.visible_text = text; self.reveal.stop()
        if not streaming or not self.motion: self.visible_text = self.full_text; self.reveal.stop()
        self._refresh()

    def _reveal(self):
        now = time.monotonic(); self.reveal_credit += (now - self.reveal_started) * 75; self.reveal_started = now
        count = int(self.reveal_credit)
        if count:
            self.reveal_credit -= count; self.visible_text = self.full_text[:len(self.visible_text) + count]; self._refresh()
            if self.visible_text == self.full_text: self.reveal.stop()

    def _refresh(self):
        self.body.setText(self.visible_text or ('正在计算…' if self.streaming else ''))
        self.dots.setVisible(self.streaming); self.dots.motion = self.motion


class ConversationView(QWidget):
    suggested = Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent); self.motion = True; self.nodes = []
        self.stack = QStackedLayout(self); self.stack.setContentsMargins(0, 0, 0, 0)
        empty = QWidget(); v = QVBoxLayout(empty); v.setContentsMargins(28, 20, 28, 20); v.setSpacing(12); v.addStretch()
        self.orb = AmbientOrb(); v.addWidget(self.orb, 0, Qt.AlignmentFlag.AlignHCenter)
        title = QLabel('你好，今天聊些什么？'); title.setObjectName('EmptyTitle'); title.setAlignment(Qt.AlignmentFlag.AlignCenter); v.addWidget(title)
        hint = QLabel('选择运行模式，连接 Coral，开始本地对话'); hint.setObjectName('Muted'); hint.setAlignment(Qt.AlignmentFlag.AlignCenter); v.addWidget(hint)
        chips = QHBoxLayout(); chips.setSpacing(8)
        for text in ('解释矩阵乘法', '写一个 Python 示例', '帮我梳理思路'):
            button = QPushButton(text); button.setObjectName('Suggestion'); button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda checked=False, prompt=text: self.suggested.emit(prompt)); chips.addWidget(button)
        v.addSpacing(14); v.addLayout(chips); v.addStretch()
        self.stack.addWidget(empty)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.follow = True
        self.scroll.verticalScrollBar().valueChanged.connect(self._scroll_changed)
        self.scroll.verticalScrollBar().rangeChanged.connect(self._range_changed)
        self.surface = QWidget(); self.surface.setObjectName('MessageSurface')
        self.rows = QVBoxLayout(self.surface); self.rows.setContentsMargins(22, 24, 22, 24); self.rows.setSpacing(16); self.rows.addStretch()
        self.scroll.setWidget(self.surface); self.stack.addWidget(self.scroll)

    def set_conversation(self, messages, pending=None):
        wanted = [(item['role'], item['content']) for item in messages]
        if pending is not None: wanted.append(('assistant', pending))
        bar = self.scroll.verticalScrollBar(); follow = bar.maximum() - bar.value() < 60
        self.follow = follow
        shared = 0
        while shared < min(len(wanted), len(self.nodes)) and self.nodes[shared][0] == wanted[shared][0]: shared += 1
        while len(self.nodes) > shared:
            _, _, row = self.nodes.pop(); self.rows.removeWidget(row); row.deleteLater()
        for index, (role, text) in enumerate(wanted):
            streaming = pending is not None and index == len(wanted) - 1
            if index >= len(self.nodes):
                row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0)
                bubble = ChatBubble(role, text, motion=self.motion, streaming=streaming)
                if role == 'user': h.addStretch(1)
                h.addWidget(bubble, 5)
                if role != 'user': h.addStretch(1)
                self.rows.insertWidget(self.rows.count() - 1, row); self.nodes.append((role, bubble, row))
            self.nodes[index][1].set_target(text, streaming=streaming)
            self._fit_bubble(self.nodes[index][1])
        self.stack.setCurrentIndex(1 if wanted else 0)
        if follow: QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))

    def toPlainText(self): return '\n\n'.join(bubble.full_text for _, bubble, _ in self.nodes)

    def _fit_bubble(self, bubble):
        bubble.ensurePolished(); bubble.body.ensurePolished()
        lines = (bubble.full_text or '正在计算…').splitlines() or ['']
        natural = max(bubble.body.fontMetrics().horizontalAdvance(line) for line in lines) + 44
        maximum = max(220, int(self.width() * .84))
        bubble.setMaximumWidth(min(maximum, max(112 if bubble.role == 'user' else 188, natural)))

    def _scroll_changed(self, value):
        self.follow = self.scroll.verticalScrollBar().maximum() - value < 60

    def _range_changed(self, minimum, maximum):
        if self.follow: self.scroll.verticalScrollBar().setValue(maximum)

    def tick(self):
        if not self.motion: return
        if self.stack.currentIndex() == 0: self.orb.update()
        for _, bubble, _ in self.nodes:
            if bubble.streaming: bubble.dots.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for _, bubble, _ in self.nodes: self._fit_bubble(bubble)
