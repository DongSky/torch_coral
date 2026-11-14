"""Headless GUI event/settings contracts; these tests never open a Coral."""
import json
import os
import pytest
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6.QtWidgets')
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import torch_coral.qwen_chat_gui as gui


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, 'workspace', lambda: tmp_path)
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(gui.STYLE)
    view = gui.QwenChat(); view.show(); app.processEvents()
    yield view
    view.close(); view.deleteLater(); app.processEvents()


def test_parameter_inputs_live_only_in_settings(window):
    assert window.profile.isVisible()
    assert not window.python.isVisible() and not window.limit.isVisible()
    assert window.python.window() is window.settings_dialog
    window.settings_dialog.show(); QApplication.processEvents()
    assert window.python.isVisible()
    window.settings_dialog.reject()
    window.resize(980, 720); QApplication.processEvents()
    assert window.width() == 980 and window.height() == 720
    assert window.sidebar.width() >= 260
    assert window.profile.currentData()=='turbo-v2'
    assert window.settings_dialog.v2_clock.currentData()=='max'
    assert window.profile.findData('turbo')>=0
    assert window.profile.findData('turbo-v3')>=0
    assert window.profile.findData('turbo-v4')>=0
    assert window.settings_dialog.v3_speculative.isChecked()
    assert window.settings_dialog.v4_blocks.currentData()==2


def test_frequency_change_releases_worker_and_keeps_chat(window,monkeypatch):
    window.loaded=True; window.messages=[dict(role='user',content='保留')]
    def save():
        clock=window.settings_dialog.v2_clock
        clock.setCurrentIndex(clock.findData('high')); window.settings_dialog.accept()
    QTimer.singleShot(0,save); window.open_settings()
    assert window.preferences['v2_clock']=='high'
    assert not window.loaded and window.messages[0]['content']=='保留'


def test_speed_events_do_not_count_first_token_as_decode(window):
    window.loaded = window.busy = True
    window.messages = [dict(role='user', content='<script>literal text</script>')]; window.render()
    base = dict(phase='prefill', elapsed_seconds=10., prefill_seconds=10., decode_seconds=0.,
        generated_tokens=0, decode_tokens=0, ttft_seconds=None, prefill_complete=False,
        last_decode_tokens_per_second=None, prompt_tokens=13, tpu_calls=30, progress=.5)
    window.handle_event(dict(event='telemetry', **base))
    assert window.rate.text() == '—' and len(window.rate_chart.samples) == 0
    first = dict(base, phase='sampling', generated_tokens=1, ttft_seconds=10., prefill_complete=True, prefill_tokens_per_second=1.3)
    window.handle_event(dict(event='text', text='你', metrics=first))
    assert window.rate.text() == '—'
    second = dict(first, phase='complete', elapsed_seconds=14., decode_seconds=4., generated_tokens=2,
                  decode_tokens=1, last_decode_tokens_per_second=.25, tpu_calls=45, progress=1.)
    window.handle_event(dict(event='done', text='你好', new_tokens=2, elapsed_seconds=14., metrics=second))
    assert window.rate.text() == '0.25'
    assert len(window.rate_chart.samples) == 1
    assert window.call_count.text().startswith('45')
    assert '<script>' in window.chat.toPlainText()
    assert window.chat.toPlainText().endswith('你好')
    assert window.loaded and not window.busy


def test_cancelled_settings_are_discarded_and_saved_settings_are_local(window):
    original = window.limit.value()
    def cancel(): window.limit.setValue(31); window.settings_dialog.reject()
    QTimer.singleShot(0, cancel); window.open_settings()
    assert window.limit.value() == original
    assert not (window.root / '.gui/qwen-chat-settings.json').exists()
    def save(): window.limit.setValue(32); window.settings_dialog.accept()
    QTimer.singleShot(0, save); window.open_settings()
    stored = json.loads((window.root / '.gui/qwen-chat-settings.json').read_text(encoding='utf-8'))
    assert stored['limit'] == 32 and 'messages' not in stored


def test_mode_switch_reloads_without_dropping_chat_history(window, monkeypatch):
    window.loaded = True; window.messages = [dict(role='user', content='保留对话')]
    called = []; monkeypatch.setattr(window, 'load_model', lambda: called.append(True))
    window.profile.setCurrentIndex(window.profile.findData('precise'))
    assert called == [True]
    assert window.messages[0]['content'] == '保留对话'
    assert window.preferences['matrix_profile'] == 'precise'


def test_worker_exit_stops_pending_animation_and_restores_input(window):
    window.loaded = window.busy = True
    window.messages = [dict(role='user', content='恢复这个问题')]; window.reply = '部分回复'; window.render()
    window.finished(1, None)
    assert not window.loaded and not window.busy
    assert window.input.toPlainText() == '恢复这个问题'
    assert not window.messages and not window.chat.nodes
