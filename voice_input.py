"""
Voice Input — GUI client for local Whisper server.

- System tray icon with menu
- Settings window (mic selection, server URL, language, hotkey)
- Global hotkey (default Win+Space) to record & type
- Overlay indicator, silence auto-stop
- Config persisted to voice_input_config.json

Run:   python voice_input.py
Build: pyinstaller --noconsole --onefile --icon=app.ico voice_input.py
"""
from __future__ import annotations

import io
import json
import logging
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd

# Global hotkeys via 'keyboard' lib (native Windows hook, no six/shiboken conflict).
try:
    import keyboard as _kb
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False

from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize, Slot, QMetaObject, QRect
from PySide6.QtGui import (
    QAction, QBrush, QColor, QCursor, QFont, QPen, QLinearGradient,
    QRadialGradient, QIcon, QImage, QPainter, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMenu, QPushButton, QSpinBox, QSystemTrayIcon, QVBoxLayout,
    QWidget, QMessageBox,
)

APP_NAME = "Voice Input"
ORG_NAME = "VoiceInput"
CONFIG_FILE = Path(__file__).parent / "voice_input_config.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("voice-input")

# ============================================================================
# CONFIG
# ============================================================================
DEFAULTS = {
    "api_url": "http://localhost:8000/v1/audio/transcriptions",
    "health_url": "http://localhost:8000/health",
    "sample_rate": 16000,
    "language": "ru",            # "ru", "en", "" = auto
    "device_index": None,        # int or None
    "silence_threshold": 0.005,
    "silence_seconds": 0.8,
    "max_record_seconds": 300,
    "timeout_s": 60,
    "mode": "ptt",                # "ptt" = push-to-talk (hold to record) | "toggle"
    "hotkey": "f8",               # main key (ptt: hold; toggle: press twice)
    "hotkey_cancel": "ctrl+alt+esc",
    "ptt_disable_silence_autostop": True,
    "auto_start_recording": False,
    "paste_with_ctrl_v": True,
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        if CONFIG_FILE.exists():
            cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
    except Exception as e:
        log.warning("config load failed: %s", e)
    return cfg


def save_config(cfg: dict) -> None:
    try:
        CONFIG_FILE.write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        log.error("config save failed: %s", e)


# ============================================================================
# AUDIO HELPERS
# ============================================================================
def list_input_devices() -> list[tuple[int, str]]:
    out = []
    try:
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                out.append((i, d["name"]))
    except Exception as e:
        log.error("query_devices failed: %s", e)
    return out


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(samples.reshape(-1), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ============================================================================
# ICON (generated programmatically — no external files needed)
# ============================================================================
def make_icon(color: str = "#4CAF50", text: str = "") -> QIcon:
    """Draw a simple circular mic icon."""
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(4, 4, 56, 56)
    # mic body
    p.setBrush(QColor("white"))
    p.drawRoundedRect(26, 16, 12, 24, 6, 6)
    p.drawEllipse(24, 36, 16, 8)
    p.drawRect(31, 42, 2, 8)
    p.drawRect(26, 46, 12, 2)
    if text:
        p.setPen(QColor("white"))
        f = QFont("Segoe UI", 18, QFont.Bold)
        p.setFont(f)
        p.drawText(pm.rect().adjusted(0, 0, 0, -30), Qt.AlignCenter, text)
    p.end()
    return QIcon(pm)


def make_rec_icon() -> QIcon:
    return make_icon("#F44336", "●")


def make_idle_icon() -> QIcon:
    return make_icon("#607D8B")


def make_done_icon() -> QIcon:
    return make_icon("#4CAF50", "✓")


def make_off_icon() -> QIcon:
    return make_icon("#9E9E9E", "✕")


# ============================================================================
# WORKERS
# ============================================================================
class RecordWorker(QThread):
    """Records audio from mic in background."""
    started = Signal()
    level = Signal(float)          # current RMS level 0..1
    stopped = Signal(np.ndarray)   # full recording

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self._stop_flag = threading.Event()
        self._chunks: list[np.ndarray] = []

    def run(self):
        self._stop_flag.clear()
        self._chunks = []
        dev = self.cfg.get("device_index")
        sr = self.cfg.get("sample_rate", 16000)
        block = int(sr * 0.1)  # 100ms
        silence_blocks = 0
        max_silence_blocks = int(self.cfg.get("silence_seconds", 0.8) / 0.1)
        threshold = float(self.cfg.get("silence_threshold", 0.005))
        max_seconds = int(self.cfg.get("max_record_seconds", 60))
        # in PTT mode silence auto-stop is disabled — key release stops recording
        use_autostop = not bool(self.cfg.get("ptt_disable_silence_autostop", True))
        started_at = time.time()
        saw_speech = False

        def callback(indata, frames, t, status):
            nonlocal silence_blocks, saw_speech
            chunk = indata.copy().reshape(-1)
            self._chunks.append(chunk)
            r = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            self.level.emit(min(1.0, r * 20))
            if r > threshold:
                silence_blocks = 0
                saw_speech = True
            else:
                silence_blocks += 1

        try:
            with sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                device=dev, blocksize=block,
                                callback=callback) as stream:
                self.started.emit()
                while not self._stop_flag.is_set():
                    time.sleep(0.05)
                    if use_autostop and saw_speech and silence_blocks >= max_silence_blocks:
                        log.info("silence auto-stop")
                        break
                    if time.time() - started_at > max_seconds:
                        log.info("max duration stop")
                        break
        except Exception as e:
            log.exception("record error")
            self.stopped.emit(np.zeros(0, dtype=np.float32))
            return

        audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)
        self.stopped.emit(audio)

    def stop(self):
        self._stop_flag.set()


class TranscribeWorker(QThread):
    """Sends WAV to server, emits resulting text."""
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, cfg: dict, wav_bytes: bytes):
        super().__init__()
        self.cfg = cfg
        self.wav = wav_bytes

    def run(self):
        try:
            files = {"file": ("audio.wav", self.wav, "audio/wav")}
            data = {"response_format": "text", "temperature": "0"}
            lang = self.cfg.get("language", "")
            if lang:
                data["language"] = lang
            t0 = time.time()
            r = requests.post(self.cfg["api_url"], files=files, data=data,
                              timeout=int(self.cfg.get("timeout_s", 60)))
            dt = time.time() - t0
            if r.status_code != 200:
                self.failed.emit(f"server {r.status_code}: {r.text[:200]}")
                return
            text = r.text.strip()
            log.info("transcribed in %.2fs: %r", dt, text[:100])
            self.finished_ok.emit(text)
        except Exception as e:
            log.exception("transcribe error")
            self.failed.emit(str(e))


# ============================================================================
# OVERLAY — premium animated status indicator (dot + waveform + label)
# ============================================================================
def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _ease(t: float) -> float:
    """Smooth ease-in-out cubic."""
    return 3 * t * t - 2 * t * t * t


def _mix_colors(c1: QColor, c2: QColor, t: float) -> QColor:
    return QColor(
        int(_lerp(c1.red(), c2.red(), t)),
        int(_lerp(c1.green(), c2.green(), t)),
        int(_lerp(c1.blue(), c2.blue(), t)),
        int(_lerp(c1.alpha(), c2.alpha(), t)),
    )


class Overlay(QWidget):
    """Premium animated status indicator, top-right corner.

    - Smooth color/size transitions between states (eased interpolation)
    - Recording: green dot + live voice waveform bars reacting to mic level
    - Processing: orange dot with elegant radar sweep
    - Idle: tiny dim dot, labelless — nearly invisible
    - Soft radial glow behind the dot
    - Click-through, frameless, always-on-top
    """

    IDLE_C   = QColor(255, 255, 255, 70)
    GREEN_C  = QColor(52, 199, 89, 255)
    ORANGE_C = QColor(255, 149, 0, 255)
    RED_C    = QColor(255, 59, 48, 255)
    TEXT_C   = QColor(255, 255, 255, 220)

    LABELS = {
        "idle":       "",
        "recording":  "",
        "processing": "",
        "done":       "",
        "error":      "",
    }

    STATE_COLORS = {
        "idle": IDLE_C, "recording": GREEN_C,
        "processing": ORANGE_C, "done": GREEN_C, "error": RED_C,
    }

    FPS = 60
    TRANSITION_MS = 350

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedSize(40, 28)

        # animation state
        self._state = "idle"
        self._label_override = ""
        self._t_ms = 0.0                 # running clock
        self._trans_t = 1.0              # 0..1 transition progress
        self._prev_color = QColor(self.IDLE_C)
        self._cur_color = QColor(self.IDLE_C)
        self._level = 0.0                # smoothed mic level 0..1
        self._target_level = 0.0
        self._bars = [0.0] * 5           # waveform bars history
        self._sweep = 0.0

        # cursor-follow position (smoothed)
        self._mouse_pos = None
        self._shown_pos = None

        self._t = QTimer(self)
        self._t.timeout.connect(self._tick)
        self._t.start(int(1000 / self.FPS))
        self.show()
        # apply click-through AFTER the native window exists
        self._apply_click_through()
        self._follow_cursor()

    def _apply_click_through(self):
        """WS_EX_TRANSPARENT: mouse events pass to windows below."""
        try:
            import ctypes
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x80000
            WS_EX_TRANSPARENT = 0x20
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                                  style | WS_EX_LAYERED | WS_EX_TRANSPARENT)
        except Exception:
            log.warning("click-through style failed", exc_info=True)

    def _follow_cursor(self):
        """Anchor the dot right next to the mouse arrow."""
        pos = QCursor.pos()
        # widget center lands at cursor.x + 16, cursor.y (just right of arrow tip)
        target_x = pos.x() + 16 - self.width() // 2
        target_y = pos.y() - self.height() // 2
        if self._shown_pos is None:
            self._shown_pos = (target_x, target_y)
        else:
            x = _lerp(self._shown_pos[0], target_x, 0.5)
            y = _lerp(self._shown_pos[1], target_y, 0.5)
            self._shown_pos = (x, y)
        self.move(int(self._shown_pos[0]), int(self._shown_pos[1]))
        self.update()

    def set_state(self, state: str, text: str = ""):
        if state != self._state:
            # start eased color transition
            self._prev_color = QColor(self._cur_color)
            self._trans_t = 0.0
        self._state = state
        self._label_override = text
        self._bars = [0.0] * 5
        self._level = 0.0
        self._target_level = 0.0
        self.update()
        self.show()

    def set_level(self, level: float):
        """Mic input level 0..1 — call during recording for live waveform."""
        self._target_level = max(0.0, min(1.0, level))

    def _tick(self):
        dt = 1000 / self.FPS
        self._t_ms += dt
        self._follow_cursor()
        # transition progress
        if self._trans_t < 1.0:
            self._trans_t = min(1.0, self._trans_t + dt / self.TRANSITION_MS)
            t = _ease(self._trans_t)
            self._cur_color = _mix_colors(self._prev_color,
                                          self.STATE_COLORS.get(self._state, self.IDLE_C),
                                          t)
        # smooth level
        self._level = _lerp(self._level, self._target_level, 0.25)
        # waveform bars: shift history, feed current level + noise
        self._bars.pop(0)
        import math
        jitter = 0.5 + 0.5 * math.sin(self._t_ms / 90 + id(self) % 7)
        self._bars.append(max(0.08, self._level * (0.7 + 0.5 * jitter)))
        # radar sweep
        self._sweep = (self._sweep + dt / 1400) % 1.0
        self.update()

    # ---------------------------------------------------------------- paint
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        state = self._state
        color = self._cur_color
        label = self._label_override or self.LABELS.get(state, "")
        W, H = self.width(), self.height()
        cy = H // 2
        dot_cx = W // 2 + 4  # slightly right of widget center

        if state == "idle":
            # tiny dim breathing dot — nearly invisible
            import math
            breath = 0.5 + 0.5 * math.sin(self._t_ms / 1600)
            r = 2.5 + 0.8 * breath
            p.setPen(Qt.NoPen)
            p.setBrush(self.IDLE_C)
            p.drawEllipse(int(dot_cx - r), int(cy - r), int(2 * r), int(2 * r))
            return

        # ---- animated dot radius (small & subtle)
        if state == "recording":
            import math
            pulse = 0.5 + 0.5 * math.sin(self._t_ms / 500)
            r = 2.8 + 0.5 * pulse + 0.8 * self._level
        elif state == "processing":
            r = 2.8
        else:
            r = 2.8

        # ---- radial glow
        glow_r = r + 7
        gradient = QRadialGradient(dot_cx, cy, glow_r)
        gc = QColor(color)
        gc.setAlpha(55)
        gradient.setColorAt(0.0, gc)
        gc2 = QColor(color)
        gc2.setAlpha(0)
        gradient.setColorAt(1.0, gc2)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(gradient))
        p.drawEllipse(int(dot_cx - glow_r), int(cy - glow_r),
                      int(2 * glow_r), int(2 * glow_r))

        # ---- processing: radar sweep ring (compact)
        if state == "processing":
            sweep_r = r + 2 + 6 * self._sweep
            sweep_a = int(90 * (1 - self._sweep))
            if sweep_a > 0:
                ring = QColor(color)
                ring.setAlpha(sweep_a)
                pen = QPen(ring, 1.4)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(int(dot_cx - sweep_r), int(cy - sweep_r),
                              int(2 * sweep_r), int(2 * sweep_r))

        # ---- dot core
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        p.drawEllipse(int(dot_cx - r), int(cy - r), int(2 * r), int(2 * r))
        # inner highlight for 3D feel
        hi = QColor(255, 255, 255, 90)
        p.setBrush(hi)
        p.drawEllipse(int(dot_cx - r * 0.45), int(cy - r * 0.55),
                      int(r * 0.7), int(r * 0.7))

        # ---- recording: waveform removed — dot only, reacts to voice via size
        # (kept: level-driven radius pulse in the recording branch above)

        # ---- label (disabled — indicator only, no text)
        # if label: ...



# ============================================================================
# SETTINGS DIALOG
# ============================================================================
class SettingsDialog(QDialog):
    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle(f"{APP_NAME} — настройки")
        self.setMinimumWidth(440)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        form = QFormLayout()

        # server
        self.ed_url = QLineEdit(self.cfg["api_url"])
        form.addRow("URL сервера:", self.ed_url)

        # mic
        self.cb_mic = QComboBox()
        self.cb_mic.addItem("По умолчанию (системный)", None)
        for idx, name in list_input_devices():
            self.cb_mic.addItem(f"[{idx}] {name}", idx)
        cur = self.cfg.get("device_index")
        pos = self.cb_mic.findData(cur)
        if pos >= 0:
            self.cb_mic.setCurrentIndex(pos)
        form.addRow("Микрофон:", self.cb_mic)
        btn_refresh = QPushButton("Обновить список")
        btn_refresh.clicked.connect(self._refresh_devices)
        form.addRow("", btn_refresh)

        # language
        self.cb_lang = QComboBox()
        self.cb_lang.addItem("Авто-определение", "")
        self.cb_lang.addItem("Русский", "ru")
        self.cb_lang.addItem("English", "en")
        self.cb_lang.addItem("Deutsch", "de")
        self.cb_lang.addItem("Français", "fr")
        self.cb_lang.addItem("Español", "es")
        pos = self.cb_lang.findData(self.cfg.get("language", ""))
        self.cb_lang.setCurrentIndex(max(0, pos))
        form.addRow("Язык речи:", self.cb_lang)

        # silence auto-stop (toggle mode only)
        self.chk_autostop = QCheckBox("Автостоп по тишине (для режима «Переключатель»)")
        self.chk_autostop.setChecked(not bool(self.cfg.get("ptt_disable_silence_autostop", True)))
        form.addRow("", self.chk_autostop)

        self.sp_silence = QDoubleSpinBox()
        self.sp_silence.setRange(0.3, 10.0)
        self.sp_silence.setSingleStep(0.1)
        self.sp_silence.setSuffix(" сек")
        self.sp_silence.setValue(float(self.cfg.get("silence_seconds", 0.8)))
        form.addRow("Тишина до автостопа:", self.sp_silence)

        self.sp_thresh = QDoubleSpinBox()
        self.sp_thresh.setRange(0.0005, 0.5)
        self.sp_thresh.setDecimals(4)
        self.sp_thresh.setSingleStep(0.001)
        self.sp_thresh.setValue(float(self.cfg.get("silence_threshold", 0.005)))
        form.addRow("Порог тишины (RMS):", self.sp_thresh)

        self.sp_max = QSpinBox()
        self.sp_max.setRange(5, 3600)
        self.sp_max.setSuffix(" сек")
        self.sp_max.setValue(int(self.cfg.get("max_record_seconds", 300)))
        form.addRow("Макс. длительность:", self.sp_max)

        # paste behavior
        # mode
        self.cb_mode = QComboBox()
        self.cb_mode.addItem("Рация (удерживать клавишу)", "ptt")
        self.cb_mode.addItem("Переключатель (нажать начать/остановить)", "toggle")
        pos = self.cb_mode.findData(self.cfg.get("mode", "ptt"))
        self.cb_mode.setCurrentIndex(max(0, pos))
        form.addRow("Режим:", self.cb_mode)

        # hotkeys
        self.ed_hotkey = QLineEdit(self.cfg.get("hotkey", "f8"))
        form.addRow("Клавиша записи:", self.ed_hotkey)
        self.ed_hotkey_cancel = QLineEdit(self.cfg.get("hotkey_cancel", "ctrl+alt+esc"))
        form.addRow("Хоткей отмены:", self.ed_hotkey_cancel)

        # paste behavior
        self.chk_paste = QCheckBox("Вставлять текст автоматически (Ctrl+V)")
        self.chk_paste.setChecked(bool(self.cfg.get("paste_with_ctrl_v", True)))
        form.addRow("", self.chk_paste)

        lay.addLayout(form)

        # status line for server check
        self.lbl_status = QLabel("Сервер: не проверен")
        self.lbl_status.setStyleSheet("color: gray;")
        lay.addWidget(self.lbl_status)
        btn_check = QPushButton("Проверить сервер")
        btn_check.clicked.connect(self._check_server)
        lay.addWidget(btn_check)

        # ok / cancel
        btns = QHBoxLayout()
        btn_ok = QPushButton("Сохранить")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Отмена")
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        lay.addLayout(btns)

    def _refresh_devices(self):
        self.cb_mic.clear()
        self.cb_mic.addItem("По умолчанию (системный)", None)
        for idx, name in list_input_devices():
            self.cb_mic.addItem(f"[{idx}] {name}", idx)
        cur = self.cfg.get("device_index")
        pos = self.cb_mic.findData(cur)
        if pos >= 0:
            self.cb_mic.setCurrentIndex(pos)

    def _check_server(self):
        url = self.ed_url.text().strip()
        # strip path: http://host:port/v1/audio/transcriptions -> http://host:port
        from urllib.parse import urlsplit
        parts = urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        health = base + "/health"
        try:
            r = requests.get(health, timeout=3)
            if r.status_code == 200:
                info = r.json()
                self.lbl_status.setText(
                    f"Сервер: OK — model={info.get('model','?')} device={info.get('device','?')}")
                self.lbl_status.setStyleSheet("color: #2E7D32;")
            else:
                self.lbl_status.setText(f"Сервер: HTTP {r.status_code}")
                self.lbl_status.setStyleSheet("color: #C62828;")
        except Exception as e:
            self.lbl_status.setText(f"Сервер: недоступен ({e.__class__.__name__})")
            self.lbl_status.setStyleSheet("color: #C62828;")

    def accept(self):
        self.cfg["api_url"] = self.ed_url.text().strip()
        self.cfg["device_index"] = self.cb_mic.currentData()
        self.cfg["language"] = self.cb_lang.currentData()
        self.cfg["silence_seconds"] = self.sp_silence.value()
        self.cfg["ptt_disable_silence_autostop"] = not self.chk_autostop.isChecked()
        self.cfg["silence_threshold"] = self.sp_thresh.value()
        self.cfg["max_record_seconds"] = self.sp_max.value()
        self.cfg["paste_with_ctrl_v"] = self.chk_paste.isChecked()
        self.cfg["mode"] = self.cb_mode.currentData()
        self.cfg["hotkey"] = self.ed_hotkey.text().strip() or "f8"
        self.cfg["hotkey_cancel"] = self.ed_hotkey_cancel.text().strip() or "ctrl+alt+esc"
        save_config(self.cfg)
        super().accept()


# ============================================================================
# MAIN WINDOW (hidden — all interaction via tray)
# ============================================================================
class MainWindow(QMainWindow):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle(APP_NAME)
        self.setWindowFlag(Qt.Tool)  # don't show in taskbar

        self.recorder: RecordWorker | None = None
        self.transcriber: TranscribeWorker | None = None
        self._processing = False

        # overlay
        self.overlay = Overlay()

        # tray
        self._icons = {
            "idle": make_idle_icon(),
            "recording": make_rec_icon(),
            "processing": make_icon("#FFC107"),
            "done": make_done_icon(),
            "error": make_off_icon(),
        }
        self.tray = QSystemTrayIcon(self._icons["idle"], self)
        self.tray.setToolTip(f"{APP_NAME} — готов")
        menu = QMenu()

        self.act_record = QAction(f"Записать ({self.cfg.get('hotkey', 'f8')})", self)
        act_record = self.act_record
        act_record.triggered.connect(self.toggle_record)
        menu.addAction(act_record)

        act_settings = QAction("Настройки…", self)
        act_settings.triggered.connect(self.show_settings)
        menu.addAction(act_settings)

        menu.addSeparator()

        act_quit = QAction("Выход", self)
        act_quit.triggered.connect(self.quit)
        menu.addAction(act_quit)

        self.tray.setContextMenu(menu)
        self.tray.show()
        self.tray.activated.connect(self._on_tray_activated)

        # global hotkey via keyboard lib
        self._setup_hotkey()

        # timer to restore idle icon
        self._state_timer = QTimer(self)
        self._state_timer.setSingleShot(True)
        self._state_timer.timeout.connect(lambda: self.set_state("idle"))

    # ---------------------------------------------------------------- hotkey
    def _setup_hotkey(self):
        if not HAS_KEYBOARD:
            log.warning("keyboard lib not installed — no global hotkey")
            return

        hk = self.cfg.get("hotkey", "f8").strip().lower()
        hk_cancel = self.cfg.get("hotkey_cancel", "ctrl+alt+esc")
        self._mode = self.cfg.get("mode", "ptt")
        suppress = self._mode == "ptt"

        def on_event(e):
            try:
                if e.name != hk:
                    return
                if e.event_type == "down":
                    if self._mode == "ptt":
                        log.debug("ptt down")
                        QMetaObject.invokeMethod(self, "_ptt_press_qt",
                                                 Qt.ConnectionType.QueuedConnection)
                    else:
                        QMetaObject.invokeMethod(self, "toggle_record_qt",
                                                 Qt.ConnectionType.QueuedConnection)
                elif e.event_type == "up" and self._mode == "ptt":
                    log.debug("ptt up")
                    QMetaObject.invokeMethod(self, "_ptt_release_qt",
                                             Qt.ConnectionType.QueuedConnection)
            except Exception:
                log.exception("hotkey handler error")

        # kb.hook fires reliably on Windows; suppress=False so we never
        # block the keyboard — the F8 keypress just also reaches other apps.
        _kb.hook(on_event, suppress=False)

        def fire_cancel():
            QMetaObject.invokeMethod(self, "cancel_record_qt",
                                     Qt.ConnectionType.QueuedConnection)

        _kb.add_hotkey(hk_cancel, fire_cancel, suppress=False, trigger_on_release=False)
        log.info("hotkeys: %s (%s mode), %s (cancel) — keyboard NOT blocked",
                 hk, self._mode, hk_cancel)

    @Slot()
    def _ptt_press_qt(self):
        self._ptt_press()

    @Slot()
    def _ptt_release_qt(self):
        self._ptt_release()

    @Slot()
    def toggle_record_qt(self):
        self.toggle_record()

    @Slot()
    def cancel_record_qt(self):
        self.cancel_record()

    def _ptt_press(self):
        if self._processing:
            log.info("busy — ignoring ptt press")
            return
        if self.recorder and self.recorder.isRunning():
            return  # already recording
        self._start_record()

    def _ptt_release(self):
        if self._processing:
            return
        if self.recorder and self.recorder.isRunning():
            self.recorder.stop()

    # ---------------------------------------------------------------- state
    def set_state(self, state: str, tooltip: str = ""):
        icon = self._icons.get(state, self._icons["idle"])
        self.tray.setIcon(icon)
        tips = {
            "idle": f"{APP_NAME} — готов",
            "recording": f"{APP_NAME} — ЗАПИСЬ…",
            "processing": f"{APP_NAME} — обработка…",
            "done": f"{APP_NAME} — готово",
            "error": f"{APP_NAME} — ошибка",
        }
        self.tray.setToolTip(tooltip or tips.get(state, APP_NAME))
        mapping = {
            "idle": "idle",
            "recording": "recording",
            "processing": "processing",
            "done": "done",
            "error": "error",
        }
        self.overlay.set_state(mapping.get(state, "idle"))
        if state in ("done", "error"):
            self._state_timer.start(1500)
        else:
            self._state_timer.stop()

    # ---------------------------------------------------------------- record
    def toggle_record(self):
        if self._processing:
            log.info("busy — ignoring toggle")
            return
        if self.recorder and self.recorder.isRunning():
            self.recorder.stop()
        else:
            self._start_record()

    def _start_record(self):
        self.recorder = RecordWorker(self.cfg)
        self.recorder.started.connect(lambda: self.set_state("recording"))
        self.recorder.level.connect(self.overlay.set_level)
        self.recorder.stopped.connect(self._on_recorded)
        self.recorder.finished.connect(self._on_recorder_finished)
        self.recorder.start()
        # watchdog: force-stop if thread hangs
        max_s = int(self.cfg.get("max_record_seconds", 60)) + 10
        QTimer.singleShot(max_s * 1000, self._watchdog)

    def _watchdog(self):
        if self.recorder and self.recorder.isRunning():
            log.warning("watchdog: force-stopping hung recorder")
            try:
                self.recorder.stop()
            except Exception:
                pass

    def _on_recorder_finished(self):
        """QThread finished signal — clean up if stopped without emitting audio."""
        if not self._processing and self.tray.toolTip().startswith(f"{APP_NAME} — ЗАПИСЬ"):
            log.warning("recorder finished while still showing RECORDING — resetting")
            self.set_state("idle")

    def _on_recorded(self, audio: np.ndarray):
        self.set_state("processing")
        if audio.size == 0:
            self.set_state("error", "нет аудио")
            self._state_timer.start(1500)
            return
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 1e-4:
            log.info("too quiet: rms=%s", rms)
            self.set_state("error", "слишком тихо")
            self._state_timer.start(1500)
            return
        wav = encode_wav(audio, int(self.cfg.get("sample_rate", 16000)))
        self._processing = True
        self.transcriber = TranscribeWorker(self.cfg, wav)
        self.transcriber.finished_ok.connect(self._on_transcribed)
        self.transcriber.failed.connect(self._on_transcribe_failed)
        self.transcriber.start()

    def _on_transcribed(self, text: str):
        self._processing = False
        if not text:
            self.set_state("done", "пустой текст")
            return
        if self.cfg.get("paste_with_ctrl_v", True):
            self._paste_text(text)
        else:
            self._win32_clipboard_copy(text)
        self.set_state("done")

    def _on_transcribe_failed(self, err: str):
        self._processing = False
        log.error("transcribe failed: %s", err)
        self.set_state("error", err[:60])
        self.tray.showMessage(
            APP_NAME, f"Ошибка:\n{err[:200]}",
            QSystemTrayIcon.MessageIcon.Critical, 4000)

    def _paste_text(self, text: str):
        """Copy to clipboard via Win32 API, then send Ctrl+V via SendInput."""
        if not self._win32_clipboard_copy(text):
            self.tray.showMessage(
                APP_NAME, "Не удалось записать в буфер обмена",
                QSystemTrayIcon.MessageIcon.Warning, 4000)
            return
        # give clipboard a moment, then send Ctrl+V
        time.sleep(0.08)
        self._win32_ctrl_v()

    @staticmethod
    def _win32_clipboard_copy(text: str) -> bool:
        import ctypes
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        CF_UNICODETEXT = 13
        GMEM_MOVEABLE = 0x0002
        # explicit 64-bit-safe prototypes
        kernel32.GlobalAlloc.restype = ctypes.c_void_p
        kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
        kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
        user32.SetClipboardData.restype = ctypes.c_void_p
        user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]

        opened = False
        for _ in range(10):
            if user32.OpenClipboard(None):
                opened = True
                break
            time.sleep(0.05)
        if not opened:
            log.error("OpenClipboard failed")
            return False
        try:
            user32.EmptyClipboard()
            data = text.encode("utf-16-le") + b"\x00\x00"
            h = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
            if not h:
                log.error("GlobalAlloc failed")
                return False
            p = kernel32.GlobalLock(h)
            if not p:
                log.error("GlobalLock failed")
                return False
            try:
                ctypes.memmove(p, data, len(data))
            finally:
                kernel32.GlobalUnlock(h)
            if not user32.SetClipboardData(CF_UNICODETEXT, h):
                log.error("SetClipboardData failed err=%s", kernel32.GetLastError())
                return False
            return True
        finally:
            user32.CloseClipboard()

    @staticmethod
    def _win32_ctrl_v():
        """Send Ctrl+V via SendInput (keybd_event fallback)."""
        import ctypes
        user32 = ctypes.windll.user32
        VK_CONTROL = 0x11
        VK_V = 0x56
        KEYEVENTF_KEYUP = 0x0002
        user32.keybd_event(VK_CONTROL, 0, 0, None)
        user32.keybd_event(VK_V, 0, 0, None)
        user32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, None)
        user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, None)

    def cancel_record(self):
        if self.recorder and self.recorder.isRunning():
            self.recorder.stopped.disconnect(self._on_recorded)
            self.recorder.stop()
        self.set_state("idle")
        log.info("recording cancelled")

    # ---------------------------------------------------------------- misc
    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_settings()

    def show_settings(self):
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec():
            log.info("settings saved")
            self._rehook_hotkeys()

    def _rehook_hotkeys(self):
        """Re-register global hotkeys after settings change."""
        if not HAS_KEYBOARD:
            return
        try:
            _kb.unhook_all()
        except Exception:
            pass
        self._setup_hotkey()
        self.act_record.setText(f"Записать ({self.cfg.get('hotkey', 'f8')})")

    def quit(self):
        if self.recorder and self.recorder.isRunning():
            self.recorder.stop()
            self.recorder.wait(2000)
        if HAS_KEYBOARD:
            try:
                _kb.unhook_all()
            except Exception:
                pass
        self.tray.hide()
        QApplication.quit()


# ============================================================================
# ENTRY
# ============================================================================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setQuitOnLastWindowClosed(False)

    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.warning(None, APP_NAME,
                            "Системный трей недоступен в этой среде.")
        sys.exit(1)

    cfg = load_config()
    win = MainWindow(cfg)
    win.set_state("idle")
    win.tray.showMessage(
        APP_NAME, f"{APP_NAME} запущен.\n{cfg.get('hotkey')} — запись, {cfg.get('hotkey_cancel')} — отмена.",
        QSystemTrayIcon.MessageIcon.Information, 3000)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
