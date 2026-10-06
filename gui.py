"""
Desktop GUI for the step tracker.

    ~/venvs/steptracker/bin/python gui.py [--refs refs] [--source 0]

Manage reference states (add a state, capture frames from the live feed, import files),
pick a crop region by dragging on the video, tune the thresholds live and watch steps complete.
"""
import argparse
import os
import re
import shutil
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# PySide6 must be imported before cv2: opencv-python ships its own Qt plugins and points
# QT_QPA_PLATFORM_PLUGIN_PATH at them, which breaks PySide6's platform plugin loading.
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSlider, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    QAbstractItemView,
)

import cv2
import numpy as np
import torch

for var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
    os.environ.pop(var, None)

from track import IMG_EXT, VID_EXT, Embedder, StepTracker, crop, load_reference_frames

GREEN = QColor(46, 160, 67)
AMBER = QColor(210, 140, 20)


def parse_source(text):
    text = text.strip()
    return int(text) if text.isdigit() else text


def scan_refs(refs):
    """[(folder, n_files)] for each state folder, in procedure order."""
    root = Path(refs)
    if not root.is_dir():
        return []
    out = []
    for f in sorted(p for p in root.iterdir() if p.is_dir()):
        n = sum(1 for p in f.iterdir() if p.suffix.lower() in IMG_EXT | VID_EXT)
        out.append((f, n))
    return out


class Worker(QThread):
    """Capture + embedding loop. Plays the source even when references are not ready (preview mode)."""
    frame = Signal(object)                   # raw BGR frame
    loaded = Signal(list)                    # state names after (re)loading references
    result = Signal(object, int, str)        # similarities, current state index, status text
    position = Signal(int, int)              # frame index, total frames (video files only)
    log = Signal(str)
    error = Signal(str)

    def __init__(self, refs, source, model, params):
        super().__init__()
        self.refs, self.source, self.model = refs, source, model
        self.params = params                 # dict mutated live from the GUI thread
        self.paused = False
        self.seek_to = None
        self._stop = False
        self._reload = False

    def stop(self):
        self._stop = True

    def request_reload(self):
        self._reload = True

    def run(self):
        try:
            self._run()
        except Exception as e:  # surface anything (no camera, bad model) in the GUI
            self.error.emit(str(e))

    def _load_refs(self, embed):
        """(names, protos), or None (with the reason logged) if the references are not usable yet."""
        root = Path(self.refs)
        folders = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        if len(folders) < 2:
            self.log.emit("Preview only: add at least 2 states (start + step 1) to begin tracking.")
            return None
        names, protos = [], []
        for f in folders:
            frames = load_reference_frames(f, self.params["roi"])
            if not frames:
                self.log.emit(f"Preview only: state {f.name} has no pictures yet.")
                return None
            names.append(f.name)
            protos.append(embed(frames))
            self.log.emit(f"loaded {f.name}: {len(frames)} reference frames")
        return names, protos

    def _make_tracker(self, names, old_current=0):
        p = self.params
        tracker = StepTracker(names, p["window"], p["margin"], p["confirm"])
        tracker.current = min(old_current, len(names) - 1)
        return tracker

    def _run(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log.emit(f"Loading model on {device}...")
        embed = Embedder(self.model, device)
        refs = self._load_refs(embed)
        names, protos = refs if refs else ([], [])
        tracker = self._make_tracker(names) if refs else None
        if refs:
            self.loaded.emit(names)

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source {self.source!r}")
        p = self.params
        is_file = isinstance(self.source, str)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
        frame_time = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.log.emit("Tracking started" if refs else "Video started (preview)")

        i, t0, ended = 0, time.time(), False
        while not self._stop:
            t_frame = time.time()
            if self._reload:
                self._reload = False
                self.log.emit("Reloading references...")
                refs = self._load_refs(embed)
                old = tracker.current if tracker else 0
                names, protos = refs if refs else ([], [])
                tracker = self._make_tracker(names, old) if refs else None
                if refs:
                    self.loaded.emit(names)
            if self.seek_to is not None:        # user dragged the slider
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_to)
                self.seek_to, ended = None, False
                ok, frame = cap.read()
                if ok:
                    self.frame.emit(frame)
                    self.position.emit(int(cap.get(cv2.CAP_PROP_POS_FRAMES)), total)
                continue
            if self.paused or ended:
                time.sleep(0.03)
                continue
            ok, frame = cap.read()
            if not ok:
                if is_file:                     # stay open so the user can scrub back and capture
                    ended = True
                    self.log.emit("Video ended. Drag the slider to go back, or press Stop.")
                    continue
                break
            self.frame.emit(frame)
            if is_file:
                self.position.emit(int(cap.get(cv2.CAP_PROP_POS_FRAMES)), total)
            i += 1
            if tracker and i % p["every"] == 0:
                tracker.confirm, tracker.margin = p["confirm"], p["margin"]
                if tracker.window.maxlen != p["window"]:
                    tracker.window = deque(tracker.window, maxlen=p["window"])
                e = embed([crop(frame, p["roi"])])[0]
                sims = np.array([float((pr @ e).max()) for pr in protos])
                status, avg, _ = tracker.update(sims)
                if status != f"state: {names[tracker.current]}":
                    self.log.emit(f"[{time.time() - t0:6.1f}s] {status}")
                self.result.emit(avg, tracker.current, status)
            if is_file:  # play recordings at their native speed
                time.sleep(max(0.0, frame_time - (time.time() - t_frame)))
        cap.release()
        if tracker:
            self.log.emit(f"Stopped. Final state: {names[tracker.current]} ({tracker.current}/{len(names) - 1} steps)")
        else:
            self.log.emit("Stopped.")


class VideoView(QWidget):
    """Shows the live frame, the crop region, and lets the user drag a new crop region."""
    roi_selected = Signal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 320)
        self.image = None
        self.roi = None
        self.selecting = False
        self._drag = None
        self.setMouseTracking(True)

    def set_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        self.image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.update()

    def set_selecting(self, on):
        self.selecting = on
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def _target(self):
        iw, ih = self.image.width(), self.image.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h), s

    def _to_image(self, pos):
        rect, s = self._target()
        x = min(max((pos.x() - rect.x()) / s, 0), self.image.width())
        y = min(max((pos.y() - rect.y()) / s, 0), self.image.height())
        return QPointF(x, y)

    def _to_widget(self, r):
        rect, s = self._target()
        return QRectF(rect.x() + r[0] * s, rect.y() + r[1] * s, r[2] * s, r[3] * s)

    def mousePressEvent(self, ev):
        if self.selecting and self.image is not None and ev.button() == Qt.LeftButton:
            p = self._to_image(ev.position())
            self._drag = (p, p)

    def mouseMoveEvent(self, ev):
        if self._drag:
            self._drag = (self._drag[0], self._to_image(ev.position()))
            self.update()

    def mouseReleaseEvent(self, ev):
        if not self._drag:
            return
        a, b = self._drag
        self._drag = None
        x, y = int(min(a.x(), b.x())), int(min(a.y(), b.y()))
        w, h = int(abs(a.x() - b.x())), int(abs(a.y() - b.y()))
        self.set_selecting(False)
        if w >= 16 and h >= 16:
            self.roi = (x, y, w, h)
            self.roi_selected.emit(self.roi)
        else:
            self.roi_selected.emit(self.roi)  # too small: keep what we had
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        if self.image is None:
            p.setPen(QColor(150, 150, 150))
            p.drawText(self.rect(), Qt.AlignCenter, "No video. Press Start.")
            return
        rect, _s = self._target()
        p.drawImage(rect, self.image)
        pen = QPen(QColor(255, 200, 0), 2, Qt.DashLine)
        if self._drag:
            a, b = self._drag
            r = (min(a.x(), b.x()), min(a.y(), b.y()), abs(a.x() - b.x()), abs(a.y() - b.y()))
            self._paint_roi(p, self._to_widget(r), rect, pen)
        elif self.roi:
            self._paint_roi(p, self._to_widget(self.roi), rect, pen)

    @staticmethod
    def _paint_roi(p, r, bounds, pen):
        # dim everything outside the crop region so it is obvious what the tracker sees
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 110)))
        p.drawRect(QRectF(bounds.left(), bounds.top(), bounds.width(), r.top() - bounds.top()))
        p.drawRect(QRectF(bounds.left(), r.bottom(), bounds.width(), bounds.bottom() - r.bottom()))
        p.drawRect(QRectF(bounds.left(), r.top(), r.left() - bounds.left(), r.height()))
        p.drawRect(QRectF(r.right(), r.top(), bounds.right() - r.right(), r.height()))
        p.setBrush(Qt.NoBrush)
        p.setPen(pen)
        p.drawRect(r)


class MainWindow(QMainWindow):
    def __init__(self, refs, source, model):
        super().__init__()
        self.setWindowTitle("Step Tracker")
        self.resize(1280, 760)
        self.model = model
        self.worker = None
        self.last_frame = None
        self.has_event = False
        self.params = {"roi": None, "confirm": 8, "margin": 0.02, "window": 5, "every": 2}

        # --- top bar -------------------------------------------------------------------------
        self.refs_edit = QLineEdit(refs)
        self.refs_edit.editingFinished.connect(self.refresh_states)
        refs_btn = QPushButton("Browse...")
        refs_btn.clicked.connect(self.browse_refs)
        self.source_edit = QLineEdit(source)
        self.source_edit.setToolTip("Camera index (0, 1, ...) or a video file path")
        self.source_edit.setMaximumWidth(260)
        src_btn = QPushButton("Video file...")
        src_btn.clicked.connect(self.browse_source)
        self.start_btn = QPushButton("Start")
        self.start_btn.setMinimumWidth(90)
        self.start_btn.clicked.connect(self.toggle)
        top = QHBoxLayout()
        for w in (QLabel("References:"), self.refs_edit, refs_btn, QLabel("Source:"), self.source_edit,
                  src_btn, self.start_btn):
            top.addWidget(w)
        top.setStretch(1, 1)

        # --- video side ----------------------------------------------------------------------
        self.video = VideoView()
        self.video.roi_selected.connect(self.on_roi_selected)
        self.status = QLabel("Idle")
        f = QFont()
        f.setPointSize(16)
        f.setBold(True)
        self.status.setFont(f)
        self.status.setMinimumHeight(40)
        self.status.setAlignment(Qt.AlignCenter)
        self.roi_btn = QPushButton("Select crop region")
        self.roi_btn.setCheckable(True)
        self.roi_btn.toggled.connect(self.toggle_roi_select)
        self.roi_clear_btn = QPushButton("Clear crop")
        self.roi_clear_btn.clicked.connect(lambda: self.set_roi(None))
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setToolTip("Freeze the video so you can capture the exact frame you want")
        self.pause_btn.toggled.connect(self.on_pause)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setToolTip("Scrub through a video file (pauses it)")
        self.slider.sliderMoved.connect(self.on_seek)
        self.pos_label = QLabel("")
        play_row = QHBoxLayout()
        play_row.addWidget(self.pause_btn)
        play_row.addWidget(self.slider, 1)
        play_row.addWidget(self.pos_label)
        roi_row = QHBoxLayout()
        roi_row.addWidget(self.roi_btn)
        roi_row.addWidget(self.roi_clear_btn)
        roi_row.addStretch(1)
        left = QWidget()
        lay = QVBoxLayout(left)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.video, 1)
        lay.addWidget(self.status)
        lay.addLayout(play_row)
        lay.addLayout(roi_row)

        # --- states side ---------------------------------------------------------------------
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["State", "Refs", "Similarity"])
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Fixed)
        self.table.setColumnWidth(2, 130)

        add_state = QPushButton("Add state")
        add_state.clicked.connect(self.add_state)
        add_files = QPushButton("Add files...")
        add_files.setToolTip("Copy images/videos into the selected state")
        add_files.clicked.connect(self.add_files)
        capture = QPushButton("Capture frame")
        capture.setToolTip("Save the current video frame as a reference for the selected state")
        capture.clicked.connect(self.capture_frame)
        open_dir = QPushButton("Open folder")
        open_dir.clicked.connect(self.open_folder)
        btns = QHBoxLayout()
        for b in (add_state, add_files, capture, open_dir):
            btns.addWidget(b)

        self.confirm_sb = QSpinBox()
        self.confirm_sb.setRange(1, 200)
        self.confirm_sb.setValue(self.params["confirm"])
        self.confirm_sb.setToolTip("Consecutive frames a state must match before a step counts")
        self.margin_sb = QDoubleSpinBox()
        self.margin_sb.setRange(0.0, 1.0)
        self.margin_sb.setDecimals(3)
        self.margin_sb.setSingleStep(0.005)
        self.margin_sb.setValue(self.params["margin"])
        self.margin_sb.setToolTip("Required similarity lead over the second-best state")
        self.window_sb = QSpinBox()
        self.window_sb.setRange(1, 100)
        self.window_sb.setValue(self.params["window"])
        self.window_sb.setToolTip("Frames of smoothing")
        self.every_sb = QSpinBox()
        self.every_sb.setRange(1, 30)
        self.every_sb.setValue(self.params["every"])
        self.every_sb.setToolTip("Process every Nth frame")
        for sb in (self.confirm_sb, self.margin_sb, self.window_sb, self.every_sb):
            sb.valueChanged.connect(self.update_params)
        tuning = QGroupBox("Tuning (applies live)")
        form = QFormLayout(tuning)
        form.addRow("Confirm frames", self.confirm_sb)
        form.addRow("Margin", self.margin_sb)
        form.addRow("Smoothing window", self.window_sb)
        form.addRow("Process every Nth", self.every_sb)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(1000)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.table, 3)
        rl.addLayout(btns)
        rl.addWidget(tuning)
        rl.addWidget(self.log_box, 2)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        central = QWidget()
        cl = QVBoxLayout(central)
        cl.addLayout(top)
        cl.addWidget(split, 1)
        self.setCentralWidget(central)

        self.bars = []
        self.refresh_states()

    # --- helpers ---------------------------------------------------------------------------------
    def log(self, text):
        self.log_box.appendPlainText(text)

    def set_status(self, text, color=None):
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {color.name()};" if color else "")

    def refs_dir(self):
        return Path(self.refs_edit.text().strip() or "refs")

    def running(self):
        return self.worker is not None and self.worker.isRunning()

    def selected_folder(self):
        row = self.table.currentRow()
        states = scan_refs(self.refs_dir())
        if row < 0 or row >= len(states):
            QMessageBox.information(self, "Select a state", "Select a state in the table first.")
            return None
        return states[row][0]

    def reload_if_running(self):
        if self.running():
            self.worker.request_reload()

    # --- state table -----------------------------------------------------------------------------
    def refresh_states(self, keep_row=None):
        states = scan_refs(self.refs_dir())
        self.table.setRowCount(len(states))
        self.bars = []
        for r, (folder, n) in enumerate(states):
            self.table.setItem(r, 0, QTableWidgetItem(folder.name))
            item = QTableWidgetItem(str(n))
            item.setTextAlignment(Qt.AlignCenter)
            if n == 0:
                item.setForeground(QBrush(AMBER))
            self.table.setItem(r, 1, item)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("-")
            self.table.setCellWidget(r, 2, bar)
            self.bars.append(bar)
        if keep_row is not None and keep_row < len(states):
            self.table.selectRow(keep_row)
        self.mark_progress(-1)

    def mark_progress(self, current):
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            font = item.font()
            font.setBold(r == current)
            for c in (0, 1):
                it = self.table.item(r, c)
                it.setFont(font)
                it.setBackground(QBrush(QColor(GREEN.red(), GREEN.green(), GREEN.blue(), 70)) if 0 <= r <= current
                                 else QBrush())

    def on_loaded(self, names):
        self.refresh_states(self.table.currentRow() if self.table.currentRow() >= 0 else None)

    def on_result(self, avg, current, status):
        for bar, s in zip(self.bars, avg):
            bar.setValue(int(max(0.0, float(s)) * 100))
            bar.setFormat(f"{float(s):.2f}")
        self.mark_progress(current)
        if status.startswith("STEP"):
            self.has_event = True
            self.set_status(status, GREEN)
        elif status.startswith("WARNING"):
            self.has_event = True
            self.set_status(status, AMBER)
        elif not self.has_event:  # once a step/warning fired, keep showing it until the next one
            self.set_status(status)

    # --- reference management --------------------------------------------------------------------
    def browse_refs(self):
        d = QFileDialog.getExistingDirectory(self, "Reference folder", str(self.refs_dir().resolve()))
        if d:
            self.refs_edit.setText(d)
            self.refresh_states()

    def add_state(self):
        name, ok = QInputDialog.getText(self, "Add state", "Name of the new state (after the last one):")
        name = re.sub(r"[^\w\-]+", "_", name.strip()).strip("_")
        if not ok or not name:
            return
        root = self.refs_dir()
        root.mkdir(parents=True, exist_ok=True)
        nums = [int(m.group(1)) for f, _ in scan_refs(root) if (m := re.match(r"(\d+)_", f.name))]
        folder = root / f"{(max(nums) + 1) if nums else 0:02d}_{name}"
        folder.mkdir()
        self.refresh_states(keep_row=len(scan_refs(root)) - 1)
        self.reload_if_running()
        self.log(f"Created {folder}")

    def add_files(self):
        folder = self.selected_folder()
        if folder is None:
            return
        exts = " ".join(f"*{e}" for e in sorted(IMG_EXT | VID_EXT))
        files, _ = QFileDialog.getOpenFileNames(self, f"Add to {folder.name}", "", f"Images/videos ({exts})")
        for f in files:
            shutil.copy2(f, folder / Path(f).name)
        if files:
            self.log(f"Added {len(files)} file(s) to {folder.name}")
            self.refresh_states(keep_row=self.table.currentRow())
            self.reload_if_running()

    def capture_frame(self):
        folder = self.selected_folder()
        if folder is None:
            return
        if self.last_frame is None:
            QMessageBox.information(self, "No frame", "Start the source first so there is a frame to capture.")
            return
        path = folder / f"cap_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        cv2.imwrite(str(path), self.last_frame)
        self.log(f"Captured {folder.name}/{path.name}")
        self.refresh_states(keep_row=self.table.currentRow())
        self.reload_if_running()

    def open_folder(self):
        row = self.table.currentRow()
        states = scan_refs(self.refs_dir())
        target = states[row][0] if 0 <= row < len(states) else self.refs_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))

    # --- crop region -----------------------------------------------------------------------------
    def toggle_roi_select(self, on):
        if on and self.video.image is None:
            frame = self.grab_one_frame()
            if frame is None:
                self.roi_btn.setChecked(False)
                QMessageBox.information(self, "No video", "Could not read a frame from the source.")
                return
            self.last_frame = frame
            self.video.set_frame(frame)
        self.video.set_selecting(on)
        if on:
            self.log("Drag a rectangle on the video around the unit.")

    def grab_one_frame(self):
        cap = cv2.VideoCapture(parse_source(self.source_edit.text()))
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None

    def on_roi_selected(self, roi):
        self.roi_btn.setChecked(False)
        self.set_roi(roi)

    def set_roi(self, roi):
        changed = roi != self.params["roi"]
        self.params["roi"] = roi
        self.video.roi = roi
        self.video.update()
        if changed:
            self.log(f"Crop region: {roi}" if roi else "Crop cleared")
            self.reload_if_running()

    # --- run control -----------------------------------------------------------------------------
    def update_params(self):
        self.params.update(confirm=self.confirm_sb.value(), margin=self.margin_sb.value(),
                           window=self.window_sb.value(), every=self.every_sb.value())

    def toggle(self):
        if self.running():
            self.worker.stop()
            self.start_btn.setEnabled(False)
            self.start_btn.setText("Stopping...")
        else:
            self.start()

    def start(self):
        self.update_params()
        self.has_event = False
        self.set_status("Loading...")
        self.worker = Worker(str(self.refs_dir()), parse_source(self.source_edit.text()), self.model, self.params)
        self.worker.frame.connect(self.on_frame)
        self.worker.position.connect(self.on_position)
        self.slider.setEnabled(isinstance(self.worker.source, str))
        self.pause_btn.setChecked(False)
        self.worker.loaded.connect(self.on_loaded)
        self.worker.result.connect(self.on_result)
        self.worker.log.connect(self.log)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.start_btn.setText("Stop")
        self.worker.start()

    def on_pause(self, on):
        self.pause_btn.setText("Resume" if on else "Pause")
        if self.worker:
            self.worker.paused = on

    def on_seek(self, value):
        if self.worker:
            self.pause_btn.setChecked(True)
            self.worker.seek_to = value

    def on_position(self, pos, total):
        if total > 0 and not self.slider.isSliderDown():
            self.slider.setRange(0, total - 1)
            self.slider.setValue(pos)
            self.pos_label.setText(f"{pos}/{total}")

    def on_frame(self, frame):
        self.last_frame = frame
        self.video.set_frame(frame)

    def on_error(self, msg):
        self.log(f"ERROR: {msg}")
        self.set_status("Error", AMBER)
        QMessageBox.warning(self, "Step Tracker", msg)

    def on_finished(self):
        self.start_btn.setText("Start")
        self.start_btn.setEnabled(True)
        self.slider.setEnabled(False)
        if self.status.text() in ("Loading...",):
            self.set_status("Idle")

    def browse_source(self):
        exts = " ".join(f"*{e}" for e in sorted(VID_EXT))
        f, _ = QFileDialog.getOpenFileName(self, "Video file", "", f"Videos ({exts})")
        if f:
            self.source_edit.setText(f)

    def closeEvent(self, ev):
        if self.running():
            self.worker.stop()
            self.worker.wait(5000)
        ev.accept()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="refs")
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--model", default="facebook/dinov2-small")
    args = ap.parse_args()
    app = QApplication(sys.argv)
    win = MainWindow(args.refs, args.source, args.model)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
