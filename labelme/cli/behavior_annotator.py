#!/usr/bin/env python3
"""
SmarterLabelMe Behaviour Annotator (MVP)

Two first-class input modes:
  1) Tracklet video mode:
       Load an already-cropped tracklet video and label temporal behaviour.
  2) Full video + SmarterLabelMe mode:
       Load the original video plus a directory of LabelMe/SmarterLabelMe JSON
       annotations. Select an individual track label; the GUI dynamically crops
       around that animal and linearly interpolates bounding boxes between
       annotated frames.

Design goals:
  - fast continuous temporal annotation with behaviour buttons / number keys
  - frame-accurate labels stored as start_frame/end_frame intervals
  - play/pause, frame stepping, seeking, playback speed
  - timeline editing, boundary dragging, relabel selected interval
  - undo
  - crop/full-scene toggle in full-video mode
  - save/load behaviour JSON
  - next/previous tracklet or track

Recommended invocation from the repository:
  python /home/aamir/Work/Animal-Behaviour-Inference-Framework/labelme/cli/behavior_annotator.py

Dependencies:
  qtpy, a Qt binding supported by SmarterLabelMe (normally PyQt5), opencv-python[-headless], numpy
"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".webm"}
DEFAULT_BEHAVIORS = [
    "grazing",
    "standing",
    "walking",
    "running",
    "lying",
    "other",
    "uncertain",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    start: int
    end: int
    behavior: str

    def to_dict(self) -> dict:
        return {
            "start_frame": int(self.start),
            "end_frame": int(self.end),
            "behavior": self.behavior,
        }


@dataclass
class TrackSample:
    frame: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    json_path: str
    image_path: str = ""
    timestamp_ms: Optional[int] = None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_segments(segments: Sequence[Segment], total_frames: int) -> List[Segment]:
    """Sort, clamp and merge adjacent equal labels. Gaps are preserved."""
    if total_frames <= 0:
        return []
    cleaned: List[Segment] = []
    for s in sorted(segments, key=lambda x: (x.start, x.end)):
        start = clamp(int(s.start), 0, total_frames - 1)
        end = clamp(int(s.end), 0, total_frames - 1)
        if end < start:
            start, end = end, start
        if not s.behavior:
            continue

        if cleaned and start <= cleaned[-1].end:
            # Resolve accidental overlap by beginning after the previous segment.
            start = cleaned[-1].end + 1
            if start > end:
                continue

        if (
            cleaned
            and cleaned[-1].behavior == s.behavior
            and cleaned[-1].end + 1 == start
        ):
            cleaned[-1].end = end
        else:
            cleaned.append(Segment(start, end, s.behavior))
    return cleaned


def set_behavior_from_frame(
    segments: Sequence[Segment],
    frame: int,
    behavior: str,
    total_frames: int,
) -> List[Segment]:
    """
    Continuous-mode annotation primitive:
    set `behavior` from `frame` onward until the next existing change point,
    or to the end of the video.

    If frame falls inside an existing segment, that segment is split.
    If a later segment already starts after frame, it remains intact.
    """
    if total_frames <= 0:
        return []
    frame = clamp(int(frame), 0, total_frames - 1)
    src = normalize_segments(copy.deepcopy(list(segments)), total_frames)

    # Determine next boundary strictly after frame.
    future_starts = [s.start for s in src if s.start > frame]
    new_end = (min(future_starts) - 1) if future_starts else total_frames - 1

    out: List[Segment] = []
    for s in src:
        if s.end < frame or s.start > new_end:
            out.append(copy.deepcopy(s))
            continue

        # overlap with [frame, new_end]; preserve left/right pieces
        if s.start < frame:
            out.append(Segment(s.start, frame - 1, s.behavior))
        if s.end > new_end:
            out.append(Segment(new_end + 1, s.end, s.behavior))

    out.append(Segment(frame, new_end, behavior))
    return normalize_segments(out, total_frames)


def relabel_segment(
    segments: Sequence[Segment], index: int, behavior: str, total_frames: int
) -> List[Segment]:
    src = copy.deepcopy(list(segments))
    if 0 <= index < len(src):
        src[index].behavior = behavior
    return normalize_segments(src, total_frames)


# ---------------------------------------------------------------------------
# Video source
# ---------------------------------------------------------------------------

class VideoSource:
    def __init__(self, path: str):
        self.path = str(Path(path).expanduser().resolve())
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.fps = 25.0

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        self._last_index = -1
        self._last_frame: Optional[np.ndarray] = None

    def close(self):
        if self.cap is not None:
            self.cap.release()

    def read(self, frame_index: int) -> Optional[np.ndarray]:
        if self.total_frames > 0:
            frame_index = clamp(frame_index, 0, self.total_frames - 1)
        else:
            frame_index = max(0, frame_index)

        if frame_index == self._last_index and self._last_frame is not None:
            return self._last_frame.copy()

        # Sequential read is cheaper than seeking every frame.
        if frame_index == self._last_index + 1:
            ok, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = self.cap.read()

        if not ok or frame is None:
            return None

        self._last_index = frame_index
        self._last_frame = frame
        return frame.copy()


# ---------------------------------------------------------------------------
# SmarterLabelMe track loader
# ---------------------------------------------------------------------------

F_TIMELINE_RE = re.compile(r"(?:^|[^A-Za-z0-9])f(\d+)_t(\d+)", re.IGNORECASE)
FRAME_RE = re.compile(r"(?:frame[_-]?)(\d+)", re.IGNORECASE)
LAST_INTEGER_RE = re.compile(r"(\d+)(?!.*\d)")


def _shape_bbox(shape: dict) -> Optional[Tuple[float, float, float, float]]:
    pts = shape.get("points") or []
    if not pts:
        return None

    xs, ys = [], []
    for p in pts:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        try:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
        except Exception:
            pass
    if not xs or not ys:
        return None

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _parse_frame_identity(
    json_path: Path,
    data: dict,
    source_fps: float,
    annotation_fps: Optional[float],
    legacy_min_index: int = 0,
) -> Tuple[Optional[int], Optional[int], str]:
    """
    Returns (source_frame, timestamp_ms, image_path).

    Preferred modern format:
      fXXXXXXXX_tXXXXXXXXX.jpg
      -> exact source frame + timestamp.

    Legacy format:
      frame_000123.jpg
      If --annotation-fps is supplied, interpret the number as an index in the
      annotation/extraction stream and convert by time to source-video frame.
      Otherwise treat it directly as source-video frame/PTS index.
    """
    image_path = str(data.get("imagePath") or "")
    candidates = [Path(image_path).name if image_path else "", json_path.stem]

    for name in candidates:
        m = F_TIMELINE_RE.search(name)
        if m:
            return int(m.group(1)), int(m.group(2)), image_path

    legacy_num = None
    for name in candidates:
        m = FRAME_RE.search(name)
        if m:
            legacy_num = int(m.group(1))
            break

    if legacy_num is None:
        for name in candidates:
            m = LAST_INTEGER_RE.search(Path(name).stem)
            if m:
                legacy_num = int(m.group(1))
                break

    if legacy_num is None:
        return None, None, image_path

    if annotation_fps and annotation_fps > 0 and source_fps > 0:
        zero_based = max(0, legacy_num - legacy_min_index)
        seconds = zero_based / float(annotation_fps)
        source_frame = int(round(seconds * source_fps))
        return source_frame, int(round(seconds * 1000.0)), image_path

    return legacy_num, None, image_path


class SmarterLabelMeTracks:
    def __init__(
        self,
        annotation_dir: str,
        source_fps: float,
        source_total_frames: int,
        annotation_fps: Optional[float] = None,
    ):
        self.annotation_dir = str(Path(annotation_dir).expanduser().resolve())
        self.source_fps = source_fps
        self.source_total_frames = source_total_frames
        self.annotation_fps = annotation_fps
        self.tracks: Dict[str, List[TrackSample]] = {}
        self.errors: List[str] = []
        self._load()

    @staticmethod
    def _all_json_files(annotation_dir: Path) -> List[Path]:
        files = sorted(annotation_dir.glob("*.json"))
        if not files:
            files = sorted(annotation_dir.rglob("*.json"))
        return files

    def _load(self):
        root = Path(self.annotation_dir)
        files = self._all_json_files(root)
        if not files:
            raise RuntimeError(f"No JSON files found under: {root}")

        # For sampled legacy frame_N names, normalize sequence numbering from the
        # smallest observed value (commonly 0 or 1).
        legacy_indices = []
        for p in files:
            m = FRAME_RE.search(p.stem)
            if m:
                legacy_indices.append(int(m.group(1)))
        legacy_min = min(legacy_indices) if legacy_indices else 0

        for p in files:
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                self.errors.append(f"{p}: {e}")
                continue

            frame_idx, timestamp_ms, image_path = _parse_frame_identity(
                p, data, self.source_fps, self.annotation_fps, legacy_min
            )
            if frame_idx is None:
                self.errors.append(f"{p}: could not infer frame number")
                continue

            if self.source_total_frames > 0:
                frame_idx = clamp(frame_idx, 0, self.source_total_frames - 1)

            for shape in data.get("shapes") or []:
                label = str(shape.get("label") or "").strip()
                if not label:
                    continue
                bbox = _shape_bbox(shape)
                if bbox is None:
                    continue
                self.tracks.setdefault(label, []).append(
                    TrackSample(
                        frame=int(frame_idx),
                        bbox=bbox,
                        json_path=str(p),
                        image_path=image_path,
                        timestamp_ms=timestamp_ms,
                    )
                )

        # Deduplicate same track/frame (keep last) and sort.
        for label, samples in list(self.tracks.items()):
            by_frame: Dict[int, TrackSample] = {}
            for sample in samples:
                by_frame[sample.frame] = sample
            self.tracks[label] = [by_frame[k] for k in sorted(by_frame)]

        self.tracks = {
            label: samples
            for label, samples in self.tracks.items()
            if samples
        }

        if not self.tracks:
            raise RuntimeError(
                "JSON files were found, but no labeled shapes with valid bounding boxes "
                "could be read."
            )

    def labels(self) -> List[str]:
        return sorted(self.tracks.keys(), key=natural_key)

    def bounds(self, label: str) -> Tuple[int, int]:
        samples = self.tracks[label]
        return samples[0].frame, samples[-1].frame

    def interpolated_bbox(
        self, label: str, frame: int
    ) -> Optional[Tuple[float, float, float, float]]:
        samples = self.tracks.get(label)
        if not samples:
            return None

        frames = [s.frame for s in samples]
        pos = bisect.bisect_left(frames, frame)

        if pos == 0:
            return samples[0].bbox
        if pos >= len(samples):
            return samples[-1].bbox

        a, b = samples[pos - 1], samples[pos]
        if b.frame == a.frame:
            return a.bbox
        alpha = (frame - a.frame) / float(b.frame - a.frame)
        return tuple(
            float(a.bbox[i] + alpha * (b.bbox[i] - a.bbox[i]))
            for i in range(4)
        )


def natural_key(s: str):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", s)
    ]


# ---------------------------------------------------------------------------
# Timeline widget
# ---------------------------------------------------------------------------

class TimelineWidget(QtWidgets.QWidget):
    seekRequested = QtCore.Signal(int)
    segmentsChanged = QtCore.Signal()
    selectionChanged = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(105)
        self.setMouseTracking(True)
        self.total_frames = 1
        self.active_start = 0
        self.active_end = 0
        self.current_frame = 0
        self.item_start = 0
        self.item_end = 0
        self.segments: List[Segment] = []
        self.selected_index = -1
        self._drag_boundary: Optional[Tuple[int, int]] = None  # left idx, right idx
        self._palette: Dict[str, QtGui.QColor] = {}

    def sizeHint(self):
        return QtCore.QSize(700, 115)

    def set_total_frames(self, n: int):
        self.total_frames = max(1, int(n))
        self.active_start = 0
        self.active_end = self.total_frames - 1
        self.update()

    def set_active_range(self, start: int, end: int):
        start = clamp(int(start), 0, self.total_frames - 1)
        end = clamp(int(end), 0, self.total_frames - 1)
        if end < start:
            start, end = end, start
        self.active_start = start
        self.active_end = end
        self.current_frame = clamp(self.current_frame, start, end)
        self.update()

    def set_current_frame(self, frame: int):
        self.current_frame = clamp(int(frame), 0, self.total_frames - 1)
        self.update()

    def set_segments(self, segments: Sequence[Segment]):
        # Deep copy is deliberate: boundary dragging must not mutate the main
        # window's model until the edit is committed, otherwise Undo snapshots
        # would capture the already-modified state.
        self.segments = copy.deepcopy(list(segments))
        if self.selected_index >= len(self.segments):
            self.selected_index = -1
        self.update()

    def clear_selection(self):
        if self.selected_index != -1:
            self.selected_index = -1
            self.selectionChanged.emit(-1)
            self.update()

    def _track_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(12, 34, max(10, self.width() - 24), 45)

    def _frame_to_x(self, frame: int) -> float:
        r = self._track_rect()
        span = self.active_end - self.active_start
        if span <= 0:
            return r.left()
        alpha = (frame - self.active_start) / float(span)
        alpha = clamp(alpha, 0.0, 1.0)
        return r.left() + alpha * r.width()

    def _x_to_frame(self, x: float) -> int:
        r = self._track_rect()
        if r.width() <= 0:
            return self.active_start
        alpha = clamp((x - r.left()) / r.width(), 0.0, 1.0)
        return int(round(self.active_start + alpha * (self.active_end - self.active_start)))

    def _color_for(self, label: str) -> QtGui.QColor:
        if label not in self._palette:
            # Deterministic, visually distinct HSV colour without maintaining a
            # hard-coded taxonomy palette.
            h = abs(hash(label)) % 360
            self._palette[label] = QtGui.QColor.fromHsv(h, 150, 225)
        return self._palette[label]

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        r = self._track_rect()
        p.fillRect(r, self.palette().alternateBase())

        # tick labels
        p.setPen(self.palette().text().color())
        duration_s = max(0, self.active_end - self.active_start)
        parent = self.window()
        fps = getattr(parent, "video_fps", 25.0) or 25.0
        duration_s /= fps

        ticks = 6
        for i in range(ticks):
            a = i / float(ticks - 1)
            x = r.left() + a * r.width()
            sec = a * duration_s
            p.drawLine(QtCore.QPointF(x, r.top() - 5), QtCore.QPointF(x, r.top()))
            p.drawText(
                QtCore.QRectF(x - 35, 4, 70, 22),
                QtCore.Qt.AlignCenter,
                f"{sec:.1f}s",
            )

        # segments
        for i, s in enumerate(self.segments):
            x1 = self._frame_to_x(s.start)
            x2 = self._frame_to_x(s.end)
            if s.end >= s.start:
                x2 = max(x2, x1 + 2)
            sr = QtCore.QRectF(x1, r.top() + 2, max(2, x2 - x1), r.height() - 4)
            p.fillRect(sr, self._color_for(s.behavior))
            p.setPen(QtGui.QPen(QtGui.QColor(40, 40, 40), 1))
            p.drawRect(sr)

            if i == self.selected_index:
                p.setPen(QtGui.QPen(self.palette().highlight().color(), 3))
                p.drawRect(sr.adjusted(1, 1, -1, -1))

            if sr.width() > 42:
                p.setPen(QtGui.QColor(20, 20, 20))
                text = p.fontMetrics().elidedText(
                    s.behavior, QtCore.Qt.ElideRight, int(sr.width() - 8)
                )
                p.drawText(sr.adjusted(4, 0, -4, 0), QtCore.Qt.AlignCenter, text)

        # playhead
        x = self._frame_to_x(self.current_frame)
        p.setPen(QtGui.QPen(QtGui.QColor(220, 30, 30), 2))
        p.drawLine(QtCore.QPointF(x, r.top() - 8), QtCore.QPointF(x, r.bottom() + 7))

        # frame number
        p.setPen(self.palette().text().color())
        p.drawText(
            QtCore.QRectF(12, 82, self.width() - 24, 20),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            f"source frame {self.current_frame}   |   active range {self.active_start}–{self.active_end}",
        )

    def _segment_at(self, frame: int) -> int:
        for i, s in enumerate(self.segments):
            if s.start <= frame <= s.end:
                return i
        return -1

    def _nearest_internal_boundary(self, x: float, tolerance_px: float = 8.0):
        if len(self.segments) < 2:
            return None
        best = None
        best_dist = tolerance_px + 1
        for i in range(len(self.segments) - 1):
            left, right = self.segments[i], self.segments[i + 1]
            if left.end + 1 != right.start:
                continue
            bx = self._frame_to_x(right.start)
            d = abs(x - bx)
            if d <= tolerance_px and d < best_dist:
                best = (i, i + 1)
                best_dist = d
        return best

    def mousePressEvent(self, e):
        if e.button() != QtCore.Qt.LeftButton:
            return
        boundary = self._nearest_internal_boundary(e.position().x() if hasattr(e, "position") else e.x())
        if boundary is not None:
            self._drag_boundary = boundary
            return

        x = e.position().x() if hasattr(e, "position") else e.x()
        frame = self._x_to_frame(x)
        idx = self._segment_at(frame)
        self.selected_index = idx
        self.selectionChanged.emit(idx)
        self.seekRequested.emit(frame)
        self.update()

    def mouseMoveEvent(self, e):
        x = e.position().x() if hasattr(e, "position") else e.x()
        if self._drag_boundary is None:
            if self._nearest_internal_boundary(x) is not None:
                self.setCursor(QtCore.Qt.SplitHCursor)
            else:
                self.setCursor(QtCore.Qt.ArrowCursor)
            return

        li, ri = self._drag_boundary
        if not (0 <= li < len(self.segments) and 0 <= ri < len(self.segments)):
            return
        left, right = self.segments[li], self.segments[ri]
        frame = self._x_to_frame(x)
        min_start = left.start + 1
        max_start = right.end
        new_right_start = clamp(frame, min_start, max_start)
        left.end = new_right_start - 1
        right.start = new_right_start
        self.update()

    def mouseReleaseEvent(self, e):
        if self._drag_boundary is not None:
            self._drag_boundary = None
            self.segmentsChanged.emit()
            self.update()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class BehaviorAnnotator(QtWidgets.QMainWindow):
    def __init__(
        self,
        behaviors: Sequence[str],
        annotation_fps: Optional[float] = None,
        crop_margin: float = 0.40,
        output_dir: Optional[str] = None,
    ):
        super().__init__()
        self.setWindowTitle("SmarterLabelMe Behaviour Annotator")
        self.resize(1180, 820)

        self.behaviors = list(dict.fromkeys([x.strip() for x in behaviors if x.strip()]))
        if not self.behaviors:
            self.behaviors = DEFAULT_BEHAVIORS[:]

        self.annotation_fps = annotation_fps
        self.crop_margin = max(0.0, float(crop_margin))
        self.explicit_output_dir = output_dir

        self.mode: Optional[str] = None  # "tracklet" | "full"
        self.video: Optional[VideoSource] = None
        self.video_fps = 25.0
        self.current_frame = 0
        self.item_start = 0
        self.item_end = 0
        self.segments: List[Segment] = []
        self.undo_stack: List[List[Segment]] = []
        self.output_path: Optional[Path] = None

        self.tracklet_files: List[Path] = []
        self.tracklet_index = -1

        self.track_db: Optional[SmarterLabelMeTracks] = None
        self.track_labels: List[str] = []
        self.track_index = -1
        self.current_track: Optional[str] = None
        self.annotation_dir: Optional[str] = None

        self.crop_view = True
        self.playback_speed = 1.0
        self.is_playing = False

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._play_tick)

        self._build_ui()
        self._build_shortcuts()
        self._update_enabled_state()

        QtCore.QTimer.singleShot(0, self._show_start_dialog)

    # ---- UI construction -------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Top line
        top = QtWidgets.QHBoxLayout()
        self.source_label = QtWidgets.QLabel("No video loaded")
        self.source_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        top.addWidget(self.source_label, 1)

        self.track_combo = QtWidgets.QComboBox()
        self.track_combo.setMinimumWidth(180)
        self.track_combo.currentIndexChanged.connect(self._track_combo_changed)
        top.addWidget(self.track_combo)

        self.crop_button = QtWidgets.QPushButton("Show full scene")
        self.crop_button.clicked.connect(self._toggle_crop_view)
        top.addWidget(self.crop_button)
        root.addLayout(top)

        # Video display
        self.video_label = QtWidgets.QLabel("Open a tracklet or full video to begin")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 400)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.video_label.setStyleSheet("QLabel { background: #111; color: #ddd; }")
        root.addWidget(self.video_label, 1)

        # Transport
        transport = QtWidgets.QHBoxLayout()

        self.prev_item_btn = QtWidgets.QPushButton("◀ Prev item")
        self.prev_item_btn.clicked.connect(lambda: self._change_item(-1))
        transport.addWidget(self.prev_item_btn)

        self.step_back_btn = QtWidgets.QPushButton("← frame")
        self.step_back_btn.clicked.connect(lambda: self._step(-1))
        transport.addWidget(self.step_back_btn)

        self.play_btn = QtWidgets.QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        transport.addWidget(self.play_btn)

        self.step_fwd_btn = QtWidgets.QPushButton("frame →")
        self.step_fwd_btn.clicked.connect(lambda: self._step(1))
        transport.addWidget(self.step_fwd_btn)

        self.next_item_btn = QtWidgets.QPushButton("Next item ▶")
        self.next_item_btn.clicked.connect(lambda: self._change_item(1))
        transport.addWidget(self.next_item_btn)

        transport.addSpacing(15)
        transport.addWidget(QtWidgets.QLabel("Speed:"))
        self.speed_combo = QtWidgets.QComboBox()
        for s in [0.25, 0.5, 1.0, 1.5, 2.0]:
            self.speed_combo.addItem(f"{s:g}×", s)
        self.speed_combo.setCurrentText("1×")
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)
        transport.addWidget(self.speed_combo)

        transport.addStretch(1)
        self.time_label = QtWidgets.QLabel("00:00.000 / 00:00.000")
        transport.addWidget(self.time_label)
        root.addLayout(transport)

        # Timeline
        self.timeline = TimelineWidget()
        self.timeline.seekRequested.connect(self._seek)
        self.timeline.segmentsChanged.connect(self._timeline_segments_edited)
        self.timeline.selectionChanged.connect(self._timeline_selection_changed)
        root.addWidget(self.timeline)

        # Behaviour buttons
        behavior_group = QtWidgets.QGroupBox(
            "Behaviours — press number key at the instant the behaviour changes"
        )
        behavior_layout = QtWidgets.QGridLayout(behavior_group)
        self.behavior_buttons: List[QtWidgets.QPushButton] = []

        for i, behavior in enumerate(self.behaviors):
            shortcut = str(i + 1) if i < 9 else ""
            title = f"{shortcut}  {behavior}" if shortcut else behavior
            btn = QtWidgets.QPushButton(title)
            btn.setMinimumHeight(42)
            btn.clicked.connect(lambda checked=False, b=behavior: self._behavior_pressed(b))
            self.behavior_buttons.append(btn)
            behavior_layout.addWidget(btn, i // 4, i % 4)

        root.addWidget(behavior_group)

        # Bottom controls
        bottom = QtWidgets.QHBoxLayout()
        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.undo_btn.clicked.connect(self._undo)
        bottom.addWidget(self.undo_btn)

        self.clear_selection_btn = QtWidgets.QPushButton("Clear selection")
        self.clear_selection_btn.clicked.connect(self.timeline.clear_selection)
        bottom.addWidget(self.clear_selection_btn)

        self.save_btn = QtWidgets.QPushButton("Save annotations")
        self.save_btn.clicked.connect(self._save)
        bottom.addWidget(self.save_btn)

        self.open_tracklet_btn = QtWidgets.QPushButton("Open tracklet…")
        self.open_tracklet_btn.clicked.connect(self._choose_tracklet)
        bottom.addWidget(self.open_tracklet_btn)

        self.open_full_btn = QtWidgets.QPushButton("Open full video + JSONs…")
        self.open_full_btn.clicked.connect(self._choose_full_mode)
        bottom.addWidget(self.open_full_btn)

        bottom.addStretch(1)
        self.status_label = QtWidgets.QLabel("")
        bottom.addWidget(self.status_label)
        root.addLayout(bottom)

        self.statusBar().showMessage(
            "Space: play/pause | ←/→: frame | 1…9: behaviours | Ctrl+S: save | Ctrl+Z: undo"
        )

    def _build_shortcuts(self):
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self._toggle_play)
        QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self, activated=lambda: self._step(-1))
        QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self, activated=lambda: self._step(1))
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, activated=self._save)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self, activated=self._undo)
        QtWidgets.QShortcut(QtGui.QKeySequence("N"), self, activated=lambda: self._change_item(1))
        QtWidgets.QShortcut(QtGui.QKeySequence("P"), self, activated=lambda: self._change_item(-1))

        for i, behavior in enumerate(self.behaviors[:9]):
            QtWidgets.QShortcut(
                QtGui.QKeySequence(str(i + 1)),
                self,
                activated=lambda b=behavior: self._behavior_pressed(b),
            )

    # ---- opening sources -------------------------------------------------

    def _show_start_dialog(self):
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Open behaviour annotation source")
        box.setText("Choose the source type.")
        tracklet = box.addButton("Tracklet video", QtWidgets.QMessageBox.AcceptRole)
        full = box.addButton("Full video + SmarterLabelMe JSONs", QtWidgets.QMessageBox.AcceptRole)
        box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.exec_()
        clicked = box.clickedButton()
        if clicked == tracklet:
            self._choose_tracklet()
        elif clicked == full:
            self._choose_full_mode()

    def _choose_tracklet(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open tracklet video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.mpg *.mpeg *.webm);;All files (*)",
        )
        if path:
            self.open_tracklet(path)

    def open_tracklet(self, path: str):
        p = Path(path).expanduser().resolve()
        siblings = sorted(
            [x for x in p.parent.iterdir() if x.is_file() and x.suffix.lower() in VIDEO_EXTS],
            key=lambda x: natural_key(x.name),
        )
        self.tracklet_files = siblings
        self.tracklet_index = siblings.index(p) if p in siblings else 0
        self._open_tracklet_at_index(self.tracklet_index)

    def _open_tracklet_at_index(self, index: int):
        if not self.tracklet_files:
            return
        index = clamp(index, 0, len(self.tracklet_files) - 1)
        self._maybe_save_before_switch()

        self.mode = "tracklet"
        self.tracklet_index = index
        p = self.tracklet_files[index]
        self._set_video(str(p))

        self.track_db = None
        self.track_labels = []
        self.current_track = None
        self.track_combo.blockSignals(True)
        self.track_combo.clear()
        self.track_combo.addItem(f"Tracklet {index + 1}/{len(self.tracklet_files)}")
        self.track_combo.blockSignals(False)
        self.track_combo.setEnabled(False)

        self.crop_view = False
        self.crop_button.setVisible(False)
        self.item_start = 0
        self.item_end = max(0, self.video.total_frames - 1)
        self.timeline.set_active_range(self.item_start, self.item_end)
        self.current_frame = self.item_start
        self._set_output_path_for_tracklet(p)
        self._load_existing_behavior_json()
        self.source_label.setText(f"Tracklet: {p.name}")
        self._update_enabled_state()
        self._show_current_frame()

    def _choose_full_mode(self):
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open original/source video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.mpg *.mpeg *.webm);;All files (*)",
        )
        if not video_path:
            return

        annotation_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose SmarterLabelMe annotation directory (e.g. Annotations)",
            str(Path(video_path).parent),
        )
        if not annotation_dir:
            return

        self.open_full(video_path, annotation_dir)

    def open_full(self, video_path: str, annotation_dir: str):
        self._maybe_save_before_switch()
        self.mode = "full"
        self._set_video(video_path)
        self.annotation_dir = str(Path(annotation_dir).expanduser().resolve())

        try:
            self.track_db = SmarterLabelMeTracks(
                self.annotation_dir,
                self.video_fps,
                self.video.total_frames if self.video else 0,
                self.annotation_fps,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Could not load tracks", str(e))
            self.track_db = None
            return

        self.track_labels = self.track_db.labels()
        self.track_combo.blockSignals(True)
        self.track_combo.clear()
        self.track_combo.addItems(self.track_labels)
        self.track_combo.blockSignals(False)
        self.track_combo.setEnabled(True)
        self.crop_button.setVisible(True)
        self.crop_view = True
        self.crop_button.setText("Show full scene")

        if self.track_labels:
            self._open_track_at_index(0)

        msg = (
            f"Loaded {len(self.track_labels)} tracks from {len(list(Path(self.annotation_dir).rglob('*.json')))} JSON files."
        )
        if self.track_db.errors:
            msg += f" {len(self.track_db.errors)} JSON/frame warnings."
        self.statusBar().showMessage(msg, 8000)
        self._update_enabled_state()

    def _set_video(self, path: str):
        self._pause()
        if self.video is not None:
            self.video.close()
        self.video = VideoSource(path)
        self.video_fps = self.video.fps
        self.timeline.set_total_frames(max(1, self.video.total_frames))
        self.item_start = 0
        self.item_end = max(0, self.video.total_frames - 1)
        self.timeline.set_active_range(self.item_start, self.item_end)

    def _track_combo_changed(self, index: int):
        if self.mode == "full" and index >= 0 and index != self.track_index:
            self._open_track_at_index(index)

    def _open_track_at_index(self, index: int):
        if not self.track_db or not self.track_labels:
            return
        index = clamp(index, 0, len(self.track_labels) - 1)
        self._maybe_save_before_switch()

        self.track_index = index
        self.current_track = self.track_labels[index]
        self.track_combo.blockSignals(True)
        self.track_combo.setCurrentIndex(index)
        self.track_combo.blockSignals(False)

        start, end = self.track_db.bounds(self.current_track)
        self.item_start = start
        self.item_end = end
        self.timeline.set_active_range(self.item_start, self.item_end)
        self.current_frame = self.item_start
        self._set_output_path_for_full()
        self._load_existing_behavior_json()

        video_name = Path(self.video.path).name if self.video else ""
        self.source_label.setText(
            f"Full video: {video_name}   |   track: {self.current_track}"
        )
        self._show_current_frame()

    # ---- output paths / persistence -------------------------------------

    def _behavior_output_dir(self) -> Path:
        if self.explicit_output_dir:
            out = Path(self.explicit_output_dir).expanduser().resolve()
        elif self.mode == "tracklet" and self.video:
            out = Path(self.video.path).parent / "BehaviorAnnotations"
        elif self.video:
            out = Path(self.video.path).parent / "BehaviorAnnotations"
        else:
            out = Path.cwd() / "BehaviorAnnotations"
        out.mkdir(parents=True, exist_ok=True)
        return out

    @staticmethod
    def _safe_name(s: str) -> str:
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
        return s.strip("_") or "track"

    def _set_output_path_for_tracklet(self, p: Path):
        self.output_path = self._behavior_output_dir() / f"{p.stem}.behavior.json"

    def _set_output_path_for_full(self):
        if not self.video or not self.current_track:
            self.output_path = None
            return
        video_stem = Path(self.video.path).stem
        track = self._safe_name(self.current_track)
        self.output_path = self._behavior_output_dir() / f"{video_stem}__{track}.behavior.json"

    def _load_existing_behavior_json(self):
        self.segments = []
        self.undo_stack = []
        if self.output_path and self.output_path.exists():
            try:
                with self.output_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                loaded = []
                for s in data.get("segments") or []:
                    loaded.append(
                        Segment(
                            int(s["start_frame"]),
                            int(s["end_frame"]),
                            str(s["behavior"]),
                        )
                    )
                self.segments = normalize_segments(
                    loaded, self.video.total_frames if self.video else 1
                )
                self.statusBar().showMessage(
                    f"Loaded existing annotations: {self.output_path}", 5000
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Could not load behaviour JSON",
                    f"{self.output_path}\n\n{e}",
                )

        self.timeline.selected_index = -1
        self.timeline.set_segments(self.segments)

    def _save(self):
        if not self.video or not self.output_path:
            return

        data = {
            "format": "smarterlabelme-behavior-annotations-v1",
            "mode": self.mode,
            "video": self.video.path,
            "fps": self.video_fps,
            "total_frames": self.video.total_frames,
            "behaviors": self.behaviors,
            "segments": [s.to_dict() for s in self.segments],
            "segment_frame_reference": "source_video" if self.mode == "full" else "tracklet_video",
            "active_start_frame": self.item_start,
            "active_end_frame": self.item_end,
        }

        if self.mode == "tracklet":
            data.update(
                {
                    "tracklet": self.video.path,
                    "source_video": None,
                    "track_id": None,
                    "source_start_frame": None,
                }
            )
        elif self.mode == "full":
            start, end = self.track_db.bounds(self.current_track) if self.track_db and self.current_track else (None, None)
            data.update(
                {
                    "source_video": self.video.path,
                    "annotation_dir": self.annotation_dir,
                    "track_id": self.current_track,
                    "track_first_annotated_frame": start,
                    "track_last_annotated_frame": end,
                    "annotation_fps_override": self.annotation_fps,
                    "crop_margin": self.crop_margin,
                }
            )

        tmp = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
            os.replace(str(tmp), str(self.output_path))
            self.status_label.setText(f"Saved: {self.output_path.name}")
            self.statusBar().showMessage(f"Saved {self.output_path}", 4000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def _maybe_save_before_switch(self):
        if self.video is not None and self.output_path is not None and self.segments:
            self._save()

    # ---- annotation actions ---------------------------------------------

    def _push_undo(self):
        self.undo_stack.append(copy.deepcopy(self.segments))
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)

    def _behavior_pressed(self, behavior: str):
        if not self.video:
            return

        selected = self.timeline.selected_index
        self._push_undo()

        if 0 <= selected < len(self.segments):
            self.segments = relabel_segment(
                self.segments, selected, behavior, self.video.total_frames
            )
            # selection may change after adjacent-equal merge
            self.timeline.selected_index = -1
        else:
            self.segments = set_behavior_from_frame(
                self.segments,
                self.current_frame,
                behavior,
                self.video.total_frames,
            )
            # In full-video mode, annotations belong only to the selected
            # individual's track lifetime, never to the rest of the source video.
            clipped = []
            for seg in self.segments:
                a = max(seg.start, self.item_start)
                b = min(seg.end, self.item_end)
                if a <= b:
                    clipped.append(Segment(a, b, seg.behavior))
            self.segments = normalize_segments(clipped, self.video.total_frames)

        self.timeline.set_segments(self.segments)
        self.status_label.setText(f"{behavior} @ frame {self.current_frame}")

    def _undo(self):
        if not self.undo_stack:
            return
        self.segments = self.undo_stack.pop()
        self.timeline.selected_index = -1
        self.timeline.set_segments(self.segments)
        self.status_label.setText("Undo")

    def _timeline_segments_edited(self):
        if not self.video:
            return
        self._push_undo()
        self.segments = normalize_segments(self.timeline.segments, self.video.total_frames)
        self.timeline.set_segments(self.segments)
        self.status_label.setText("Boundary adjusted")

    def _timeline_selection_changed(self, index: int):
        if 0 <= index < len(self.segments):
            s = self.segments[index]
            self.status_label.setText(
                f"Selected {s.behavior}: frames {s.start}–{s.end}; press a behaviour button to relabel"
            )

    # ---- transport -------------------------------------------------------

    def _toggle_play(self):
        if not self.video:
            return
        if self.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        if not self.video:
            return
        self.timeline.clear_selection()
        self.is_playing = True
        self.play_btn.setText("❚❚ Pause")
        interval_ms = max(1, int(round(1000.0 / (self.video_fps * self.playback_speed))))
        self.timer.start(interval_ms)

    def _pause(self):
        self.is_playing = False
        self.timer.stop()
        if hasattr(self, "play_btn"):
            self.play_btn.setText("▶ Play")

    def _play_tick(self):
        if not self.video:
            return
        if self.current_frame >= self.item_end:
            self._pause()
            return
        self.current_frame += 1
        self._show_current_frame()

    def _step(self, delta: int):
        if not self.video:
            return
        self._pause()
        self.timeline.clear_selection()
        self.current_frame = clamp(
            self.current_frame + delta, self.item_start, self.item_end
        )
        self._show_current_frame()

    def _seek(self, frame: int):
        if not self.video:
            return
        self._pause()
        self.current_frame = clamp(frame, self.item_start, self.item_end)
        self._show_current_frame()

    def _speed_changed(self):
        data = self.speed_combo.currentData()
        self.playback_speed = float(data or 1.0)
        if self.is_playing:
            self._play()

    def _change_item(self, delta: int):
        if self.mode == "tracklet" and self.tracklet_files:
            target = self.tracklet_index + delta
            if 0 <= target < len(self.tracklet_files):
                self._open_tracklet_at_index(target)
        elif self.mode == "full" and self.track_labels:
            target = self.track_index + delta
            if 0 <= target < len(self.track_labels):
                self._open_track_at_index(target)

    # ---- rendering -------------------------------------------------------

    def _toggle_crop_view(self):
        if self.mode != "full":
            return
        self.crop_view = not self.crop_view
        self.crop_button.setText("Show full scene" if self.crop_view else "Show crop")
        self._show_current_frame()

    def _crop_for_track(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(2.0, x2 - x1)
        bh = max(2.0, y2 - y1)

        # Use a square-ish crop around the animal with generous context.
        side = max(bw, bh) * (1.0 + 2.0 * self.crop_margin)
        side = max(side, 64.0)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        xa = int(round(cx - side / 2.0))
        ya = int(round(cy - side / 2.0))
        xb = int(round(cx + side / 2.0))
        yb = int(round(cy + side / 2.0))

        xa, ya = max(0, xa), max(0, ya)
        xb, yb = min(w, xb), min(h, yb)
        if xb <= xa or yb <= ya:
            return frame

        crop = frame[ya:yb, xa:xb].copy()

        # Draw bbox in crop coordinates for traceability.
        rx1 = int(round(x1 - xa))
        ry1 = int(round(y1 - ya))
        rx2 = int(round(x2 - xa))
        ry2 = int(round(y2 - ya))
        cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
        return crop

    def _full_scene_with_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> np.ndarray:
        out = frame.copy()
        x1, y1, x2, y2 = [int(round(x)) for x in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 3)
        if self.current_track:
            cv2.putText(
                out,
                self.current_track,
                (max(0, x1), max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return out

    def _show_current_frame(self):
        if not self.video:
            return

        frame = self.video.read(self.current_frame)
        if frame is None:
            return

        if self.mode == "full" and self.track_db and self.current_track:
            bbox = self.track_db.interpolated_bbox(self.current_track, self.current_frame)
            if bbox is not None:
                if self.crop_view:
                    frame = self._crop_for_track(frame, bbox)
                else:
                    frame = self._full_scene_with_bbox(frame, bbox)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(
            rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        ).copy()

        target = self.video_label.size()
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            target,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

        self.timeline.set_current_frame(self.current_frame)
        self.timeline.set_segments(self.segments)

        elapsed_s = max(0, self.current_frame - self.item_start) / self.video_fps
        active_s = max(0, self.item_end - self.item_start) / self.video_fps
        self.time_label.setText(
            f"{format_time(elapsed_s)} / {format_time(active_s)}"
        )

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.video:
            self._show_current_frame()

    def _update_enabled_state(self):
        loaded = self.video is not None
        for w in [
            self.prev_item_btn,
            self.step_back_btn,
            self.play_btn,
            self.step_fwd_btn,
            self.next_item_btn,
            self.undo_btn,
            self.save_btn,
        ] + self.behavior_buttons:
            w.setEnabled(loaded)
        self.crop_button.setVisible(self.mode == "full")

    def closeEvent(self, e):
        try:
            if self.video and self.output_path and self.segments:
                self._save()
            if self.video:
                self.video.close()
        finally:
            e.accept()


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    return f"{minutes:02d}:{sec:06.3f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Fast temporal behaviour annotation for SmarterLabelMe tracks/tracklets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--behaviors",
        nargs="+",
        default=DEFAULT_BEHAVIORS,
        help="Behaviour button labels. First nine get number-key shortcuts.",
    )
    p.add_argument(
        "--annotation-fps",
        type=float,
        default=None,
        help=(
            "For legacy sampled frame_XXXXXX SmarterLabelMe datasets only: FPS at "
            "which annotation frames were extracted. Modern fXXXXXXXX_tXXXXXXXXX "
            "filenames do not need this."
        ),
    )
    p.add_argument(
        "--crop-margin",
        type=float,
        default=0.40,
        help="Extra crop context around each side, as a fraction of bbox size.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for *.behavior.json files. Default: BehaviorAnnotations beside video.",
    )
    p.add_argument("--tracklet", type=str, default=None, help="Open this tracklet video directly.")
    p.add_argument("--video", type=str, default=None, help="Original/source video for full mode.")
    p.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="SmarterLabelMe JSON annotation directory for full mode.",
    )
    p.add_argument(
        "--track",
        type=str,
        default=None,
        help="Track label to select initially in full mode.",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = BehaviorAnnotator(
        behaviors=args.behaviors,
        annotation_fps=args.annotation_fps,
        crop_margin=args.crop_margin,
        output_dir=args.output_dir,
    )
    win.show()

    if args.tracklet:
        win.open_tracklet(args.tracklet)
    elif args.video and args.annotations:
        win.open_full(args.video, args.annotations)
        if args.track and args.track in win.track_labels:
            win._open_track_at_index(win.track_labels.index(args.track))

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
