"""PySide6 desktop interface for the audio interleaver."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QCloseEvent, QFont, QPaintEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .audio import (
    AudioEngine,
    AudioError,
    InterleavePattern,
    LoadedAudio,
    RenderingCancelled,
    SourceId,
    load_wav,
    write_wav,
)
from .playback import PlaybackController


SOURCE_A_COLOR = "#6ea8fe"
SOURCE_B_COLOR = "#f08cba"
DEFAULT_CHUNK_MS = 360
DEFAULT_CROSSFADE_MS = 0
MIN_CHUNK_MS = 50
MAX_CHUNK_MS = 2000
MAX_CROSSFADE_MS = 50


def _format_time(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, remainder = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remainder:02d}"
    return f"{minutes:d}:{remainder:02d}"


class _UiSignals(QObject):
    playback_position = Signal(float)
    playback_finished = Signal(bool)
    playback_error = Signal(str)
    export_progress = Signal(int)
    export_finished = Signal(str)
    export_error = Signal(str)


class SourceCard(QFrame):
    load_requested = Signal()

    def __init__(self, source_name: str, accent: str) -> None:
        super().__init__()
        self.setObjectName("sourceCard")
        self._accent = accent

        title = QLabel(f"SOURCE {source_name}")
        title.setObjectName("sourceTitle")
        title.setStyleSheet(f"color: {accent};")
        self.file_label = QLabel("No WAV selected")
        self.file_label.setObjectName("fileName")
        self.file_label.setWordWrap(True)
        self.details_label = QLabel("Load a mono or stereo WAV file")
        self.details_label.setObjectName("sourceDetails")
        self.load_button = QPushButton(f"Load {source_name}")
        self.load_button.clicked.connect(self.load_requested)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addWidget(self.file_label)
        layout.addWidget(self.details_label)
        layout.addSpacing(5)
        layout.addWidget(self.load_button, alignment=Qt.AlignmentFlag.AlignLeft)

    def display_audio(self, audio: LoadedAudio) -> None:
        self.file_label.setText(audio.path.name if audio.path else "Loaded WAV")
        channel_text = "mono" if audio.channels == 1 else "stereo"
        self.details_label.setText(
            f"{_format_time(audio.duration)}  •  {audio.sample_rate:,} Hz  •  {channel_text}"
        )


class InterleaveTimeline(QWidget):
    """Two-lane preview of the source selected for every timeline slot."""

    def __init__(self) -> None:
        super().__init__()
        self._engine: AudioEngine | None = None
        self._pattern = InterleavePattern()
        self._position = 0.0
        self._slot_sources: tuple[str, ...] = ()
        self.setMinimumHeight(58)
        self.setSizePolicy(
            self.sizePolicy().horizontalPolicy(),
            self.sizePolicy().Policy.Fixed,
        )
        self.setAccessibleName("Interleave preview")
        self.setAccessibleDescription(
            "Two timeline lanes showing which audio chunks use source A or B."
        )

    @property
    def slot_sources(self) -> tuple[str, ...]:
        return self._slot_sources

    @property
    def position(self) -> float:
        return self._position

    def sizeHint(self) -> QSize:
        return QSize(640, 58)

    def set_engine(self, engine: AudioEngine | None) -> None:
        self._engine = engine
        self._position = 0.0
        self._rebuild_slots()

    def set_pattern(self, pattern: InterleavePattern) -> None:
        self._pattern = pattern
        self._rebuild_slots()

    def set_position(self, seconds: float) -> None:
        self._position = max(0.0, seconds)
        self.update()

    def _rebuild_slots(self) -> None:
        if self._engine is None:
            self._slot_sources = ()
        else:
            self._slot_sources = tuple(
                self._engine.source_for_slot(index, self._pattern)
                for index in range(self._engine.slot_count)
            )
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        label_width = 22.0
        right_margin = 1.0
        track_left = label_width
        track_width = max(1.0, self.width() - track_left - right_margin)
        lane_height = 19.0
        lane_gap = 6.0
        top = 5.0
        track_color = QColor("#30343d")
        border_color = QColor("#434956")

        painter.setFont(self.font())
        painter.setPen(QColor(SOURCE_A_COLOR))
        painter.drawText(
            QRectF(0, top, label_width - 5, lane_height),
            Qt.AlignmentFlag.AlignCenter,
            "A",
        )
        painter.setPen(QColor(SOURCE_B_COLOR))
        painter.drawText(
            QRectF(0, top + lane_height + lane_gap, label_width - 5, lane_height),
            Qt.AlignmentFlag.AlignCenter,
            "B",
        )

        lane_a = QRectF(track_left, top, track_width, lane_height)
        lane_b = QRectF(
            track_left, top + lane_height + lane_gap, track_width, lane_height
        )
        painter.fillRect(lane_a, track_color)
        painter.fillRect(lane_b, track_color)

        if self._engine is not None and self._slot_sources:
            total_frames = self._engine.total_frames
            slot_frames = self._engine.slot_frames
            for index, source_id in enumerate(self._slot_sources):
                start_frame = index * slot_frames
                end_frame = min(start_frame + slot_frames, total_frames)
                x1 = track_left + track_width * start_frame / total_frames
                x2 = track_left + track_width * end_frame / total_frames
                slot_rect = QRectF(
                    x1,
                    lane_a.top() if source_id == "A" else lane_b.top(),
                    max(1.0, x2 - x1),
                    lane_height,
                )
                painter.fillRect(
                    slot_rect,
                    QColor(SOURCE_A_COLOR if source_id == "A" else SOURCE_B_COLOR),
                )
                if x2 - x1 >= 4.0:
                    painter.setPen(QColor("#20232b"))
                    painter.drawLine(
                        round(x2),
                        round(slot_rect.top()),
                        round(x2),
                        round(slot_rect.bottom()),
                    )

            if self._engine.duration > 0:
                progress = min(1.0, self._position / self._engine.duration)
                marker_x = track_left + track_width * progress
                painter.setPen(QPen(QColor("#f4f6fa"), 2))
                painter.drawLine(
                    round(marker_x),
                    round(lane_a.top() - 2),
                    round(marker_x),
                    round(lane_b.bottom() + 2),
                )

        painter.setPen(QPen(border_color, 1))
        painter.drawRect(lane_a)
        painter.drawRect(lane_b)
        painter.end()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Audio Interleaver")
        self.setMinimumSize(760, 900)
        self.resize(860, 940)

        self._source_a: LoadedAudio | None = None
        self._source_b: LoadedAudio | None = None
        self._engine: AudioEngine | None = None
        self._fill = 0.5
        self._starts_with: SourceId = "A"
        self._first_alternate_slot = 1
        self._b_chunks_per_occurrence = 1
        self._chunk_ms = DEFAULT_CHUNK_MS
        self._crossfade_ms = DEFAULT_CROSSFADE_MS
        self._loop = False
        self._export_thread: threading.Thread | None = None
        self._export_cancel = threading.Event()
        self._exporting = False

        self._signals = _UiSignals()
        self._signals.playback_position.connect(self._on_playback_position)
        self._signals.playback_finished.connect(self._on_playback_finished)
        self._signals.playback_error.connect(self._on_playback_error)
        self._signals.export_progress.connect(self._on_export_progress)
        self._signals.export_finished.connect(self._on_export_finished)
        self._signals.export_error.connect(self._on_export_error)

        self._playback = PlaybackController(
            on_position=self._signals.playback_position.emit,
            on_finished=self._signals.playback_finished.emit,
            on_error=self._signals.playback_error.emit,
        )

        self._build_ui()
        self._apply_style()
        self._refresh_actions()

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(32, 20, 32, 20)
        root.setSpacing(14)

        heading = QLabel("Audio Interleaver")
        heading.setObjectName("heading")
        product_subheading = QLabel("A product of DG1TAL Compute Sweatshop")
        product_subheading.setObjectName("productSubheading")
        subtitle = QLabel(
            "Build repeating chunk patterns from two independent audio sources."
        )
        subtitle.setObjectName("subtitle")
        title_block = QVBoxLayout()
        title_block.setSpacing(3)
        title_block.addWidget(heading)
        title_block.addWidget(product_subheading)
        title_block.addSpacing(7)
        title_block.addWidget(subtitle)
        root.addLayout(title_block)

        cards = QHBoxLayout()
        cards.setSpacing(14)
        self.source_a_card = SourceCard("A", SOURCE_A_COLOR)
        self.source_b_card = SourceCard("B", SOURCE_B_COLOR)
        self.source_a_card.load_requested.connect(lambda: self._load_source("A"))
        self.source_b_card.load_requested.connect(lambda: self._load_source("B"))
        cards.addWidget(self.source_a_card)
        cards.addWidget(self.source_b_card)
        root.addLayout(cards)

        fader_panel = QFrame()
        fader_panel.setObjectName("faderPanel")
        fader_layout = QVBoxLayout(fader_panel)
        fader_layout.setContentsMargins(22, 18, 22, 18)
        fader_layout.setSpacing(6)

        fader_header = QHBoxLayout()
        fader_title = QLabel("B OCCURRENCE FILL")
        fader_title.setObjectName("sectionTitle")
        self.mix_label = QLabel("50%")
        self.mix_label.setObjectName("mixLabel")
        fader_header.addWidget(fader_title)
        fader_header.addStretch()
        fader_header.addWidget(self.mix_label)
        fader_layout.addLayout(fader_header)

        self.crossfader = QSlider(Qt.Orientation.Horizontal)
        self.crossfader.setRange(0, 100)
        self.crossfader.setValue(50)
        self.crossfader.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.crossfader.setTickInterval(10)
        self.crossfader.valueChanged.connect(self._on_crossfader_changed)
        fader_layout.addWidget(self.crossfader)

        fader_labels = QHBoxLayout()
        fader_labels.addWidget(QLabel("Minimum pattern"))
        center_label = QLabel("Reveal left to right")
        center_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fader_labels.addWidget(center_label, 1)
        right_label = QLabel("All occurrences")
        right_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        fader_labels.addWidget(right_label)
        fader_layout.addLayout(fader_labels)

        pattern_options = QHBoxLayout()
        self.start_with_b_checkbox = QCheckBox("Start with B")
        self.start_with_b_checkbox.setToolTip(
            "Make output chunk 1 use source B instead of source A"
        )
        self.start_with_b_checkbox.toggled.connect(self._on_start_source_changed)
        pattern_options.addWidget(self.start_with_b_checkbox)
        pattern_options.addStretch()
        fader_layout.addSpacing(6)
        fader_layout.addLayout(pattern_options)

        alternate_header = QHBoxLayout()
        self.first_alternate_title = QLabel("FIRST B CHUNK")
        self.first_alternate_title.setObjectName("sectionTitle")
        self.first_alternate_label = QLabel("Chunk 2")
        self.first_alternate_label.setObjectName("settingValue")
        alternate_header.addWidget(self.first_alternate_title)
        alternate_header.addStretch()
        alternate_header.addWidget(self.first_alternate_label)
        fader_layout.addLayout(alternate_header)

        self.first_alternate_slider = QSlider(Qt.Orientation.Horizontal)
        self.first_alternate_slider.setRange(2, 2)
        self.first_alternate_slider.setValue(2)
        self.first_alternate_slider.setEnabled(False)
        self.first_alternate_slider.setSingleStep(1)
        self.first_alternate_slider.setTickPosition(
            QSlider.TickPosition.TicksBelow
        )
        self.first_alternate_slider.setTickInterval(1)
        self.first_alternate_slider.valueChanged.connect(
            self._on_first_alternate_changed
        )
        fader_layout.addWidget(self.first_alternate_slider)

        burst_header = QHBoxLayout()
        burst_title = QLabel("B CHUNKS PER OCCURRENCE")
        burst_title.setObjectName("sectionTitle")
        self.burst_size_label = QLabel("1")
        self.burst_size_label.setObjectName("settingValue")
        burst_header.addWidget(burst_title)
        burst_header.addStretch()
        burst_header.addWidget(self.burst_size_label)
        fader_layout.addLayout(burst_header)

        self.burst_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.burst_size_slider.setRange(1, 1)
        self.burst_size_slider.setValue(1)
        self.burst_size_slider.setEnabled(False)
        self.burst_size_slider.setSingleStep(1)
        self.burst_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.burst_size_slider.setTickInterval(1)
        self.burst_size_slider.valueChanged.connect(self._on_burst_size_changed)
        fader_layout.addWidget(self.burst_size_slider)

        chunk_header = QHBoxLayout()
        chunk_title = QLabel("CHUNK DURATION")
        chunk_title.setObjectName("sectionTitle")
        self.chunk_duration_input = QSpinBox()
        self.chunk_duration_input.setRange(MIN_CHUNK_MS, MAX_CHUNK_MS)
        self.chunk_duration_input.setValue(self._chunk_ms)
        self.chunk_duration_input.setSuffix(" ms")
        self.chunk_duration_input.setKeyboardTracking(False)
        self.chunk_duration_input.setObjectName("durationInput")
        self.chunk_duration_input.valueChanged.connect(
            self._on_chunk_duration_changed
        )
        chunk_header.addWidget(chunk_title)
        chunk_header.addStretch()
        chunk_header.addWidget(self.chunk_duration_input)
        fader_layout.addSpacing(8)
        fader_layout.addLayout(chunk_header)

        self.chunk_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.chunk_duration_slider.setRange(MIN_CHUNK_MS, MAX_CHUNK_MS)
        self.chunk_duration_slider.setValue(self._chunk_ms)
        self.chunk_duration_slider.setSingleStep(10)
        self.chunk_duration_slider.setPageStep(50)
        self.chunk_duration_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.chunk_duration_slider.setTickInterval(250)
        self.chunk_duration_slider.valueChanged.connect(
            self._on_chunk_duration_changed
        )
        fader_layout.addWidget(self.chunk_duration_slider)

        transition_header = QHBoxLayout()
        transition_title = QLabel("CHUNK CROSSFADE")
        transition_title.setObjectName("sectionTitle")
        self.crossfade_duration_input = QSpinBox()
        self.crossfade_duration_input.setRange(0, MAX_CROSSFADE_MS)
        self.crossfade_duration_input.setValue(self._crossfade_ms)
        self.crossfade_duration_input.setSuffix(" ms")
        self.crossfade_duration_input.setKeyboardTracking(False)
        self.crossfade_duration_input.setObjectName("durationInput")
        self.crossfade_duration_input.valueChanged.connect(
            self._on_crossfade_duration_changed
        )
        transition_header.addWidget(transition_title)
        transition_header.addStretch()
        transition_header.addWidget(self.crossfade_duration_input)
        fader_layout.addLayout(transition_header)

        self.crossfade_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.crossfade_duration_slider.setRange(0, MAX_CROSSFADE_MS)
        self.crossfade_duration_slider.setValue(self._crossfade_ms)
        self.crossfade_duration_slider.setSingleStep(1)
        self.crossfade_duration_slider.setPageStep(5)
        self.crossfade_duration_slider.setTickPosition(
            QSlider.TickPosition.TicksBelow
        )
        self.crossfade_duration_slider.setTickInterval(5)
        self.crossfade_duration_slider.setToolTip(
            "Equal-power transition when consecutive chunks switch sources"
        )
        self.crossfade_duration_slider.valueChanged.connect(
            self._on_crossfade_duration_changed
        )
        fader_layout.addWidget(self.crossfade_duration_slider)

        preview_header = QHBoxLayout()
        preview_title = QLabel("INTERLEAVE PREVIEW")
        preview_title.setObjectName("sectionTitle")
        self.preview_detail = QLabel(f"Each block = {self._chunk_ms} ms")
        self.preview_detail.setObjectName("previewDetail")
        preview_header.addWidget(preview_title)
        preview_header.addStretch()
        preview_header.addWidget(self.preview_detail)
        fader_layout.addSpacing(8)
        fader_layout.addLayout(preview_header)

        self.interleave_timeline = InterleaveTimeline()
        fader_layout.addWidget(self.interleave_timeline)
        root.addWidget(fader_panel)

        transport = QFrame()
        transport.setObjectName("transport")
        transport_layout = QVBoxLayout(transport)
        transport_layout.setContentsMargins(20, 16, 20, 16)
        transport_layout.setSpacing(10)

        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("primaryButton")
        self.play_button.clicked.connect(self._toggle_playback)
        self.export_button = QPushButton("Export WAV…")
        self.export_button.clicked.connect(self._export_wav)
        self.loop_checkbox = QCheckBox("Loop")
        self.loop_checkbox.setToolTip("Restart the complete result when playback ends")
        self.loop_checkbox.toggled.connect(self._on_loop_changed)
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("timeLabel")
        controls.addWidget(self.play_button)
        controls.addWidget(self.export_button)
        controls.addWidget(self.loop_checkbox)
        controls.addStretch()
        controls.addWidget(self.time_label)
        transport_layout.addLayout(controls)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        transport_layout.addWidget(self.progress)
        root.addWidget(transport)

        self.status_label = QLabel("Load two WAV files to begin.")
        self.status_label.setObjectName("status")
        root.addWidget(self.status_label)
        root.addStretch()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { color: #e8eaf0; }
            QMainWindow, QWidget#central { background: #15171c; }
            QLabel, QSlider { background: transparent; }
            QLabel#heading { font-size: 30px; font-weight: 700; }
            QLabel#productSubheading { color: #7f8797; font-size: 11px; }
            QLabel#subtitle { color: #9ca3b2; font-size: 14px; }
            QFrame#sourceCard, QFrame#faderPanel, QFrame#transport {
                background: #20232b; border: 1px solid #30343e; border-radius: 10px;
            }
            QLabel#sourceTitle, QLabel#sectionTitle { font-size: 11px; font-weight: 700; }
            QLabel#fileName { font-size: 16px; font-weight: 600; }
            QLabel#sourceDetails, QLabel#status { color: #9ca3b2; }
            QLabel#mixLabel, QLabel#settingValue { color: #c4c8d2; font-weight: 600; }
            QSpinBox#durationInput {
                color: #c4c8d2; background: #303541; border: 1px solid #434956;
                border-radius: 4px; padding: 3px 6px; font-weight: 600;
            }
            QLabel#previewDetail { color: #858c9b; font-size: 11px; }
            QLabel#timeLabel { color: #b8bdc9; }
            QPushButton {
                background: #303541; border: 1px solid #434956; border-radius: 6px;
                padding: 8px 15px; font-weight: 600;
            }
            QPushButton:hover { background: #3a404d; }
            QPushButton:pressed { background: #282c35; }
            QPushButton:disabled { color: #686d78; background: #252831; border-color: #30343c; }
            QPushButton#primaryButton { background: #4778d0; border-color: #5789e2; }
            QPushButton#primaryButton:hover { background: #5486df; }
            QCheckBox { spacing: 7px; color: #c4c8d2; font-weight: 600; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QSlider::groove:horizontal { height: 6px; background: #3b404b; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #6ea8fe; border-radius: 3px; }
            QSlider::handle:horizontal {
                background: #f4f6fa; border: 2px solid #15171c; width: 18px;
                margin: -7px 0; border-radius: 9px;
            }
            QProgressBar { background: #30343d; border: none; border-radius: 3px; height: 6px; }
            QProgressBar::chunk { background: #6ea8fe; border-radius: 3px; }
            """
        )

    def _load_source(self, source_id: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Load source {source_id}",
            "",
            "WAV audio (*.wav *.wave);;All files (*)",
        )
        if not path:
            return
        try:
            audio = load_wav(path)
        except AudioError as exc:
            QMessageBox.critical(self, "Could not load WAV", str(exc))
            return

        self._stop_playback(wait=True, reset=True)
        if source_id == "A":
            self._source_a = audio
            self.source_a_card.display_audio(audio)
        else:
            self._source_b = audio
            self.source_b_card.display_audio(audio)

        self._rebuild_engine()

    def _on_crossfader_changed(self, value: int) -> None:
        self._fill = value / 100.0
        self.mix_label.setText(f"{value}%")
        self._pattern_changed()

    def _on_start_source_changed(self, starts_with_b: bool) -> None:
        self._starts_with = "B" if starts_with_b else "A"
        alternate = "A" if starts_with_b else "B"
        self.first_alternate_title.setText(f"FIRST {alternate} CHUNK")
        self._pattern_changed()

    def _on_first_alternate_changed(self, chunk_number: int) -> None:
        self._first_alternate_slot = chunk_number - 1
        self.first_alternate_label.setText(f"Chunk {chunk_number}")
        self._pattern_changed()

    def _on_burst_size_changed(self, chunks: int) -> None:
        self._b_chunks_per_occurrence = chunks
        self.burst_size_label.setText(str(chunks))
        self._pattern_changed()

    def _pattern(self) -> InterleavePattern:
        return InterleavePattern(
            fill=self._fill,
            starts_with=self._starts_with,
            first_alternate_slot=self._first_alternate_slot,
            b_chunks_per_occurrence=self._b_chunks_per_occurrence,
        )

    def _pattern_changed(self) -> None:
        self.interleave_timeline.set_pattern(self._pattern())

    def _on_chunk_duration_changed(self, value: int) -> None:
        self.chunk_duration_slider.blockSignals(True)
        self.chunk_duration_input.blockSignals(True)
        self.chunk_duration_slider.setValue(value)
        self.chunk_duration_input.setValue(value)
        self.chunk_duration_slider.blockSignals(False)
        self.chunk_duration_input.blockSignals(False)
        self._chunk_ms = value
        self.preview_detail.setText(f"Each block = {value} ms")
        self._configuration_changed()

    def _on_crossfade_duration_changed(self, value: int) -> None:
        self.crossfade_duration_slider.blockSignals(True)
        self.crossfade_duration_input.blockSignals(True)
        self.crossfade_duration_slider.setValue(value)
        self.crossfade_duration_input.setValue(value)
        self.crossfade_duration_slider.blockSignals(False)
        self.crossfade_duration_input.blockSignals(False)
        self._crossfade_ms = value
        self._configuration_changed()

    def _configuration_changed(self) -> None:
        self._stop_playback(wait=True, reset=True)
        self._rebuild_engine()

    def _rebuild_engine(self) -> None:
        try:
            self._engine = (
                AudioEngine(
                    self._source_a,
                    self._source_b,
                    slot_ms=self._chunk_ms,
                    smoothing_ms=self._crossfade_ms,
                )
                if self._source_a is not None and self._source_b is not None
                else None
            )
        except (AudioError, ValueError) as exc:
            self._engine = None
            QMessageBox.critical(self, "Could not prepare audio", str(exc))

        if self._engine is not None:
            self.status_label.setText(
                f"Ready • {self._chunk_ms} ms chunks • "
                f"{self._crossfade_ms} ms crossfade • "
                f"{_format_time(self._engine.duration)} output"
            )
        else:
            self.status_label.setText("Load the other WAV file to begin.")
        self._sync_pattern_controls()
        self.interleave_timeline.set_engine(self._engine)
        self.interleave_timeline.set_pattern(self._pattern())
        self._update_time(0.0)
        self._refresh_actions()

    def _sync_pattern_controls(self) -> None:
        slot_count = self._engine.slot_count if self._engine is not None else 1
        has_alternate_slot = slot_count >= 2

        self.first_alternate_slider.blockSignals(True)
        self.burst_size_slider.blockSignals(True)
        if has_alternate_slot:
            self._first_alternate_slot = min(
                max(1, self._first_alternate_slot), slot_count - 1
            )
            self._b_chunks_per_occurrence = min(
                max(1, self._b_chunks_per_occurrence), slot_count
            )
            self.first_alternate_slider.setRange(2, slot_count)
            self.first_alternate_slider.setValue(self._first_alternate_slot + 1)
            self.burst_size_slider.setRange(1, slot_count)
            self.burst_size_slider.setValue(self._b_chunks_per_occurrence)
        else:
            self._first_alternate_slot = 1
            self._b_chunks_per_occurrence = 1
            self.first_alternate_slider.setRange(2, 2)
            self.first_alternate_slider.setValue(2)
            self.burst_size_slider.setRange(1, 1)
            self.burst_size_slider.setValue(1)
        self.first_alternate_slider.setEnabled(has_alternate_slot)
        self.burst_size_slider.setEnabled(has_alternate_slot)
        self.first_alternate_label.setText(
            f"Chunk {self._first_alternate_slot + 1}"
            if has_alternate_slot
            else "Unavailable"
        )
        self.burst_size_label.setText(str(self._b_chunks_per_occurrence))
        self.first_alternate_slider.blockSignals(False)
        self.burst_size_slider.blockSignals(False)

    def _on_loop_changed(self, checked: bool) -> None:
        self._loop = checked

    def _toggle_playback(self) -> None:
        if self._playback.is_playing:
            self._stop_playback(reset=True)
            return
        if self._engine is None:
            return
        self.progress.setValue(0)
        self._update_time(0.0)
        if self._playback.start(self._engine, self._pattern, lambda: self._loop):
            self.play_button.setText("Stop")
            self.status_label.setText(
                f"Playing • pattern changes apply at the next "
                f"{self._chunk_ms} ms boundary"
            )

    def _stop_playback(self, wait: bool = False, reset: bool = False) -> None:
        self._playback.stop(wait=wait)
        self.play_button.setText("Play")
        if reset:
            self.progress.setValue(0)
            self._update_time(0.0)
        if self._engine is not None and not self._exporting:
            self.status_label.setText("Ready")

    def _on_playback_position(self, seconds: float) -> None:
        if self._engine is None:
            return
        self.progress.setValue(round(seconds / self._engine.duration * 1000))
        self._update_time(seconds)

    def _on_playback_finished(self, natural: bool) -> None:
        self.play_button.setText("Play")
        if self._engine is None or self._exporting:
            return
        if natural:
            self.progress.setValue(1000)
            self._update_time(self._engine.duration)
            self.status_label.setText("Playback finished")
        else:
            self.status_label.setText("Ready")

    def _on_playback_error(self, message: str) -> None:
        QMessageBox.critical(self, "Audio output error", message)

    def _update_time(self, current: float) -> None:
        duration = self._engine.duration if self._engine is not None else 0.0
        self.interleave_timeline.set_position(current)
        self.time_label.setText(f"{_format_time(current)} / {_format_time(duration)}")

    def _export_wav(self) -> None:
        if self._engine is None or self._exporting:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export interleaved WAV",
            "interleaved.wav",
            "WAV audio (*.wav)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() not in (".wav", ".wave"):
            output_path = output_path.with_suffix(".wav")

        self._stop_playback(wait=True, reset=True)
        engine = self._engine
        snapshot = self._pattern()
        self._export_cancel.clear()
        self._exporting = True
        self.progress.setValue(0)
        self.status_label.setText(
            f"Exporting pattern at {round(snapshot.fill * 100)}% occurrence fill…"
        )
        self._refresh_actions()

        self._export_thread = threading.Thread(
            target=self._render_export,
            args=(engine, snapshot, output_path),
            name="audio-export",
            daemon=True,
        )
        self._export_thread.start()

    def _render_export(
        self, engine: AudioEngine, pattern: InterleavePattern, output_path: Path
    ) -> None:
        try:
            rendered = engine.render(
                pattern,
                progress=lambda value: self._signals.export_progress.emit(round(value * 1000)),
                cancel_event=self._export_cancel,
            )
            write_wav(output_path, rendered, engine.sample_rate)
        except RenderingCancelled:
            return
        except Exception as exc:
            self._signals.export_error.emit(str(exc))
        else:
            self._signals.export_finished.emit(str(output_path))

    def _on_export_progress(self, value: int) -> None:
        self.progress.setValue(value)

    def _on_export_finished(self, path: str) -> None:
        self._exporting = False
        self._export_thread = None
        self.progress.setValue(1000)
        self.status_label.setText(f"Exported {Path(path).name}")
        self._refresh_actions()
        QMessageBox.information(self, "Export complete", f"Saved WAV to:\n{path}")

    def _on_export_error(self, message: str) -> None:
        self._exporting = False
        self._export_thread = None
        self.status_label.setText("Export failed")
        self._refresh_actions()
        QMessageBox.critical(self, "Export failed", message)

    def _refresh_actions(self) -> None:
        ready = self._engine is not None and not self._exporting
        self.play_button.setEnabled(ready)
        self.export_button.setEnabled(ready)
        self.source_a_card.load_button.setEnabled(not self._exporting)
        self.source_b_card.load_button.setEnabled(not self._exporting)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._playback.stop(wait=True)
        self._export_cancel.set()
        if self._export_thread is not None:
            self._export_thread.join(timeout=2.0)
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Audio Interleaver")
    app.setOrganizationName("Audio Interleaver")
    app.setStyle("Fusion")
    font = QFont()
    font.setPointSize(11)
    app.setFont(font)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
