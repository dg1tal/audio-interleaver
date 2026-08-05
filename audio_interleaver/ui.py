"""PySide6 desktop interface for the audio interleaver."""

from __future__ import annotations

from functools import lru_cache
import math
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QCloseEvent, QFont, QPaintEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .audio import (
    AudioEngine,
    AudioError,
    InterleavePattern,
    InterleaveSettings,
    LoadedAudio,
    RegionInsert,
    RenderingCancelled,
    SourceId,
    load_wav,
    occurrence_capacity,
    write_wav,
)
from .acelp import (
    ACELP_FRAME_MS,
    ACELP_MAX_CHUNK_MS,
    AcelpEngine,
    BEncoderMode,
    ProcessingStage,
    snap_acelp_chunk_ms,
)
from .playback import (
    AudioPreviewController,
    PlaybackController,
    RenderedPlaybackController,
)


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


def _format_position_ms(milliseconds: int) -> str:
    total_minutes, remainder_ms = divmod(max(0, milliseconds), 60_000)
    hours, minutes = divmod(total_minutes, 60)
    seconds, milliseconds = divmod(remainder_ms, 1000)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{minutes:d}:{seconds:02d}.{milliseconds:03d}"


def _format_chunk_count(count: int) -> str:
    chunk_word = "chunk" if count == 1 else "chunks"
    return f"{count} {chunk_word}"


@lru_cache(maxsize=1)
def _commit_hash() -> str:
    configured = os.environ.get("AUDIO_INTERLEAVER_COMMIT", "").strip()
    if configured:
        return configured[:12]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            check=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


class _UiSignals(QObject):
    playback_position = Signal(float)
    playback_finished = Signal(bool)
    playback_error = Signal(str)
    preview_finished = Signal(str, bool)
    preview_error = Signal(str)
    export_progress = Signal(int)
    export_finished = Signal(str)
    export_error = Signal(str)
    acelp_ready = Signal(object)
    acelp_prepare_error = Signal(str)
    acelp_prepare_cancelled = Signal()


class SourceCard(QFrame):
    load_requested = Signal()
    preview_requested = Signal()

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
        self.preview_button = QPushButton("Play preview")
        self.preview_button.setEnabled(False)
        self.preview_button.clicked.connect(self.preview_requested)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(4)
        layout.addWidget(title)
        content_row = QHBoxLayout()
        content_row.setSpacing(10)
        file_details = QVBoxLayout()
        file_details.setSpacing(4)
        file_details.addWidget(self.file_label)
        file_details.addWidget(self.details_label)
        self.action_layout = QVBoxLayout()
        self.action_layout.setSpacing(6)
        self.action_layout.addWidget(self.load_button)
        self.action_layout.addWidget(self.preview_button)
        content_row.addLayout(file_details, 1)
        content_row.addLayout(self.action_layout)
        layout.addLayout(content_row)

    def display_audio(self, audio: LoadedAudio) -> None:
        self.file_label.setText(audio.path.name if audio.path else "Loaded WAV")
        channel_text = "mono" if audio.channels == 1 else "stereo"
        self.details_label.setText(
            f"{_format_time(audio.duration)}  •  {audio.sample_rate:,} Hz  •  {channel_text}"
        )


class InterleaveTimeline(QWidget):
    """Chunk-aligned source waveforms and two-lane interleave preview."""

    def __init__(self) -> None:
        super().__init__()
        self._engine: AudioEngine | AcelpEngine | None = None
        self._settings: InterleaveSettings = InterleavePattern()
        self._position = 0.0
        self._slot_sources: tuple[str, ...] = ()
        self._waveform_chunk_indices: dict[
            SourceId, tuple[int | None, ...]
        ] = {"A": (), "B": ()}
        self._waveform_start_frames: dict[
            SourceId, tuple[int | None, ...]
        ] = {"A": (), "B": ()}
        self.setMinimumHeight(116)
        self.setSizePolicy(
            self.sizePolicy().horizontalPolicy(),
            self.sizePolicy().Policy.Fixed,
        )
        self.setAccessibleName("Interleave preview")
        self.setAccessibleDescription(
            "Source waveforms surrounding timeline lanes for selected A and B chunks."
        )

    @property
    def slot_sources(self) -> tuple[str, ...]:
        return self._slot_sources

    @property
    def position(self) -> float:
        return self._position

    def waveform_chunk_indices(
        self, source_id: SourceId
    ) -> tuple[int | None, ...]:
        return self._waveform_chunk_indices[source_id]

    def waveform_start_frames(
        self, source_id: SourceId
    ) -> tuple[int | None, ...]:
        return self._waveform_start_frames[source_id]

    def sizeHint(self) -> QSize:
        return QSize(640, 116)

    def set_engine(self, engine: AudioEngine | AcelpEngine | None) -> None:
        self._engine = engine
        self._position = 0.0
        self._rebuild_slots()

    def set_settings(self, settings: InterleaveSettings) -> None:
        self._settings = settings
        self._rebuild_slots()

    def set_pattern(self, pattern: InterleavePattern) -> None:
        self.set_settings(pattern)

    def set_position(self, seconds: float) -> None:
        self._position = max(0.0, seconds)
        self.update()

    def _rebuild_slots(self) -> None:
        if self._engine is None:
            self._slot_sources = ()
            self._waveform_chunk_indices = {"A": (), "B": ()}
            self._waveform_start_frames = {"A": (), "B": ()}
        else:
            self._slot_sources = tuple(
                self._engine.source_for_slot(index, self._settings)
                for index in range(self._engine.slot_count)
            )
            start_frames: dict[SourceId, list[int | None]] = {"A": [], "B": []}
            chunk_indices: dict[SourceId, list[int | None]] = {"A": [], "B": []}
            for index, selected_source in enumerate(self._slot_sources):
                a_start = self._engine.source_start_frame_for_slot(
                    index, self._settings, "A"
                )
                start_frames["A"].append(a_start)
                chunk_indices["A"].append(a_start // self._engine.slot_frames)
                b_start = (
                    self._engine.source_start_frame_for_slot(
                        index, self._settings, "B"
                    )
                    if selected_source == "B"
                    else None
                )
                if (
                    b_start is not None
                    and isinstance(self._settings, RegionInsert)
                    and self._settings.silence_after_b_end
                    and b_start >= self._engine.source_b.frames
                ):
                    b_start = None
                start_frames["B"].append(b_start)
                chunk_indices["B"].append(
                    b_start // self._engine.slot_frames
                    if b_start is not None
                    else None
                )
            self._waveform_chunk_indices = {
                "A": tuple(chunk_indices["A"]),
                "B": tuple(chunk_indices["B"]),
            }
            self._waveform_start_frames = {
                "A": tuple(start_frames["A"]),
                "B": tuple(start_frames["B"]),
            }
        self.update()

    @staticmethod
    def _draw_waveform(
        painter: QPainter,
        rect: QRectF,
        samples: np.ndarray,
        peak: float,
        color: QColor,
    ) -> None:
        width = max(1, round(rect.width()))
        frame_count = len(samples)
        if frame_count == 0:
            return
        center = rect.center().y()
        half_height = max(1.0, rect.height() / 2.0 - 2.0)
        painter.setPen(QPen(color, 1))
        for pixel in range(width):
            frame_start = pixel * frame_count // width
            frame_end = max(frame_start + 1, (pixel + 1) * frame_count // width)
            section = samples[frame_start:frame_end]
            low = float(np.min(section)) / peak
            high = float(np.max(section)) / peak
            x = round(rect.left()) + pixel
            painter.drawLine(
                x,
                round(center - np.clip(high, -1.0, 1.0) * half_height),
                x,
                round(center - np.clip(low, -1.0, 1.0) * half_height),
            )

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor("#20232b"))

        label_width = 22.0
        right_margin = 1.0
        track_left = label_width
        track_width = max(1.0, self.width() - track_left - right_margin)
        waveform_height = 30.0
        lane_height = 16.0
        lane_gap = 4.0
        section_gap = 5.0
        top = 3.0
        track_color = QColor("#30343d")
        border_color = QColor("#434956")
        muted_waveform = QColor("#59606d")

        waveform_a = QRectF(track_left, top, track_width, waveform_height)
        lane_a = QRectF(
            track_left,
            waveform_a.bottom() + section_gap,
            track_width,
            lane_height,
        )
        lane_b = QRectF(
            track_left,
            lane_a.bottom() + lane_gap,
            track_width,
            lane_height,
        )
        waveform_b = QRectF(
            track_left,
            lane_b.bottom() + section_gap,
            track_width,
            waveform_height,
        )

        painter.setFont(self.font())
        painter.setPen(QColor(SOURCE_A_COLOR))
        painter.drawText(
            QRectF(0, waveform_a.top(), label_width - 5, waveform_height),
            Qt.AlignmentFlag.AlignCenter,
            "A",
        )
        painter.setPen(QColor(SOURCE_B_COLOR))
        painter.drawText(
            QRectF(0, waveform_b.top(), label_width - 5, waveform_height),
            Qt.AlignmentFlag.AlignCenter,
            "B",
        )

        painter.fillRect(waveform_a, track_color)
        painter.fillRect(lane_a, track_color)
        painter.fillRect(lane_b, track_color)
        painter.fillRect(waveform_b, track_color)

        if self._engine is not None and self._slot_sources:
            total_frames = self._engine.total_frames
            slot_frames = self._engine.slot_frames
            peaks = {
                "A": max(1e-6, float(np.max(np.abs(self._engine.source_a.samples)))),
                "B": max(1e-6, float(np.max(np.abs(self._engine.source_b.samples)))),
            }
            for index, source_id in enumerate(self._slot_sources):
                start_frame = index * slot_frames
                end_frame = min(start_frame + slot_frames, total_frames)
                x1 = track_left + track_width * start_frame / total_frames
                x2 = track_left + track_width * end_frame / total_frames

                for waveform_source, waveform_rect, active_color in (
                    ("A", waveform_a, QColor(SOURCE_A_COLOR)),
                    ("B", waveform_b, QColor(SOURCE_B_COLOR)),
                ):
                    source_start_frame = self._waveform_start_frames[waveform_source][
                        index
                    ]
                    if source_start_frame is None:
                        continue
                    chunk = self._engine.preview_source_frames(
                        waveform_source, source_start_frame, index
                    )
                    chunk_rect = QRectF(
                        x1,
                        waveform_rect.top(),
                        max(1.0, x2 - x1),
                        waveform_rect.height(),
                    )
                    self._draw_waveform(
                        painter,
                        chunk_rect,
                        chunk,
                        peaks[waveform_source],
                        active_color if waveform_source == source_id else muted_waveform,
                    )

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
                painter.setPen(QColor("#3a3f49"))
                painter.drawLine(
                    round(x2),
                    round(waveform_a.top()),
                    round(x2),
                    round(waveform_b.bottom()),
                )

            if self._engine.duration > 0:
                progress = min(1.0, self._position / self._engine.duration)
                marker_x = track_left + track_width * progress
                painter.setPen(QPen(QColor("#f4f6fa"), 2))
                painter.drawLine(
                    round(marker_x),
                    round(waveform_a.top()),
                    round(marker_x),
                    round(waveform_b.bottom()),
                )

        painter.setPen(QPen(border_color, 1))
        painter.drawRect(waveform_a)
        painter.drawRect(lane_a)
        painter.drawRect(lane_b)
        painter.drawRect(waveform_b)
        painter.end()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Audio Interleaver")
        self.setMinimumSize(760, 900)
        self.resize(860, 940)

        self._source_a: LoadedAudio | None = None
        self._source_b: LoadedAudio | None = None
        self._engine: AudioEngine | AcelpEngine | None = None
        self._stage: ProcessingStage = "raw"
        self._b_encoder_mode: BEncoderMode = "one_stream"
        self._mode = "region"
        self._fill = 0.5
        self._starts_with: SourceId = "A"
        self._first_alternate_slot = 1
        self._b_chunks_per_occurrence = 1
        self._region_b_source_ms = 0
        self._region_output_slot = 0
        self._region_length_slots = 1
        self._region_silence_after_b_end = False
        self._raw_chunk_ms = DEFAULT_CHUNK_MS
        self._acelp_chunk_ms = DEFAULT_CHUNK_MS
        self._chunk_ms = self._raw_chunk_ms
        self._crossfade_ms = DEFAULT_CROSSFADE_MS
        self._loop = False
        self._preview_target: str | None = None
        self._export_thread: threading.Thread | None = None
        self._export_cancel = threading.Event()
        self._exporting = False
        self._acelp_prepare_thread: threading.Thread | None = None
        self._acelp_prepare_cancel = threading.Event()
        self._acelp_preparing = False

        self._signals = _UiSignals()
        self._signals.playback_position.connect(self._on_playback_position)
        self._signals.playback_finished.connect(self._on_playback_finished)
        self._signals.playback_error.connect(self._on_playback_error)
        self._signals.preview_finished.connect(self._on_preview_finished)
        self._signals.preview_error.connect(self._on_preview_error)
        self._signals.export_progress.connect(self._on_export_progress)
        self._signals.export_finished.connect(self._on_export_finished)
        self._signals.export_error.connect(self._on_export_error)
        self._signals.acelp_ready.connect(self._on_acelp_ready)
        self._signals.acelp_prepare_error.connect(self._on_acelp_prepare_error)
        self._signals.acelp_prepare_cancelled.connect(
            self._on_acelp_prepare_cancelled
        )

        self._playback = PlaybackController(
            on_position=self._signals.playback_position.emit,
            on_finished=self._signals.playback_finished.emit,
            on_error=self._signals.playback_error.emit,
        )
        self._preview_playback = AudioPreviewController(
            on_finished=self._signals.preview_finished.emit,
            on_error=self._signals.preview_error.emit,
        )
        self._rendered_playback = RenderedPlaybackController(
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
        self.revision_label = QLabel(f"Commit {_commit_hash()}")
        self.revision_label.setObjectName("revisionLabel")
        subtitle = QLabel(
            "Build repeating chunk patterns from two independent audio sources."
        )
        subtitle.setObjectName("subtitle")
        title_block = QVBoxLayout()
        title_block.setSpacing(3)
        title_block.addWidget(heading)
        metadata_row = QHBoxLayout()
        metadata_row.addWidget(product_subheading)
        metadata_row.addStretch()
        metadata_row.addWidget(self.revision_label)
        title_block.addLayout(metadata_row)
        title_block.addSpacing(7)
        title_block.addWidget(subtitle)
        root.addLayout(title_block)

        cards = QHBoxLayout()
        cards.setSpacing(14)
        self.source_a_card = SourceCard("A", SOURCE_A_COLOR)
        self.source_b_card = SourceCard("B", SOURCE_B_COLOR)
        self.source_a_card.load_requested.connect(lambda: self._load_source("A"))
        self.source_b_card.load_requested.connect(lambda: self._load_source("B"))
        self.source_a_card.preview_requested.connect(
            lambda: self._toggle_source_preview("A")
        )
        self.source_b_card.preview_requested.connect(
            lambda: self._toggle_source_preview("B")
        )
        cards.addWidget(self.source_a_card)
        cards.addWidget(self.source_b_card)
        root.addLayout(cards)

        fader_panel = QFrame()
        fader_panel.setObjectName("faderPanel")
        self.configuration_panel = fader_panel
        fader_layout = QVBoxLayout(fader_panel)
        fader_layout.setContentsMargins(22, 18, 22, 18)
        fader_layout.setSpacing(5)

        stage_header = QHBoxLayout()
        stage_title = QLabel("PROCESSING STAGE")
        stage_title.setObjectName("sectionTitle")
        self.raw_stage_radio = QRadioButton("Raw wave")
        self.acelp_stage_radio = QRadioButton("ACELP symbol insertion")
        self.acelp_stage_radio.setToolTip(
            "Encode A continuously, replace selected frames with independently "
            "encoded B symbols, then decode the mixed stream continuously"
        )
        self.stage_button_group = QButtonGroup(self)
        self.stage_button_group.setExclusive(True)
        self.stage_button_group.addButton(self.raw_stage_radio)
        self.stage_button_group.addButton(self.acelp_stage_radio)
        self.raw_stage_radio.setChecked(True)
        self.acelp_stage_radio.toggled.connect(self._on_stage_changed)
        stage_header.addWidget(stage_title)
        stage_header.addStretch()
        stage_header.addWidget(self.raw_stage_radio)
        stage_header.addWidget(self.acelp_stage_radio)
        fader_layout.addLayout(stage_header)

        self.b_encoder_controls = QWidget()
        b_encoder_layout = QVBoxLayout(self.b_encoder_controls)
        b_encoder_layout.setContentsMargins(0, 0, 0, 0)
        b_encoder_layout.setSpacing(6)
        b_encoder_header = QHBoxLayout()
        self.b_encoder_title = QLabel("B SYMBOL ENCODING")
        self.b_encoder_title.setObjectName("sectionTitle")
        b_encoder_header.addWidget(self.b_encoder_title)
        b_encoder_header.addStretch()
        b_encoder_layout.addLayout(b_encoder_header)
        self.one_stream_radio = QRadioButton("Continuous selected B chunks")
        self.one_stream_radio.setToolTip(
            "Encode active B chunks as one independent stream; B does not "
            "inherit source A encoder state"
        )
        self.restart_chunk_radio = QRadioButton("Restart each B chunk")
        self.restart_chunk_radio.setToolTip(
            "Encode every active B chunk with fresh encoder state; B does not "
            "inherit source A encoder state"
        )
        self.b_encoder_button_group = QButtonGroup(self)
        self.b_encoder_button_group.setExclusive(True)
        self.b_encoder_button_group.addButton(self.one_stream_radio)
        self.b_encoder_button_group.addButton(self.restart_chunk_radio)
        self.one_stream_radio.setChecked(True)
        self.restart_chunk_radio.toggled.connect(self._on_b_encoder_mode_changed)
        b_encoder_options = QHBoxLayout()
        b_encoder_options.addWidget(self.one_stream_radio)
        b_encoder_options.addWidget(self.restart_chunk_radio)
        b_encoder_options.addStretch()
        b_encoder_layout.addLayout(b_encoder_options)
        self.b_encoder_controls.setVisible(False)

        self.acelp_banner = QLabel(
            "Continuous A encode → insert B symbols → continuous mixed decode"
        )
        self.acelp_banner.setToolTip(
            "A is encoded once. Selected A frames are replaced with B frames. "
            "The complete mixed symbol stream is decoded in one pass."
        )
        self.acelp_banner.setObjectName("acelpBanner")
        self.acelp_banner.setWordWrap(True)
        self.acelp_banner.setVisible(False)
        fader_layout.addWidget(self.acelp_banner)

        mode_header = QHBoxLayout()
        mode_title = QLabel("MODE")
        mode_title.setObjectName("sectionTitle")
        self.mode_selector = QComboBox()
        self.mode_selector.addItem("Pattern Interleave", "pattern")
        self.mode_selector.addItem("Region Insert", "region")
        self.mode_selector.setCurrentIndex(1)
        self.mode_selector.currentIndexChanged.connect(self._on_mode_changed)
        mode_header.addWidget(mode_title)
        mode_header.addStretch()
        mode_header.addWidget(self.mode_selector)
        fader_layout.addLayout(mode_header)

        self.pattern_controls = QWidget()
        pattern_layout = QVBoxLayout(self.pattern_controls)
        pattern_layout.setContentsMargins(0, 0, 0, 0)
        pattern_layout.setSpacing(6)

        fader_header = QHBoxLayout()
        fader_title = QLabel("B OCCURRENCE FILL")
        fader_title.setObjectName("sectionTitle")
        self.mix_label = QLabel("Load sources")
        self.mix_label.setObjectName("mixLabel")
        fader_header.addWidget(fader_title)
        fader_header.addStretch()
        fader_header.addWidget(self.mix_label)
        pattern_layout.addLayout(fader_header)

        self.crossfader = QSlider(Qt.Orientation.Horizontal)
        self.crossfader.setRange(0, 1)
        self.crossfader.setValue(0)
        self.crossfader.setEnabled(False)
        self.crossfader.setSingleStep(1)
        self.crossfader.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.crossfader.setTickInterval(1)
        self.crossfader.setToolTip(
            "Increasing fill enables B occurrences from left to right"
        )
        self.crossfader.valueChanged.connect(self._on_crossfader_changed)
        pattern_layout.addWidget(self.crossfader)

        fader_labels = QHBoxLayout()
        self.minimum_occurrences_label = QLabel("No optional occurrences")
        fader_labels.addWidget(self.minimum_occurrences_label)
        fader_labels.addStretch()
        right_label = QLabel("All occurrences")
        right_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        fader_labels.addWidget(right_label)
        pattern_layout.addLayout(fader_labels)

        pattern_options = QHBoxLayout()
        start_source_title = QLabel("STARTING SOURCE")
        start_source_title.setObjectName("sectionTitle")
        self.start_with_a_radio = QRadioButton("Start with A")
        self.start_with_b_radio = QRadioButton("Start with B")
        self.start_with_b_radio.setToolTip(
            "Make output chunk 1 use source B"
        )
        self.start_source_button_group = QButtonGroup(self)
        self.start_source_button_group.setExclusive(True)
        self.start_source_button_group.addButton(self.start_with_a_radio)
        self.start_source_button_group.addButton(self.start_with_b_radio)
        self.start_with_a_radio.setChecked(True)
        self.start_with_b_radio.toggled.connect(self._on_start_source_changed)
        pattern_options.addWidget(start_source_title)
        pattern_options.addStretch()
        pattern_options.addWidget(self.start_with_a_radio)
        pattern_options.addWidget(self.start_with_b_radio)
        pattern_layout.addSpacing(6)
        pattern_layout.addLayout(pattern_options)

        alternate_header = QHBoxLayout()
        self.first_alternate_title = QLabel("FIRST B CHUNK POSITION")
        self.first_alternate_title.setObjectName("sectionTitle")
        self.first_alternate_label = QLabel("Chunk 2")
        self.first_alternate_label.setObjectName("settingValue")
        alternate_header.addWidget(self.first_alternate_title)
        alternate_header.addStretch()
        alternate_header.addWidget(self.first_alternate_label)
        pattern_layout.addLayout(alternate_header)

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
        pattern_layout.addWidget(self.first_alternate_slider)

        burst_header = QHBoxLayout()
        burst_title = QLabel("B CHUNKS PER OCCURRENCE")
        burst_title.setObjectName("sectionTitle")
        self.burst_size_label = QLabel("1 chunk")
        self.burst_size_label.setObjectName("settingValue")
        burst_header.addWidget(burst_title)
        burst_header.addStretch()
        burst_header.addWidget(self.burst_size_label)
        pattern_layout.addLayout(burst_header)

        self.burst_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.burst_size_slider.setRange(1, 1)
        self.burst_size_slider.setValue(1)
        self.burst_size_slider.setEnabled(False)
        self.burst_size_slider.setSingleStep(1)
        self.burst_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.burst_size_slider.setTickInterval(1)
        self.burst_size_slider.valueChanged.connect(self._on_burst_size_changed)
        pattern_layout.addWidget(self.burst_size_slider)
        self.pattern_controls.setVisible(False)
        fader_layout.addWidget(self.pattern_controls)

        self.region_controls = QWidget()
        region_layout = QVBoxLayout(self.region_controls)
        region_layout.setContentsMargins(0, 0, 0, 0)
        region_layout.setSpacing(6)

        region_source_header = QHBoxLayout()
        region_source_title = QLabel("B SOURCE POSITION")
        region_source_title.setObjectName("sectionTitle")
        self.region_source_label = QLabel(_format_position_ms(0))
        self.region_source_label.setObjectName("settingValue")
        region_source_header.addWidget(region_source_title)
        region_source_header.addStretch()
        region_source_header.addWidget(self.region_source_label)
        region_layout.addLayout(region_source_header)
        self.region_source_slider = QSlider(Qt.Orientation.Horizontal)
        self.region_source_slider.setRange(0, 0)
        self.region_source_slider.setEnabled(False)
        self.region_source_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.region_source_slider.setTickInterval(1)
        self.region_source_slider.setToolTip(
            "Start of the continuous B selection: 1 ms steps in Raw mode, "
            "30 ms codec-frame steps in ACELP mode"
        )
        self.region_source_slider.valueChanged.connect(
            self._on_region_source_changed
        )
        region_layout.addWidget(self.region_source_slider)

        region_length_header = QHBoxLayout()
        region_length_title = QLabel("REGION LENGTH")
        region_length_title.setObjectName("sectionTitle")
        self.region_length_label = QLabel("1 chunk")
        self.region_length_label.setObjectName("settingValue")
        self.region_silence_checkbox = QCheckBox("Add silence after B ends")
        self.region_silence_checkbox.setEnabled(False)
        self.region_silence_checkbox.toggled.connect(
            self._on_region_silence_changed
        )
        region_length_header.addWidget(region_length_title)
        region_length_header.addStretch()
        region_length_header.addWidget(self.region_silence_checkbox)
        region_length_header.addSpacing(12)
        region_length_header.addWidget(self.region_length_label)
        region_layout.addLayout(region_length_header)
        self.region_length_slider = QSlider(Qt.Orientation.Horizontal)
        self.region_length_slider.setRange(1, 1)
        self.region_length_slider.setEnabled(False)
        self.region_length_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.region_length_slider.setTickInterval(1)
        self.region_length_slider.valueChanged.connect(
            self._on_region_length_changed
        )
        region_layout.addWidget(self.region_length_slider)

        region_output_header = QHBoxLayout()
        self.region_output_title = QLabel("INSERT POSITION IN A")
        self.region_output_title.setObjectName("sectionTitle")
        self.region_output_label = QLabel("Chunk 1")
        self.region_output_label.setObjectName("settingValue")
        region_output_header.addWidget(self.region_output_title)
        region_output_header.addStretch()
        region_output_header.addWidget(self.region_output_label)
        region_layout.addLayout(region_output_header)
        self.region_output_slider = QSlider(Qt.Orientation.Horizontal)
        self.region_output_slider.setRange(1, 1)
        self.region_output_slider.setEnabled(False)
        self.region_output_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.region_output_slider.setTickInterval(1)
        self.region_output_slider.valueChanged.connect(
            self._on_region_output_changed
        )
        region_layout.addWidget(self.region_output_slider)
        fader_layout.addWidget(self.region_controls)

        self.duration_controls = QWidget()
        self.duration_controls.setObjectName("durationControls")
        duration_controls_layout = QHBoxLayout(self.duration_controls)
        duration_controls_layout.setContentsMargins(0, 0, 0, 0)
        duration_controls_layout.setSpacing(18)

        self.chunk_duration_group = QWidget()
        chunk_group_layout = QVBoxLayout(self.chunk_duration_group)
        chunk_group_layout.setContentsMargins(0, 0, 0, 0)
        chunk_group_layout.setSpacing(6)
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
            self._on_chunk_input_changed
        )
        chunk_header.addWidget(chunk_title)
        chunk_header.addStretch()
        chunk_header.addWidget(self.chunk_duration_input)
        chunk_group_layout.addLayout(chunk_header)

        self.chunk_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.chunk_duration_slider.setRange(MIN_CHUNK_MS, MAX_CHUNK_MS)
        self.chunk_duration_slider.setValue(self._chunk_ms)
        self.chunk_duration_slider.setSingleStep(10)
        self.chunk_duration_slider.setPageStep(50)
        self.chunk_duration_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.chunk_duration_slider.setTickInterval(250)
        self.chunk_duration_slider.valueChanged.connect(
            self._on_chunk_slider_changed
        )
        chunk_group_layout.addWidget(self.chunk_duration_slider)

        self.crossfade_duration_group = QWidget()
        crossfade_group_layout = QVBoxLayout(self.crossfade_duration_group)
        crossfade_group_layout.setContentsMargins(0, 0, 0, 0)
        crossfade_group_layout.setSpacing(6)
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
        crossfade_group_layout.addLayout(transition_header)

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
        crossfade_group_layout.addWidget(self.crossfade_duration_slider)

        duration_controls_layout.addWidget(self.chunk_duration_group, 2)
        duration_controls_layout.addWidget(self.crossfade_duration_group, 2)
        duration_controls_layout.addWidget(self.b_encoder_controls, 3)
        fader_layout.addSpacing(8)
        fader_layout.addWidget(self.duration_controls)

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
        region_preview_row = QHBoxLayout()
        region_preview_row.addStretch()
        self.region_preview_button = QPushButton("Preview B region")
        self.region_preview_button.clicked.connect(self._toggle_region_preview)
        self.region_preview_button.setVisible(True)
        self.region_preview_button.setEnabled(False)
        region_preview_row.addWidget(self.region_preview_button)
        fader_layout.addLayout(region_preview_row)
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
        self.symbol_export_button = QPushButton("Export ACELP symbols…")
        self.symbol_export_button.clicked.connect(self._export_symbols)
        self.symbol_export_button.setVisible(False)
        self.loop_checkbox = QCheckBox("Loop")
        self.loop_checkbox.setToolTip("Restart the complete result when playback ends")
        self.loop_checkbox.toggled.connect(self._on_loop_changed)
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("timeLabel")
        controls.addWidget(self.play_button)
        controls.addWidget(self.export_button)
        controls.addWidget(self.symbol_export_button)
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
            QLabel#acelpBanner {
                color: #f5d08a; background: #3a3223; border: 1px solid #66552f;
                border-radius: 5px; padding: 4px 6px;
            }
            QFrame#sourceCard, QFrame#faderPanel, QFrame#transport {
                background: #20232b; border: 1px solid #30343e; border-radius: 10px;
            }
            QLabel#sourceTitle, QLabel#sectionTitle { font-size: 11px; font-weight: 700; }
            QLabel#fileName { font-size: 16px; font-weight: 600; }
            QLabel#sourceDetails, QLabel#status { color: #9ca3b2; }
            QLabel#mixLabel, QLabel#settingValue { color: #c4c8d2; font-weight: 600; }
            QLabel#revisionLabel { color: #6f7787; font-size: 10px; }
            QSpinBox#durationInput {
                color: #c4c8d2; background: #303541; border: 1px solid #434956;
                border-radius: 4px; padding: 3px 6px; font-weight: 600;
            }
            QComboBox {
                color: #c4c8d2; background: #303541; border: 1px solid #434956;
                border-radius: 4px; padding: 4px 8px; font-weight: 600;
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
            QCheckBox, QRadioButton { spacing: 7px; color: #c4c8d2; font-weight: 600; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QRadioButton::indicator { width: 16px; height: 16px; }
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

        self._stop_all_playback(wait=True, reset=True)
        if source_id == "A":
            self._source_a = audio
            self.source_a_card.display_audio(audio)
        else:
            self._source_b = audio
            self.source_b_card.display_audio(audio)

        self._rebuild_engine()

    def _on_crossfader_changed(self, value: int) -> None:
        capacity = self.crossfader.maximum()
        self._fill = value / capacity if capacity > 0 else 0.0
        self.mix_label.setText(f"{value} / {capacity} occurrences")
        if self._mode == "pattern":
            self.interleave_timeline.set_settings(self._settings())

    def _on_mode_changed(self, index: int) -> None:
        if self._preview_target == "region-B":
            self._stop_preview(wait=True)
        self._mode = str(self.mode_selector.itemData(index))
        self.pattern_controls.setVisible(self._mode == "pattern")
        self.region_controls.setVisible(self._mode == "region")
        self.region_preview_button.setVisible(self._mode == "region")
        self._sync_region_controls()
        self.interleave_timeline.set_settings(self._settings())
        self._refresh_actions()

    def _on_region_source_changed(self, position_step: int) -> None:
        if self._preview_target == "region-B":
            self._stop_preview(wait=True)
        step_ms = ACELP_FRAME_MS if self._stage == "acelp" else 1
        self._region_b_source_ms = position_step * step_ms
        self.region_source_label.setText(
            _format_position_ms(self._region_b_source_ms)
        )
        if self._mode == "region":
            self.interleave_timeline.set_settings(self._settings())

    def _on_region_length_changed(self, chunks: int) -> None:
        if self._preview_target == "region-B":
            self._stop_preview(wait=True)
        self._region_length_slots = chunks
        self._sync_region_controls()
        if self._mode == "region":
            self.interleave_timeline.set_settings(self._settings())

    def _on_region_output_changed(self, chunk_number: int) -> None:
        self._region_output_slot = chunk_number - 1
        self._sync_region_controls()
        if self._mode == "region":
            self.interleave_timeline.set_settings(self._settings())

    def _on_region_silence_changed(self, checked: bool) -> None:
        if self._preview_target == "region-B":
            self._stop_preview(wait=True)
        self._region_silence_after_b_end = checked
        self._sync_region_controls()
        if self._mode == "region":
            self.interleave_timeline.set_settings(self._settings())

    def _on_start_source_changed(self, starts_with_b: bool) -> None:
        self._starts_with = "B" if starts_with_b else "A"
        alternate = "A" if starts_with_b else "B"
        self.first_alternate_title.setText(
            f"FIRST {alternate} CHUNK POSITION"
        )
        if starts_with_b:
            self._set_first_alternate_slot(self._b_chunks_per_occurrence)
        self._pattern_changed()

    def _on_first_alternate_changed(self, chunk_number: int) -> None:
        self._first_alternate_slot = chunk_number - 1
        self.first_alternate_label.setText(f"Chunk {chunk_number}")
        self._pattern_changed()

    def _on_burst_size_changed(self, chunks: int) -> None:
        self._b_chunks_per_occurrence = chunks
        self.burst_size_label.setText(_format_chunk_count(chunks))
        if self._starts_with == "B":
            self._set_first_alternate_slot(chunks)
        self._pattern_changed()

    def _set_first_alternate_slot(self, slot_index: int) -> None:
        self._first_alternate_slot = slot_index
        self.first_alternate_slider.blockSignals(True)
        self.first_alternate_slider.setValue(slot_index + 1)
        self.first_alternate_slider.blockSignals(False)
        label = (
            f"Chunk {slot_index + 1}"
            if slot_index + 1 <= self.first_alternate_slider.maximum()
            else "After output"
        )
        self.first_alternate_label.setText(label)

    def _pattern(self) -> InterleavePattern:
        return InterleavePattern(
            fill=self._fill,
            starts_with=self._starts_with,
            first_alternate_slot=self._first_alternate_slot,
            b_chunks_per_occurrence=self._b_chunks_per_occurrence,
        )

    def _region(self) -> RegionInsert:
        return RegionInsert(
            b_source_ms=self._region_b_source_ms,
            output_slot=self._region_output_slot,
            length_slots=self._region_length_slots,
            silence_after_b_end=self._region_silence_after_b_end,
        )

    def _settings(self) -> InterleaveSettings:
        return self._region() if self._mode == "region" else self._pattern()

    def _on_stage_changed(self, acelp_enabled: bool) -> None:
        self._stop_all_playback(wait=True, reset=True)
        self._stage = "acelp" if acelp_enabled else "raw"
        self.b_encoder_controls.setVisible(acelp_enabled)
        self.acelp_banner.setVisible(acelp_enabled)
        self.symbol_export_button.setVisible(acelp_enabled)
        self.crossfade_duration_group.setVisible(not acelp_enabled)
        self._chunk_ms = (
            self._acelp_chunk_ms if acelp_enabled else self._raw_chunk_ms
        )
        self._sync_chunk_duration_controls()
        self._rebuild_engine()

    def _on_b_encoder_mode_changed(self, restart_each_chunk: bool) -> None:
        self._b_encoder_mode = (
            "restart_each_chunk" if restart_each_chunk else "one_stream"
        )

    def _pattern_changed(self) -> None:
        self._sync_occurrence_control()
        if self._mode == "pattern":
            self.interleave_timeline.set_settings(self._settings())

    def _on_chunk_slider_changed(self, value: int) -> None:
        duration = value * ACELP_FRAME_MS if self._stage == "acelp" else value
        self._set_chunk_duration(duration)

    def _on_chunk_input_changed(self, value: int) -> None:
        duration = snap_acelp_chunk_ms(value) if self._stage == "acelp" else value
        self._set_chunk_duration(duration)

    def _on_chunk_duration_changed(self, value: int) -> None:
        """Compatibility entry point used by callers that provide milliseconds."""
        self._on_chunk_input_changed(value)

    def _set_chunk_duration(self, value: int) -> None:
        self.chunk_duration_slider.blockSignals(True)
        self.chunk_duration_input.blockSignals(True)
        slider_value = value // ACELP_FRAME_MS if self._stage == "acelp" else value
        self.chunk_duration_slider.setValue(slider_value)
        self.chunk_duration_input.setValue(value)
        self.chunk_duration_slider.blockSignals(False)
        self.chunk_duration_input.blockSignals(False)
        self._chunk_ms = value
        if self._stage == "acelp":
            self._acelp_chunk_ms = value
        else:
            self._raw_chunk_ms = value
        self.preview_detail.setText(f"Each block = {value} ms")
        self._configuration_changed()

    def _sync_chunk_duration_controls(self) -> None:
        self.chunk_duration_slider.blockSignals(True)
        self.chunk_duration_input.blockSignals(True)
        if self._stage == "acelp":
            self.chunk_duration_slider.setRange(
                1, ACELP_MAX_CHUNK_MS // ACELP_FRAME_MS
            )
            self.chunk_duration_slider.setSingleStep(1)
            self.chunk_duration_slider.setPageStep(4)
            self.chunk_duration_slider.setTickInterval(10)
            self.chunk_duration_slider.setValue(self._chunk_ms // ACELP_FRAME_MS)
            self.chunk_duration_input.setRange(ACELP_FRAME_MS, ACELP_MAX_CHUNK_MS)
            self.chunk_duration_input.setSingleStep(ACELP_FRAME_MS)
        else:
            self.chunk_duration_slider.setRange(MIN_CHUNK_MS, MAX_CHUNK_MS)
            self.chunk_duration_slider.setSingleStep(10)
            self.chunk_duration_slider.setPageStep(50)
            self.chunk_duration_slider.setTickInterval(250)
            self.chunk_duration_slider.setValue(self._chunk_ms)
            self.chunk_duration_input.setRange(MIN_CHUNK_MS, MAX_CHUNK_MS)
            self.chunk_duration_input.setSingleStep(1)
        self.chunk_duration_input.setValue(self._chunk_ms)
        self.chunk_duration_slider.blockSignals(False)
        self.chunk_duration_input.blockSignals(False)
        self.preview_detail.setText(f"Each block = {self._chunk_ms} ms")

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
        self._stop_all_playback(wait=True, reset=True)
        self._rebuild_engine()

    def _rebuild_engine(self) -> None:
        try:
            if self._source_a is not None and self._source_b is not None:
                if self._stage == "acelp":
                    self._engine = AcelpEngine(
                        self._source_a,
                        self._source_b,
                        slot_ms=self._chunk_ms,
                    )
                else:
                    self._engine = AudioEngine(
                        self._source_a,
                        self._source_b,
                        slot_ms=self._chunk_ms,
                        smoothing_ms=self._crossfade_ms,
                    )
            else:
                self._engine = None
        except (AudioError, ValueError) as exc:
            self._engine = None
            QMessageBox.critical(self, "Could not prepare audio", str(exc))

        if self._engine is not None:
            if self._stage == "acelp":
                self.status_label.setText(
                    f"Ready • ACELP • {self._chunk_ms} ms chunks • "
                    f"{_format_time(self._engine.duration)} source-A output"
                )
            else:
                self.status_label.setText(
                    f"Ready • {self._chunk_ms} ms chunks • "
                    f"{self._crossfade_ms} ms crossfade • "
                    f"{_format_time(self._engine.duration)} output"
                )
        else:
            self.status_label.setText("Load the other WAV file to begin.")
        self._sync_pattern_controls()
        self._sync_region_controls()
        self.interleave_timeline.set_engine(self._engine)
        self.interleave_timeline.set_settings(self._settings())
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
        self.burst_size_label.setText(
            _format_chunk_count(self._b_chunks_per_occurrence)
        )
        self.first_alternate_slider.blockSignals(False)
        self.burst_size_slider.blockSignals(False)
        self._sync_occurrence_control()

    def _sync_occurrence_control(self) -> None:
        if self._engine is None:
            self.crossfader.blockSignals(True)
            self.crossfader.setRange(0, 1)
            self.crossfader.setValue(0)
            self.crossfader.setEnabled(False)
            self.crossfader.blockSignals(False)
            self.mix_label.setText("Load sources")
            return

        capacity = occurrence_capacity(self._engine.slot_count, self._pattern())
        selected = min(capacity, max(0, math.floor(self._fill * capacity + 0.5)))
        self.crossfader.blockSignals(True)
        self.crossfader.setRange(0, max(1, capacity))
        self.crossfader.setValue(selected)
        self.crossfader.setEnabled(capacity > 0)
        self.crossfader.setTickInterval(1)
        self.crossfader.blockSignals(False)
        if capacity > 0:
            self._fill = selected / capacity
            self.mix_label.setText(f"{selected} / {capacity} occurrences")
        else:
            self._fill = 0.0
            self.mix_label.setText("No occurrences")

    def _sync_region_controls(self) -> None:
        controls = (
            self.region_source_slider,
            self.region_length_slider,
            self.region_output_slider,
        )
        for control in controls:
            control.blockSignals(True)
        self.region_silence_checkbox.blockSignals(True)

        if self._engine is None:
            self._region_b_source_ms = 0
            self._region_output_slot = 0
            self._region_length_slots = 1
            self.region_source_slider.setRange(0, 0)
            self.region_source_slider.setValue(0)
            self.region_source_slider.setEnabled(False)
            for control in (self.region_length_slider, self.region_output_slider):
                control.setRange(1, 1)
                control.setValue(1)
                control.setEnabled(False)
            self.region_silence_checkbox.setEnabled(False)
        else:
            source_chunks = self._engine.source_chunk_count("B")
            output_chunks = self._engine.slot_count
            self._region_output_slot = min(
                max(0, self._region_output_slot), output_chunks - 1
            )
            if self._region_silence_after_b_end:
                max_length = output_chunks - self._region_output_slot
            else:
                max_length = source_chunks
                self._region_length_slots = min(
                    max(1, self._region_length_slots), max_length
                )
            self._region_length_slots = min(
                max(1, self._region_length_slots), max_length
            )
            step_ms = ACELP_FRAME_MS if self._stage == "acelp" else 1
            step_frames = max(
                1, round(self._engine.sample_rate * step_ms / 1000.0)
            )
            if self._region_silence_after_b_end:
                max_source_start_frame = max(0, self._engine.source_b.frames - 1)
            else:
                padded_source_frames = source_chunks * self._engine.slot_frames
                region_frames = self._region_length_slots * self._engine.slot_frames
                max_source_start_frame = max(
                    0, padded_source_frames - region_frames
                )
            max_source_step = max_source_start_frame // step_frames
            current_source_step = min(
                max_source_step,
                max(0, round(self._region_b_source_ms / step_ms)),
            )
            self._region_b_source_ms = current_source_step * step_ms
            self.region_source_slider.setRange(0, max_source_step)
            self.region_source_slider.setSingleStep(1)
            self.region_source_slider.setPageStep(
                10 if self._stage == "acelp" else 100
            )
            self.region_source_slider.setTickInterval(
                10 if self._stage == "acelp" else 1000
            )
            self.region_source_slider.setValue(current_source_step)
            self.region_length_slider.setRange(1, max_length)
            self.region_length_slider.setValue(self._region_length_slots)
            self.region_output_slider.setRange(1, output_chunks)
            self.region_output_slider.setValue(self._region_output_slot + 1)
            for control in controls:
                control.setEnabled(True)
            self.region_silence_checkbox.setEnabled(True)

        self.region_source_label.setText(
            _format_position_ms(self._region_b_source_ms)
        )
        self.region_length_label.setText(
            _format_chunk_count(self._region_length_slots)
        )
        self.region_output_label.setText(f"Chunk {self._region_output_slot + 1}")
        for control in controls:
            control.blockSignals(False)
        self.region_silence_checkbox.blockSignals(False)

    def _on_loop_changed(self, checked: bool) -> None:
        self._loop = checked

    def _toggle_playback(self) -> None:
        if (
            self._playback.is_playing
            or self._rendered_playback.is_playing
            or self._acelp_preparing
        ):
            self._stop_playback(wait=self._acelp_preparing, reset=True)
            return
        if self._engine is None:
            return
        self._stop_preview(wait=True)
        self.progress.setValue(0)
        self._update_time(0.0)
        if self._stage == "acelp":
            assert isinstance(self._engine, AcelpEngine)
            self._acelp_prepare_cancel.clear()
            self._acelp_preparing = True
            self.play_button.setText("Stop")
            self.status_label.setText("Preparing ACELP symbols and decoded audio…")
            engine = self._engine
            settings = self._settings()
            b_encoder_mode = self._b_encoder_mode
            self._acelp_prepare_thread = threading.Thread(
                target=self._prepare_acelp_playback,
                args=(engine, settings, b_encoder_mode),
                name="acelp-playback-prepare",
                daemon=True,
            )
            self._acelp_prepare_thread.start()
            self._refresh_actions()
            return
        if self._playback.start(self._engine, self._settings, lambda: self._loop):
            self.play_button.setText("Stop")
            self.status_label.setText(
                f"Playing • interleave changes apply at the next "
                f"{self._chunk_ms} ms boundary"
            )

    def _prepare_acelp_playback(
        self,
        engine: AcelpEngine,
        settings: InterleaveSettings,
        b_encoder_mode: BEncoderMode,
    ) -> None:
        try:
            rendered = engine.render(
                settings,
                b_encoder_mode,
                progress=lambda value: self._signals.export_progress.emit(
                    round(value * 1000)
                ),
                cancel_event=self._acelp_prepare_cancel,
            )
            audio = LoadedAudio(rendered, engine.sample_rate)
        except RenderingCancelled:
            self._signals.acelp_prepare_cancelled.emit()
        except Exception as exc:
            self._signals.acelp_prepare_error.emit(str(exc))
        else:
            self._signals.acelp_ready.emit(audio)

    def _on_acelp_ready(self, audio: LoadedAudio) -> None:
        if (
            self._acelp_prepare_thread is not None
            and not self._acelp_prepare_thread.is_alive()
        ):
            self._acelp_prepare_thread = None
        self._acelp_preparing = False
        if self._acelp_prepare_cancel.is_set():
            self._refresh_actions()
            return
        if self._rendered_playback.start(audio, lambda: self._loop):
            self.status_label.setText("Playing fixed ACELP symbol-stage result")
        else:
            self.play_button.setText("Play")
        self._refresh_actions()

    def _on_acelp_prepare_error(self, message: str) -> None:
        self._acelp_prepare_thread = None
        self._acelp_preparing = False
        self.play_button.setText("Play")
        self.status_label.setText("ACELP preparation failed")
        self._refresh_actions()
        QMessageBox.critical(self, "ACELP preparation failed", message)

    def _on_acelp_prepare_cancelled(self) -> None:
        self._acelp_prepare_thread = None
        self._acelp_preparing = False
        self.play_button.setText("Play")
        if self._engine is not None and not self._exporting:
            self.status_label.setText("Ready")
        self._refresh_actions()

    def _stop_playback(self, wait: bool = False, reset: bool = False) -> None:
        self._acelp_prepare_cancel.set()
        self._playback.stop(wait=wait)
        self._rendered_playback.stop(wait=wait)
        if (
            wait
            and self._acelp_prepare_thread is not None
            and self._acelp_prepare_thread is not threading.current_thread()
        ):
            self._acelp_prepare_thread.join(timeout=2.0)
        if (
            self._acelp_prepare_thread is not None
            and not self._acelp_prepare_thread.is_alive()
        ):
            self._acelp_prepare_thread = None
        self._acelp_preparing = (
            self._acelp_prepare_thread is not None
            and self._acelp_prepare_thread.is_alive()
        )
        self.play_button.setText("Play")
        if reset:
            self.progress.setValue(0)
            self._update_time(0.0)
        if self._engine is not None and not self._exporting:
            self.status_label.setText(
                "Stopping ACELP preparation…" if self._acelp_preparing else "Ready"
            )
        self._refresh_actions()

    def _toggle_source_preview(self, source_id: SourceId) -> None:
        target = f"source-{source_id}"
        if self._preview_playback.is_playing and self._preview_target == target:
            self._stop_preview()
            return
        audio = self._source_a if source_id == "A" else self._source_b
        if audio is None:
            return
        self._start_preview(audio, target)

    def _toggle_region_preview(self) -> None:
        target = "region-B"
        if self._preview_playback.is_playing and self._preview_target == target:
            self._stop_preview()
            return
        if self._engine is None or self._mode != "region":
            return
        samples = self._engine.source_region(
            "B",
            self._region_b_source_ms,
            self._region_length_slots,
            self._region_silence_after_b_end,
        )
        audio = LoadedAudio(samples, self._engine.sample_rate)
        self._start_preview(audio, target)

    def _start_preview(self, audio: LoadedAudio, target: str) -> None:
        self._stop_playback(wait=True, reset=True)
        self._stop_preview(wait=True)
        self._preview_target = target
        if self._preview_playback.start(audio, target):
            self._sync_preview_buttons()
            description = (
                "selected B region" if target == "region-B" else f"source {target[-1]}"
            )
            self.status_label.setText(f"Previewing {description}")
        else:
            self._preview_target = None
            self._sync_preview_buttons()

    def _stop_preview(self, wait: bool = False) -> None:
        self._preview_playback.stop(wait=wait)
        self._preview_target = None
        self._sync_preview_buttons()
        if self._engine is not None and not self._exporting:
            self.status_label.setText("Ready")

    def _stop_all_playback(self, wait: bool = False, reset: bool = False) -> None:
        self._stop_playback(wait=wait, reset=reset)
        self._stop_preview(wait=wait)

    def _sync_preview_buttons(self) -> None:
        self.source_a_card.preview_button.setText(
            "Stop preview" if self._preview_target == "source-A" else "Play preview"
        )
        self.source_b_card.preview_button.setText(
            "Stop preview" if self._preview_target == "source-B" else "Play preview"
        )
        self.region_preview_button.setText(
            "Stop B region"
            if self._preview_target == "region-B"
            else "Preview B region"
        )

    def _on_preview_finished(self, target: str, natural: bool) -> None:
        if target != self._preview_target:
            return
        self._preview_target = None
        self._sync_preview_buttons()
        if not self._exporting:
            self.status_label.setText("Preview finished" if natural else "Ready")

    def _on_preview_error(self, message: str) -> None:
        QMessageBox.critical(self, "Audio preview error", message)

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
        self._refresh_actions()

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

        self._stop_all_playback(wait=True, reset=True)
        engine = self._engine
        snapshot = self._settings()
        b_encoder_mode = self._b_encoder_mode
        self._export_cancel.clear()
        self._exporting = True
        self.progress.setValue(0)
        self.status_label.setText(
            "Exporting current interleave settings…"
        )
        self._refresh_actions()

        self._export_thread = threading.Thread(
            target=self._render_export,
            args=(engine, snapshot, b_encoder_mode, output_path),
            name="audio-export",
            daemon=True,
        )
        self._export_thread.start()

    def _render_export(
        self,
        engine: AudioEngine | AcelpEngine,
        settings: InterleaveSettings,
        b_encoder_mode: BEncoderMode,
        output_path: Path,
    ) -> None:
        try:
            if isinstance(engine, AcelpEngine):
                rendered = engine.render(
                    settings,
                    b_encoder_mode,
                    progress=lambda value: self._signals.export_progress.emit(
                        round(value * 1000)
                    ),
                    cancel_event=self._export_cancel,
                )
            else:
                rendered = engine.render(
                    settings,
                    progress=lambda value: self._signals.export_progress.emit(
                        round(value * 1000)
                    ),
                    cancel_event=self._export_cancel,
                )
            write_wav(output_path, rendered, engine.sample_rate)
        except RenderingCancelled:
            return
        except Exception as exc:
            self._signals.export_error.emit(str(exc))
        else:
            self._signals.export_finished.emit(str(output_path))

    def _export_symbols(self) -> None:
        if (
            not isinstance(self._engine, AcelpEngine)
            or self._exporting
            or self._acelp_preparing
            or self._rendered_playback.is_playing
        ):
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export interleaved ACELP symbols",
            "interleaved.spe",
            "ETSI ACELP symbols (*.spe)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() != ".spe":
            output_path = output_path.with_suffix(".spe")

        self._stop_all_playback(wait=True, reset=True)
        engine = self._engine
        settings = self._settings()
        b_encoder_mode = self._b_encoder_mode
        self._export_cancel.clear()
        self._exporting = True
        self.progress.setValue(0)
        self.status_label.setText("Exporting ACELP symbol stream…")
        self._refresh_actions()
        self._export_thread = threading.Thread(
            target=self._render_symbol_export,
            args=(engine, settings, b_encoder_mode, output_path),
            name="acelp-symbol-export",
            daemon=True,
        )
        self._export_thread.start()

    def _render_symbol_export(
        self,
        engine: AcelpEngine,
        settings: InterleaveSettings,
        b_encoder_mode: BEncoderMode,
        output_path: Path,
    ) -> None:
        try:
            symbols = engine.render_symbols(
                settings,
                b_encoder_mode,
                progress=lambda value: self._signals.export_progress.emit(
                    round(value * 1000)
                ),
                cancel_event=self._export_cancel,
            )
            symbols.write_spe(output_path)
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
        acelp_busy = self._stage == "acelp" and (
            self._acelp_preparing or self._rendered_playback.is_playing
        )
        self.configuration_panel.setEnabled(not acelp_busy and not self._exporting)
        self.play_button.setEnabled(
            ready
            and not (
                self._stage == "acelp"
                and self._acelp_preparing
                and self._acelp_prepare_cancel.is_set()
            )
        )
        self.export_button.setEnabled(ready and not acelp_busy)
        self.symbol_export_button.setEnabled(
            ready and self._stage == "acelp" and not acelp_busy
        )
        self.source_a_card.load_button.setEnabled(
            not self._exporting and not acelp_busy
        )
        self.source_b_card.load_button.setEnabled(
            not self._exporting and not acelp_busy
        )
        self.source_a_card.preview_button.setEnabled(
            self._source_a is not None and not self._exporting and not acelp_busy
        )
        self.source_b_card.preview_button.setEnabled(
            self._source_b is not None and not self._exporting and not acelp_busy
        )
        self.region_preview_button.setEnabled(
            ready and self._mode == "region" and not acelp_busy
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self._playback.stop(wait=True)
        self._rendered_playback.stop(wait=True)
        self._preview_playback.stop(wait=True)
        self._acelp_prepare_cancel.set()
        if self._acelp_prepare_thread is not None:
            self._acelp_prepare_thread.join(timeout=2.0)
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
