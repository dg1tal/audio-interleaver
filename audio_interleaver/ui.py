"""PySide6 desktop interface for the audio interleaver."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QCloseEvent, QFont
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
    QVBoxLayout,
    QWidget,
)

from .audio import (
    AudioEngine,
    AudioError,
    LoadedAudio,
    RenderingCancelled,
    load_wav,
    write_wav,
)
from .playback import PlaybackController


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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Audio Interleaver")
        self.setMinimumSize(760, 590)
        self.resize(860, 650)

        self._source_a: LoadedAudio | None = None
        self._source_b: LoadedAudio | None = None
        self._engine: AudioEngine | None = None
        self._crossfader = 0.5
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
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(20)

        heading = QLabel("Audio Interleaver")
        heading.setObjectName("heading")
        subtitle = QLabel(
            "Blend by selection: every 360 ms, choose a piece from A or B."
        )
        subtitle.setObjectName("subtitle")
        root.addWidget(heading)
        root.addWidget(subtitle)

        cards = QHBoxLayout()
        cards.setSpacing(14)
        self.source_a_card = SourceCard("A", "#6ea8fe")
        self.source_b_card = SourceCard("B", "#f08cba")
        self.source_a_card.load_requested.connect(lambda: self._load_source("A"))
        self.source_b_card.load_requested.connect(lambda: self._load_source("B"))
        cards.addWidget(self.source_a_card)
        cards.addWidget(self.source_b_card)
        root.addLayout(cards)

        fader_panel = QFrame()
        fader_panel.setObjectName("faderPanel")
        fader_layout = QVBoxLayout(fader_panel)
        fader_layout.setContentsMargins(22, 18, 22, 18)
        fader_layout.setSpacing(8)

        fader_header = QHBoxLayout()
        fader_title = QLabel("CROSSFADER")
        fader_title.setObjectName("sectionTitle")
        self.mix_label = QLabel("A 50%  •  B 50%")
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
        fader_labels.addWidget(QLabel("A only"))
        center_label = QLabel("Alternating")
        center_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fader_labels.addWidget(center_label, 1)
        right_label = QLabel("B only")
        right_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        fader_labels.addWidget(right_label)
        fader_layout.addLayout(fader_labels)
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
            QLabel#subtitle { color: #9ca3b2; font-size: 14px; }
            QFrame#sourceCard, QFrame#faderPanel, QFrame#transport {
                background: #20232b; border: 1px solid #30343e; border-radius: 10px;
            }
            QLabel#sourceTitle, QLabel#sectionTitle { font-size: 11px; font-weight: 700; }
            QLabel#fileName { font-size: 16px; font-weight: 600; }
            QLabel#sourceDetails, QLabel#status { color: #9ca3b2; }
            QLabel#mixLabel { color: #c4c8d2; font-weight: 600; }
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

        try:
            self._engine = (
                AudioEngine(self._source_a, self._source_b)
                if self._source_a is not None and self._source_b is not None
                else None
            )
        except AudioError as exc:
            self._engine = None
            QMessageBox.critical(self, "Could not prepare audio", str(exc))

        if self._engine is not None:
            self.status_label.setText(
                f"Ready • 360 ms slots • {_format_time(self._engine.duration)} output"
            )
        else:
            self.status_label.setText("Load the other WAV file to begin.")
        self._update_time(0.0)
        self._refresh_actions()

    def _on_crossfader_changed(self, value: int) -> None:
        self._crossfader = value / 100.0
        self.mix_label.setText(f"A {100 - value}%  •  B {value}%")

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
        if self._playback.start(
            self._engine, lambda: self._crossfader, lambda: self._loop
        ):
            self.play_button.setText("Stop")
            self.status_label.setText(
                "Playing • crossfader changes apply at the next 360 ms boundary"
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
        snapshot = self._crossfader
        self._export_cancel.clear()
        self._exporting = True
        self.progress.setValue(0)
        self.status_label.setText(
            f"Exporting snapshot at A {round((1 - snapshot) * 100)}% / B {round(snapshot * 100)}%…"
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
        self, engine: AudioEngine, crossfader: float, output_path: Path
    ) -> None:
        try:
            rendered = engine.render(
                crossfader,
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
