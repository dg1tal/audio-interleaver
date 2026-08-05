import numpy as np
from PySide6.QtWidgets import QLabel

from audio_interleaver.audio import AudioEngine, LoadedAudio
from audio_interleaver.ui import InterleaveTimeline, MainWindow


def test_window_starts_waiting_for_two_sources(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert window.windowTitle() == "Audio Interleaver"
    assert window.findChild(QLabel, "productSubheading").text() == (
        "A product of DG1TAL Compute Sweatshop"
    )
    assert not window.play_button.isEnabled()
    assert not window.export_button.isEnabled()
    assert window.crossfader.value() == 50
    assert window.chunk_duration_slider.value() == 360
    assert window.crossfade_duration_slider.value() == 5
    assert not window.loop_checkbox.isChecked()


def test_loop_checkbox_updates_playback_mode(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window.loop_checkbox.setChecked(True)

    assert window._loop is True


def test_interleave_timeline_tracks_slot_selection_and_position(qtbot):
    source = LoadedAudio(np.zeros((1440, 1), dtype=np.float32), 1000)
    engine = AudioEngine(source, source)
    timeline = InterleaveTimeline()
    qtbot.addWidget(timeline)

    timeline.set_engine(engine)
    assert timeline.slot_sources == ("B", "A", "B", "A")

    timeline.set_crossfader(0.25)
    assert timeline.slot_sources == ("B", "A", "A", "A")

    timeline.set_position(0.72)
    assert timeline.position == 0.72


def test_crossfader_updates_interleave_preview(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    source = LoadedAudio(np.zeros((1080, 1), dtype=np.float32), 1000)
    window._engine = AudioEngine(source, source)
    window.interleave_timeline.set_engine(window._engine)

    window.crossfader.setValue(100)

    assert window.interleave_timeline.slot_sources == ("B", "B", "B")


def test_duration_sliders_rebuild_engine_and_preview(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    source = LoadedAudio(np.zeros((1000, 1), dtype=np.float32), 1000)
    window._source_a = source
    window._source_b = source
    window._rebuild_engine()

    window.chunk_duration_slider.setValue(250)
    window.crossfade_duration_slider.setValue(20)

    assert window._engine is not None
    assert window._engine.slot_ms == 250
    assert window._engine.smoothing_ms == 20
    assert window._engine.slot_count == 4
    assert window.preview_detail.text() == "Each block = 250 ms"
    assert window.crossfade_duration_label.text() == "20 ms"
