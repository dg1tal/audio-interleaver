import numpy as np
from PySide6.QtWidgets import QLabel

from audio_interleaver.audio import (
    AudioEngine,
    InterleavePattern,
    LoadedAudio,
    RegionInsert,
)
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
    assert window.crossfader.value() == 0
    assert not window.crossfader.isEnabled()
    assert window.revision_label.text().startswith("Commit ")
    assert not window.start_with_b_checkbox.isChecked()
    assert window.first_alternate_slider.value() == 2
    assert window.burst_size_slider.value() == 1
    assert window.chunk_duration_slider.value() == 360
    assert window.chunk_duration_input.value() == 360
    assert window.crossfade_duration_slider.value() == 0
    assert window.crossfade_duration_input.value() == 0
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
    assert timeline.minimumHeight() == 116
    assert timeline.slot_sources == ("A", "B", "A", "A")

    timeline.set_pattern(InterleavePattern(fill=1.0, b_chunks_per_occurrence=2))
    assert timeline.slot_sources == ("A", "B", "B", "A")

    timeline.set_position(0.72)
    assert timeline.position == 0.72


def test_waveform_b_starts_at_its_first_selected_slot(qtbot):
    source = LoadedAudio(np.zeros((2160, 1), dtype=np.float32), 1000)
    engine = AudioEngine(source, source)
    timeline = InterleaveTimeline()
    qtbot.addWidget(timeline)

    timeline.set_engine(engine)
    timeline.set_pattern(InterleavePattern(fill=1.0, first_alternate_slot=3))

    assert timeline.slot_sources == tuple("AAABAB")
    assert timeline.waveform_chunk_indices("B") == (None, None, None, 0, 1, 1)


def test_occurrence_fill_updates_interleave_preview(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    source = LoadedAudio(np.zeros((1080, 1), dtype=np.float32), 1000)
    window._engine = AudioEngine(source, source)
    window.interleave_timeline.set_engine(window._engine)

    window.crossfader.setValue(100)

    assert window.interleave_timeline.slot_sources == ("A", "B", "A")


def test_pattern_controls_update_preview(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    source = LoadedAudio(np.zeros((3240, 1), dtype=np.float32), 1000)
    window._source_a = source
    window._source_b = source
    window._rebuild_engine()

    window.crossfader.setValue(100)
    window.burst_size_slider.setValue(2)
    assert window.interleave_timeline.slot_sources == tuple("ABBABBABB")

    window.start_with_b_checkbox.setChecked(True)
    assert window.first_alternate_title.text() == "FIRST A CHUNK"
    assert window.interleave_timeline.slot_sources == tuple("BABBABBAB")

    window.first_alternate_slider.setValue(4)
    assert window.interleave_timeline.slot_sources == tuple("BBBABBABB")


def test_occurrence_fill_has_one_meaningful_step_per_occurrence(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    source = LoadedAudio(np.zeros((4320, 1), dtype=np.float32), 1000)
    window._source_a = source
    window._source_b = source
    window._rebuild_engine()

    window.burst_size_slider.setValue(2)
    assert window.crossfader.minimum() == 0
    assert window.crossfader.maximum() == 4
    patterns = []
    for occurrence_count in range(5):
        window.crossfader.setValue(occurrence_count)
        patterns.append(window.interleave_timeline.slot_sources)

    assert len(set(patterns)) == 5
    assert window.mix_label.text() == "4 / 4 occurrences"


def test_region_insert_mode_controls_source_window_and_output_position(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    source = LoadedAudio(np.zeros((3240, 1), dtype=np.float32), 1000)
    window._source_a = source
    window._source_b = source
    window._rebuild_engine()

    window.mode_selector.setCurrentIndex(1)
    window.region_length_slider.setValue(3)
    window.region_source_slider.setValue(4)
    window.region_output_slider.setValue(5)

    assert window.pattern_controls.isHidden()
    assert not window.region_controls.isHidden()
    assert window._settings() == RegionInsert(
        b_source_slot=3, output_slot=4, length_slots=3
    )
    assert window.interleave_timeline.slot_sources == tuple("AAAABBBAA")
    assert window.interleave_timeline.waveform_chunk_indices("B") == (
        None,
        None,
        None,
        None,
        3,
        4,
        5,
        None,
        None,
    )


def test_duration_sliders_and_numeric_inputs_stay_synchronized(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    source = LoadedAudio(np.zeros((1000, 1), dtype=np.float32), 1000)
    window._source_a = source
    window._source_b = source
    window._rebuild_engine()

    window.chunk_duration_input.setValue(250)
    window.crossfade_duration_slider.setValue(20)

    assert window._engine is not None
    assert window._engine.slot_ms == 250
    assert window._engine.smoothing_ms == 20
    assert window._engine.slot_count == 4
    assert window.chunk_duration_slider.value() == 250
    assert window.crossfade_duration_input.value() == 20
    assert window.first_alternate_slider.maximum() == 4
    assert window.burst_size_slider.maximum() == 4
    assert window.preview_detail.text() == "Each block = 250 ms"

    window.crossfade_duration_input.setValue(7)
    window.chunk_duration_slider.setValue(300)
    assert window.crossfade_duration_slider.value() == 7
    assert window.chunk_duration_input.value() == 300
