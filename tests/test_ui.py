import numpy as np
from PySide6.QtWidgets import QLabel

from audio_interleaver.audio import (
    AudioEngine,
    InterleavePattern,
    LoadedAudio,
    RegionInsert,
)
from audio_interleaver.acelp import AcelpEngine
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
    assert window.start_with_a_radio.isChecked()
    assert not window.start_with_b_radio.isChecked()
    assert window.first_alternate_slider.value() == 2
    assert window.burst_size_slider.value() == 1
    assert window.chunk_duration_slider.value() == 360
    assert window.chunk_duration_input.value() == 360
    assert window.crossfade_duration_slider.value() == 0
    assert window.crossfade_duration_input.value() == 0
    assert window.raw_stage_radio.isChecked()
    assert not window.acelp_stage_radio.isChecked()
    assert not window.loop_checkbox.isChecked()
    assert window.source_a_card.preview_button.text() == "Play preview"
    assert window.source_b_card.preview_button.text() == "Play preview"
    assert not window.source_a_card.preview_button.isEnabled()
    assert not window.source_b_card.preview_button.isEnabled()
    assert window.first_alternate_title.text() == "FIRST B CHUNK POSITION"
    assert window.burst_size_label.text() == "1 chunk"


def test_source_card_stacks_load_above_preview_beside_file_details(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    for card in (window.source_a_card, window.source_b_card):
        assert card.action_layout.itemAt(0).widget() is card.load_button
        assert card.action_layout.itemAt(1).widget() is card.preview_button


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


def test_waveform_b_appears_only_in_selected_slots_without_ghost_chunks(qtbot):
    source = LoadedAudio(np.zeros((2160, 1), dtype=np.float32), 1000)
    engine = AudioEngine(source, source)
    timeline = InterleaveTimeline()
    qtbot.addWidget(timeline)

    timeline.set_engine(engine)
    timeline.set_pattern(InterleavePattern(fill=1.0, first_alternate_slot=3))

    assert timeline.slot_sources == tuple("AAABAB")
    assert timeline.waveform_chunk_indices("B") == (
        None,
        None,
        None,
        0,
        None,
        1,
    )
    assert timeline.waveform_chunk_indices("A") == (0, 1, 2, 3, 4, 5)


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

    window.start_with_b_radio.setChecked(True)
    assert window.first_alternate_title.text() == "FIRST A CHUNK POSITION"
    assert window.interleave_timeline.slot_sources == tuple("BABBABBAB")

    window.first_alternate_slider.setValue(4)
    assert window.interleave_timeline.slot_sources == tuple("BBBABBABB")


def test_burst_size_label_pluralizes_chunks(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    source = LoadedAudio(np.zeros((3240, 1), dtype=np.float32), 1000)
    window._source_a = source
    window._source_b = source
    window._rebuild_engine()

    assert window.burst_size_label.text() == "1 chunk"
    window.burst_size_slider.setValue(2)
    assert window.burst_size_label.text() == "2 chunks"


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
    assert not window.region_preview_button.isHidden()
    assert window.region_preview_button.isEnabled()
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


def test_duration_controls_are_arranged_side_by_side(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    layout = window.duration_controls.layout()
    assert layout.itemAt(0).widget() is window.chunk_duration_group
    assert layout.itemAt(1).widget() is window.crossfade_duration_group


def test_acelp_stage_uses_legal_frame_duration_positions(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    source = LoadedAudio(np.zeros((8000, 2), dtype=np.float32), 8000)
    window._source_a = source
    window._source_b = source
    window._rebuild_engine()
    window.chunk_duration_input.setValue(250)

    window.acelp_stage_radio.setChecked(True)

    assert isinstance(window._engine, AcelpEngine)
    assert window.acelp_stage_radio.isChecked()
    assert not window.raw_stage_radio.isChecked()
    assert window.chunk_duration_slider.minimum() == 1
    assert window.chunk_duration_slider.maximum() == 66
    assert window.chunk_duration_slider.value() == 12
    assert window.chunk_duration_input.value() == 360
    assert window.chunk_duration_input.singleStep() == 30
    assert not window.crossfade_duration_group.isEnabled()
    assert not window.b_encoder_controls.isHidden()
    assert window.one_stream_radio.isChecked()
    assert not window.restart_chunk_radio.isChecked()
    assert not window.acelp_banner.isHidden()
    assert not window.symbol_export_button.isHidden()

    window.restart_chunk_radio.setChecked(True)
    assert window._b_encoder_mode == "restart_each_chunk"
    assert not window.one_stream_radio.isChecked()

    window.one_stream_radio.setChecked(True)
    assert window._b_encoder_mode == "one_stream"
    assert not window.restart_chunk_radio.isChecked()

    window.chunk_duration_input.setValue(45)
    assert window.chunk_duration_input.value() == 60
    assert window.chunk_duration_slider.value() == 2

    window.raw_stage_radio.setChecked(True)
    assert window.chunk_duration_input.value() == 250
    assert window.chunk_duration_slider.value() == 250

    window.acelp_stage_radio.setChecked(True)
    assert window.chunk_duration_input.value() == 60


def test_acelp_configuration_locks_during_preparation(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    source = LoadedAudio(np.zeros((8000, 1), dtype=np.float32), 8000)
    window._source_a = source
    window._source_b = source
    window._rebuild_engine()
    window.acelp_stage_radio.setChecked(True)

    window._acelp_prepare_cancel.clear()
    window._acelp_preparing = True
    window._refresh_actions()

    assert not window.configuration_panel.isEnabled()
    assert not window.export_button.isEnabled()
    assert not window.symbol_export_button.isEnabled()
    assert window.play_button.isEnabled()


class FakePreviewPlayback:
    def __init__(self):
        self.is_playing = False
        self.starts = []

    def start(self, audio, preview_id):
        self.starts.append((audio, preview_id))
        self.is_playing = True
        return True

    def stop(self, wait=False):
        del wait
        self.is_playing = False


def test_source_cards_preview_complete_loaded_files_with_play_stop_labels(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    source_a = LoadedAudio(np.full((500, 1), 0.25, dtype=np.float32), 1000)
    window._source_a = source_a
    window._rebuild_engine()
    preview = FakePreviewPlayback()
    window._preview_playback = preview

    assert window.source_a_card.preview_button.isEnabled()
    assert not window.source_b_card.preview_button.isEnabled()

    window._toggle_source_preview("A")

    assert preview.starts == [(source_a, "source-A")]
    assert window.source_a_card.preview_button.text() == "Stop preview"

    window._toggle_source_preview("A")

    assert not preview.is_playing
    assert window.source_a_card.preview_button.text() == "Play preview"


def test_region_preview_plays_selected_chunk_fitted_b_portion(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    source_a = LoadedAudio(np.zeros((600, 1), dtype=np.float32), 1000)
    source_b = LoadedAudio(np.arange(450, dtype=np.float32)[:, None], 1000)
    window._source_a = source_a
    window._source_b = source_b
    window._rebuild_engine()
    window.mode_selector.setCurrentIndex(1)
    window.chunk_duration_input.setValue(100)
    window.region_length_slider.setValue(2)
    window.region_source_slider.setValue(4)
    preview = FakePreviewPlayback()
    window._preview_playback = preview

    window._toggle_region_preview()

    preview_audio, preview_id = preview.starts[-1]
    assert preview_id == "region-B"
    assert preview_audio.frames == 200
    np.testing.assert_allclose(
        preview_audio.samples[:150, 0], np.arange(300, 450, dtype=np.float32)
    )
    np.testing.assert_allclose(preview_audio.samples[150:, 0], 0.0)
    assert window.region_preview_button.text() == "Stop B region"
