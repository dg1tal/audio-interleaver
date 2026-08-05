from audio_interleaver.ui import MainWindow


def test_window_starts_waiting_for_two_sources(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert window.windowTitle() == "Audio Interleaver"
    assert not window.play_button.isEnabled()
    assert not window.export_button.isEnabled()
    assert window.crossfader.value() == 50
    assert not window.loop_checkbox.isChecked()


def test_loop_checkbox_updates_playback_mode(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window.loop_checkbox.setChecked(True)

    assert window._loop is True
