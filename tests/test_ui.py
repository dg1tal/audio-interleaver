from audio_interleaver.ui import MainWindow


def test_window_starts_waiting_for_two_sources(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert window.windowTitle() == "Audio Interleaver"
    assert not window.play_button.isEnabled()
    assert not window.export_button.isEnabled()
    assert window.crossfader.value() == 50

