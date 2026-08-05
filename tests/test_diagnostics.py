from audio_interleaver.diagnostics import diagnostics_directory


def test_diagnostics_directory_honors_override(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIO_INTERLEAVER_LOG_DIR", str(tmp_path))

    assert diagnostics_directory() == tmp_path
