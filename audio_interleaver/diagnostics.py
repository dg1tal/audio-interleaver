"""Persistent diagnostics for failures that occur below Python's exception layer."""

from __future__ import annotations

import faulthandler
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import platform
import sys
import threading

import sounddevice as sd


_FAULT_FILE = None
_CONFIGURED_PATH: Path | None = None


def diagnostics_directory() -> Path:
    """Return the platform-appropriate directory for persistent application logs."""

    override = os.environ.get("AUDIO_INTERLEAVER_LOG_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "Audio Interleaver"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Audio Interleaver" / "Logs"
    state_home = os.environ.get("XDG_STATE_HOME", "").strip()
    return (
        Path(state_home).expanduser()
        if state_home
        else Path.home() / ".local" / "state"
    ) / "audio-interleaver"


def configure_diagnostics() -> Path:
    """Enable rotating logs, uncaught-exception logging, and fatal-signal traces."""

    global _CONFIGURED_PATH, _FAULT_FILE
    if _CONFIGURED_PATH is not None:
        return _CONFIGURED_PATH

    log_directory = diagnostics_directory()
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / "audio-interleaver.log"
    handler = RotatingFileHandler(
        log_path,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(threadName)s "
            "%(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    def log_uncaught(exc_type, exc_value, exc_traceback) -> None:
        logging.getLogger(__name__).critical(
            "uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = log_uncaught

    def log_thread_uncaught(args: threading.ExceptHookArgs) -> None:
        logging.getLogger(__name__).critical(
            "uncaught thread exception thread=%s",
            args.thread.name if args.thread is not None else "unknown",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = log_thread_uncaught

    fault_path = log_directory / "native-crash.log"
    try:
        _FAULT_FILE = fault_path.open("a", encoding="utf-8", buffering=1)
        faulthandler.enable(file=_FAULT_FILE, all_threads=True)
    except (OSError, RuntimeError):
        logging.getLogger(__name__).exception("could not enable native crash log")

    _CONFIGURED_PATH = log_path
    logger = logging.getLogger(__name__)
    logger.info(
        "application starting python=%s platform=%s log=%s",
        sys.version.replace("\n", " "),
        platform.platform(),
        log_path,
    )
    try:
        logger.info(
            "audio environment portaudio=%s default_device=%r devices=%s",
            sd.get_portaudio_version(),
            sd.default.device,
            len(sd.query_devices()),
        )
    except Exception:
        logger.exception("could not query audio environment")
    return log_path
