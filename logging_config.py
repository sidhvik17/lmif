"""Centralized logging setup. Call setup_logging() once at app entry."""
import logging
import os
import sys

_CONFIGURED = False

_LEVEL_NAMES = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _resolve_level(level):
    """Accept int, level name, or None → env var → INFO."""
    if isinstance(level, int):
        return level
    name = (level or os.environ.get("LMIF_LOG_LEVEL", "INFO")).upper()
    if name not in _LEVEL_NAMES:
        name = "INFO"
    return getattr(logging, name)


def setup_logging(level=None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    resolved = _resolve_level(level)
    root = logging.getLogger()
    root.setLevel(resolved)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.handlers = [handler]
    for noisy in ("chromadb", "sentence_transformers", "urllib3",
                  "httpx", "httpcore", "huggingface_hub", "transformers",
                  "PIL", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
