"""
core/logger.py -- System-wide structured logging (Windows-safe)
"""
import io
import logging
import sys
from pathlib import Path
from datetime import datetime
import colorlog


def _safe_stream():
    """Return a UTF-8 capable stdout stream, works on Windows CP1252."""
    if sys.stdout and hasattr(sys.stdout, 'buffer'):
        try:
            return io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass
    return sys.stdout


class SafeStreamHandler(colorlog.StreamHandler):
    """StreamHandler that never crashes on unencodable characters."""

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                safe = msg.encode('ascii', errors='replace').decode('ascii')
                stream.write(safe + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler -- UTF-8 aware, Windows-safe
    stream = _safe_stream()
    ch = SafeStreamHandler(stream)
    ch.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(name)s] %(levelname)s%(reset)s  %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        }
    ))
    logger.addHandler(ch)

    # File handler -- always UTF-8
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    fh = logging.FileHandler(
        log_dir / f"quant_agent_{date_str}.log",
        encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)
    logger.propagate = False
    return logger
