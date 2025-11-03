import logging
import os
import sys


def _ensure_utf8_stdout():
    """Ensure sys.stdout uses UTF-8 encoding so logging to console can emit Unicode."""
    try:
        # Python 3.7+: reconfigure is available
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8") # type: ignore
        else:
            # Fallback for older environments
            import codecs

            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
    except Exception:
        # Best-effort only; if this fails, file logging below still uses UTF-8
        pass


def setup_logging(
    log_level: str, logs_dir: str, log_filename: str, name: str
) -> logging.Logger:
    _ensure_utf8_stdout()
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(levelname)-8s - %(name)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(logs_dir, log_filename),
                encoding="utf-8",
            ),
        ],
    )
    return logging.getLogger(name)
