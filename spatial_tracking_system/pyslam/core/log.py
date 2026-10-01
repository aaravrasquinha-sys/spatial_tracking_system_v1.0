import logging
import sys

_CONFIGURED = False


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s %(levelname)-7s %(name)-22s | %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
