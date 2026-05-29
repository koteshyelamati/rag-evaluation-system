"""
app/logger.py — Centralised logging configuration.

Call configure_logging() once at application startup (in main.py lifespan)
to apply a consistent log format and level across all modules.

Log level is controlled by the LOG_LEVEL environment variable (default: INFO).
Set LOG_LEVEL=DEBUG locally for verbose output.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logger with a structured, human-readable format.

        Parameters
            ----------
                level:
                        Logging level string (e.g. ``"DEBUG"``, ``"INFO"``).
                                Falls back to the ``LOG_LEVEL`` env var, then ``INFO``.
                                    """
                                        from app.config import settings

                                            resolved_level = (level or settings.LOG_LEVEL or "INFO").upper()
                                                numeric = getattr(logging, resolved_level, logging.INFO)

                                                    formatter = logging.Formatter(
                                                            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                                                                    datefmt="%Y-%m-%dT%H:%M:%S",
                                                                        )

                                                                            handler = logging.StreamHandler(sys.stdout)
                                                                                handler.setFormatter(formatter)

                                                                                    root = logging.getLogger()
                                                                                        # Remove any existing handlers to avoid duplicate log lines
                                                                                            root.handlers.clear()
                                                                                                root.addHandler(handler)
                                                                                                    root.setLevel(numeric)
                                                                                                    
                                                                                                        # Quieten noisy third-party libraries
                                                                                                            for noisy in ("chromadb", "httpx", "httpcore", "urllib3", "openai"):
                                                                                                                    logging.getLogger(noisy).setLevel(logging.WARNING)
                                                                                                                    
                                                                                                                        logging.getLogger(__name__).info(
                                                                                                                                "Logging configured — level=%s", resolved_level
                                                                                                                                    )
                                                                                                                                    
                                                                                                                                    
                                                                                                                                    def get_logger(name: str) -> logging.Logger:
                                                                                                                                        """Convenience wrapper: returns a named logger.
                                                                                                                                        
                                                                                                                                            Usage::
                                                                                                                                            
                                                                                                                                                    from app.logger import get_logger
                                                                                                                                                            logger = get_logger(__name__)
                                                                                                                                                                    logger.info("Hello from %s", __name__)
                                                                                                                                                                        """
                                                                                                                                                                            return logging.getLogger(name)
