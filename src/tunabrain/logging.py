from __future__ import annotations

"""Centralized logging configuration for TunaBrain.

This module installs a consistent formatter and log level so application logs are
forwarded correctly in containerized environments such as Kubernetes. The
configuration is idempotent and safe to call multiple times.
"""

import logging
import os
from typing import Final


LOG_FORMAT: Final = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging() -> None:
    """Configure root logging with sensible defaults if not already configured."""

    if logging.getLogger().handlers:
        return

    level = logging.DEBUG if os.getenv("TUNABRAIN_DEBUG") else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)

