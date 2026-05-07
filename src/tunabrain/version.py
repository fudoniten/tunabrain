"""Version endpoint implementation for Tunabrain."""
import logging
import os

logger = logging.getLogger(__name__)


def get_git_info() -> dict[str, str | None]:
    """Get git commit, timestamp, and version tag from environment variables."""
    return {
        "git-commit": os.getenv("GIT_COMMIT"),
        "git-timestamp": os.getenv("GIT_TIMESTAMP"),
        "version-tag": os.getenv("VERSION"),
    }
