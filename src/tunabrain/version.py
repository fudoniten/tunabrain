"""Version endpoint implementation for Tunabrain."""
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_git_info() -> dict[str, str | None]:
    """Get git commit, timestamp, and version tag."""
    try:
        # Get the git repository root
        git_dir = Path(__file__).parent.parent.parent / ".git"
        
        # Get commit hash
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_dir.parent,
            capture_output=True,
            text=True,
            timeout=5
        ).stdout.strip()
        
        # Get commit timestamp
        timestamp = subprocess.run(
            ["git", "log", "-1", "--format=%ct"],
            cwd=git_dir.parent,
            capture_output=True,
            text=True,
            timeout=5
        ).stdout.strip()
        
        # Get version tag (latest tag if it exists)
        version_tag = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=git_dir.parent,
            capture_output=True,
            text=True,
            timeout=5
        ).stdout.strip()
        
        return {
            "git-commit": commit or None,
            "git-timestamp": timestamp or None,
            "version-tag": version_tag or None,
        }
    except Exception as e:
        logger.warning("Failed to get git info: %s", e)
        return {
            "git-commit": None,
            "git-timestamp": None,
            "version-tag": None,
        }
