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
        
        # Method 1: Try using git command (if git is installed)
        try:
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
            
            # Get version tag
            version_tag = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                cwd=git_dir.parent,
                capture_output=True,
                text=True,
                timeout=5
            ).stdout.strip()
            
            if commit:  # If we got a commit, return these results
                return {
                    "git-commit": commit or None,
                    "git-timestamp": timestamp or None,
                    "version-tag": version_tag or None,
                }
        except FileNotFoundError:
            # git command not available, try reading from .git directory
            logger.debug("git command not available, reading from .git directory")
        
        # Method 2: Read directly from .git directory
        # Read current commit hash from HEAD
        commit = None
        head_file = git_dir / "HEAD"
        if head_file.exists():
            head_content = head_file.read_text().strip()
            if head_content.startswith("ref: "):
                # It's a symbolic ref, read the actual file
                ref_path = git_dir / head_content.replace("ref: ", "")
                if ref_path.exists():
                    commit = ref_path.read_text().strip()
            else:
                # Direct commit hash
                commit = head_content
        
        # Read commit timestamp from packed-refs or loose ref files
        timestamp = None
        if commit:
            # Try to get timestamp from git object
            commit_obj_path = git_dir / "objects" / commit[:2] / commit[2:]
            if commit_obj_path.exists():
                timestamp = str(int(commit_obj_path.stat().st_mtime))
        
        # Try to read version tag from refs/tags
        version_tag = None
        tags_dir = git_dir / "refs" / "tags"
        if tags_dir.exists():
            tags = sorted(list(tags_dir.glob("*")))
            if tags:
                # Return the last (most recent) tag
                version_tag = tags[-1].name
        
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
