import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session", autouse=True)
def setup_openai_api_key():
    """Ensure OPENAI_API_KEY is set for integration tests.

    Falls back to OPEN_API_KEY if OPENAI_API_KEY is not set.
    """
    if not os.getenv("OPENAI_API_KEY"):
        # Check if OPEN_API_KEY exists (common typo)
        open_api_key = os.getenv("OPEN_API_KEY")
        if open_api_key:
            os.environ["OPENAI_API_KEY"] = open_api_key
        else:
            pytest.fail(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it before running integration tests:\n"
                "  export OPENAI_API_KEY=sk-your-key-here"
            )
