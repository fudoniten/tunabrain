from __future__ import annotations

import argparse
import logging
import uvicorn

from tunabrain.app import app
from tunabrain.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TunaBrain API server")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 5546)",
    )
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting TunaBrain with port=%s", args.port)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

