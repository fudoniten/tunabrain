from __future__ import annotations

import argparse
import logging
import uvicorn

from tunabrain.app import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TunaBrain API server")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 5546)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()

