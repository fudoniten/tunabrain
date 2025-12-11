from __future__ import annotations

import logging
import uvicorn

from tunabrain.app import app


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)

