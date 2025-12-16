from __future__ import annotations

import logging

from fastapi import FastAPI

from tunabrain.api import routes
from tunabrain.logging import configure_logging


def create_app() -> FastAPI:
    """Construct the FastAPI application instance."""
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Initializing TunaBrain FastAPI application")

    app = FastAPI(title="TunaBrain", description="LangChain utilities for Tunarr Scheduler")
    app.include_router(routes.router)

    logger.info("Application routes registered")
    return app


app = create_app()

