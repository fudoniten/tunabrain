from __future__ import annotations

from fastapi import FastAPI

from tunabrain.api import routes


def create_app() -> FastAPI:
    """Construct the FastAPI application instance."""

    app = FastAPI(title="TunaBrain", description="LangChain utilities for Tunarr Scheduler")
    app.include_router(routes.router)
    return app


app = create_app()

