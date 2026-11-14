"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from traffic_predictor.api.middleware import RequestLoggingMiddleware
from traffic_predictor.api.routes import health, predict as predict_router
from traffic_predictor.utils.config import get_config
from traffic_predictor.utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


def _try_load_model(config):
    from traffic_predictor.models.traditional.traffic_classifier import TrafficPredictor

    model_dir = Path(config.model.model_dir)
    model_files = sorted(model_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not model_files:
        logger.warning("No trained model found in models/saved/. Run `python scripts/train.py` first.")
        return None

    logger.info(f"Loading model from {model_files[0]}")
    try:
        return TrafficPredictor.load(model_files[0])
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    setup_logger(config.log_level, config.log_file)
    logger.info(f"Starting {config.app_name} v{config.version}")

    model = _try_load_model(config)
    if model:
        predict_router.set_model(model)
        logger.info(f"Model ready: {model.model_name}")

    yield
    logger.info("Application shutdown")


def create_app() -> FastAPI:
    config = get_config()

    app = FastAPI(
        title=config.app_name,
        version=config.version,
        description=(
            "AI-Based Smart Traffic Prediction System. "
            "Predicts traffic volume and congestion level for road intersections "
            "using time, weather, and historical traffic features."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(predict_router.router)

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "type": type(exc).__name__},
        )

    return app


app = create_app()


def start() -> None:
    import uvicorn
    config = get_config()
    uvicorn.run(
        "traffic_predictor.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        workers=config.api.workers,
        log_level=config.api.log_level,
    )


if __name__ == "__main__":
    start()
