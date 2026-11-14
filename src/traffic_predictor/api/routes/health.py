"""Health and readiness endpoints."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone

from fastapi import APIRouter

import traffic_predictor

router = APIRouter(tags=["health"])

_start_time = datetime.now(timezone.utc)


@router.get("/health", summary="Liveness probe")
async def health_check() -> dict:
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/ready", summary="Readiness probe — model loaded check")
async def readiness() -> dict:
    return {"status": "ready", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/info", summary="Application metadata")
async def app_info() -> dict:
    return {
        "name": "Smart Traffic Predictor",
        "version": traffic_predictor.__version__,
        "author": traffic_predictor.__author__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "uptime_seconds": (datetime.now(timezone.utc) - _start_time).total_seconds(),
    }
