"""Centralized configuration helpers for the Demand-Forecasting project."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)


def get_env(key: str, default: str | None = None, *, required: bool = False) -> str:
    """Fetch a single environment variable, enforcing presence when required."""
    value = os.getenv(key, default)
    if required and value is None:
        raise KeyError(f"Missing required environment variable: {key}")
    if value is None:
        raise KeyError(f"Environment variable {key} is not set and has no default.")
    return value


def get_starrocks_conn_kwargs() -> Dict[str, Any]:
    """Return connection kwargs compatible with mysql-connector."""
    return {
        "host": get_env("STARROCKS_HOST", required=True),
        "port": int(get_env("STARROCKS_PORT", required=True)),
        "user": get_env("STARROCKS_USER", required=True),
        "password": get_env("STARROCKS_PASSWORD", required=True),
        "database": get_env("STARROCKS_DB", required=True),
    }


def get_path_from_env(key: str, *, is_dir: bool = False) -> Path:
    """Resolve a project-relative path defined in .env and ensure it exists."""
    relative_path = get_env(key, required=True)
    path = BASE_DIR / relative_path
    if is_dir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_prefect_settings() -> Dict[str, Any]:
    """Expose Prefect-related settings for flow parameters."""
    return {
        "api_url": get_env("PREFECT_API_URL", required=True),
        "work_queue": get_env("PREFECT_WORK_QUEUE", required=True),
        "deployment_name": get_env("PREFECT_DEPLOYMENT_NAME", required=True),
        "run_mode": get_env("RUN_MODE", default="daily"),
    }
