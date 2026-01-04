"""Centralized configuration helpers for the Demand-Forecasting project."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from dotenv import dotenv_values, load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"


@dataclass(slots=True)
class ConfigLoader:
    """Load project configuration from .env with safe defaults."""

    env_path: Path = ENV_PATH
    _cached_config: Dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.env_path.exists():
            load_dotenv(self.env_path)

    def get_env(self, key: str, default: str | None = None, *, required: bool = False) -> str:
        value = os.getenv(key) if key in os.environ else default
        if value in (None, ""):
            if required:
                raise KeyError(f"Missing required environment variable: {key}")
            if default is None:
                raise KeyError(f"Environment variable {key} is not set and has no default")
            value = default
        return value

    def load_config(self) -> Dict[str, Any]:
        if self._cached_config is not None:
            return self._cached_config

        data_root = Path(self.get_env("BASE_DATA_DIR", "data"))
        config = {
            "paths": {
                "base": BASE_DIR,
                "data_root": data_root,
                "raw_dir": data_root / "raw",
                "feature_dir": data_root / "feature",
                "forecast_dir": data_root / "forecast",
                "model_dir": Path(self.get_env("MODEL_DIR", "saved_models")),
                "log_file": Path(self.get_env("LOG_FILE", "logs/pipeline.log")),
            },
            "starrocks": {
                "host": self.get_env("STARROCKS_HOST", "localhost"),
                "port": int(self.get_env("STARROCKS_PORT", "9030")),
                "user": self.get_env("STARROCKS_USER", "root"),
                "password": self.get_env("STARROCKS_PASSWORD", ""),
                "database": self.get_env("STARROCKS_DB", "default"),
            },
            "model": {
                "seed": int(self.get_env("LGBM_SEED", "42")),
                "n_splits": int(self.get_env("LGBM_N_SPLITS", "5")),
                "seeds": self._parse_int_list(self.get_env("LGBM_SEEDS", "42,123,404")),
            },
            "format": {
                "date_suffix": self.get_env("DATE_SUFFIX_FORMAT", "%Y%m%d"),
            },
        }

        for path in config["paths"].values():
            if isinstance(path, Path) and path.name not in ("base",):
                path.parent.mkdir(parents=True, exist_ok=True)

        self._cached_config = config
        return config

    @staticmethod
    def _parse_int_list(raw: str) -> List[int]:
        return [int(token.strip()) for token in raw.split(",") if token.strip()]


_CONFIG_LOADER = ConfigLoader()


def get_env(key: str, default: str | None = None, *, required: bool = False) -> str:
    return _CONFIG_LOADER.get_env(key, default, required=required)


def get_starrocks_conn_kwargs() -> Dict[str, Any]:
    cfg = _CONFIG_LOADER.load_config()["starrocks"]
    return cfg.copy()


def get_path_from_env(key: str, *, is_dir: bool = False) -> Path:
    relative_path = get_env(key, required=True)
    path = BASE_DIR / relative_path
    if is_dir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_prefect_settings() -> Dict[str, Any]:
    work_pool = get_env("PREFECT_WORK_POOL", None)
    work_queue = get_env("PREFECT_WORK_QUEUE", None)
    if not work_pool and work_queue:
        work_pool = work_queue

    return {
        "api_url": get_env("PREFECT_API_URL", "http://prefect-server:4200/api"),
        "work_pool": work_pool or "demand-local",
        "work_queue": work_queue,
        "deployment_name": get_env(
            "PREFECT_DEPLOYMENT_NAME", "full_forecast_flow/weekly-demand-forecast"
        ),
        "run_mode": get_env("RUN_MODE", "daily"),
    }
