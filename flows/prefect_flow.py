"""Prefect flow orchestrating the Demand-Forecasting CSV pipeline."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import structlog
from prefect import flow, task

from config.config_loader import (
    get_env,
    get_path_from_env,
    get_prefect_settings,
)
from db.starrocks_oper import StarRocksOperator
from feature.feature_builder import FeatureBuilder
from model.lightgbm_runner import LightGBMRunner

logger = structlog.get_logger("demand_forecast_flow")


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d%H%M")


@task(name="Export raw data")
def export_raw(run_date: str) -> str:
    sql = get_env("STARROCKS_EXPORT_SQL", required=True).replace("{run_date}", run_date)
    raw_dir = get_path_from_env("RAW_DATA_DIR", is_dir=True)
    raw_path = raw_dir / f"raw_{run_date}.csv"
    StarRocksOperator().export_to_csv(sql, raw_path)
    logger.info("raw_export_completed", path=raw_path.as_posix())
    return raw_path.as_posix()


@task(name="Build features")
def build_features(raw_path: str, run_mode: str, run_date: str) -> str:
    feature_dir = get_path_from_env("FEATURE_DATA_DIR", is_dir=True)
    feature_path = feature_dir / f"feature_{run_date}.csv"
    FeatureBuilder(Path(raw_path), feature_path).build(run_mode=run_mode)
    logger.info("features_ready", path=feature_path.as_posix())
    return feature_path.as_posix()


@task(name="Train LightGBM model")
def train_model(feature_path: str) -> str:
    model_dir = get_path_from_env("MODEL_DIR", is_dir=True)
    model_path = model_dir / "lightgbm_model.pkl"
    LightGBMRunner(model_path).train(Path(feature_path))
    logger.info("model_trained", path=model_path.as_posix())
    return model_path.as_posix()


@task(name="Predict demand")
def predict_demand(model_path: str, feature_path: str, run_date: str) -> str:
    forecast_dir = get_path_from_env("FORECAST_DATA_DIR", is_dir=True)
    forecast_path = forecast_dir / f"forecast_{run_date}.csv"
    LightGBMRunner(Path(model_path)).predict(Path(feature_path), forecast_path, run_date)
    logger.info("forecast_generated", path=forecast_path.as_posix())
    return forecast_path.as_posix()


@task(name="Import forecast into StarRocks")
def import_forecast(csv_path: str) -> int:
    table = get_env("STARROCKS_IMPORT_TABLE", required=True)
    columns = ("ds", "prediction", "run_date")
    affected = StarRocksOperator().import_from_csv(table, Path(csv_path), columns)
    logger.info("forecast_loaded", table=table, rows=affected)
    return affected


@flow(name="Demand Forecast Flow", retries=1, retry_delay_seconds=60)
def demand_forecast_flow(run_mode: str | None = None, run_date: str | None = None) -> int:
    settings = get_prefect_settings()
    run_mode = run_mode or settings["run_mode"]
    run_date = run_date or _timestamp()

    raw_path = export_raw(run_date)
    feature_path = build_features(raw_path, run_mode, run_date)
    model_path = train_model(feature_path)
    forecast_path = predict_demand(model_path, feature_path, run_date)
    affected_rows = import_forecast(forecast_path)

    logger.info(
        "flow_completed",
        run_mode=run_mode,
        run_date=run_date,
        affected_rows=affected_rows,
    )
    return affected_rows


if __name__ == "__main__":
    demand_forecast_flow()
