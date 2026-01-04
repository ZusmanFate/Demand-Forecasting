"""Prefect flow definition for the weekly demand forecasting pipeline."""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Optional, Dict, Any

import structlog
from prefect import flow, get_run_logger, task
from prefect.client.schemas.schedules import CronSchedule


from config.config_loader import ConfigLoader, get_env, get_prefect_settings
from db.starrocks_oper import StarRocksOper
from feature.feature_builder import FeatureBuilder
from model.lightgbm_runner import LightGBMRunner
from utils.backup_manager import BackupManager
from utils.logger_config import configure_logging, set_request_id

LOGGER = structlog.get_logger("prefect_forecast_flow")
CONFIG_LOADER = ConfigLoader()
COMMON_TAGS = ["weekly-forecast", "sales-prediction"]
TASK_RETRIES = 3
TASK_RETRY_DELAY_SECONDS = 60
DEFAULT_WEEKLY_SCHEDULE = CronSchedule(cron="0 23 * * 0", timezone="Asia/Shanghai")
BACKUP_MANAGER = BackupManager()
configure_logging()


def _runtime_config() -> Dict[str, Any]:
    cfg = CONFIG_LOADER.load_config()
    return {
        "paths": {name: str(path) for name, path in cfg["paths"].items()},
        "format": cfg["format"],
    }


def _resolve_run_date(run_date: str | None, *, date_format: str) -> str:
    if run_date:
        return run_date
    return dt.date.today().strftime(date_format)


@task(
    name="Export raw weekly data",
    description="Pulls weekly sales aggregates from StarRocks and stores them as CSV.",
    retries=TASK_RETRIES,
    retry_delay_seconds=TASK_RETRY_DELAY_SECONDS,
    tags=COMMON_TAGS,
    log_prints=True,
)
def export_raw_data_task(run_date: str, config: Dict[str, Any]) -> str:
    set_request_id(f"prefect-export-{run_date}")
    logger = structlog.get_logger("prefect.export_raw")
    prefect_logger = get_run_logger()
    raw_dir = Path(config["paths"]["raw_dir"])
    raw_path = raw_dir / f"raw_{run_date}.csv"

    sql_template = get_env("STARROCKS_EXPORT_SQL", required=True)
    sql = sql_template.replace("{run_date}", run_date)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    starrocks = StarRocksOper()
    exported = starrocks.export_query_to_csv(sql, raw_path)

    BACKUP_MANAGER.backup_csv_files("raw", run_date, [exported])

    payload = {"run_date": run_date, "raw_csv": exported.as_posix()}

    # ✅ structlog 支持传递上下文字段
    logger.info("raw_export_completed", **payload)

    # ✅ Prefect logger 只支持字符串消息，不能展开 kwargs
    prefect_logger.info(
        f"Raw export finished | run_date={run_date} | raw_csv={exported.as_posix()}"
    )

    return exported.as_posix()



@task(
    name="Build engineered features",
    description="Transforms raw weekly sales into the feature set expected by LightGBM.",
    retries=TASK_RETRIES,
    retry_delay_seconds=TASK_RETRY_DELAY_SECONDS,
    tags=COMMON_TAGS,
    log_prints=True,
)
def build_feature_task(raw_csv_path: str, run_date: str, config: Dict[str, Any]) -> str:
    set_request_id(f"prefect-feature-{run_date}")
    logger = structlog.get_logger("prefect.build_feature")
    prefect_logger = get_run_logger()
    feature_dir = Path(config["paths"]["feature_dir"])
    feature_path = feature_dir / f"feature_{run_date}.csv"

    builder = FeatureBuilder(raw_csv=Path(raw_csv_path), feature_csv=feature_path)
    output = builder.process()
    BACKUP_MANAGER.backup_csv_files("feature", run_date, [output])

    payload = {
        "feature_csv": output.as_posix(),
        "input_raw_csv": raw_csv_path,
        "bytes": output.stat().st_size,
    }
    logger.info("feature_build_completed", extra=payload)
    prefect_logger.info("Feature engineering completed", extra=payload)
    return output.as_posix()


@task(
    name="Train & predict LightGBM",
    description="Runs LightGBM training, tuning, and inference to build the forecast CSV.",
    retries=TASK_RETRIES,
    retry_delay_seconds=TASK_RETRY_DELAY_SECONDS,
    tags=COMMON_TAGS,
    log_prints=True,
)
def train_predict_model_task(feature_csv_path: str, run_date: str, config: Dict[str, Any]) -> Dict[str, Any]:
    set_request_id(f"prefect-train-{run_date}")
    logger = structlog.get_logger("prefect.train_predict")
    prefect_logger = get_run_logger()
    forecast_dir = Path(config["paths"]["forecast_dir"])
    forecast_path = forecast_dir / f"forecast_{run_date}.csv"

    runner = LightGBMRunner()
    output_path = runner.train_and_predict(Path(feature_csv_path), forecast_path)
    metrics = runner.get_model_metrics()
    BACKUP_MANAGER.backup_csv_files("forecast", run_date, [output_path])

    payload = {"forecast_csv": output_path.as_posix(), "metrics": metrics}
    logger.info("training_prediction_completed", **payload)
    prefect_logger.info("Model training + prediction completed", **payload)
    return payload


@task(
    name="Write forecast back to StarRocks",
    description="Loads the sanitized forecast CSV into the StarRocks reporting table.",
    retries=TASK_RETRIES,
    retry_delay_seconds=TASK_RETRY_DELAY_SECONDS,
    tags=COMMON_TAGS,
    log_prints=True,
)
def write_forecast_data_task(forecast_csv: str) -> int:
    set_request_id("prefect-write")
    logger = structlog.get_logger("prefect.write_forecast")
    prefect_logger = get_run_logger()
    table_name = get_env("STARROCKS_IMPORT_TABLE", "weekly_sales_forecast")

    starrocks = StarRocksOper()
    affected_rows = starrocks.write_forecast_data_to_starrocks(Path(forecast_csv), table_name)
    payload = {"table": table_name, "rows": affected_rows, "forecast_csv": forecast_csv}
    logger.info("forecast_import_completed", **payload)
    prefect_logger.info("Forecast loaded into StarRocks", **payload)
    return affected_rows


@flow(
    name="full_forecast_flow",
    description="Weekly demand forecasting flow with export, feature engineering, training, and load steps.",
    log_prints=True,
    task_runner=None,
    validate_parameters=False,
    retries=0,
)
def full_forecast_flow(run_date: Optional[str] = None) -> Dict[str, Any]:
    config = _runtime_config()
    date_format = config["format"]["date_suffix"]
    resolved_run_date = _resolve_run_date(run_date, date_format=date_format)

    set_request_id(f"prefect-flow-{resolved_run_date}")
    flow_logger = get_run_logger()
    flow_logger.info(f"Flow started for {resolved_run_date} with tags {COMMON_TAGS}")
    start_time = time.time()

    raw_csv = export_raw_data_task(run_date=resolved_run_date, config=config)
    feature_csv = build_feature_task(raw_csv_path=raw_csv, run_date=resolved_run_date, config=config)
    model_output = train_predict_model_task(
        feature_csv_path=feature_csv,
        run_date=resolved_run_date,
        config=config,
    )
    affected_rows = write_forecast_data_task(forecast_csv=model_output["forecast_csv"])

    duration = time.time() - start_time
    summary = {
        "run_date": resolved_run_date,
        "raw_csv": raw_csv,
        "feature_csv": feature_csv,
        "forecast_csv": model_output["forecast_csv"],
        "metrics": model_output["metrics"],
        "rows_written": affected_rows,
        "duration_seconds": round(duration, 2),
    }
    BACKUP_MANAGER.cleanup_old_backups()
    LOGGER.info("flow_completed", **summary)
    flow_logger.info("Flow completed", **summary)
    return summary


def create_weekly_deployment(*, apply: bool = False):
    """Create (and optionally register) a weekly deployment compatible with Prefect 3."""

    settings = get_prefect_settings()
    deployment = full_forecast_flow.to_deployment(
        name="weekly-demand-forecast",
        description="Weekly 23:00 Asia/Shanghai demand forecast",
        schedule=DEFAULT_WEEKLY_SCHEDULE,
        work_pool_name=settings["work_pool"],
        work_queue_name=settings["work_queue"],
        tags=COMMON_TAGS,
    )
    if apply:
        deployment.apply()
    return deployment


if __name__ == "__main__":
    full_forecast_flow()
