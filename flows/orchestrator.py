"""Flow orchestrator chaining FeatureBuilder and LightGBMRunner."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

from feature.feature_builder import FeatureBuilder
from model.lightgbm_runner import LightGBMRunner
from utils.backup_manager import BackupManager
from utils.logger_config import configure_logging, set_request_id

LOGGER = structlog.get_logger("forecast_orchestrator")
configure_logging()


def _ensure_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required file missing or empty: {path}")


@dataclass(slots=True)
class ForecastOrchestrator:
    base_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    feature_dir: Path = Path("data/feature")
    forecast_dir: Path = Path("data/forecast")
    logger: structlog.stdlib.BoundLogger = field(
        default_factory=lambda: structlog.get_logger("forecast_orchestrator")
    )
    builder: Optional[FeatureBuilder] = None
    runner: Optional[LightGBMRunner] = None
    backup_manager: BackupManager = field(default_factory=BackupManager)

    def run_full_pipeline(self, *, run_date: dt.date | None = None) -> Path:
        run_date = run_date or dt.date.today()
        suffix = run_date.strftime("%Y%m%d")
        set_request_id(f"orchestrator-{suffix}")
        self._ensure_directories()

        raw_csv = self.raw_dir / f"raw_{suffix}.csv"
        feature_csv = self.feature_dir / f"feature_{suffix}.csv"
        forecast_csv = self.forecast_dir / f"forecast_{suffix}.csv"

        self.logger.info("✅ pipeline_start", run_date=suffix)

        builder = self.builder or FeatureBuilder(raw_csv=raw_csv, feature_csv=feature_csv)
        future_feature_csv = self.feature_dir / f"future_{suffix}.csv"
        try:
            self.logger.info(
                "✅ feature_build_start", raw=raw_csv.as_posix(), out=feature_csv.as_posix()
            )
            builder.process()
            builder.build_future_features(horizon_weeks=4, future_csv_path=future_feature_csv)
            _ensure_file(feature_csv)
            if raw_csv.exists():
                self.backup_manager.backup_csv_files("raw", suffix, [raw_csv])
            self.backup_manager.backup_csv_files("feature", suffix, [feature_csv])
            self.backup_manager.backup_csv_files("future_feature", suffix, [future_feature_csv])
            self.logger.info("✅ feature_build_complete", feature_csv=feature_csv.as_posix())
        except Exception as exc:
            self.logger.error("❌ feature_build_failed", error=str(exc))
            raise

        runner = self.runner or LightGBMRunner()
        future_forecast_csv = self.forecast_dir / f"future_forecast_{suffix}.csv"
        try:
            self.logger.info("✅ model_run_start", feature_csv=feature_csv.as_posix())
            runner.train_and_predict(
                feature_csv,
                forecast_csv,
                future_feature_csv_path=future_feature_csv,
                future_forecast_csv_path=future_forecast_csv,
            )
            _ensure_file(forecast_csv)
            metrics = runner.get_model_metrics()
            self.logger.info(
                "✅ model_run_complete", forecast_csv=forecast_csv.as_posix(), metrics=metrics
            )
        except Exception as exc:
            self.logger.error("❌ model_run_failed", error=str(exc))
            raise

        sanitized = runner.export_forecast_to_starrocks_format(forecast_csv)
        future_forecast_path = runner.get_future_forecast_path()
        self.backup_manager.backup_csv_files("forecast", suffix, [sanitized])
        if future_forecast_path and future_forecast_path.exists():
            self.backup_manager.backup_csv_files("forecast_future", suffix, [future_forecast_path])
        self.backup_manager.cleanup_old_backups()
        self.logger.info(
            "✅ pipeline_complete",
            sanitized_csv=sanitized.as_posix(),
            future_forecast=future_forecast_path.as_posix() if future_forecast_path else None,
            run_date=suffix,
        )
        return sanitized

    def _ensure_directories(self) -> None:
        for directory in {self.base_dir, self.raw_dir, self.feature_dir, self.forecast_dir}:
            directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    orchestrator = ForecastOrchestrator()
    orchestrator.run_full_pipeline()
