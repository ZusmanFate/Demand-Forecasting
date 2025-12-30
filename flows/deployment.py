"""Prefect deployment definitions for the Demand-Forecasting project."""
from __future__ import annotations

from pathlib import Path

from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

from config.config_loader import get_prefect_settings
from flows.prefect_flow import demand_forecast_flow


def build_deployments() -> list[Deployment]:
    settings = get_prefect_settings()
    base_params = {"run_mode": settings["run_mode"]}

    daily = Deployment.build_from_flow(
        flow=demand_forecast_flow,
        name="daily",
        schedule=CronSchedule(cron="0 0 * * *", timezone="Asia/Shanghai"),
        parameters=base_params,
        work_queue_name=settings["work_queue"],
        tags=["demand-forecast", "daily"],
    )

    weekly = Deployment.build_from_flow(
        flow=demand_forecast_flow,
        name="weekly",
        schedule=CronSchedule(cron="0 0 * * 1", timezone="Asia/Shanghai"),
        parameters={"run_mode": "weekly"},
        work_queue_name=settings["work_queue"],
        tags=["demand-forecast", "weekly"],
    )

    return [daily, weekly]


def export_deployments(artifacts_dir: Path | None = None) -> list[Path]:
    artifacts_dir = artifacts_dir or Path("flows")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for deployment in build_deployments():
        path = artifacts_dir / f"{deployment.flow_name}-{deployment.name}-deployment.yaml"
        deployment.apply(path)
        outputs.append(path)
    return outputs


if __name__ == "__main__":
    for yaml_path in export_deployments(Path("flows")):
        print(f"Deployment manifest written to {yaml_path}")
