"""StarRocks import/export helpers for the demand forecast pipeline."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import mysql.connector
import pandas as pd
import structlog

from config.config_loader import get_env

DEFAULT_EXPORT_COLUMNS = (
    "gcode",
    "gname",
    "week_end_date",
    "week_end_year",
    "total_weekly_sales",
    "weekly_purchase_tax_rate_mean",
    "weekly_invoice_tax_rate_mean",
)

FORECAST_COLUMNS = (
    "week_end_date",
    "total_weekly_sales",
    "pred_sales",
    "error",
    "abs_error",
    "rmse",
)


def _default_conn_kwargs() -> Dict[str, Any]:
    return {
        "host": get_env("STARROCKS_HOST", required=True),
        "port": int(get_env("STARROCKS_PORT", required=True)),
        "user": get_env("STARROCKS_USER", required=True),
        "password": get_env("STARROCKS_PASSWORD", required=True),
        "database": get_env("STARROCKS_DB", required=True),
        "charset": "utf8mb4",
    }


_DEFAULT_CONN = _default_conn_kwargs()


@dataclass(slots=True)
class StarRocksOper:
    """Encapsulates StarRocks export/import logic for the pipeline."""

    host: str = _DEFAULT_CONN["host"]
    port: int = _DEFAULT_CONN["port"]
    user: str = _DEFAULT_CONN["user"]
    password: str = _DEFAULT_CONN["password"]
    database: str = _DEFAULT_CONN["database"]
    logger: structlog.stdlib.BoundLogger = field(
        default_factory=lambda: structlog.get_logger("starrocks_oper")
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def export_raw_data_to_csv(
        self,
        raw_csv_path: Path,
        table_name: str = "weekly_sales_raw",
        columns: Iterable[str] = DEFAULT_EXPORT_COLUMNS,
    ) -> Path:
        raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
        query = self._build_export_sql(table_name, columns)
        with self._connect() as connection:
            cursor = connection.cursor(dictionary=True)
            self._ensure_table_has_rows(cursor, table_name)
            cursor.execute(query)
            rows = cursor.fetchall()
        if not rows:
            raise ValueError(f"No data returned from {table_name}")

        fieldnames = list(columns)
        with raw_csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col) for col in fieldnames})
        self.logger.info(
            "raw_export_finished",
            table=table_name,
            rows=len(rows),
            path=raw_csv_path.as_posix(),
        )
        return raw_csv_path

    def export_query_to_csv(self, sql: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
        with output_path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            writer.writerows(rows)
        self.logger.info(
            "query_export_finished",
            rows=len(rows),
            path=output_path.as_posix(),
        )
        return output_path

    # Backwards compatibility for legacy scripts ---------------------------------
    def export_to_csv(self, sql: str, output_path: Path) -> Path:  # pragma: no cover
        self.logger.warning(
            "deprecated_export_to_csv",
            hint="Use export_raw_data_to_csv or export_query_to_csv",
        )
        return self.export_query_to_csv(sql, output_path)

    def write_forecast_data_to_starrocks(
        self,
        forecast_csv_path: Path,
        table_name: str = "weekly_sales_forecast",
        batch_size: int = 500,
    ) -> int:
        if not forecast_csv_path.exists():
            raise FileNotFoundError(f"Forecast CSV not found: {forecast_csv_path}")
        df = pd.read_csv(forecast_csv_path)
        missing = sorted(set(FORECAST_COLUMNS) - set(df.columns))
        if missing:
            raise KeyError(f"Forecast CSV missing columns: {missing}")
        df["week_end_date"] = pd.to_datetime(df["week_end_date"]).dt.strftime("%Y-%m-%d")
        rows = df[list(FORECAST_COLUMNS)].values.tolist()
        if not rows:
            raise ValueError("Forecast CSV contains no records to import")

        insert_sql = self._build_insert_sql(table_name, FORECAST_COLUMNS)
        affected = 0
        with self._connect() as connection:
            cursor = connection.cursor()
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                cursor.executemany(insert_sql, batch)
                affected += cursor.rowcount
            connection.commit()
        self.logger.info(
            "forecast_import_finished",
            table=table_name,
            rows=affected,
            path=forecast_csv_path.as_posix(),
        )
        return affected

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _connect(self):  # type: ignore[override]
        try:
            return mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                autocommit=False,
            )
        except mysql.connector.Error as exc:  # pragma: no cover - network failure
            self.logger.error("starrocks_connection_failed", error=str(exc))
            raise

    @staticmethod
    def _build_export_sql(table: str, columns: Iterable[str]) -> str:
        column_list = ",".join(columns)
        return f"SELECT {column_list} FROM {table} ORDER BY week_end_date"

    @staticmethod
    def _build_insert_sql(table: str, columns: Iterable[str]) -> str:
        placeholders = ",".join(["%s"] * len(columns))
        column_list = ",".join(columns)
        return f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})"

    def _ensure_table_has_rows(self, cursor, table_name: str) -> None:
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        if cursor.fetchone() is None:
            raise ValueError(f"Table {table_name} does not exist in StarRocks")
        cursor.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
        count = cursor.fetchone()["cnt"]
        if count == 0:
            raise ValueError(f"Table {table_name} has no data to export")


class StarRocksOperator(StarRocksOper):  # pragma: no cover - legacy alias
    """Maintain compatibility with older imports."""

    pass


if __name__ == "__main__":
    oper = StarRocksOper()
    raw_csv = Path("data/raw/raw_20251230.csv")
    oper.export_raw_data_to_csv(raw_csv)
    oper.write_forecast_data_to_starrocks(Path("data/forecast/forecast_20251230.csv"))
