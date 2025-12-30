"""StarRocks CSV import/export helpers using mysql-connector."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import mysql.connector

from config.config_loader import get_starrocks_conn_kwargs


class StarRocksOperator:
    """Utility class that isolates all StarRocks interactions."""

    def __init__(self) -> None:
        self._conn_kwargs = get_starrocks_conn_kwargs()

    def _connect(self):
        return mysql.connector.connect(**self._conn_kwargs)

    def export_to_csv(self, sql: str, output_path: Path) -> Path:
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute(sql)
        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([col[0] for col in cursor.description])
            writer.writerows(cursor)
        cursor.close()
        connection.close()
        return output_path

    def import_from_csv(self, table: str, csv_path: Path, columns: Iterable[str]) -> int:
        connection = self._connect()
        cursor = connection.cursor()
        placeholders = ",".join(["%s"] * len(columns))
        column_list = ",".join(columns)
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = [tuple(row[col] for col in columns) for row in reader]
        cursor.executemany(
            f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})",
            rows,
        )
        connection.commit()
        affected = cursor.rowcount
        cursor.close()
        connection.close()
        return affected
