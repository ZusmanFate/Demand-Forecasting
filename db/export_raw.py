"""Standalone helper to export StarRocks data into CSV."""
from __future__ import annotations

from pathlib import Path

from config.config_loader import get_env, get_path_from_env
from db.starrocks_oper import StarRocksOperator


def export_csv(run_date: str | None = None) -> Path:
    sql_template = get_env("STARROCKS_EXPORT_SQL", required=True)
    sql = sql_template.replace("{run_date}", run_date or "CURRENT_DATE")
    raw_dir = get_path_from_env("RAW_DATA_DIR", is_dir=True)
    filename = f"raw_{run_date or 'latest'}.csv"
    target = raw_dir / filename
    StarRocksOperator().export_to_csv(sql, target)
    return target


if __name__ == "__main__":
    output = export_csv()
    print(f"CSV exported to {output}")
