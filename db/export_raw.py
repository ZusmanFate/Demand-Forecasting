"""Standalone helper to export StarRocks data into CSV."""
from __future__ import annotations

import datetime
from pathlib import Path

from config.config_loader import get_env, get_path_from_env
from db.starrocks_oper import StarRocksOperator


def export_csv(run_date: str | None = None) -> Path:
    """
    导出 StarRocks 数据到 CSV
    
    Args:
        run_date: 指定查询的日期（如 '2025-12-25'），None 表示使用当前日期
    
    Returns:
        导出的 CSV 文件路径
    """
    sql_template = get_env("STARROCKS_EXPORT_SQL", required=True)
    
    # 如果没传 run_date，就用今天
    if run_date is None:
        today = datetime.date.today()
        run_date_str = today.strftime("%Y%m%d")           # 用于文件名：20251230
        query_date = "CURRENT_DATE()"                     # 用于 SQL 替换
    else:
        run_date_str = run_date.replace("-", "")
        query_date = f"'{run_date}'"

    # 替换 SQL 中的占位符
    sql = sql_template.replace("{run_date}", query_date)

    # 输出路径和文件名
    raw_dir = get_path_from_env("RAW_DATA_DIR", is_dir=True)
    raw_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    filename = f"raw_{run_date_str}.csv"   # ← 关键：按日期命名！
    target = raw_dir / filename

    # 导出
    StarRocksOperator().export_to_csv(sql, target)
    
    return target


if __name__ == "__main__":
    output = export_csv()  # 默认导出今天的数据
    print(f"CSV exported to {output}")
    print(f"文件名为: {output.name}")
