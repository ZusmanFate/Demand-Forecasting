# Demand-Forecasting 故障排查指南

面向运维/工程人员的排障手册，覆盖配置加载 → 数据导出 → 特征工程 → 模型训练 → 结果写回 → Prefect 调度的常见故障、排查步骤与解决思路。

---
## 1. 配置加载（ConfigLoader / .env）

| 现象 | 典型报错 | 排查步骤 | 解决方案 |
| --- | --- | --- | --- |
| `.env` 缺少 StarRocks 信息 | `KeyError: Missing required environment variable` | 1) 打开 `.env` 检查键名是否匹配；2) 运行 `python - <<'PY'` 脚本读取变量确认 | 补齐 `STARROCKS_*` 等必填项，重新运行流程。|
| 日志级别未生效 | 日志仍大量输出 INFO | 查看 `LOG_LEVEL` 环境变量；运行 `python -m utils.logger_config` 验证 | 在 `.env` 中设置 `LOG_LEVEL=WARNING`，重新启动流程。|

**常用命令**：
```bash
# 检查当前环境变量（PowerShell）
Get-ChildItem Env:STARROCKS*

# 快速验证 ConfigLoader 输出
python - <<'PY'
from config.config_loader import ConfigLoader
print(ConfigLoader().load_config())
PY
```

---
## 2. 数据导出（StarRocks → CSV）

| 现象 | 典型报错 | 排查步骤 | 解决方案 |
| --- | --- | --- | --- |
| 无法连接 StarRocks | `Can't connect to MySQL server on 'host:9030'` | 1) 检查网络/防火墙；2) 使用 `mysql -h host -P 9030` 测试；3) 查看 `logs/pipeline.log` | 确认 StarRocks 服务可达、账号密码正确，必要时联系 DBA。|
| CSV 为空 | `ValueError: No data returned from weekly_sales_raw` | 1) 在 StarRocks 手动运行导出 SQL；2) 查看 SQL 中的 SKU/日期过滤 | 调整 SQL，确认源表数据存在。|

**命令示例**：
```bash
# 测试数据库连通性
mysql -h 172.17.224.210 -P 9030 -u erplysrser -p

# 独立执行导出脚本
python -m db.export_raw --run_date 2025-12-30
```

---
## 3. 特征工程（FeatureBuilder）

| 现象 | 典型报错 | 排查步骤 | 解决方案 |
| --- | --- | --- | --- |
| 特征 CSV 为空 | `Required file missing or empty` 或 `train/test split empty` | 1) 检查 `data/raw/raw_YYYYMMDD.csv` 是否存在/有数据；2) 重新跑 `FeatureBuilder.process()` | 确保原始 CSV 不为空，并且满足最少日期范围。|
| 节假日加载异常 | `ModuleNotFoundError: chinese_calendar` 或 日志 `holiday_compute_failed` | 1) 确认依赖安装完整；2) 检查 `.env` 中是否设置了节假日相关开关 | 重新运行 `pip install -r requirements.txt`，确保 `chinese-calendar==1.9.0` 安装成功。|

**命令示例**：
```bash
# 单独运行特征工程
python test_feature_pipeline.py

# 检查 CSV 完整性
python - <<'PY'
from pathlib import Path
from pandas import read_csv
df = read_csv(Path('data/feature/feature_20260104.csv'))
print(df.tail())
PY
```

---
## 4. 模型训练（LightGBMRunner）

| 现象 | 典型报错 | 排查步骤 | 解决方案 |
| --- | --- | --- | --- |
| 训练阶段崩溃 | `ImportError: Missing required packages` | 检查虚拟环境、重新安装依赖 | `pip install -r requirements.txt`，并重新激活 venv。|
| 特征缺失导致训练失败 | `KeyError: Missing columns in feature CSV` | 1) 打开 feature CSV 检查列；2) 确认 FeatureBuilder 是否输出完整 | 确保 `week_end_date/total_weekly_sales` 等必需列存在。|

**命令示例**：
```bash
# 查看最近模型日志
Get-Content -Tail 200 logs/pipeline.log

# 直接调用 LightGBMRunner
python test_feature_single.py
```

---
## 5. 结果写回（StarRocks 导入）

| 现象 | 典型报错 | 排查步骤 | 解决方案 |
| --- | --- | --- | --- |
| 写入失败 | `KeyError: Forecast CSV missing columns` 或 `starrocks_connection_failed` | 1) 检查 `data/forecast/forecast_*.csv` 列头；2) 确认 StarRocks 连接；3) 查看 `logs/pipeline.log` | 保证 CSV 包含 `week_end_date,total_weekly_sales,pred_sales,error,abs_error,rmse` 并为 UTF-8-SIG。|
| 表不存在 | `Table weekly_sales_forecast does not exist` | 登录 StarRocks，检查目标表建表语句 | 创建或修复目标表，赋予写入权限。|

**命令示例**：
```bash
python - <<'PY'
from db.starrocks_oper import StarRocksOper
from pathlib import Path
StarRocksOper().write_forecast_data_to_starrocks(Path('data/forecast/forecast_20260104.csv'))
PY
```

---
## 6. Prefect 调度 / 任务重试

| 现象 | 典型报错 | 排查步骤 | 解决方案 |
| --- | --- | --- | --- |
| Task 重试 3 次仍失败 | Prefect UI 显示 `Failed` | 1) 打开 Flow Run detail 查看日志；2) 对照本指南按阶段排查 | 按出错 Task（导出/特征/模型/写回）执行对应修复，必要时本地单步运行。|
| Prefect Agent 无法消费任务 | `Queue demand-local not found` | 检查 Prefect server 是否运行，Agent 是否指向正确队列 | 执行 `prefect server start` 后再 `prefect agent start -q demand-local`。|

**命令示例**：
```bash
prefect server start
prefect agent start -q demand-local
prefect flow-run ls -l 5
prefect deployment run full_forecast_flow/weekly-demand-forecast
```

---
## 7. 文件备份与清理

- `utils.backup_manager.BackupManager` 会在 `backups/<RUN_DATE>/<step>/` 下保存 `_backup.csv`，并根据 `BACKUP_RETENTION_DAYS`（默认 30）自动清理过期目录。
- 如需手动清理：
```bash
Remove-Item -Recurse -Force backups\20250101
```

---
## 8. 依赖/版本确认

| 场景 | 命令 |
| --- | --- |
| 列出关键依赖版本 | `pip show pandas lightgbm optuna prefect` |
| 检查 Python 版本 | `python --version` |
| 重新安装全部依赖 | `pip install -r requirements.txt --upgrade --force-reinstall` |

---
## 9. 日志分析技巧

- 所有模块使用 `utils.logger_config.configure_logging()` 输出 JSON 日志，包含 `timestamp/level/module/func_name/request_id/消息`。
- Prefect UI 会同步展示 structlog 日志，可通过标签 `weekly-forecast` 筛选。
- 本地日志位于 `logs/pipeline.log`，按天自动滚动保存。

```bash
# 过滤指定 request_id
Get-Content logs/pipeline.log | Select-String "orchestrator-20260104"
```

---
如遇未覆盖的问题，建议：
1. 先在 Prefect UI / logs 中定位出错步骤。
2. 依据步骤阅读对应模块源代码（`feature_builder.py`、`lightgbm_runner.py`等）。
3. 必要时在 `issues` 中记录复现步骤与日志，便于进一步支持。
