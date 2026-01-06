# Demand-Forecasting · 端到端需求预测闭环

本项目实现「StarRocks 原始数据 → 特征工程 → LightGBM 训练/预测 → 预测结果写回 StarRocks」的自动化闭环，适配本地脚本、Prefect 流程以及 Docker 部署场景。

## 目录结构
```
demand_forecast_light/
├── README.md                   # 使用说明、排障指南
├── requirements.txt            # 分类依赖列表（带用途注释）
├── .env                        # StarRocks / 目录配置（示例见下）
├── config/
│   └── config_loader.py        # ConfigLoader：集中加载 .env，暴露给其他模块
├── db/
│   └── starrocks_oper.py       # StarRocks 导出&写入实现（兼容 FeatureBuilder / Runner）
├── feature/
│   └── feature_builder.py      # CSV → 特征工程（节假日增强、拆分、日志）
├── model/
│   └── lightgbm_runner.py      # LightGBM 训练、调参、结果输出
├── flows/
│   ├── orchestrator.py         # 本地 orchestrator 脚本
│   └── prefect_forecast_flow.py# Prefect Flow（拆分任务、调度、未来预测入库）
├── data/
│   ├── raw/                    # StarRocks 导出的原始 CSV
│   ├── feature/                # 特征 CSV（FeatureBuilder 输出）
│   └── forecast/               # 预测 CSV（Runner 输出，并写回 StarRocks 前可复用）
├── saved_models/               # LightGBM 模型与重要性产物
└── logs/                       # 结构化日志
```

## 环境准备
1. **Python 版本**：建议 3.11+（开发环境为 3.12）。
2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
3. **配置 .env（示例）**：
   ```env
   # StarRocks
   STARROCKS_HOST=172.17.224.210
   STARROCKS_PORT=9030
   STARROCKS_USER=erplysrser
   STARROCKS_PASSWORD=******
   STARROCKS_DB=datasense_dlink_erpservice
   STARROCKS_EXPORT_SQL="WITH ..."  # 见实际 SQL
   STARROCKS_IMPORT_TABLE=weekly_sales_forecast
   STARROCKS_FUTURE_TABLE=forecast_result

   # 数据目录
   BASE_DATA_DIR=data
   MODEL_DIR=saved_models
   LOG_FILE=logs/pipeline.log

   # Prefect（如使用）
   PREFECT_API_URL=http://prefect-server:4200/api
   PREFECT_WORK_QUEUE=demand-local
   PREFECT_DEPLOYMENT_NAME=demand-forecast/daily
   RUN_MODE=daily
   ```

`config/config_loader.py` 会尝试加载 `.env` 并提供默认值，确保缺少配置时仍可运行。

## 一键执行全流程
```bash
python flows/orchestrator.py
```
流程包含：
1. `db/StarRocksOper.export_raw_data_to_csv`（或已有的 Prefect/脚本导出）
2. `feature/FeatureBuilder.process()` 生成特征 CSV
3. `model/LightGBMRunner.train_and_predict()` 训练+预测
4. `StarRocksOper.write_forecast_data_to_starrocks()` 写回数据库
5. `StarRocksOper.write_future_forecast_to_starrocks()` 将未来 4 周的预测写入 `forecast_result`（可通过 `STARROCKS_FUTURE_TABLE` 覆写表名）。

Prefect 场景推荐使用 `flows/prefect_forecast_flow.py`：

```bash
# 本地快速验证（默认以当天日期 suffix）
python -m flows.prefect_forecast_flow

# 或者传入自定义日期
python - <<'PY'
from flows.prefect_forecast_flow import full_forecast_flow
full_forecast_flow(run_date="20260104")
PY
```

## 模块说明
- **config_loader.ConfigLoader**：加载目录/StarRocks/模型配置，可传入 `FeatureBuilder`、`LightGBMRunner`、`StarRocksOper`。
- **db.starrocks_oper**：
  - `export_raw_data_to_csv`：校验表、执行查询、写出 UTF-8-SIG CSV。
  - `write_forecast_data_to_starrocks`：校验预测 CSV 字段、批量写入目标表。
  - `write_future_forecast_to_starrocks`：写入未来预测结果到 `forecast_result` 表。
- **feature.feature_builder.FeatureBuilder**：节假日增强、统计特征、98.7% 时序拆分（最近 6 年训练）并生成特征 CSV。
- **model.lightgbm_runner.LightGBMRunner**：Optuna 调参、多种子训练、SHAP/重要度、预测 CSV、StarRocks 兼容输出。
- **flows.orchestrator.ForecastOrchestrator**：本地脚本 orchestrator，串联“原始 CSV → 特征 → 模型 → 预测”，带完整日志与异常处理。

## 输出目录
- `data/raw/raw_YYYYMMDD.csv`：StarRocks 原始导出（FeatureBuilder 输入）。
- `data/feature/feature_YYYYMMDD.csv`：特征工程结果。
- `data/forecast/forecast_YYYYMMDD.csv`：模型预测（含 `week_end_date/total_weekly_sales/pred_sales/error/abs_error/rmse`）。
- `data/forecast/future_forecast_YYYYMMDD.csv`：未来 4 周 horizon 的预测，仅包含 `week_end_date/pred_sales`。
- `feature_importance/*.csv|.png`：特征重要度+SHAP 结果。
- `saved_models/lightgbm_model.pkl`：训练模型与配置 bundle。

## 常见问题排查
| 问题 | 排查步骤 |
| --- | --- |
| `config_loader` 报缺少变量 | 检查 `.env` 是否存在/键名是否一致，或在 `.env` 中补全必填项。
| `FeatureBuilder` 找不到原始 CSV | 确认 `data/raw/raw_YYYYMMDD.csv` 已生成，文件路径与日期后缀匹配.
| 模型训练报 ImportError（如 lightgbm/optuna/shap） | 重新执行 `pip install -r requirements.txt`，并确保虚拟环境激活.
| 写回 StarRocks 失败 | 1) 校验 `forecast_csv` 字段是否完整；2) 检查 StarRocks 连接权限；3) 查看日志中 `starrocks_connection_failed`.
| Prefect Flow 无法运行 | 确保 Prefect API/工作队列可访问，并在 `.env` 中配置正确.

## Prefect 部署与监控

1. **构建并应用 Deployment（默认周日 23:00 执行）**
   ```bash
   # 构建部署（名称可自定义）
   prefect deployment build flows/prefect_forecast_flow.py:full_forecast_flow \
       --name weekly-demand-forecast \
       --tag weekly-forecast --tag sales-prediction \
       --cron "0 23 * * 0" --timezone "Asia/Shanghai"

   # 应用部署到 Prefect API（需先登录 Prefect）
   prefect deployment apply full_forecast_flow-deployment.yaml
   ```
2. **启动 Prefect UI + Agent**
   ```bash
   prefect server start         # 启动 UI + API（默认 http://127.0.0.1:4200）
   prefect agent start -q demand-local   # 绑定工作队列，消费 flow run
   ```
   Prefect UI 中可看到：
   - Flow/Task 标签：`weekly-forecast`, `sales-prediction`
   - 每个 Task 日志（结构化 JSON 依旧由 structlog 输出）
   - Task/Flow 的耗时、输入输出路径、失败重试历史

3. **手动触发/查看运行结果**
   ```bash
   # 手动触发一次 Deployment（可传参数）
   prefect deployment run full_forecast_flow/weekly-demand-forecast \
       --params '{"run_date": "20260104"}'

   # 查看最近运行状态
   prefect flow-run ls -l 10
   prefect task-run ls -l 10
   ```
   Prefect UI 中的 Flow Run detail 会展示 `raw_csv / feature_csv / forecast_csv / metrics / rows_written / duration_seconds` 等 summary。
   Flow 的 summary 会包含 `future_feature_csv/future_forecast_csv/future_rows_written` 字段，用于确认 horizon 结果写入 `forecast_result`.

4. **日志兼容**：Prefect 自动收集 `structlog` 输出，可在 UI 或 `logs/pipeline.log` 中分析；失败 Task 会按配置重试 3 次，间隔 60 秒.

5. **依赖提醒**：Prefect Flow 依赖 `prefect==3.4.25`，已在 `requirements.txt` → Workflow & Orchestration 分类列出.

---
## Terraform/Docker/CI 适配
- 所有路径基于 `pathlib.Path`，兼容 Linux/Windows.
- `.env` 由 `python-dotenv` 读取，可被容器/CI 通过环境变量覆写.
- `.env` 由 `python-dotenv` 读取，可被容器/CI 通过环境变量覆写。
- 日志统一使用 structlog，便于后续接入 ELK、Datadog 等.

## 参考命令
- 导出单日数据：`python db/export_raw.py --run_date 2025-12-30`
- 仅运行特征工程：`python test_feature_pipeline.py`
- 仅运行模型：`python test_feature_single.py`
- 详细排障指南：`docs/troubleshooting_guide.md`

如需扩展（多 SKU、更多时间粒度、CI/CD 集成），可在当前模块基础上逐步拆分。欢迎提 Issue/PR.