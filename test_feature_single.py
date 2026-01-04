from pathlib import Path
from model.lightgbm_runner import LightGBMRunner

# 实例化并执行
runner = LightGBMRunner()
runner.train_and_predict(
    feature_csv_path=Path("data/feature/feature_20251230.csv"),
    forecast_csv_path=Path("data/forecast/forecast_20251230.csv"),
)