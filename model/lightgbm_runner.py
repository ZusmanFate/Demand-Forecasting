"""LightGBM training/prediction utilities operating purely on CSV files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import lightgbm as lgb
import pandas as pd


@dataclass(slots=True)
class LightGBMRunner:
    model_path: Path
    target_col: str = "demand"
    drop_columns: Sequence[str] = ("ds", "run_mode")

    def _select_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [c for c in df.columns if c not in {*self.drop_columns, self.target_col}]
        return df[feature_cols]

    def train(self, feature_csv: Path) -> Path:
        df = pd.read_csv(feature_csv)
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not present in {feature_csv}")
        features = self._select_feature_frame(df)
        train_data = lgb.Dataset(features, label=df[self.target_col])
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "learning_rate": 0.05,
            "num_leaves": 31,
        }
        model = lgb.train(params, train_data, num_boost_round=200)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        return self.model_path

    def predict(self, feature_csv: Path, output_csv: Path, run_date: str) -> Path:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}. Train it first.")
        model = joblib.load(self.model_path)
        df = pd.read_csv(feature_csv)
        features = self._select_feature_frame(df)
        predictions = model.predict(features)
        result = pd.DataFrame({
            "ds": df.get("ds"),
            "prediction": predictions,
            "run_date": run_date,
        })
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        return output_csv
