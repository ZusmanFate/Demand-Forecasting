"""Feature engineering utilities for demand forecasting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from feature_engine.creation import LagFeatures


@dataclass(slots=True)
class FeatureBuilder:
    raw_csv: Path
    feature_csv: Path
    date_col: str = "ds"
    target_col: str = "demand"

    def build(self, *, run_mode: Literal["daily", "weekly"] = "daily") -> Path:
        df = pd.read_csv(self.raw_csv)
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(self.date_col).reset_index(drop=True)
        df = df.drop_duplicates(subset=[self.date_col])

        lagger = LagFeatures(variables=[self.target_col], periods=[1, 7, 14])
        df = lagger.fit_transform(df)

        df["rolling_mean_7"] = (
            df[self.target_col].rolling(window=7, min_periods=1).mean().round(4)
        )
        df["run_mode"] = run_mode

        self.feature_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.feature_csv, index=False)
        return self.feature_csv
