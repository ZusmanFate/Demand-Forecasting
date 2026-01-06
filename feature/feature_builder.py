"""Feature engineering pipeline with enhanced Chinese holiday features.

This module is tailored for the ``demand_forecast_light`` project and focuses on
transforming aggregated weekly sales into model-ready features while keeping
module boundaries low-coupled. Holiday awareness is powered by the
``chinese-calendar==1.9.0`` package to ensure business-critical peaks are
captured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import structlog
from chinese_calendar import get_holidays, is_workday


LOGGER = structlog.get_logger("feature_builder")
HOLIDAY_LIB_VERSION = "1.9.0"
MIN_YEAR = 2019


def load_chinese_holidays(start_year: int, end_year: int, logger=LOGGER) -> set[pd.Timestamp]:
    """Load and normalize Chinese legal holidays across a year range.

    The chinese_calendar.get_holidays API returns holiday date objects (with
    names). We normalize them into ``pd.Timestamp`` and deduplicate to keep the
    rest of the pipeline type-safe.
    """

    holiday_dates: set[pd.Timestamp] = set()
    try:
        for year in range(start_year, end_year + 1):
            try:
                year_holidays = get_holidays(start=f"{year}-01-01", end=f"{year}-12-31")
            except Exception as year_err:  # pragma: no cover - defensive
                logger.warning(
                    "holiday_year_load_failed",
                    year=year,
                    error=str(year_err),
                    lib_version=HOLIDAY_LIB_VERSION,
                )
                continue

            for entry in year_holidays:
                # new versions return tuples (date, name)
                date_obj = entry[0] if isinstance(entry, (tuple, list)) else entry
                holiday_dates.add(pd.Timestamp(date_obj).normalize())

        if not holiday_dates:
            logger.error(
                "holiday_load_empty",
                start_year=start_year,
                end_year=end_year,
                lib_version=HOLIDAY_LIB_VERSION,
            )
    except Exception as exc:  # pragma: no cover - fail safe logging
        logger.exception(
            "holiday_load_failed",
            error=str(exc),
            lib_version=HOLIDAY_LIB_VERSION,
        )
        return set()

    logger.info(
        "holiday_loaded",
        count=len(holiday_dates),
        start_year=start_year,
        end_year=end_year,
        lib_version=HOLIDAY_LIB_VERSION,
    )
    return holiday_dates


def calc_holiday_features(
    timestamps: pd.Series,
    holiday_dates: set[pd.Timestamp],
    logger=LOGGER,
) -> pd.DataFrame:
    """Compute week-level holiday features using normalized holiday dates."""

    if not holiday_dates:
        logger.warning("holiday_features_zero_fallback")
        return pd.DataFrame(
            {
                "num_holidays_in_week": np.zeros(len(timestamps), dtype=int),
                "min_days_to_next_holiday_in_week": np.full(len(timestamps), np.nan),
                "num_days_before_holiday_period_in_week": np.zeros(len(timestamps), dtype=int),
                "num_workday_holidays_in_week": np.zeros(len(timestamps), dtype=int),
            }
        )

    holiday_set = {ts.normalize() for ts in holiday_dates}
    holiday_array = np.array(sorted(holiday_set), dtype="datetime64[D]")

    def week_range(ts: pd.Timestamp) -> Iterable[pd.Timestamp]:
        start = ts - pd.Timedelta(days=6)
        return pd.date_range(start=start.normalize(), end=ts.normalize(), freq="D")

    def days_to_next_holiday(day: pd.Timestamp) -> float:
        if holiday_array.size == 0:
            return np.nan
        normalized = np.datetime64(day.normalize().date())
        idx = holiday_array.searchsorted(normalized)
        if idx >= holiday_array.size:
            return np.nan
        delta = (holiday_array[idx] - normalized).astype(int)
        return float(delta)

    def summarize_week(ts: pd.Timestamp) -> Dict[str, float]:
        week_days = list(week_range(ts))
        holidays_in_week = sum(1 for day in week_days if day.normalize() in holiday_set)

        min_days = np.nan
        before_count = 0
        workday_holidays = 0

        for day in week_days:
            delta = days_to_next_holiday(day)
            if np.isnan(min_days) or (not np.isnan(delta) and delta < min_days):
                min_days = delta

            for offset in range(1, 4):
                if day.normalize() + pd.Timedelta(days=offset) in holiday_set:
                    before_count += 1

            if day.weekday() >= 5 and is_workday(day.date()):
                workday_holidays += 1

        return {
            "num_holidays_in_week": holidays_in_week,
            "min_days_to_next_holiday_in_week": min_days,
            "num_days_before_holiday_period_in_week": before_count,
            "num_workday_holidays_in_week": workday_holidays,
        }

    records = [summarize_week(ts) for ts in timestamps]
    holiday_df = pd.DataFrame(records)
    # replace NaN when no future holiday is available
    holiday_df["min_days_to_next_holiday_in_week"].fillna(-1.0, inplace=True)
    return holiday_df


@dataclass(slots=True)
class FeatureBuilder:
    """Convert StarRocks weekly aggregates into model features."""

    raw_csv: Path
    feature_csv: Path
    date_col: str = "week_end_date"
    target_col: str = "total_weekly_sales"
    logger: structlog.stdlib.BoundLogger = field(
        default_factory=lambda: structlog.get_logger("feature_builder")
    )

    def process(self) -> Path:
        """Public entrypoint executed by flows and CLI scripts."""

        df = self._load_raw()
        df = self._filter_year_range(df)
        df = self._enrich_temporal_features(df)
        df = self._add_statistical_features(df)
        df = self._add_holiday_features(df)
        df = self._assign_data_split(df)
        df = self._finalize(df)

        self.feature_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.feature_csv, index=False)
        self.logger.info(
            "feature_csv_written",
            path=self.feature_csv.as_posix(),
            rows=len(df),
            columns=len(df.columns),
        )
        return self.feature_csv

    def build_future_features(
        self,
        *,
        horizon_weeks: int = 4,
        future_csv_path: Path | None = None,
    ) -> Path:
        """Create placeholder features for future weeks to feed into inference."""

        if horizon_weeks <= 0:
            raise ValueError("horizon_weeks must be positive")
        if not self.feature_csv.exists():
            raise FileNotFoundError(
                f"Feature CSV not found at {self.feature_csv}. Run process() first."
            )

        df = pd.read_csv(self.feature_csv)
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        future_df = self._generate_future_rows(df, horizon_weeks)

        if future_csv_path is None:
            future_csv_path = self._default_future_csv_path()

        future_csv_path.parent.mkdir(parents=True, exist_ok=True)
        future_df.to_csv(future_csv_path, index=False)
        self.logger.info(
            "future_feature_csv_written",
            path=future_csv_path.as_posix(),
            rows=len(future_df),
            horizon_weeks=horizon_weeks,
        )
        return future_csv_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_raw(self) -> pd.DataFrame:
        df = pd.read_csv(self.raw_csv)
        if self.date_col not in df.columns:
            raise KeyError(f"Missing date column '{self.date_col}' in raw CSV")
        df[self.date_col] = pd.to_datetime(df[self.date_col]).dt.normalize()
        df = df.sort_values(self.date_col).drop_duplicates(subset=[self.date_col]).reset_index(
            drop=True
        )
        if self.target_col not in df.columns:
            raise KeyError(f"Missing target column '{self.target_col}' in raw CSV")
        self.logger.info("raw_loaded", rows=len(df), columns=list(df.columns))
        return df

    def _filter_year_range(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df[self.date_col] >= pd.Timestamp(year=MIN_YEAR, month=1, day=1)
        filtered = df.loc[mask].copy()
        self.logger.info(
            "recent_years_filtered",
            min_year=MIN_YEAR,
            removed=len(df) - len(filtered),
            remaining=len(filtered),
        )
        return filtered

    def _enrich_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["year"] = df[self.date_col].dt.year
        df["month"] = df[self.date_col].dt.month
        df["quarter"] = df[self.date_col].dt.quarter
        df["weekofyear"] = df[self.date_col].dt.isocalendar().week.astype(int)
        df["weekofmonth"] = ((df[self.date_col].dt.day - 1) // 7) + 1
        df["is_month_start"] = df[self.date_col].dt.is_month_start.astype(int)
        df["is_month_end"] = df[self.date_col].dt.is_month_end.astype(int)
        df["days_in_month"] = df[self.date_col].dt.days_in_month
        df["elapsed_weeks"] = (
            (df[self.date_col] - df[self.date_col].min()).dt.days // 7
        )
        self.logger.info("temporal_features_added")
        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        lag_periods = (1, 2, 4, 8)
        for period in lag_periods:
            df[f"{self.target_col}_lag_{period}"] = df[self.target_col].shift(period)

        # Use only historical lags for derived diff features to avoid leakage
        for period in lag_periods:
            if period == 1:
                continue
            df[f"lag_{period}_diff"] = (
                df[f"{self.target_col}_lag_1"] - df[f"{self.target_col}_lag_{period}"]
            )

        shifted_target = df[self.target_col].shift(1)
        for window in (4, 8, 12, 26):
            df[f"rolling_mean_{window}"] = (
                shifted_target.rolling(window=window, min_periods=1).mean()
            )
            df[f"rolling_std_{window}"] = (
                shifted_target.rolling(window=window, min_periods=2).std().fillna(0.0)
            )

        df["pct_change_1w"] = shifted_target.pct_change(periods=1).fillna(0.0)
        df["pct_change_4w"] = shifted_target.pct_change(periods=4).fillna(0.0)
        df["rolling_mean_ratio_4_12"] = (
            df["rolling_mean_4"] / (df["rolling_mean_12"].replace(0, np.nan))
        ).fillna(1.0)

        rolling_mean_12 = df["rolling_mean_12"]
        rolling_std_12 = df["rolling_std_12"].replace(0, np.nan)
        df["target_zscore"] = (
            (df[f"{self.target_col}_lag_1"] - rolling_mean_12) / rolling_std_12
        ).fillna(0.0)

        self.logger.info("statistical_features_added")
        return df

    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        start_year = int(df[self.date_col].dt.year.min())
        end_year = int(df[self.date_col].dt.year.max())
        holidays = load_chinese_holidays(start_year, end_year, logger=self.logger)
        holiday_df = calc_holiday_features(df[self.date_col], holidays, logger=self.logger)
        df = pd.concat([df.reset_index(drop=True), holiday_df], axis=1)
        self.logger.info("holiday_features_added", extra_columns=list(holiday_df.columns))
        return df

    def _assign_data_split(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(self.date_col).reset_index(drop=True)
        n_rows = len(df)
        if n_rows < 2:
            df["data_split"] = "train"
            df["is_train"] = True
            return df

        split_idx = max(1, int(np.floor(n_rows * 0.987)))
        split_idx = min(split_idx, n_rows - 1)
        split_date = df.loc[split_idx - 1, self.date_col]
        latest_date = df[self.date_col].max()
        train_cutoff = latest_date - pd.DateOffset(years=6)

        df["data_split"] = "test"
        train_mask = (df[self.date_col] <= split_date) & (df[self.date_col] >= train_cutoff)
        df.loc[train_mask, "data_split"] = "train"
        df.loc[(df[self.date_col] <= split_date) & (~train_mask), "data_split"] = "discard_pre_6y"
        df["is_train"] = df["data_split"] == "train"

        self.logger.info(
            "data_split_assigned",
            split_date=split_date.strftime("%Y-%m-%d"),
            train_rows=int(train_mask.sum()),
            test_rows=int((df["data_split"] == "test").sum()),
        )
        return df

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
        df = df.sort_values(self.date_col).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Future feature helpers
    # ------------------------------------------------------------------
    def _default_future_csv_path(self) -> Path:
        current_name = self.feature_csv.stem
        if current_name.startswith("feature_"):
            replacement = current_name.replace("feature_", "future_", 1)
        else:
            replacement = f"{current_name}_future"
        return self.feature_csv.with_name(f"{replacement}{self.feature_csv.suffix}")

    def _generate_future_rows(self, df: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Cannot generate future features from empty dataframe")

        last_row = df.iloc[-1].copy()
        last_date = pd.to_datetime(last_row[self.date_col])
        elapsed_base = int(last_row.get("elapsed_weeks", len(df) - 1))
        min_year = int(df[self.date_col].dt.year.min())
        max_future_year = int((last_date + pd.DateOffset(weeks=horizon_weeks)).year)
        holidays = load_chinese_holidays(min_year, max_future_year, logger=self.logger)

        future_rows: list[pd.Series] = []
        future_dates: list[pd.Timestamp] = []
        for step in range(1, horizon_weeks + 1):
            new_row = last_row.copy()
            new_date = last_date + pd.DateOffset(weeks=step)
            future_dates.append(new_date)

            new_row[self.date_col] = new_date
            new_row[self.target_col] = 0.0
            new_row["data_split"] = "future"
            new_row["is_train"] = False
            new_row["elapsed_weeks"] = elapsed_base + step

            temporal = self._temporal_features_for_date(df[self.date_col].min(), new_date)
            for key, value in temporal.items():
                if key in new_row:
                    new_row[key] = value

            future_rows.append(new_row)

        holiday_features = calc_holiday_features(pd.Series(future_dates), holidays, logger=self.logger)
        for idx, col in enumerate(holiday_features.columns):
            values = holiday_features[col].tolist()
            for row_idx, value in enumerate(values):
                future_rows[row_idx][col] = value

        future_df = pd.DataFrame(future_rows)
        future_df[self.date_col] = pd.to_datetime(future_df[self.date_col])
        return future_df

    @staticmethod
    def _temporal_features_for_date(
        min_date: pd.Timestamp, new_date: pd.Timestamp
    ) -> Dict[str, int | float]:
        return {
            "year": new_date.year,
            "month": new_date.month,
            "quarter": new_date.quarter,
            "weekofyear": int(new_date.isocalendar().week),
            "weekofmonth": ((new_date.day - 1) // 7) + 1,
            "is_month_start": int(new_date.is_month_start),
            "is_month_end": int(new_date.is_month_end),
            "days_in_month": new_date.days_in_month,
            "elapsed_weeks": int(((new_date - min_date).days) // 7),
        }
