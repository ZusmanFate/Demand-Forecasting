"""Production-ready LightGBM runner enriched with notebook features."""
from __future__ import annotations

import gc
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import importlib
import joblib
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
import structlog
from dotenv import dotenv_values
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LOGGER = structlog.get_logger("lightgbm_runner")


def set_seed(seed: int) -> None:
    """Set random seeds for numpy/python/lightgbm reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    # lgb.register_logger(None)  # 可选，防止旧 logger 残留

def generate_time_weights(dates: pd.Series, half_life_weeks: float) -> np.ndarray:
    """Generate exponential decay weights based on recency."""

    max_date = pd.to_datetime(dates).max()
    weeks_ago = ((max_date - pd.to_datetime(dates)).dt.days / 7).clip(lower=0)
    decay = 0.5 ** (weeks_ago / max(half_life_weeks, 1))
    return (decay / decay.mean()).to_numpy()


def ts_augmentation(df: pd.DataFrame, target_col: str, ratio: float = 0.05) -> pd.DataFrame:
    """Simple time-series augmentation by injecting noisy copies near recent weeks."""

    if df.empty:
        return df
    aug_rows = int(len(df) * ratio)
    if aug_rows <= 0:
        return df
    recent = df.tail(aug_rows).copy()
    noise = np.random.normal(loc=0.0, scale=0.02, size=recent[target_col].shape)
    recent[target_col] = (recent[target_col] * (1 + noise)).clip(lower=0)
    recent["is_augmented"] = 1
    df = df.copy()
    df["is_augmented"] = 0
    return pd.concat([df, recent], ignore_index=True)


def add_time_robust_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add harmonic features to make the model more robust to periodicity."""

    df = df.copy()
    day_count = (df[date_col] - df[date_col].min()).dt.days
    df["sin_annual"] = np.sin(2 * np.pi * day_count / 365.25)
    df["cos_annual"] = np.cos(2 * np.pi * day_count / 365.25)
    df["sin_quarter"] = np.sin(2 * np.pi * day_count / 91.31)
    df["cos_quarter"] = np.cos(2 * np.pi * day_count / 91.31)
    return df


def plot_optimized_results(
    actual: pd.Series,
    predicted: pd.Series,
    save_path: Path,
) -> Path:
    """Plot actual vs predicted curve for sanity check."""

    plt.figure(figsize=(10, 4))
    plt.plot(actual.index, actual.values, label="actual", linewidth=2)
    plt.plot(predicted.index, predicted.values, label="prediction", linewidth=2)
    plt.legend()
    plt.title("LightGBM Optimized Results")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path


def save_and_print_dataset(df: pd.DataFrame, save_path: Path, logger: structlog.stdlib.BoundLogger) -> None:
    """Persist dataset snapshot for debugging/inspection."""

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.head(200).to_csv(save_path, index=False)
    logger.info(
        "dataset_snapshot",
        path=save_path.as_posix(),
        rows=len(df),
        columns=list(df.columns[:10]),
    )


class FixedLengthTimeSeriesSplit:
    """Custom time series split with fixed-length training windows."""

    def __init__(self, window_size: int, step_size: int):
        self.window_size = window_size
        self.step_size = step_size

    def split(self, df: pd.DataFrame) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(df)
        if n_samples <= self.window_size:
            yield np.arange(0, n_samples - 1), np.arange(n_samples - 1, n_samples)
            return

        start = 0
        while start + self.window_size < n_samples:
            train_end = start + self.window_size
            val_end = min(train_end + self.step_size, n_samples)
            train_idx = np.arange(start, train_end)
            val_idx = np.arange(train_end, val_end)
            if len(val_idx) == 0:
                break
            yield train_idx, val_idx
            start += self.step_size


DEFAULT_CONFIG: Dict[str, Any] = {
    "tuning_trials": 25,
    "num_boost_round": 3000,
    "early_stopping_rounds": 200,
    "cv_window_size": 104,
    "cv_step_size": 4,
    "weight_half_life_weeks": 52,
    "holdout_weeks": 12,
    "seeds": [42, 123, 404],
    "top_feature_fraction": 0.8,
    "shap_sample_rows": 512,
    "augmentation_ratio": 0.05,
}

LIGHTGBM_PARAMS: Dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.9,
    "bagging_freq": 3,
    "min_data_in_leaf": 50,
    "lambda_l1": 0.01,
    "lambda_l2": 0.01,
}


@dataclass(slots=True)
class LightGBMRunner:
    """Encapsulates LightGBM training, tuning, feature selection, and forecasting."""

    target_col: str = "total_weekly_sales"
    date_col: str = "week_end_date"
    is_train_col: str = "is_train"
    drop_columns: Sequence[str] = ("data_split",)
    config: Dict[str, Any] = field(default_factory=lambda: DEFAULT_CONFIG.copy())
    model_dir: Path = Path("saved_models")
    importance_dir: Path = Path("feature_importance")
    logger: structlog.stdlib.BoundLogger = field(
        default_factory=lambda: structlog.get_logger("lightgbm_runner")
    )
    _last_metrics: Optional[Dict[str, float]] = field(default=None, init=False, repr=False)
    _last_future_forecast_path: Optional[Path] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._verify_dependencies()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.importance_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train_and_predict(
        self,
        feature_csv_path: Path,
        forecast_csv_path: Path,
        *,
        future_feature_csv_path: Path | None = None,
        future_forecast_csv_path: Path | None = None,
    ) -> Path:
        start_time = time.time()
        self.logger.info(
            "✅ training_started",
            feature_csv=feature_csv_path.as_posix(),
            forecast_csv=forecast_csv_path.as_posix(),
        )
        df = self._load_feature_csv(feature_csv_path)
        df = add_time_robust_features(df, self.date_col)
        df = ts_augmentation(df, self.target_col, ratio=self.config["augmentation_ratio"])

        train_df, test_df = self._split_train_test(df)
        feature_cols = self._select_feature_columns(df)

        weights = generate_time_weights(train_df[self.date_col], self.config["weight_half_life_weeks"])

        try:
            best_params = self.tune_lgb_params(train_df, feature_cols, weights)
            self.logger.info("✅ tuning_completed", params=best_params)
        except Exception as exc:  # pragma: no cover - degrade gracefully
            self.logger.warning("⚠️ tuning_failed_using_default", error=str(exc))
            best_params = LIGHTGBM_PARAMS.copy()

        models, val_scores = self.multi_seed_training(train_df, feature_cols, best_params, weights)

        preds = self._average_predictions(models, test_df[feature_cols])
        forecast_df = self._build_forecast(test_df, preds)

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        forecast_csv_path.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.sort_values(self.date_col).to_csv(
            forecast_csv_path, index=False, encoding="utf-8-sig"
        )
        sanitized_path = self.export_forecast_to_starrocks_format(forecast_csv_path)
        self.logger.info("✅ forecast_saved", path=sanitized_path.as_posix(), rows=len(forecast_df))

        importance_paths, selected_features = self.calculate_feature_importance(
            models, feature_cols, train_df
        )
        retrain_rmse = self._retrain_with_selected_features(
            train_df,
            test_df,
            selected_features,
            best_params,
            weights,
        )

        model_artifact = self._save_model_bundle(models[0], feature_cols, best_params, val_scores)

        plot_path = plot_optimized_results(
            forecast_df.set_index(self.date_col)[self.target_col],
            forecast_df.set_index(self.date_col)["pred_sales"],
            self.importance_dir / f"optimized_curve_{timestamp}.png",
        )

        save_and_print_dataset(
            forecast_df,
            self.importance_dir / f"forecast_sample_{timestamp}.csv",
            self.logger,
        )

        future_forecast_path: Path | None = None
        if future_feature_csv_path:
            future_forecast_path = self._run_future_forecast(
                models,
                feature_cols,
                future_feature_csv_path,
                future_forecast_csv_path,
            )
        self._last_future_forecast_path = future_forecast_path

        duration = time.time() - start_time
        log_payload = {
            "forecast_path": forecast_csv_path.as_posix(),
            "model_artifact": model_artifact.as_posix(),
            "importance_paths": [p.as_posix() for p in importance_paths],
            "retrain_rmse": retrain_rmse,
            "cv_rmse": np.mean(val_scores),
            "val_rmse_std": np.std(val_scores),
            "feature_count": len(feature_cols),
            "duration_seconds": duration,
            "plot_path": plot_path.as_posix(),
        }
        if future_forecast_path is not None:
            log_payload["future_forecast_path"] = future_forecast_path.as_posix()

        self.logger.info("✅ train_and_predict_completed", **log_payload)
        gc.collect()
        return forecast_csv_path

    def load_saved_model(self, model_path: Path | None = None) -> Dict[str, Any]:
        target_path = model_path or (self.model_dir / "lightgbm_model.pkl")
        if not target_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {target_path}")
        artifact = joblib.load(target_path)
        self.logger.info("model_loaded", path=target_path.as_posix())
        return artifact

    # ------------------------------------------------------------------
    # Core pipeline helpers
    # ------------------------------------------------------------------
    def _load_feature_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Feature CSV not found: {path}")
        df = pd.read_csv(path)
        required_cols = {self.target_col, self.date_col}
        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise KeyError(f"Missing columns in feature CSV: {missing}")
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        if self.is_train_col not in df.columns:
            split_idx = max(1, int(math.floor(len(df) * 0.987)))
            df[self.is_train_col] = False
            df.loc[: split_idx - 1, self.is_train_col] = True
        df[self.is_train_col] = df[self.is_train_col].astype(bool)
        self.logger.info("features_loaded", rows=len(df), path=path.as_posix())
        return df.sort_values(self.date_col).reset_index(drop=True)

    def _split_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if "data_split" not in df.columns:
            raise KeyError("Feature CSV missing 'data_split' column for split assignment")

        train_mask = (df["data_split"] == "train") & (df[self.is_train_col])
        test_mask = (df["data_split"] == "test") & (~df[self.is_train_col])

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if train_df.empty:
            raise ValueError("Training dataframe is empty after applying train split filters")
        if test_df.empty:
            raise ValueError(
                "Test dataframe is empty after applying test split filters (expected future 4 weeks)"
            )

        self.logger.info(
            "dataset_split_summary",
            train_rows=len(train_df),
            test_rows=len(test_df),
            discard_rows=int((df["data_split"] == "discard_pre_6y").sum()),
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        drop_set = set(self.drop_columns) | {self.target_col, self.is_train_col}
        numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns
        feature_cols = [col for col in numeric_cols if col not in drop_set]
        if not feature_cols:
            raise ValueError("No numeric feature columns available after filtering.")
        self.logger.info("feature_columns_selected", count=len(feature_cols))
        return feature_cols

    # ------------------------------------------------------------------
    # Hyper-parameter tuning & training
    # ------------------------------------------------------------------
    def tune_lgb_params(
        self,
        train_df: pd.DataFrame,
        feature_cols: Sequence[str],
        sample_weight: np.ndarray,
    ) -> Dict[str, Any]:
        splitter = FixedLengthTimeSeriesSplit(
            window_size=self.config["cv_window_size"],
            step_size=self.config["cv_step_size"],
        )

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial)
            rmses: List[float] = []
            for train_idx, val_idx in splitter.split(train_df):
                X_train = train_df.iloc[train_idx][feature_cols]
                y_train = train_df.iloc[train_idx][self.target_col]
                w_train = sample_weight[train_idx]
                X_val = train_df.iloc[val_idx][feature_cols]
                y_val = train_df.iloc[val_idx][self.target_col]

                lgb_train = lgb.Dataset(X_train, label=y_train, weight=w_train)
                lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
                booster = lgb.train(
                    params,
                    lgb_train,
                    valid_sets=[lgb_val],
                    # verbose_eval=False,
                    num_boost_round=self.config["num_boost_round"],
                    callbacks=[
                        early_stopping(stopping_rounds=self.config["early_stopping_rounds"]),
                        log_evaluation(period=100),  # 每 100 轮打印一次日志，可改成 0 禁止输出
                    ],
                )
                preds = booster.predict(X_val)
                rmse = mean_squared_error(y_val, preds, squared=False)
                rmses.append(rmse)
            return float(np.mean(rmses))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.config["tuning_trials"], show_progress_bar=False)
        best_params = study.best_params
        best_params.update({"objective": "regression", "metric": "rmse", "verbosity": -1})
        self.logger.info(
            "tuning_completed",
            best_value=study.best_value,
            trials=study.best_trial.number,
        )
        return best_params

    def multi_seed_training(
        self,
        train_df: pd.DataFrame,
        feature_cols: Sequence[str],
        base_params: Dict[str, Any],
        sample_weight: np.ndarray,
    ) -> Tuple[List[lgb.Booster], List[float]]:
        models: List[lgb.Booster] = []
        val_scores: List[float] = []
        holdout_weeks = max(4, self.config["holdout_weeks"])
        split_point = max(1, len(train_df) - holdout_weeks)
        train_idx = np.arange(0, split_point)
        val_idx = np.arange(split_point, len(train_df))
        X_val = train_df.iloc[val_idx][feature_cols]
        y_val = train_df.iloc[val_idx][self.target_col]

        for seed in self.config["seeds"]:
            params = base_params.copy()
            params.update({"seed": seed, "feature_pre_filter": False})
            set_seed(seed)

            X_train = train_df.iloc[train_idx][feature_cols]
            y_train = train_df.iloc[train_idx][self.target_col]
            w_train = sample_weight[train_idx]
            lgb_train = lgb.Dataset(X_train, label=y_train, weight=w_train)
            lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

            booster = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_valid],
                num_boost_round=self.config["num_boost_round"],
                callbacks=[
                    early_stopping(stopping_rounds=self.config["early_stopping_rounds"]),
                    log_evaluation(period=100),  # 每 100 轮打印一次日志，可改成 0 禁止输出
                ],
                # verbose_eval=False,
            )
            preds = booster.predict(X_val)
            rmse = mean_squared_error(y_val, preds, squared=False)
            models.append(booster)
            val_scores.append(float(rmse))
            self.logger.info(
                "model_trained",
                seed=seed,
                rmse=rmse,
                best_iteration=booster.best_iteration,
            )

        self.logger.info(
            "seed_stability",
            mean_rmse=float(np.mean(val_scores)),
            std_rmse=float(np.std(val_scores)),
        )
        return models, val_scores

    # ------------------------------------------------------------------
    # Reporting & artifact helpers
    # ------------------------------------------------------------------
    def calculate_feature_importance(
        self,
        models: Sequence[lgb.Booster],
        feature_cols: Sequence[str],
        train_df: pd.DataFrame,
    ) -> Tuple[List[Path], List[str]]:
        importance_df = pd.DataFrame({"feature": feature_cols})
        for imp_type in ("gain", "split"):
            values = np.mean(
                [model.feature_importance(importance_type=imp_type) for model in models], axis=0
            )
            importance_df[f"importance_{imp_type}"] = values

        shap_path = None
        try:
            sample_rows = min(self.config["shap_sample_rows"], len(train_df))
            explainer = shap.TreeExplainer(models[0])
            shap_sample = train_df.tail(sample_rows)[feature_cols]
            shap_values = explainer.shap_values(shap_sample)
            importance_df["importance_shap"] = np.abs(shap_values).mean(axis=0)
        except Exception as exc:  # pragma: no cover - best effort only
            self.logger.warning("shap_failed", error=str(exc))

        importance_df = importance_df.sort_values("importance_gain", ascending=False)
        cumulative_gain = importance_df["importance_gain"].cumsum() / importance_df[
            "importance_gain"
        ].sum()
        cutoff = cumulative_gain.le(self.config["top_feature_fraction"]).sum()
        selected_features = importance_df.head(cutoff or len(importance_df))["feature"].tolist()
        self.logger.info("feature_selection", selected_features=selected_features[:10])

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        csv_path = self.importance_dir / f"feature_importance_{timestamp}.csv"
        importance_df.to_csv(csv_path, index=False)

        png_path = self.importance_dir / f"feature_importance_top20_{timestamp}.png"
        plt.figure(figsize=(8, 10))
        top20 = importance_df.head(20)
        plt.barh(top20["feature"][::-1], top20["importance_gain"][::-1])
        plt.xlabel("Gain Importance")
        plt.title("Top 20 Feature Importance (Gain)")
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()

        paths = [csv_path, png_path]
        if shap_path:
            paths.append(shap_path)
        self.logger.info("feature_importance_saved", paths=[p.as_posix() for p in paths])
        return paths, selected_features

    def _build_forecast(self, test_df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
        result = pd.DataFrame(
            {
                self.date_col: test_df[self.date_col].values,
                self.target_col: test_df[self.target_col].values,
                "pred_sales": preds,
            }
        )
        result["error"] = result["pred_sales"] - result[self.target_col]
        result["abs_error"] = result["error"].abs()
        result["rmse"] = np.sqrt(result["error"] ** 2)
        metrics = self._log_metrics(result)
        self._last_metrics = metrics
        return result

    def _average_predictions(self, models: Sequence[lgb.Booster], X_test: pd.DataFrame) -> np.ndarray:
        preds = np.array([model.predict(X_test) for model in models])
        return preds.mean(axis=0)

    def _load_future_features(
        self, future_csv_path: Path, feature_cols: Sequence[str]
    ) -> pd.DataFrame:
        if not future_csv_path.exists():
            raise FileNotFoundError(f"Future feature CSV not found: {future_csv_path}")
        future_df = pd.read_csv(future_csv_path)
        if self.date_col not in future_df.columns:
            raise KeyError(f"Missing '{self.date_col}' column in future feature CSV")
        future_df[self.date_col] = pd.to_datetime(future_df[self.date_col])

        # Align derived columns with the training dataframe before validation
        future_df = add_time_robust_features(future_df, self.date_col)
        if "is_augmented" not in future_df.columns:
            future_df["is_augmented"] = 0

        missing = sorted(set(feature_cols) - set(future_df.columns))
        if missing:
            raise KeyError(f"Future feature CSV missing columns: {missing}")
        future_df = future_df.sort_values(self.date_col).reset_index(drop=True)
        self.logger.info(
            "future_features_loaded",
            path=future_csv_path.as_posix(),
            rows=len(future_df),
        )
        return future_df

    def _run_future_forecast(
        self,
        models: Sequence[lgb.Booster],
        feature_cols: Sequence[str],
        future_feature_csv_path: Path,
        future_forecast_csv_path: Path | None,
    ) -> Path:
        future_df = self._load_future_features(future_feature_csv_path, feature_cols)
        preds = self._average_predictions(models, future_df[feature_cols])
        result = pd.DataFrame(
            {
                self.date_col: future_df[self.date_col].values,
                "total_weekly_sales": preds,
            }
        )
        result = result.sort_values(self.date_col).reset_index(drop=True)

        if future_forecast_csv_path is None:
            future_forecast_csv_path = future_feature_csv_path.with_name(
                f"forecast_{future_feature_csv_path.stem}.csv"
            )
        future_forecast_csv_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(future_forecast_csv_path, index=False, encoding="utf-8-sig")
        self.logger.info(
            "future_forecast_generated",
            path=future_forecast_csv_path.as_posix(),
            rows=len(result),
        )
        return future_forecast_csv_path

    def get_future_forecast_path(self) -> Optional[Path]:
        return self._last_future_forecast_path

    def _log_metrics(self, forecast_df: pd.DataFrame) -> Dict[str, float]:
        rmse = mean_squared_error(forecast_df[self.target_col], forecast_df["pred_sales"], squared=False)
        mae = mean_absolute_error(forecast_df[self.target_col], forecast_df["pred_sales"])
        mape = (
            (forecast_df["abs_error"] / forecast_df[self.target_col].replace(0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .mean()
        ) * 100
        r2 = r2_score(forecast_df[self.target_col], forecast_df["pred_sales"])
        metrics = {"rmse": float(rmse), "mae": float(mae), "mape": float(mape), "r2": float(r2)}
        self.logger.info("✅ forecast_metrics", **metrics)
        return metrics

    def _save_model_bundle(
        self,
        model: lgb.Booster,
        feature_cols: Sequence[str],
        params: Dict[str, Any],
        val_scores: Sequence[float],
    ) -> Path:
        artifact = {
            "model": model,
            "feature_columns": list(feature_cols),
            "params": params,
            "val_scores": list(val_scores),
        }
        path = self.model_dir / "lightgbm_model.pkl"
        joblib.dump(artifact, path)
        self.logger.info("model_saved", path=path.as_posix())
        return path

    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 32, 128),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }

    def _retrain_with_selected_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        selected_features: Sequence[str],
        params: Dict[str, Any],
        weights: np.ndarray,
    ) -> float:
        if not selected_features:
            return float("nan")

        best_seed = self.config["seeds"][0]
        best_rmse = float("inf")
        holdout_weeks = max(4, self.config["holdout_weeks"])
        split_point = max(1, len(train_df) - holdout_weeks)
        train_idx = np.arange(0, split_point)
        val_idx = np.arange(split_point, len(train_df))

        for seed in self.config["seeds"]:
            set_seed(seed)
            lgb_train = lgb.Dataset(
                train_df.iloc[train_idx][selected_features],
                label=train_df.iloc[train_idx][self.target_col],
                weight=weights[train_idx],
            )
            lgb_valid = lgb.Dataset(
                train_df.iloc[val_idx][selected_features],
                label=train_df.iloc[val_idx][self.target_col],
                reference=lgb_train,
            )
            booster = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_valid],
                num_boost_round=self.config["num_boost_round"],
                callbacks=[
                    early_stopping(stopping_rounds=self.config["early_stopping_rounds"]),
                    log_evaluation(period=100),  # 每 100 轮打印一次日志，可改成 0 禁止输出
                ],
                # verbose_eval=False,
            )
            preds = booster.predict(train_df.iloc[val_idx][selected_features])
            rmse = mean_squared_error(
                train_df.iloc[val_idx][self.target_col], preds, squared=False
            )
            if rmse < best_rmse:
                best_rmse = rmse
                best_seed = seed

        self.logger.info("retrain_validation", best_seed=best_seed, rmse=best_rmse)

        final_params = params.copy()
        final_params.pop("early_stopping_rounds", None)
        final_params["seed"] = best_seed
        set_seed(best_seed)
        final_dataset = lgb.Dataset(
            train_df[selected_features], label=train_df[self.target_col], weight=weights
        )
        booster = lgb.train(
            final_params,
            final_dataset,
            num_boost_round=self.config["num_boost_round"],
            # verbose_eval=False,
        )
        preds = booster.predict(test_df[selected_features])
        rmse = mean_squared_error(test_df[self.target_col], preds, squared=False)
        self.logger.info("retrain_vs_original", selected_rmse=rmse)
        return rmse

    def _verify_dependencies(self) -> None:
        missing = []
        for package_name in ("lightgbm", "optuna", "shap"):
            try:
                importlib.import_module(package_name)
            except ImportError:
                missing.append(package_name)
        if missing:
            raise ImportError(
                f"Missing required packages: {missing}. Please install via requirements.txt."
            )

    # ------------------------------------------------------------------
    # Integration helpers
    # ------------------------------------------------------------------
    def get_model_metrics(self) -> Dict[str, float]:
        """Expose the latest training metrics for orchestration logging."""

        return self._last_metrics.copy() if self._last_metrics else {}

    def export_forecast_to_starrocks_format(
        self, forecast_csv: Path, output_csv: Optional[Path] = None
    ) -> Path:
        """Normalize forecast CSV for StarRocks ingestion (dates, column names)."""

        if not forecast_csv.exists():
            raise FileNotFoundError(f"Forecast CSV not found: {forecast_csv}")
        df = pd.read_csv(forecast_csv)
        required_cols = {
            self.date_col,
            self.target_col,
            "pred_sales",
            "error",
            "abs_error",
            "rmse",
        }
        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise KeyError(f"Forecast CSV missing columns: {missing}")
        df[self.date_col] = pd.to_datetime(df[self.date_col]).dt.strftime("%Y-%m-%d")
        sanitized = df.rename(
            columns={
                self.target_col: "total_weekly_sales",
                "pred_sales": "pred_sales",
                "error": "error",
                "abs_error": "abs_error",
                "rmse": "rmse",
            }
        )
        out_path = output_csv or forecast_csv
        sanitized.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path

    def load_config_from_env(self, env_path: Path) -> None:
        """Placeholder to merge .env overrides into runner config."""

        if not env_path.exists():
            self.logger.warning("⚠️ config_env_not_found", path=env_path.as_posix())
            return
        env_data = dotenv_values(env_path)
        updates: Dict[str, Any] = {}
        if "LGBM_TUNING_TRIALS" in env_data:
            updates["tuning_trials"] = int(env_data["LGBM_TUNING_TRIALS"])  # type: ignore[arg-type]
        if "LGBM_SEEDS" in env_data:
            updates["seeds"] = [int(x.strip()) for x in env_data["LGBM_SEEDS"].split(",")]
        if "LGBM_TOP_FEATURE_FRACTION" in env_data:
            updates["top_feature_fraction"] = float(env_data["LGBM_TOP_FEATURE_FRACTION"])
        if updates:
            self.config.update(updates)
            self.logger.info("✅ config_env_loaded", keys=list(updates.keys()))
        else:
            self.logger.info("⚠️ config_env_no_overrides", path=env_path.as_posix())


# ----------------------------------------------------------------------
# Usage Example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    runner = LightGBMRunner()
    runner.train_and_predict(
        feature_csv_path=Path("data/feature/feature_20251230.csv"),
        forecast_csv_path=Path("data/forecast/forecast_20251230.csv"),
    )
