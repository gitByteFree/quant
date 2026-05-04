"""模型训练器.

滚动窗口/扩展窗口训练、样本权重、超参搜索.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_system.models.base import BaseModel
from quant_system.utils.calendar import TradingCalendar
from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainWindow:
    """训练/验证窗口."""

    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    split_idx_train: np.ndarray | None = None
    split_idx_valid: np.ndarray | None = None


class ModelTrainer:
    """模型训练调度器.

    支持滚动窗口和扩展窗口两种训练模式.

    Args:
        model: 模型实例
        window_type: rolling | expanding
        train_window_years: 训练窗口（年）
        valid_window_years: 验证窗口（年）
        step_months: 滚动步长（月）
        time_decay_alpha: 时间衰减因子（<1时越近的样本权重越高）
        use_volatility_weight: 是否使用波动率倒数加权
        seed: 随机种子
    """

    def __init__(
        self,
        model: BaseModel,
        window_type: str = "rolling",
        train_window_years: int = 5,
        valid_window_years: int = 1,
        step_months: int = 6,
        time_decay_alpha: float = 0.95,
        use_volatility_weight: bool = True,
        seed: int = 42,
    ):
        self._model = model
        self._window_type = window_type
        self._train_years = train_window_years
        self._valid_years = valid_window_years
        self._step_months = step_months
        self._time_decay_alpha = time_decay_alpha
        self._use_vol_weight = use_volatility_weight
        self._seed = seed
        self._calendar = TradingCalendar()

        self._trained_models: list[BaseModel] = []
        self._train_windows: list[TrainWindow] = []
        self._evaluations: list[dict[str, float]] = []

    @property
    def trained_models(self) -> list[BaseModel]:
        return self._trained_models

    @property
    def evaluations(self) -> list[dict[str, float]]:
        return self._evaluations

    def _generate_windows(
        self,
        data_start: pd.Timestamp,
        data_end: pd.Timestamp,
    ) -> list[TrainWindow]:
        """生成训练/验证时间窗口."""
        windows: list[TrainWindow] = []
        train_delta = pd.DateOffset(years=self._train_years)
        valid_delta = pd.DateOffset(years=self._valid_years)
        step_delta = pd.DateOffset(months=self._step_months)

        if self._window_type == "expanding":
            train_start = data_start
            train_end = train_start + train_delta
            while train_end + valid_delta <= data_end:
                valid_start = train_end
                valid_end = min(valid_start + valid_delta, data_end)
                windows.append(TrainWindow(
                    train_start=str(train_start.date()),
                    train_end=str(train_end.date()),
                    valid_start=str(valid_start.date()),
                    valid_end=str(valid_end.date()),
                ))
                train_end = min(train_end + step_delta, data_end)
        else:
            train_start = data_start
            train_end = train_start + train_delta
            while train_end + valid_delta <= data_end:
                valid_start = train_end
                valid_end = min(valid_start + valid_delta, data_end)
                windows.append(TrainWindow(
                    train_start=str(train_start.date()),
                    train_end=str(train_end.date()),
                    valid_start=str(valid_start.date()),
                    valid_end=str(valid_end.date()),
                ))
                train_start = train_start + step_delta
                train_end = min(train_start + train_delta, data_end)

        logger.info("生成 %d 个训练窗口 (%s)", len(windows), self._window_type)
        return windows

    def _compute_sample_weights(
        self,
        dates: pd.DatetimeIndex,
        returns: pd.Series | None = None,
    ) -> np.ndarray:
        """计算样本权重.

        权重 = 时间衰减 * 波动率倒数（可选）.

        Args:
            dates: 每个样本对应的日期
            returns: 收益率数据（用于计算波动率倒数权重）

        Returns:
            样本权重数组
        """
        n = len(dates)
        weights = np.ones(n)

        # 时间衰减
        latest = dates.max()
        days_from_end = (latest - dates).days.values
        total_days = max(1, days_from_end.max())
        weights *= self._time_decay_alpha ** (days_from_end / total_days)

        # 波动率倒数加权
        if self._use_vol_weight and returns is not None:
            if len(returns) == n:
                vol = returns.groupby(returns.index).transform(lambda x: x.std() + 1e-6)
                weights *= 1.0 / (vol.values + 1e-6)

        # 归一化
        weights = weights / weights.sum() * n
        return weights

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        date_col: str | None = None,
        **kwargs: Any,
    ) -> "ModelTrainer":
        """按时间窗口训练模型.

        Args:
            X: 特征DataFrame，索引需包含date信息（或通过date_col指定）
            y: 目标值Series
            date_col: 日期列名（如果date在index中则为None）
        """
        # 确定日期索引
        if date_col and date_col in X.columns:
            dates = pd.DatetimeIndex(X[date_col])
        elif isinstance(X.index, pd.MultiIndex) and "trade_date" in X.index.names:
            dates = pd.DatetimeIndex(X.index.get_level_values("trade_date"))
        else:
            raise ValueError("无法确定日期列，请使用MultiIndex (trade_date, symbol) 或指定 date_col")

        data_start = dates.min()
        data_end = dates.max()
        windows = self._generate_windows(data_start, data_end)

        for i, window in enumerate(windows):
            logger.info(
                "训练窗口 %d/%d: train=%s~%s, valid=%s~%s",
                i + 1, len(windows),
                window.train_start, window.train_end,
                window.valid_start, window.valid_end,
            )

            # 划分训练/验证集
            if isinstance(X.index, pd.MultiIndex):
                date_level = X.index.get_level_values("trade_date")
                train_mask = (date_level >= window.train_start) & (date_level < window.train_end)
                valid_mask = (date_level >= window.valid_start) & (date_level < window.valid_end)
                X_train = X.loc[train_mask]
                y_train = y.loc[train_mask]
                X_valid = X.loc[valid_mask]
                y_valid = y.loc[valid_mask]
                train_dates = pd.DatetimeIndex(date_level[train_mask])
            else:
                d = pd.DatetimeIndex(dates)
                train_mask = (d >= window.train_start) & (d < window.train_end)
                valid_mask = (d >= window.valid_start) & (d < window.valid_end)
                X_train = X.loc[train_mask]
                y_train = y.loc[train_mask]
                X_valid = X.loc[valid_mask]
                y_valid = y.loc[valid_mask]
                train_dates = pd.DatetimeIndex(d[train_mask])

            if len(X_train) < 100 or len(X_valid) < 20:
                logger.warning("跳过窗口（样本太少）")
                continue

            # 计算样本权重
            sample_weights = self._compute_sample_weights(train_dates, y_train)

            # 训练模型
            model_clone = self._clone_model()
            model_clone.fit(
                X_train.values.astype(np.float32),
                y_train.values.astype(np.float32),
                sample_weight=sample_weights,
                eval_set=[(X_valid.values.astype(np.float32), y_valid.values.astype(np.float32))],
                **kwargs,
            )

            self._trained_models.append(model_clone)
            self._train_windows.append(window)

            # 评估
            pred = model_clone.predict(X_valid.values.astype(np.float32))
            valid_ic = np.corrcoef(pred, y_valid.values)[0, 1] if len(pred) > 1 else 0
            valid_mse = np.mean((pred - y_valid.values) ** 2)
            self._evaluations.append({
                "window": i,
                "valid_ic": valid_ic,
                "valid_mse": valid_mse,
                "n_train": len(X_train),
                "n_valid": len(X_valid),
            })
            logger.info("  窗口评估: IC=%.4f, MSE=%.6f", valid_ic, valid_mse)

        logger.info("训练完成: %d 个窗口", len(self._trained_models))
        return self

    def predict(self, X: pd.DataFrame, method: str = "latest") -> np.ndarray:
        """使用训练好的模型进行预测.

        Args:
            X: 特征DataFrame
            method: 预测方法
                - latest: 使用最新训练的模型
                - average: 使用所有模型的平均预测
                - weighted: 按验证IC加权平均

        Returns:
            预测值数组
        """
        if not self._trained_models:
            raise RuntimeError("没有已训练的模型")

        X_arr = X.values.astype(np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)

        if method == "latest":
            return self._trained_models[-1].predict(X_arr)

        all_preds = np.column_stack([m.predict(X_arr) for m in self._trained_models])

        if method == "average":
            return all_preds.mean(axis=1)

        if method == "weighted":
            weights = np.array([
                max(0, e["valid_ic"]) for e in self._evaluations
            ])
            if weights.sum() == 0:
                weights = np.ones(len(weights))
            weights = weights / weights.sum()
            return all_preds @ weights

        raise ValueError(f"未知预测方法: {method}")

    def _clone_model(self) -> BaseModel:
        """创建模型副本."""
        import copy
        return copy.deepcopy(self._model)

    def get_feature_importance(self) -> pd.DataFrame:
        """获取最新模型的特征重要性."""
        if not self._trained_models:
            return pd.DataFrame(columns=["feature", "importance"])
        return self._trained_models[-1].feature_importance()

    def summary(self) -> pd.DataFrame:
        """训练评估摘要."""
        if not self._evaluations:
            return pd.DataFrame()
        return pd.DataFrame(self._evaluations)
