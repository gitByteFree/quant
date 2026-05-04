"""模型抽象基类."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """所有模型的抽象基类.

    sklearn风格接口: fit(X, y) / predict(X).
    """

    def __init__(self, seed: int = 42, **kwargs: Any):
        self._seed = seed
        self._is_fitted = False
        self._feature_names: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @abstractmethod
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | None = None,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
        **kwargs: Any,
    ) -> "BaseModel":
        """训练模型.

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标值 (n_samples,)
            sample_weight: 样本权重 (n_samples,)
            eval_set: 验证集列表 [(X_val, y_val), ...]
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """预测.

        Args:
            X: 特征矩阵 (n_samples, n_features)

        Returns:
            预测值 (n_samples,)
        """
        ...

    @abstractmethod
    def feature_importance(self) -> pd.DataFrame:
        """返回特征重要性.

        Returns:
            DataFrame: columns=[feature, importance], 按重要性降序排列
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """保存模型."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """加载模型."""
        ...

    def _validate_inputs(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """验证和标准化输入."""
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        y_arr = None
        if y is not None:
            if isinstance(y, pd.Series):
                y_arr = y.values.astype(np.float32)
            else:
                y_arr = np.asarray(y, dtype=np.float32)

        return X_arr, y_arr
