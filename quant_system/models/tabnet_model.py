"""TabNet模型封装.

使用pytorch-tabnet，提供注意力掩码可解释性.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from quant_system.models.base import BaseModel
from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


class TabNetModel(BaseModel):
    """TabNet回归模型.

    Args:
        n_d: 决策步骤的特征维度
        n_a: 注意力步骤的特征维度
        n_steps: 决策步骤数
        gamma: 特征复用系数（>1时后续步骤可重用前面的特征）
        n_independent: 独立门控线性单元层数
        n_shared: 共享门控线性单元层数
        momentum: BatchNorm动量
        mask_type: 注意力掩码类型 sparsemax | entmax
        learning_rate: 学习率
        scheduler_patience: 学习率调度器耐心值
        seed: 随机种子
        device: 训练设备
    """

    def __init__(
        self,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_independent: int = 2,
        n_shared: int = 2,
        momentum: float = 0.3,
        mask_type: str = "entmax",
        learning_rate: float = 0.001,
        scheduler_patience: int = 10,
        seed: int = 42,
        device: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(seed=seed)
        self._params = {
            "n_d": n_d,
            "n_a": n_a,
            "n_steps": n_steps,
            "gamma": gamma,
            "n_independent": n_independent,
            "n_shared": n_shared,
            "momentum": momentum,
            "mask_type": mask_type,
            "optimizer_fn": torch.optim.Adam,
            "optimizer_params": dict(lr=learning_rate),
            "scheduler_fn": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "scheduler_params": dict(mode="min", patience=scheduler_patience, factor=0.5),
            "seed": seed,
            "verbose": 0,
        }
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Any = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | None = None,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
        **kwargs: Any,
    ) -> "TabNetModel":
        from pytorch_tabnet.tab_model import TabNetRegressor

        X_arr, y_arr = self._validate_inputs(X, y)
        if y_arr is not None:
            y_arr = y_arr.reshape(-1, 1)

        eval_set_pt = None
        if eval_set is not None and len(eval_set) > 0:
            eval_set_pt = [(eval_set[0][0], eval_set[0][1].reshape(-1, 1))]

        self._model = TabNetRegressor(**self._params)
        self._model.fit(
            X_arr, y_arr,
            eval_set=eval_set_pt,
            weights=sample_weight,
            max_epochs=kwargs.get("max_epochs", 200),
            patience=kwargs.get("patience", 20),
            batch_size=kwargs.get("batch_size", 256),
            virtual_batch_size=kwargs.get("virtual_batch_size", 128),
        )
        self._is_fitted = True
        logger.info("TabNet训练完成")
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        X_arr, _ = self._validate_inputs(X)
        return self._model.predict(X_arr).flatten()

    def feature_importance(self) -> pd.DataFrame:
        if self._model is None:
            return pd.DataFrame(columns=["feature", "importance"])
        # TabNet的全局特征重要性
        importances = self._model.feature_importances_
        if not self._feature_names:
            self._feature_names = [f"f_{i}" for i in range(len(importances))]
        df = pd.DataFrame({
            "feature": self._feature_names[:len(importances)],
            "importance": importances,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def explain(self, X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """获取TabNet的注意力掩码（可解释性）.

        Args:
            X: 特征矩阵

        Returns:
            (masks, predictions): masks shape (n_steps, n_samples, n_features)
        """
        if self._model is None:
            raise RuntimeError("模型未训练")
        X_arr, _ = self._validate_inputs(X)
        masks, predictions = self._model.explain(X_arr)
        return masks, predictions

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("模型未训练")
        self._model.save_model(str(path))
        logger.info("TabNet模型已保存: %s", path)

    def load(self, path: str | Path) -> None:
        from pytorch_tabnet.tab_model import TabNetRegressor
        self._model = TabNetRegressor()
        self._model.load_model(str(path))
        self._is_fitted = True
        logger.info("TabNet模型已加载: %s", path)
