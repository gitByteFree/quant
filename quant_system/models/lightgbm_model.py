"""LightGBM模型封装.

支持早停、自定义目标、排序学习.
"""

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from quant_system.models.base import BaseModel
from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


class LightGBMModel(BaseModel):
    """LightGBM回归/排序模型.

    Args:
        n_estimators: 树的数量
        learning_rate: 学习率
        max_depth: 最大深度
        num_leaves: 叶子数
        subsample: 行采样比例
        colsample_bytree: 列采样比例
        min_child_samples: 叶子节点最小样本数
        reg_alpha: L1正则
        reg_lambda: L2正则
        early_stopping_rounds: 早停轮数
        objective: 目标函数 regression | lambdarank
        boosting_type: gbdt | dart | goss
        seed: 随机种子
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 8,
        num_leaves: int = 128,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 100,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        objective: str = "regression",
        boosting_type: str = "gbdt",
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(seed=seed)
        self._params = {
            "objective": objective,
            "metric": "rmse" if objective == "regression" else "ndcg",
            "boosting_type": boosting_type,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_samples": min_child_samples,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": seed,
            "verbosity": -1,
            "n_jobs": -1,
        }
        self._early_stopping_rounds = early_stopping_rounds
        self._model: lgb.Booster | lgb.LGBMModel | None = None
        self._best_iteration: int = 0

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | None = None,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
        **kwargs: Any,
    ) -> "LightGBMModel":
        X_arr, y_arr = self._validate_inputs(X, y)

        callbacks: list = []
        if self._params["objective"] == "regression":
            callbacks.append(
                lgb.early_stopping(self._early_stopping_rounds, verbose=False)
            )
            callbacks.append(lgb.log_evaluation(period=0))

        n_estimators = self._params.pop("n_estimators")
        try:
            self._model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=self._params["learning_rate"],
                max_depth=self._params["max_depth"],
                num_leaves=self._params["num_leaves"],
                subsample=self._params["subsample"],
                colsample_bytree=self._params["colsample_bytree"],
                min_child_samples=self._params["min_child_samples"],
                reg_alpha=self._params["reg_alpha"],
                reg_lambda=self._params["reg_lambda"],
                random_state=self._params["random_state"],
                n_jobs=-1,
                verbosity=-1,
            )

            eval_set_lgb = None
            eval_sample_weight = None
            if eval_set is not None and len(eval_set) > 0:
                eval_set_lgb = [(eval_set[0][0], eval_set[0][1])]
                if sample_weight is not None:
                    n_train = len(X_arr)
                    eval_sample_weight = (
                        [np.ones(len(eval_set[0][1]))]
                        if len(eval_set) > 0 else None
                    )

            self._model.fit(
                X_arr, y_arr,
                sample_weight=sample_weight,
                eval_set=eval_set_lgb,
                eval_sample_weight=eval_sample_weight,
            )
            self._best_iteration = self._model.best_iteration_ or n_estimators
        finally:
            self._params["n_estimators"] = n_estimators

        self._is_fitted = True
        logger.info(
            "LightGBM训练完成: best_iteration=%d",
            self._best_iteration,
        )
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        X_arr, _ = self._validate_inputs(X)
        return self._model.predict(X_arr)

    def feature_importance(self) -> pd.DataFrame:
        if self._model is None:
            return pd.DataFrame(columns=["feature", "importance"])
        importances = self._model.feature_importances_
        if not self._feature_names:
            self._feature_names = [f"f_{i}" for i in range(len(importances))]
        df = pd.DataFrame({
            "feature": self._feature_names[:len(importances)],
            "importance": importances,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("模型未训练")
        import joblib
        joblib.dump(self._model, str(path))
        logger.info("LightGBM模型已保存: %s", path)

    def load(self, path: str | Path) -> None:
        import joblib
        self._model = joblib.load(str(path))
        self._is_fitted = True
        logger.info("LightGBM模型已加载: %s", path)

    @property
    def best_iteration(self) -> int:
        return self._best_iteration
