"""Stacking集成模型.

K-fold OOF预测 + LightGBM元模型.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from quant_system.models.base import BaseModel
from quant_system.models.lightgbm_model import LightGBMModel
from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


class StackingEnsemble(BaseModel):
    """Stacking集成模型.

    使用K折交叉验证生成OOF预测，再训练一个元模型融合所有基模型.

    Args:
        base_models: 基模型列表 [(name, model), ...]
        meta_model: 元模型（默认LightGBM）
        n_folds: K折数
        use_oof_features: 是否使用OOF特征（同时保留原始特征）
        seed: 随机种子
    """

    def __init__(
        self,
        base_models: list[tuple[str, BaseModel]] | None = None,
        meta_model: BaseModel | None = None,
        n_folds: int = 5,
        use_oof_features: bool = True,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(seed=seed)
        self._base_models = base_models or []
        self._meta_model = meta_model or LightGBMModel(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=64,
            seed=seed,
        )
        self._n_folds = n_folds
        self._use_oof = use_oof_features
        self._oof_predictions: dict[str, np.ndarray] = {}
        self._fitted_base_models: list[BaseModel] = []

    def add_base_model(self, name: str, model: BaseModel) -> None:
        """添加基模型."""
        self._base_models.append((name, model))

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | None = None,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
        **kwargs: Any,
    ) -> "StackingEnsemble":
        X_arr, y_arr = self._validate_inputs(X, y)
        n_samples = len(X_arr)

        if not self._base_models:
            raise ValueError("没有基模型，请先调用 add_base_model()")

        kf = KFold(n_splits=self._n_folds, shuffle=True, random_state=self._seed)

        meta_features = np.zeros((n_samples, len(self._base_models)))
        if self._use_oof:
            meta_features = np.hstack([meta_features, X_arr])

        for i, (name, model) in enumerate(self._base_models):
            logger.info("训练基模型: %s (%d/%d)", name, i + 1, len(self._base_models))
            oof_preds = np.zeros(n_samples)

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_arr)):
                X_train, X_val = X_arr[train_idx], X_arr[val_idx]
                y_train, y_val = y_arr[train_idx], y_arr[val_idx]

                sw_train = sample_weight[train_idx] if sample_weight is not None else None

                model_copy = self._clone_model(model)
                try:
                    model_copy.fit(X_train, y_train, sample_weight=sw_train)
                    oof_preds[val_idx] = model_copy.predict(X_val)
                except Exception as e:
                    logger.warning("基模型 %s fold %d 训练失败: %s", name, fold_idx, e)
                    oof_preds[val_idx] = 0.0

            meta_features[:n_samples, i] = oof_preds
            self._oof_predictions[name] = oof_preds

            # 最终在全量数据上训练
            final_model = self._clone_model(model)
            final_model.fit(X_arr, y_arr, sample_weight=sample_weight)
            self._fitted_base_models.append(final_model)

        # 训练元模型
        logger.info("训练元模型...")
        eval_set_meta = None
        if eval_set is not None and len(eval_set) > 0:
            X_val, y_val = eval_set[0]
            meta_val = self._build_meta_features(X_val, X_val)
            eval_set_meta = [(meta_val, y_val)]

        self._meta_model.fit(meta_features, y_arr, eval_set=eval_set_meta)
        self._is_fitted = True
        logger.info("Stacking集成训练完成")
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("模型未训练")
        X_arr, _ = self._validate_inputs(X)
        meta_features = self._build_meta_features(X_arr, X_arr)
        return self._meta_model.predict(meta_features)

    def _build_meta_features(
        self,
        X_train: np.ndarray,
        X_pred: np.ndarray,
    ) -> np.ndarray:
        """构建元模型输入特征."""
        base_preds = np.zeros((len(X_pred), len(self._fitted_base_models)))
        for i, model in enumerate(self._fitted_base_models):
            base_preds[:, i] = model.predict(X_pred)

        if self._use_oof:
            return np.hstack([base_preds, X_pred])
        return base_preds

    def _clone_model(self, model: BaseModel) -> BaseModel:
        """创建模型副本（浅拷贝+重建关键参数）."""
        if isinstance(model, LightGBMModel):
            return LightGBMModel(seed=self._seed)
        # 对于其他模型类型，使用简单的重建
        import copy
        return copy.deepcopy(model)

    def feature_importance(self) -> pd.DataFrame:
        return self._meta_model.feature_importance()

    def get_base_model_weights(self) -> dict[str, float]:
        """获取基模型在元模型中的权重（近似）."""
        imp = self._meta_model.feature_importance()
        if imp.empty or len(self._base_models) == 0:
            return {}
        base_imp = imp.head(len(self._base_models))
        total = base_imp["importance"].sum()
        if total == 0:
            return {}
        weights: dict[str, float] = {}
        for i, (name, _) in enumerate(self._base_models):
            if i < len(base_imp):
                weights[name] = float(base_imp.iloc[i]["importance"] / total)
        return weights

    def save(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._meta_model.save(path / "meta_model.pkl")
        for i, model in enumerate(self._fitted_base_models):
            model.save(path / f"base_model_{i}.pkl")
        logger.info("Stacking集成已保存: %s", path)

    def load(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        self._meta_model.load(path / "meta_model.pkl")
        self._fitted_base_models = []
        i = 0
        while (path / f"base_model_{i}.pkl").exists():
            model = LightGBMModel()
            model.load(path / f"base_model_{i}.pkl")
            self._fitted_base_models.append(model)
            i += 1
        self._is_fitted = True
        logger.info("Stacking集成已加载: %d 个基模型", len(self._fitted_base_models))
