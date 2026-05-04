"""Transformer时序模型.

Encoder-only架构：可学习位置编码、多头自注意力、全局平均池化、线性头.
输入: (batch, seq_len=60, n_features)
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from quant_system.models.base import BaseModel
from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """可学习的位置编码."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]


class TransformerEncoder(nn.Module):
    """Encoder-only Transformer用于收益预测.

    Args:
        n_features: 输入特征数
        d_model: 模型维度
        n_heads: 注意力头数
        n_layers: Encoder层数
        d_feedforward: 前馈网络维度
        dropout: Dropout比例
        seq_len: 序列长度
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_feedforward: int = 512,
        dropout: float = 0.1,
        seq_len: int = 60,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len + 10)
        self.dropout_embed = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.dropout_embed(x)
        x = self.encoder(x)
        # Global average pooling: (batch, seq_len, d_model) -> (batch, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        x = self.head(x)  # (batch, 1)
        return x.squeeze(-1)


class TransformerModel(BaseModel):
    """Transformer时序预测模型.

    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        n_layers: Encoder层数
        d_feedforward: 前馈网络维度
        dropout: Dropout比例
        seq_len: 序列长度
        learning_rate: 学习率
        batch_size: 批次大小
        max_epochs: 最大训练轮数
        patience: 早停耐心值
        seed: 随机种子
        device: 训练设备
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_feedforward: int = 512,
        dropout: float = 0.1,
        seq_len: int = 60,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 10,
        seed: int = 42,
        device: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(seed=seed)
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._d_feedforward = d_feedforward
        self._dropout = dropout
        self._seq_len = seq_len
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model: TransformerEncoder | None = None
        self._n_features: int = 0

    def _build_model(self, n_features: int) -> TransformerEncoder:
        return TransformerEncoder(
            n_features=n_features,
            d_model=self._d_model,
            n_heads=self._n_heads,
            n_layers=self._n_layers,
            d_feedforward=self._d_feedforward,
            dropout=self._dropout,
            seq_len=self._seq_len,
        )

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | None = None,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
        **kwargs: Any,
    ) -> "TransformerModel":
        X_arr, y_arr = self._validate_inputs(X, y)

        # 假设数据已经是 (n_samples, n_features)，需要reshape为序列格式
        # 如果X的形状暗示了序列 (n_samples, seq_len, n_features)，直接使用
        if X_arr.ndim == 3:
            n_features = X_arr.shape[2]
            seq_len = X_arr.shape[1]
        else:
            n_features = X_arr.shape[1]
            seq_len = 1
            X_arr = X_arr[:, np.newaxis, :]

        self._n_features = n_features
        self._model = self._build_model(n_features)
        self._model.to(self._device)

        # 转换为tensor
        X_tensor = torch.tensor(X_arr, dtype=torch.float32)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32)

        if sample_weight is not None:
            sw = torch.tensor(sample_weight, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, y_tensor, sw)
        else:
            dataset = TensorDataset(X_tensor, y_tensor)

        train_loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        val_loader = None
        if eval_set is not None and len(eval_set) > 0:
            X_val, y_val = eval_set[0]
            if X_val.ndim == 2:
                X_val = X_val[:, np.newaxis, :]
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            )
            val_loader = DataLoader(val_dataset, batch_size=self._batch_size * 2, shuffle=False)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=self._patience // 2, factor=0.5
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict | None = None

        for epoch in range(self._max_epochs):
            self._model.train()
            train_loss = 0.0
            for batch in train_loader:
                if len(batch) == 3:
                    bx, by, bw = batch
                    bw = bw.to(self._device)
                else:
                    bx, by = batch
                    bw = None
                bx, by = bx.to(self._device), by.to(self._device)

                optimizer.zero_grad()
                pred = self._model(bx)
                loss = criterion(pred, by)
                if bw is not None:
                    loss = (loss * bw).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(bx)

            train_loss /= len(train_loader.dataset) if train_loader.dataset else 1

            # 验证
            val_loss = float("inf")
            if val_loader is not None:
                self._model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(self._device), by.to(self._device)
                        pred = self._model(bx)
                        val_loss += criterion(pred, by).item() * len(bx)
                val_loss /= len(val_loader.dataset) if val_loader.dataset else 1
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self._patience:
                    logger.info("Transformer早停: epoch=%d, val_loss=%.6f", epoch + 1, val_loss)
                    break
            else:
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}

            if (epoch + 1) % 20 == 0:
                logger.debug("Transformer epoch %d: train_loss=%.6f", epoch + 1, train_loss)

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._is_fitted = True
        logger.info("Transformer训练完成: best_loss=%.6f", best_val_loss)
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        X_arr, _ = self._validate_inputs(X)
        if X_arr.ndim == 2:
            X_arr = X_arr[:, np.newaxis, :]

        self._model.eval()
        self._model.to(self._device)
        X_tensor = torch.tensor(X_arr, dtype=torch.float32)

        preds: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), self._batch_size * 2):
                batch = X_tensor[i: i + self._batch_size * 2].to(self._device)
                preds.append(self._model(batch).cpu().numpy())

        return np.concatenate(preds) if preds else np.array([])

    def feature_importance(self) -> pd.DataFrame:
        # Transformer特征重要性通过输入投影层权重的绝对值来估计
        if self._model is None:
            return pd.DataFrame(columns=["feature", "importance"])
        weights = self._model.input_proj.weight.detach().cpu().numpy()
        importance = np.abs(weights).mean(axis=0)
        if not self._feature_names:
            self._feature_names = [f"f_{i}" for i in range(len(importance))]
        df = pd.DataFrame({
            "feature": self._feature_names[:len(importance)],
            "importance": importance,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("模型未训练")
        torch.save({
            "model_state": self._model.state_dict(),
            "n_features": self._n_features,
            "config": {
                "d_model": self._d_model,
                "n_heads": self._n_heads,
                "n_layers": self._n_layers,
                "d_feedforward": self._d_feedforward,
                "dropout": self._dropout,
                "seq_len": self._seq_len,
            },
        }, str(path))
        logger.info("Transformer模型已保存: %s", path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(str(path), map_location="cpu")
        self._n_features = checkpoint["n_features"]
        cfg = checkpoint["config"]
        self._d_model = cfg["d_model"]
        self._n_heads = cfg["n_heads"]
        self._n_layers = cfg["n_layers"]
        self._d_feedforward = cfg["d_feedforward"]
        self._dropout = cfg["dropout"]
        self._seq_len = cfg["seq_len"]
        self._model = self._build_model(self._n_features)
        self._model.load_state_dict(checkpoint["model_state"])
        self._is_fitted = True
        logger.info("Transformer模型已加载: %s", path)


def create_sequences(
    factor_panel: pd.DataFrame,
    returns: pd.Series,
    seq_len: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """将因子面板数据转换为Transformer序列格式.

    Args:
        factor_panel: DataFrame, index=MultiIndex (trade_date, symbol), columns=因子名
        returns: Series, MultiIndex (trade_date, symbol), 值为未来收益（标签）
        seq_len: 序列长度

    Returns:
        X: (n_samples, seq_len, n_features)
        y: (n_samples,)
    """
    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    for symbol, sym_data in factor_panel.groupby(level="symbol"):
        sym_data = sym_data.sort_index(level="trade_date")
        values = sym_data.values  # (T, n_features)
        ret_values = returns.xs(symbol, level="symbol") if symbol in returns.index.get_level_values("symbol") else pd.Series(dtype=float)

        if len(values) < seq_len + 1:
            continue

        for t in range(seq_len, len(values)):
            seq = values[t - seq_len: t]  # (seq_len, n_features)
            if np.isnan(seq).any():
                continue
            # 获取t时刻的标签
            date_t = sym_data.index[t]
            if date_t in ret_values.index:
                y_val = ret_values.loc[date_t]
                if not np.isnan(y_val):
                    X_list.append(seq)
                    y_list.append(float(y_val))

    if not X_list:
        return np.array([]).reshape(0, seq_len, 0), np.array([])

    return np.stack(X_list), np.array(y_list)
