"""自定义损失函数.

IC加权MSE、LambdaRank损失（PyTorch实现）.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICWeightedMSELoss(nn.Module):
    """IC加权MSE损失.

    在MSE基础上，用因子IC作为权重，让模型更关注高IC样本.
    Loss = mean(ic_weight * (pred - target)^2)

    Args:
        ic_weights: 每个样本的IC权重 (n_samples,)
        reduction: mean | sum | none
    """

    def __init__(
        self,
        ic_weights: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        if ic_weights is not None:
            self.register_buffer("ic_weights", ic_weights)
        else:
            self.ic_weights = None
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        se = (pred - target) ** 2
        if self.ic_weights is not None and len(self.ic_weights) == len(pred):
            se = se * self.ic_weights
        if self.reduction == "mean":
            return se.mean()
        elif self.reduction == "sum":
            return se.sum()
        return se


class LambdaRankLoss(nn.Module):
    """LambdaRank排序损失.

    用于排序学习，使模型更关注Top-K的排序精度.
    简化的pairwise ranking loss.

    Args:
        sigma: 缩放参数，控制sigmoid的陡峭程度
    """

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算LambdaRank损失.

        Args:
            pred: 预测值 (batch_size,)
            target: 真实值 (batch_size,)

        Returns:
            loss值
        """
        pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
        target_diff = target.unsqueeze(0) - target.unsqueeze(1)

        # 只有target_diff > 0的pair有损失（真实排序更高）
        weight = torch.sign(target_diff).clamp(min=0)

        # 用NDCG增益近似LambdaRank的delta_NDCG权重
        rank_pred = pred.argsort(descending=True).argsort().float() + 1
        rank_target = target.argsort(descending=True).argsort().float() + 1
        gain_pred = 1.0 / torch.log2(rank_pred + 1)
        gain_target = 1.0 / torch.log2(rank_target + 1)
        delta_ndcg = (gain_target.unsqueeze(0) - gain_target.unsqueeze(1)).abs()

        # LambdaRank loss
        prob = torch.sigmoid(self.sigma * pred_diff)
        loss_matrix = -weight * delta_ndcg * torch.log(prob + 1e-8)
        return loss_matrix.mean()


class CorrelationLoss(nn.Module):
    """皮尔逊相关系数损失.

    最小化预测值与目标值的负相关系数.
    Loss = -corr(pred, target)
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.mean()
        target_mean = target.mean()
        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        cov = (pred_centered * target_centered).mean()
        pred_std = pred_centered.std() + 1e-8
        target_std = target_centered.std() + 1e-8
        corr = cov / (pred_std * target_std)
        return -corr


class CombinedLoss(nn.Module):
    """组合损失 = alpha * MSE + beta * CorrLoss + gamma * LambdaRankLoss.

    Args:
        alpha: MSE权重
        beta: 相关系数损失权重
        gamma: 排序损失权重
        sigma: LambdaRank缩放参数
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.05,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.corr = CorrelationLoss()
        self.rank = LambdaRankLoss(sigma=sigma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.alpha * self.mse(pred, target)
        if self.beta > 0:
            loss = loss + self.beta * self.corr(pred, target)
        if self.gamma > 0:
            loss = loss + self.gamma * self.rank(pred, target)
        return loss


def compute_ic_weight(
    factor_ic: float,
    alpha: float = 1.0,
) -> float:
    """根据IC值计算样本权重.

    weight = exp(alpha * abs(ic))

    Args:
        factor_ic: 因子IC值
        alpha: 缩放系数

    Returns:
        样本权重
    """
    return float(torch.exp(torch.tensor(alpha * abs(factor_ic))))


def compute_time_decay_weight(
    days_from_end: int,
    total_days: int,
    alpha: float = 0.95,
) -> float:
    """时间衰减权重.

    weight = alpha^(days_from_end / total_days)

    Args:
        days_from_end: 距离最新日期的天数
        total_days: 总天数（用于归一化）
        alpha: 衰减因子（<1时越近权重越高）

    Returns:
        时间衰减权重
    """
    return float(alpha ** (days_from_end / max(total_days, 1)))


def compute_volatility_inverse_weight(
    volatility: float,
    eps: float = 1e-6,
) -> float:
    """波动率倒数权重.

    weight = 1 / (volatility + eps)

    低波动样本权重更高（更可靠）.

    Args:
        volatility: 波动率值
        eps: 防零

    Returns:
        波动率倒数权重
    """
    return 1.0 / (volatility + eps)
