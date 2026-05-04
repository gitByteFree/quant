"""风险控制模块.

行业偏离度、Barra因子暴露、个股集中度、回撤预警.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from quant_system.backtest.events import OrderEvent, OrderSide
from quant_system.utils.constants import (
    DRAWDOWN_POSITION_REDUCTION,
    MAX_INDUSTRY_DEVIATION,
    MAX_ROLLING_DRAWDOWN,
    MAX_SINGLE_STOCK_WEIGHT,
)
from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


class RiskManager:
    """风险管理器.

    Args:
        max_single_weight: 个股最大权重
        max_industry_deviation: 行业偏离度上限
        max_drawdown_60d: 60日最大回撤阈值
        drawdown_reduction: 回撤触发时仓位降至比例
    """

    def __init__(
        self,
        max_single_weight: float = MAX_SINGLE_STOCK_WEIGHT,
        max_industry_deviation: float = MAX_INDUSTRY_DEVIATION,
        max_drawdown_60d: float = MAX_ROLLING_DRAWDOWN,
        drawdown_reduction: float = DRAWDOWN_POSITION_REDUCTION,
    ):
        self._max_single_weight = max_single_weight
        self._max_industry_deviation = max_industry_deviation
        self._max_drawdown = max_drawdown_60d
        self._drawdown_reduction = drawdown_reduction

        self._equity_curve: list[float] = []
        self._equity_dates: list[datetime] = []

    def update_equity(self, trade_date: datetime, total_value: float) -> None:
        """更新净值曲线."""
        self._equity_dates.append(trade_date)
        self._equity_curve.append(total_value)

    def check_drawdown(self) -> tuple[bool, float]:
        """检查60日滚动最大回撤.

        Returns:
            (breach, current_drawdown)
        """
        if len(self._equity_curve) < 60:
            return False, 0.0

        recent = self._equity_curve[-60:]
        peak = max(recent)
        current = recent[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0

        return drawdown > self._max_drawdown, drawdown

    def get_risk_adjustment(self) -> float:
        """获取风控仓位调整系数.

        Returns:
            1.0 = 满仓, 0.5 = 半仓 (触发回撤降仓)
        """
        breached, dd = self.check_drawdown()
        if breached:
            logger.warning("回撤预警触发! 60日回撤=%.2f%%, 仓位降至%.0f%%",
                           dd * 100, self._drawdown_reduction * 100)
            return self._drawdown_reduction
        return 1.0

    def check_single_stock_weight(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """检查并限制个股权重.

        Args:
            weights: 目标权重 {symbol: weight}

        Returns:
            调整后的权重
        """
        adjusted = weights.copy()
        symbols_to_adjust = [s for s, w in adjusted.items() if w > self._max_single_weight]

        for sym in symbols_to_adjust:
            excess = adjusted[sym] - self._max_single_weight
            adjusted[sym] = self._max_single_weight
            other = [s for s in adjusted if s != sym]
            if other and excess > 0:
                per_other = excess / len(other)
                for o in other:
                    adjusted[o] += per_other

        return adjusted

    def check_industry_deviation(
        self,
        weights: dict[str, float],
        benchmark_weights: dict[str, float],
        stock_industries: dict[str, str],
    ) -> tuple[bool, dict[str, float]]:
        """检查行业偏离度.

        Args:
            weights: 组合权重
            benchmark_weights: 基准行业权重 {industry: weight}
            stock_industries: 股票行业映射 {symbol: industry}

        Returns:
            (is_breach, industry_deviations)
        """
        # 计算组合行业权重
        portfolio_industry: dict[str, float] = {}
        for sym, w in weights.items():
            ind = stock_industries.get(sym, "未知")
            portfolio_industry[ind] = portfolio_industry.get(ind, 0) + w

        deviations: dict[str, float] = {}
        for ind in set(portfolio_industry.keys()) | set(benchmark_weights.keys()):
            pw = portfolio_industry.get(ind, 0)
            bw = benchmark_weights.get(ind, 0)
            deviations[ind] = abs(pw - bw)

        max_dev = max(deviations.values()) if deviations else 0
        is_breach = max_dev > self._max_industry_deviation

        if is_breach:
            logger.warning("行业偏离度超限: max=%.2f%%", max_dev * 100)

        return is_breach, deviations

    def apply_risk_controls(
        self,
        target_weights: dict[str, float],
        benchmark_weights: dict[str, float] | None = None,
        stock_industries: dict[str, str] | None = None,
    ) -> dict[str, float]:
        """应用所有风控规则.

        Args:
            target_weights: 原始目标权重
            benchmark_weights: 基准行业权重
            stock_industries: 股票行业映射

        Returns:
            风控调整后的权重
        """
        weights = target_weights.copy()

        # 1. 回撤检查 -> 整体降仓
        risk_adj = self.get_risk_adjustment()
        if risk_adj < 1.0:
            weights = {s: w * risk_adj for s, w in weights.items()}

        # 2. 个股集中度
        weights = self.check_single_stock_weight(weights)

        # 3. 行业偏离度
        if benchmark_weights and stock_industries:
            breached, _ = self.check_industry_deviation(weights, benchmark_weights, stock_industries)
            if breached:
                logger.warning("行业偏离度超限，按基准权重缩放")
                # 简单的缩放调整：朝基准方向移动
                for sym, w in weights.items():
                    ind = stock_industries.get(sym, "未知")
                    bw = benchmark_weights.get(ind, 0)
                    # 如果偏离超过阈值，向基准方向调整
                    pw = sum(weights.get(s, 0) for s, i in stock_industries.items() if i == ind)
                    if pw > 0 and abs(pw - bw) > self._max_industry_deviation:
                        scale = bw / pw if pw > 0 else 1
                        for s in [s for s, i in stock_industries.items() if i == ind]:
                            if s in weights:
                                weights[s] *= scale

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}

        return weights

    def filter_orders_by_risk(
        self,
        orders: list[OrderEvent],
    ) -> list[OrderEvent]:
        """在回撤超限时，过滤掉新的买入订单（只保留卖出）.

        Args:
            orders: 原始订单列表

        Returns:
            过滤后的订单列表
        """
        breached, _ = self.check_drawdown()
        if not breached:
            return orders
        # 只保留卖出订单
        return [o for o in orders if o.side == OrderSide.SELL]
