"""组合管理.

Top-K选股 + 等权/市值加权配置.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from quant_system.backtest.events import OrderEvent, OrderSide, OrderType, SignalEvent
from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioManager:
    """投资组合管理器.

    根据信号生成目标组合和调仓订单.

    Args:
        top_k: 持仓股票数量
        weighting: 加权方式 equal | market_cap
        max_single_weight: 个股最大权重
    """

    def __init__(
        self,
        top_k: int = 100,
        weighting: str = "equal",
        max_single_weight: float = 0.03,
    ):
        self._top_k = top_k
        self._weighting = weighting
        self._max_single_weight = max_single_weight

    def generate_target_weights(
        self,
        signals: list[SignalEvent],
        market_caps: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """根据信号生成目标权重.

        Args:
            signals: 当日信号事件列表
            market_caps: 市值数据 {symbol: market_cap}

        Returns:
            {symbol: target_weight}
        """
        if not signals:
            return {}

        # 按得分降序排列
        sorted_signals = sorted(signals, key=lambda s: s.score, reverse=True)
        top_signals = sorted_signals[:self._top_k]

        if self._weighting == "equal":
            weight = 1.0 / len(top_signals)
            weights = {s.symbol: weight for s in top_signals}
        elif self._weighting == "market_cap":
            if market_caps:
                total_cap = sum(market_caps.get(s.symbol, 0) for s in top_signals)
                if total_cap > 0:
                    weights = {
                        s.symbol: market_caps.get(s.symbol, 0) / total_cap
                        for s in top_signals
                    }
                else:
                    weight = 1.0 / len(top_signals)
                    weights = {s.symbol: weight for s in top_signals}
            else:
                weight = 1.0 / len(top_signals)
                weights = {s.symbol: weight for s in top_signals}
        else:
            weight = 1.0 / len(top_signals)
            weights = {s.symbol: weight for s in top_signals}

        # 限制个股最大权重
        for sym in weights:
            if weights[sym] > self._max_single_weight:
                excess = weights[sym] - self._max_single_weight
                weights[sym] = self._max_single_weight
                # 将超额权重分配给其他股票
                other = [s for s in weights if s != sym]
                if other:
                    per_other = excess / len(other)
                    for o in other:
                        weights[o] += per_other

        return weights

    def generate_orders(
        self,
        trade_date: datetime,
        current_positions: dict[str, int],
        target_weights: dict[str, float],
        prices: dict[str, float],
        total_capital: float,
    ) -> list[OrderEvent]:
        """生成调仓订单.

        比较当前持仓和目标准确，生成买入/卖出订单.

        Args:
            trade_date: 交易日期
            current_positions: 当前持仓 {symbol: shares}
            target_weights: 目标权重 {symbol: weight}
            prices: 当前价格 {symbol: price}
            total_capital: 总资金

        Returns:
            订单列表
        """
        orders: list[OrderEvent] = []
        symbols_to_trade = set(current_positions.keys()) | set(target_weights.keys())

        for symbol in symbols_to_trade:
            target_weight = target_weights.get(symbol, 0)
            target_value = total_capital * target_weight
            price = prices.get(symbol, 0)

            if price <= 0:
                continue

            target_shares = int(target_value / price / 100) * 100  # A股整手
            current_shares = current_positions.get(symbol, 0)

            if target_shares > current_shares:
                buy_qty = target_shares - current_shares
                if buy_qty >= 100:
                    orders.append(OrderEvent(
                        trade_date=trade_date,
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,
                        quantity=buy_qty,
                    ))
            elif target_shares < current_shares:
                sell_qty = current_shares - target_shares
                if sell_qty >= 100:
                    orders.append(OrderEvent(
                        trade_date=trade_date,
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL,
                        quantity=sell_qty,
                    ))

        return orders

    def compute_turnover(
        self,
        old_weights: dict[str, float],
        new_weights: dict[str, float],
    ) -> float:
        """计算单边换手率.

        turnover = sum(|new_weight - old_weight|) / 2

        Args:
            old_weights: 上一期权重
            new_weights: 新权重

        Returns:
            单边换手率
        """
        symbols = set(old_weights.keys()) | set(new_weights.keys())
        total_change = sum(abs(new_weights.get(s, 0) - old_weights.get(s, 0)) for s in symbols)
        return total_change / 2.0
