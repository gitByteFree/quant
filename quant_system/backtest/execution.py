"""订单执行引擎.

T+1结算、涨跌停不可交易、撮合逻辑.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from quant_system.backtest.cost import CostModel, TransactionCost
from quant_system.backtest.events import (
    FillEvent,
    OrderEvent,
    OrderSide,
    OrderStatus,
    TradeRecord,
)
from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionEngine:
    """订单执行引擎.

    处理A股特有的T+1结算和涨跌停规则.

    Args:
        cost_model: 交易成本模型
        t_plus: T+N 结算（A股为1）
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        t_plus: int = 1,
    ):
        self._cost_model = cost_model or CostModel()
        self._t_plus = t_plus

        # 记录T+1结算状态
        self._pending_sells: dict[str, datetime] = {}  # symbol -> buy_date
        self._trade_history: list[TradeRecord] = []
        self._fill_history: list[FillEvent] = []

    @property
    def fills(self) -> list[FillEvent]:
        return self._fill_history

    @property
    def trades(self) -> list[TradeRecord]:
        return self._trade_history

    def can_sell(self, symbol: str, trade_date: datetime) -> bool:
        """检查T+1规则：当日买入的股票不可卖出."""
        if symbol in self._pending_sells:
            buy_date = self._pending_sells[symbol]
            if trade_date <= buy_date:
                return False
        return True

    def can_trade(
        self,
        symbol: str,
        price: float,
        pre_close: float,
        is_limit_up: bool = False,
        is_limit_down: bool = False,
    ) -> tuple[bool, bool]:
        """检查涨跌停限制.

        Args:
            symbol: 股票代码
            price: 当前价格
            pre_close: 前收盘价
            is_limit_up: 是否涨停
            is_limit_down: 是否跌停

        Returns:
            (can_buy, can_sell)
        """
        can_buy = not is_limit_up
        can_sell = not is_limit_down
        return can_buy, can_sell

    def execute_order(
        self,
        order: OrderEvent,
        price: float,
        pre_close: float,
        volume: float = 0,
        is_limit_up: bool = False,
        is_limit_down: bool = False,
    ) -> FillEvent | None:
        """执行单个订单.

        Args:
            order: 订单事件
            price: 当前成交价
            pre_close: 前收盘价
            volume: 当日成交量
            is_limit_up: 是否涨停
            is_limit_down: 是否跌停

        Returns:
            成交事件，如果无法成交则返回None
        """
        symbol = order.symbol
        trade_date = order.trade_date

        # 检查涨跌停
        can_buy, can_sell = self.can_trade(symbol, price, pre_close, is_limit_up, is_limit_down)

        if order.side == OrderSide.BUY and not can_buy:
            order.status = OrderStatus.REJECTED
            logger.debug("%s 涨停无法买入: %s", trade_date.date(), symbol)
            return None

        if order.side == OrderSide.SELL and not can_sell:
            order.status = OrderStatus.REJECTED
            logger.debug("%s 跌停无法卖出: %s", trade_date.date(), symbol)
            return None

        # 检查T+1
        if order.side == OrderSide.SELL and not self.can_sell(symbol, trade_date):
            order.status = OrderStatus.REJECTED
            logger.debug("%s T+1限制无法卖出: %s", trade_date.date(), symbol)
            return None

        # 计算成本
        if order.side == OrderSide.BUY:
            cost = self._cost_model.calculate_buy(price, order.quantity, volume)
            self._pending_sells[symbol] = trade_date
        else:
            cost = self._cost_model.calculate_sell(price, order.quantity, volume)

        # 生成成交事件
        fill = FillEvent(
            trade_date=trade_date,
            symbol=symbol,
            side=order.side,
            quantity=order.quantity,
            price=price,
            commission=cost.commission,
            stamp_duty=cost.stamp_duty,
            slippage=cost.slippage,
        )

        order.status = OrderStatus.FILLED
        self._fill_history.append(fill)
        return fill

    def execute_orders(
        self,
        orders: list[OrderEvent],
        prices: dict[str, float],
        pre_closes: dict[str, float],
        volumes: dict[str, float],
        limit_ups: set[str] | None = None,
        limit_downs: set[str] | None = None,
    ) -> list[FillEvent]:
        """批量执行订单.

        Args:
            orders: 订单列表
            prices: 当日价格 {symbol: close_price}
            pre_closes: 前收盘价 {symbol: pre_close}
            volumes: 当日成交量 {symbol: volume}
            limit_ups: 涨停股票集合
            limit_downs: 跌停股票集合

        Returns:
            成交事件列表
        """
        if not orders:
            return []

        limit_ups = limit_ups or set()
        limit_downs = limit_downs or set()
        fills: list[FillEvent] = []

        for order in orders:
            symbol = order.symbol
            price = prices.get(symbol, 0)
            pre_close = pre_closes.get(symbol, price)
            volume = volumes.get(symbol, 0)

            if price <= 0:
                order.status = OrderStatus.REJECTED
                continue

            is_lu = symbol in limit_ups
            is_ld = symbol in limit_downs

            fill = self.execute_order(order, price, pre_close, volume, is_lu, is_ld)
            if fill:
                fills.append(fill)

        logger.debug("执行 %d 个订单, 成交 %d 笔", len(orders), len(fills))
        return fills

    def record_trade(self, record: TradeRecord) -> None:
        """记录交易."""
        self._trade_history.append(record)

    def clear_day(self) -> None:
        """清理过期的T+1限制."""
        # T+1只在下一个交易日解除，此处保留以便引擎调用
        pass

    def get_trade_summary(self) -> pd.DataFrame:
        """获取交易摘要."""
        if not self._trade_history:
            return pd.DataFrame()
        return pd.DataFrame([{
            "symbol": t.symbol,
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "return_pct": t.return_pct,
            "holding_days": t.holding_days,
            "side": t.side.value,
        } for t in self._trade_history])
