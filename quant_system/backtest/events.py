"""回测事件系统.

事件驱动的回测引擎核心数据结构.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class EventType(Enum):
    MARKET = "market"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class MarketEvent:
    """市场数据事件.

    每个交易日开始时触发，包含当天可交易的所有股票数据.
    """

    trade_date: datetime
    event_type: EventType = field(default=EventType.MARKET, init=False)


@dataclass
class SignalEvent:
    """交易信号事件.

    由模型预测产生，每个交易日每个股票一个信号.
    """

    trade_date: datetime
    symbol: str
    score: float  # 预测收益/排序得分，越高越好
    event_type: EventType = field(default=EventType.SIGNAL, init=False)


@dataclass
class OrderEvent:
    """订单事件."""

    trade_date: datetime
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: int  # 股数
    price: float | None = None  # limit_price，market订单为None
    status: OrderStatus = OrderStatus.PENDING
    signal_score: float = 0.0
    event_type: EventType = field(default=EventType.ORDER, init=False)
    created_date: datetime | None = None


@dataclass
class FillEvent:
    """成交事件."""

    trade_date: datetime
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float = 0.0
    stamp_duty: float = 0.0
    slippage: float = 0.0
    event_type: EventType = field(default=EventType.FILL, init=False)


@dataclass
class PortfolioState:
    """组合状态快照."""

    trade_date: datetime
    cash: float
    positions: dict[str, int] = field(default_factory=dict)  # symbol -> shares
    market_values: dict[str, float] = field(default_factory=dict)  # symbol -> market_value
    total_value: float = 0.0
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    drawdown: float = 0.0


@dataclass
class TradeRecord:
    """交易记录（用于绩效分析）."""

    entry_date: datetime
    exit_date: datetime
    symbol: str
    side: OrderSide
    entry_price: float
    exit_price: float
    quantity: int
    return_pct: float
    holding_days: int
    entry_signal_score: float = 0.0
