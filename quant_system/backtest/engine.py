"""事件驱动回测引擎.

整合数据、因子、模型、风控、执行的完整回测主循环.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from quant_system.backtest.cost import CostModel
from quant_system.backtest.events import (
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderSide,
    PortfolioState,
    SignalEvent,
    TradeRecord,
)
from quant_system.backtest.execution import ExecutionEngine
from quant_system.backtest.portfolio import PortfolioManager
from quant_system.backtest.risk_manager import RiskManager
from quant_system.models.base import BaseModel
from quant_system.utils.calendar import TradingCalendar
from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """事件驱动回测引擎.

    Args:
        model: 预测模型
        data: 市场数据DataFrame（需包含日线全字段）
        config: 回测配置字典
        calendar: 交易日历
    """

    def __init__(
        self,
        model: BaseModel,
        data: pd.DataFrame,
        config: dict | None = None,
        calendar: TradingCalendar | None = None,
    ):
        if not model.is_fitted:
            raise ValueError("模型未训练，请先调用 model.fit()")

        self._model = model
        self._data = data.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
        self._config = config or {}
        self._calendar = calendar or TradingCalendar()

        backtest_cfg = self._config.get("backtest", {})
        self._initial_capital = backtest_cfg.get("initial_capital", 10_000_000.0)
        self._rebalance_freq = backtest_cfg.get("rebalance_freq", "daily")
        self._top_k = backtest_cfg.get("top_k", 100)
        self._weighting = backtest_cfg.get("weighting", "equal")

        # 组件
        cost_model = CostModel(
            stamp_duty_rate=backtest_cfg.get("stamp_duty", 0.001),
            commission_rate=backtest_cfg.get("commission", 0.00025),
            slippage_bps=backtest_cfg.get("slippage", 0.001),
        )
        self._execution = ExecutionEngine(cost_model=cost_model)
        self._portfolio_mgr = PortfolioManager(
            top_k=self._top_k,
            weighting=self._weighting,
            max_single_weight=backtest_cfg.get("max_single_weight", 0.03),
        )
        self._risk_mgr = RiskManager(
            max_single_weight=backtest_cfg.get("max_single_weight", 0.03),
            max_industry_deviation=backtest_cfg.get("max_industry_deviation", 0.05),
            max_drawdown_60d=backtest_cfg.get("max_drawdown_threshold", 0.15),
            drawdown_reduction=backtest_cfg.get("drawdown_reduction", 0.5),
        )

        # 状态
        self._cash: float = self._initial_capital
        self._positions: dict[str, int] = {}
        self._portfolio_history: list[PortfolioState] = []
        self._trades: list[TradeRecord] = []
        self._signal_history: list[list[SignalEvent]] = []

        # 数据预处理
        self._prepare_data()

    def _prepare_data(self) -> None:
        """预处理数据：构建查找索引."""
        # 按日期+股票建立字典
        self._price_map: dict[tuple[pd.Timestamp, str], float] = {}
        self._pre_close_map: dict[tuple[pd.Timestamp, str], float] = {}
        self._volume_map: dict[tuple[pd.Timestamp, str], float] = {}
        self._limit_up_map: dict[pd.Timestamp, set[str]] = {}
        self._limit_down_map: dict[pd.Timestamp, set[str]] = {}

        required_cols = {"trade_date", "symbol", "close", "volume"}
        missing = required_cols - set(self._data.columns)
        if missing:
            raise ValueError(f"数据缺少必要列: {missing}")

        for _, row in self._data.iterrows():
            key = (row["trade_date"], row["symbol"])
            self._price_map[key] = row.get("close", 0)
            self._volume_map[key] = row.get("volume", 0)
            pre_close = row.get("pre_close", row["close"])
            if not pre_close or pd.isna(pre_close):
                pre_close = row["close"]
            self._pre_close_map[key] = pre_close

        if "is_limit_up" in self._data.columns:
            for date, group in self._data[self._data["is_limit_up"]].groupby("trade_date"):
                self._limit_up_map[date] = set(group["symbol"])
        if "is_limit_down" in self._data.columns:
            for date, group in self._data[self._data["is_limit_down"]].groupby("trade_date"):
                self._limit_down_map[date] = set(group["symbol"])

        logger.info("回测数据准备完成: %d 条记录, %d 个交易日",
                     len(self._data), self._data["trade_date"].nunique())

    def run(
        self,
        start_date: str,
        end_date: str,
        feature_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """执行回测.

        Args:
            start_date: 回测起始日期
            end_date: 回测结束日期
            feature_cols: 用于模型预测的特征列名

        Returns:
            DataFrame: 每日组合状态
        """
        trade_dates = self._calendar.trade_dates_between(start_date, end_date)
        if len(trade_dates) == 0:
            logger.warning("回测区间无交易日")
            return pd.DataFrame()

        logger.info("开始回测: %s ~ %s (%d 个交易日)", start_date, end_date, len(trade_dates))
        self._cash = self._initial_capital
        self._positions = {}
        self._portfolio_history = []

        for i, date in enumerate(trade_dates):
            date_dt = date.to_pydatetime()
            date_ts = pd.Timestamp(date_dt)

            # 1. 获取当日可交易股票数据
            day_data = self._data[self._data["trade_date"] == date_ts]
            if day_data.empty:
                logger.debug("%s 无交易数据", date.date())
                continue

            # 2. 生成信号（使用模型预测）
            signals = self._generate_signals(date_ts, day_data, feature_cols)

            # 3. 判断是否需要调仓
            should_rebalance = self._should_rebalance(i, date_ts)
            fills: list[FillEvent] = []

            if should_rebalance:
                # 4. 组合管理：生成目标权重和订单
                prices_today = {
                    row["symbol"]: row["close"]
                    for _, row in day_data.iterrows()
                }
                market_caps = {
                    row["symbol"]: row.get("amount", row["close"] * row.get("volume", 1))
                    for _, row in day_data.iterrows()
                }
                target_weights = self._portfolio_mgr.generate_target_weights(signals, market_caps)

                # 5. 风控
                target_weights = self._risk_mgr.apply_risk_controls(target_weights)

                # 6. 生成订单
                orders = self._portfolio_mgr.generate_orders(
                    trade_date=date_dt,
                    current_positions=self._positions,
                    target_weights=target_weights,
                    prices=prices_today,
                    total_capital=self._get_total_value(prices_today),
                )

                # 7. 风控过滤
                orders = self._risk_mgr.filter_orders_by_risk(orders)

                # 8. 执行订单
                pre_closes = {
                    row["symbol"]: self._pre_close_map.get((date_ts, row["symbol"]), row["close"])
                    for _, row in day_data.iterrows()
                }
                volumes = {
                    row["symbol"]: self._volume_map.get((date_ts, row["symbol"]), 0)
                    for _, row in day_data.iterrows()
                }
                fills = self._execution.execute_orders(
                    orders=orders,
                    prices=prices_today,
                    pre_closes=pre_closes,
                    volumes=volumes,
                    limit_ups=self._limit_up_map.get(date_ts, set()),
                    limit_downs=self._limit_down_map.get(date_ts, set()),
                )

                # 9. 更新持仓和资金
                self._apply_fills(fills)

            # 10. 记录组合状态
            prices_today = {
                row["symbol"]: row["close"]
                for _, row in day_data.iterrows()
            }
            state = self._record_portfolio_state(date_dt, prices_today)
            self._portfolio_history.append(state)

            # 11. 更新风控净值
            self._risk_mgr.update_equity(date_dt, state.total_value)

            if (i + 1) % 60 == 0 or i == 0:
                logger.info(
                    "%s: 总值=%.2f万, 现金=%.2f万, 持仓=%d只",
                    date.date(),
                    state.total_value / 1e4,
                    state.cash / 1e4,
                    len(self._positions),
                )

        logger.info("回测完成: %d 个交易日", len(self._portfolio_history))
        return self.get_portfolio_df()

    def _generate_signals(
        self,
        date: pd.Timestamp,
        day_data: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> list[SignalEvent]:
        """使用模型生成当日交易信号."""
        if feature_cols is None:
            feature_cols = [c for c in day_data.columns if c not in (
                "trade_date", "symbol", "open", "high", "low", "close",
                "volume", "amount", "pre_close", "is_st", "is_limit_up",
                "is_limit_down", "is_long_suspended", "adjust_type",
            )]

        available_cols = [c for c in feature_cols if c in day_data.columns]
        if not available_cols:
            return []

        # 准备特征矩阵
        features = day_data[available_cols].copy()
        symbols = day_data["symbol"].tolist()

        # 处理缺失值
        features = features.fillna(features.median(numeric_only=True))
        features = features.fillna(0)

        try:
            scores = self._model.predict(features.values.astype(np.float32))
        except Exception as e:
            logger.error("模型预测失败: %s", e)
            return []

        signals = []
        for i, symbol in enumerate(symbols):
            if i < len(scores) and not np.isnan(scores[i]):
                signals.append(SignalEvent(
                    trade_date=date.to_pydatetime(),
                    symbol=symbol,
                    score=float(scores[i]),
                ))

        self._signal_history.append(signals)
        return signals

    def _should_rebalance(self, day_idx: int, date: pd.Timestamp) -> bool:
        """判断当日是否需要调仓."""
        if self._rebalance_freq == "daily":
            return True
        if self._rebalance_freq == "weekly":
            return date.dayofweek == 4  # 周五调仓
        return True

    def _get_total_value(self, prices: dict[str, float]) -> float:
        """计算当前总资产."""
        position_value = sum(
            shares * prices.get(sym, 0)
            for sym, shares in self._positions.items()
        )
        return self._cash + position_value

    def _apply_fills(self, fills: list[FillEvent]) -> None:
        """应用成交事件更新持仓和资金."""
        for fill in fills:
            trade_value = fill.price * fill.quantity
            if fill.side == OrderSide.BUY:
                cost = trade_value + fill.commission + fill.slippage
                if cost <= self._cash:
                    self._cash -= cost
                    self._positions[fill.symbol] = (
                        self._positions.get(fill.symbol, 0) + fill.quantity
                    )
            else:
                current_shares = self._positions.get(fill.symbol, 0)
                actual_qty = min(fill.quantity, current_shares)
                if actual_qty > 0:
                    revenue = fill.price * actual_qty
                    cost = fill.stamp_duty + fill.commission + fill.slippage
                    # 按比例调整成本
                    if fill.quantity > 0:
                        ratio = actual_qty / fill.quantity
                        cost *= ratio
                    self._cash += revenue - cost
                    remaining = current_shares - actual_qty
                    if remaining <= 0:
                        del self._positions[fill.symbol]
                    else:
                        self._positions[fill.symbol] = remaining

    def _record_portfolio_state(
        self,
        date: datetime,
        prices: dict[str, float],
    ) -> PortfolioState:
        """记录组合状态."""
        position_value = sum(
            shares * prices.get(sym, 0)
            for sym, shares in self._positions.items()
        )
        total = self._cash + position_value
        prev_total = self._initial_capital
        if self._portfolio_history:
            prev_total = self._portfolio_history[-1].total_value
        daily_ret = (total / prev_total - 1) if prev_total > 0 else 0
        peak = max(s.total_value for s in self._portfolio_history) if self._portfolio_history else total
        drawdown = (peak - total) / peak if peak > 0 else 0

        market_values = {
            sym: shares * prices.get(sym, 0)
            for sym, shares in self._positions.items()
        }

        return PortfolioState(
            trade_date=date,
            cash=self._cash,
            positions=self._positions.copy(),
            market_values=market_values,
            total_value=total,
            daily_return=daily_ret,
            cumulative_return=total / self._initial_capital - 1,
            drawdown=drawdown,
        )

    def get_portfolio_df(self) -> pd.DataFrame:
        """获取组合历史DataFrame."""
        if not self._portfolio_history:
            return pd.DataFrame()
        return pd.DataFrame([{
            "trade_date": s.trade_date,
            "total_value": s.total_value,
            "cash": s.cash,
            "daily_return": s.daily_return,
            "cumulative_return": s.cumulative_return,
            "drawdown": s.drawdown,
            "n_positions": len(s.positions),
        } for s in self._portfolio_history]).set_index("trade_date")

    def get_trades_df(self) -> pd.DataFrame:
        """获取交易记录DataFrame."""
        return self._execution.get_trade_summary()

    @property
    def portfolio_history(self) -> list[PortfolioState]:
        return self._portfolio_history

    @property
    def fills(self) -> list[FillEvent]:
        return self._execution.fills

    @property
    def trades(self) -> list[TradeRecord]:
        return self._execution.trades
