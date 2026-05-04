"""交易成本模型.

A股: 印花税0.1%（仅卖出）、佣金0.025%、滑点0.1%、最低佣金5元.
"""

from dataclasses import dataclass

from quant_system.utils.constants import COMMISSION_RATE, SLIPPAGE_BASIS_POINTS, STAMP_DUTY_RATE


@dataclass
class TransactionCost:
    """单笔交易成本."""

    stamp_duty: float  # 印花税
    commission: float  # 佣金
    slippage: float  # 滑点成本
    total: float  # 总成本

    def __repr__(self) -> str:
        return (
            f"Cost(stamp={self.stamp_duty:.2f}, "
            f"commission={self.commission:.2f}, "
            f"slippage={self.slippage:.2f}, "
            f"total={self.total:.2f})"
        )


class CostModel:
    """A股交易成本模型.

    Args:
        stamp_duty_rate: 印花税率（卖出单边0.1%）
        commission_rate: 佣金率（0.025%）
        min_commission: 最低佣金（5元）
        slippage_bps: 滑点（0.1% = 10bps）
        consider_impact: 是否考虑市场冲击
        impact_coefficient: 冲击系数
    """

    def __init__(
        self,
        stamp_duty_rate: float = STAMP_DUTY_RATE,
        commission_rate: float = COMMISSION_RATE,
        min_commission: float = 5.0,
        slippage_bps: float = SLIPPAGE_BASIS_POINTS,
        consider_impact: bool = True,
        impact_coefficient: float = 0.1,
    ):
        self._stamp_duty_rate = stamp_duty_rate
        self._commission_rate = commission_rate
        self._min_commission = min_commission
        self._slippage_rate = slippage_bps
        self._consider_impact = consider_impact
        self._impact_coeff = impact_coefficient

    def calculate_buy(
        self,
        price: float,
        quantity: int,
        daily_volume: float | None = None,
    ) -> TransactionCost:
        """计算买入成本.

        A股买入不收印花税，只收佣金和滑点.

        Args:
            price: 成交价格
            quantity: 成交股数（手）
            daily_volume: 当日成交量（用于估算市场冲击）

        Returns:
            TransactionCost
        """
        trade_value = price * quantity
        commission = max(self._min_commission, trade_value * self._commission_rate)
        slippage = trade_value * self._slippage_rate

        # 市场冲击估计
        if self._consider_impact and daily_volume and daily_volume > 0:
            participation_rate = trade_value / daily_volume
            impact = trade_value * participation_rate * self._impact_coeff
            slippage += impact

        return TransactionCost(
            stamp_duty=0.0,
            commission=commission,
            slippage=slippage,
            total=commission + slippage,
        )

    def calculate_sell(
        self,
        price: float,
        quantity: int,
        daily_volume: float | None = None,
    ) -> TransactionCost:
        """计算卖出成本.

        A股卖出收印花税0.1% + 佣金 + 滑点.

        Args:
            price: 成交价格
            quantity: 成交股数
            daily_volume: 当日成交量

        Returns:
            TransactionCost
        """
        trade_value = price * quantity
        stamp_duty = trade_value * self._stamp_duty_rate
        commission = max(self._min_commission, trade_value * self._commission_rate)
        slippage = trade_value * self._slippage_rate

        if self._consider_impact and daily_volume and daily_volume > 0:
            participation_rate = trade_value / daily_volume
            impact = trade_value * participation_rate * self._impact_coeff
            slippage += impact

        return TransactionCost(
            stamp_duty=stamp_duty,
            commission=commission,
            slippage=slippage,
            total=stamp_duty + commission + slippage,
        )

    def calculate_turnover_cost(
        self,
        price: float,
        quantity: int,
        daily_volume: float | None = None,
    ) -> TransactionCost:
        """计算完整换手成本（买入+卖出）."""
        buy = self.calculate_buy(price, quantity, daily_volume)
        sell = self.calculate_sell(price, quantity, daily_volume)
        return TransactionCost(
            stamp_duty=buy.stamp_duty + sell.stamp_duty,
            commission=buy.commission + sell.commission,
            slippage=buy.slippage + sell.slippage,
            total=buy.total + sell.total,
        )

    def effective_buy_price(
        self,
        price: float,
        quantity: int,
        daily_volume: float | None = None,
    ) -> float:
        """计算有效买入价格（含成本）."""
        cost = self.calculate_buy(price, quantity, daily_volume)
        return price + cost.total / quantity if quantity > 0 else price

    def effective_sell_price(
        self,
        price: float,
        quantity: int,
        daily_volume: float | None = None,
    ) -> float:
        """计算有效卖出价格（扣除成本后）."""
        cost = self.calculate_sell(price, quantity, daily_volume)
        return price - cost.total / quantity if quantity > 0 else price
