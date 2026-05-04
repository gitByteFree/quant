"""波动类因子（5个）.

ATR_14D、HV_20D/60D、下行波动率、最大回撤.
"""

import numpy as np
import pandas as pd

from quant_system.factors.base import BaseFactor, FactorResult, register_factor


@register_factor("atr_14d", "volatility", "14日平均真实波幅")
class ATR14D(BaseFactor):
    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        grouped = df.groupby("symbol")

        df["prev_close"] = grouped["close"].shift(1)
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = (df["high"] - df["prev_close"]).abs()
        df["tr3"] = (df["low"] - df["prev_close"]).abs()
        df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        df["atr_14d"] = grouped["true_range"].transform(
            lambda x: x.rolling(14, min_periods=7).mean()
        )
        # 归一化为价格的百分比
        df["atr_14d"] = df["atr_14d"] / df["close"]
        return self._make_series(df, "atr_14d")


@register_factor("hv_20d", "volatility", "20日历史波动率（年化）")
class HV20D(BaseFactor):
    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        df["log_ret"] = df.groupby("symbol")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )
        df["hv_20d"] = df.groupby("symbol")["log_ret"].transform(
            lambda x: x.rolling(20, min_periods=10).std() * np.sqrt(252)
        )
        return self._make_series(df, "hv_20d")


@register_factor("hv_60d", "volatility", "60日历史波动率（年化）")
class HV60D(BaseFactor):
    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        df["log_ret"] = df.groupby("symbol")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )
        df["hv_60d"] = df.groupby("symbol")["log_ret"].transform(
            lambda x: x.rolling(60, min_periods=30).std() * np.sqrt(252)
        )
        return self._make_series(df, "hv_60d")


@register_factor("downside_volatility", "volatility", "下行波动率（只考虑负收益的波动）")
class DownsideVolatility(BaseFactor):
    """下行波动率 = std(min(日收益, 0)) * sqrt(252).

    只衡量下行风险，比总波动率更能反映投资者关心的风险.
    """

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        df["log_ret"] = df.groupby("symbol")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )
        df["neg_ret"] = df["log_ret"].clip(upper=0)
        df["downside_volatility"] = df.groupby("symbol")["neg_ret"].transform(
            lambda x: x.rolling(60, min_periods=30).std() * np.sqrt(252)
        )
        return self._make_series(df, "downside_volatility")


@register_factor("max_drawdown_60d", "volatility", "60日最大回撤")
class MaxDrawdown60D(BaseFactor):
    """滚动60日最大回撤.

    正值表示回撤幅度大，是反向因子（回撤越大，预期收益越低）.
    """

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)

        def _max_drawdown(series: pd.Series, window: int = 60) -> pd.Series:
            rolling_max = series.rolling(window, min_periods=window).max()
            drawdown = series / rolling_max - 1
            return drawdown.rolling(window, min_periods=window).min()

        df["max_drawdown_60d"] = df.groupby("symbol")["close"].transform(
            lambda x: _max_drawdown(x, 60)
        )
        return self._make_series(df, "max_drawdown_60d")
