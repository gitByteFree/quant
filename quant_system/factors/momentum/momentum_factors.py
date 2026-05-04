"""动量类因子（8个）.

5D/20D/60D/120D收益率、RSI_14D、MACD背离度、加权动量、动量稳定性.
"""

import numpy as np
import pandas as pd

from quant_system.factors.base import BaseFactor, FactorResult, register_factor


@register_factor("momentum_5d", "momentum", "5日收益率")
class Momentum5D(BaseFactor):
    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        df["ret_1d"] = df.groupby("symbol")["close"].pct_change()
        df["momentum_5d"] = df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(periods=5)
        )
        return self._make_series(df, "momentum_5d")


@register_factor("momentum_20d", "momentum", "20日收益率")
class Momentum20D(BaseFactor):
    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        df["momentum_20d"] = df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(periods=20)
        )
        return self._make_series(df, "momentum_20d")


@register_factor("momentum_60d", "momentum", "60日收益率")
class Momentum60D(BaseFactor):
    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        df["momentum_60d"] = df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(periods=60)
        )
        return self._make_series(df, "momentum_60d")


@register_factor("momentum_120d", "momentum", "120日收益率（半年动量）")
class Momentum120D(BaseFactor):
    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        df["momentum_120d"] = df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(periods=120)
        )
        return self._make_series(df, "momentum_120d")


@register_factor("rsi_14d", "momentum", "14日相对强弱指标")
class RSI14D(BaseFactor):
    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)

        def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.rolling(period, min_periods=period).mean()
            avg_loss = loss.rolling(period, min_periods=period).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            return 100.0 - (100.0 / (1.0 + rs))

        df["rsi_14d"] = df.groupby("symbol")["close"].transform(_rsi)
        return self._make_series(df, "rsi_14d")


@register_factor("macd_divergence", "momentum", "MACD背离度")
class MACDDivergence(BaseFactor):
    """MACD背离度 = (DIF - DEA) / close.

    正值表示MACD金叉或多头，负值表示死叉或空头.
    """

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)

        def _macd_div(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            dif = ema_fast - ema_slow
            dea = dif.ewm(span=signal, adjust=False).mean()
            return (dif - dea) / series.replace(0, np.nan)

        df["macd_divergence"] = df.groupby("symbol")["close"].transform(_macd_div)
        return self._make_series(df, "macd_divergence")


@register_factor("weighted_momentum", "momentum", "加权动量（近期权重更高）")
class WeightedMomentum(BaseFactor):
    """加权动量: 0.4*ret_5d + 0.3*ret_20d + 0.2*ret_60d + 0.1*ret_120d."""

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        grouped = df.groupby("symbol")["close"]
        df["ret_5d"] = grouped.transform(lambda x: x.pct_change(5))
        df["ret_20d"] = grouped.transform(lambda x: x.pct_change(20))
        df["ret_60d"] = grouped.transform(lambda x: x.pct_change(60))
        df["ret_120d"] = grouped.transform(lambda x: x.pct_change(120))
        df["weighted_momentum"] = (
            0.4 * df["ret_5d"] + 0.3 * df["ret_20d"]
            + 0.2 * df["ret_60d"] + 0.1 * df["ret_120d"]
        )
        return self._make_series(df, "weighted_momentum")


@register_factor("momentum_stability", "momentum", "动量稳定性（20日收益率/收益波动）")
class MomentumStability(BaseFactor):
    """动量稳定性 = 20日平均日收益 / 20日日收益标准差.

    值越高表示上涨趋势越稳定.
    """

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        df["ret_1d"] = df.groupby("symbol")["close"].pct_change()
        df["ret_mean_20d"] = df.groupby("symbol")["ret_1d"].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        df["ret_std_20d"] = df.groupby("symbol")["ret_1d"].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        )
        df["momentum_stability"] = df["ret_mean_20d"] / (df["ret_std_20d"] + 1e-8)
        return self._make_series(df, "momentum_stability")
