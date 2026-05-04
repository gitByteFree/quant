"""情绪类因子（5个）.

异常换手率、量比、北向资金占比变化、融资余额变化、龙虎榜情绪.
"""

import numpy as np
import pandas as pd

from quant_system.factors.base import BaseFactor, FactorResult, register_factor


@register_factor("abnormal_turnover", "sentiment", "异常换手率")
class AbnormalTurnover(BaseFactor):
    """异常换手率 = (当日换手率 - 20日均值) / 20日均值.

    正值表示交易异常活跃，可能反映情绪变化.
    """

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        if "turnover" not in df.columns:
            df["turnover"] = df["volume"] / df.groupby("symbol")["volume"].transform(
                lambda x: x.rolling(252, min_periods=60).mean()
            )
        df["turnover_ma20"] = df.groupby("symbol")["turnover"].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        df["abnormal_turnover"] = (df["turnover"] - df["turnover_ma20"]) / (
            df["turnover_ma20"] + 1e-8
        )
        return self._make_series(df, "abnormal_turnover")


@register_factor("volume_ratio", "sentiment", "量比（当日成交量 / 5日均量）")
class VolumeRatio(BaseFactor):
    """量比 = volume / volume_ma5.

    >1 表示放量，<1 表示缩量.
    """

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        df["volume_ma5"] = df.groupby("symbol")["volume"].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        )
        df["volume_ratio"] = df["volume"] / (df["volume_ma5"] + 1)
        # log化处理极端值
        df["volume_ratio"] = np.log(df["volume_ratio"] + 0.5)
        return self._make_series(df, "volume_ratio")


@register_factor("north_flow_change", "sentiment", "北向资金占比变化")
class NorthFlowChange(BaseFactor):
    """北向资金占比变化代理.

    精确值需北向资金持股数据. 代理: 使用大盘成交额变化率.
    北向资金持续流入通常表示外资看好.
    """

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        # 使用成交额放大率作为北向资金活跃度的代理
        df["amount_ma5"] = df.groupby("symbol")["amount"].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        )
        df["amount_ma20"] = df.groupby("symbol")["amount"].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        df["north_flow_change"] = (df["amount_ma5"] - df["amount_ma20"]) / (df["amount_ma20"] + 1)
        return self._make_series(df, "north_flow_change")


@register_factor("margin_change", "sentiment", "融资余额变化")
class MarginChange(BaseFactor):
    """融资余额变化代理.

    精确值需融资融券数据. 代理: 使用价格和成交量联动性.
    融资买入驱动上涨的股票通常量价齐升.
    """

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        df["ret_5d"] = df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(5)
        )
        df["vol_change_5d"] = df.groupby("symbol")["volume"].transform(
            lambda x: x.pct_change(5)
        )
        # 量价相关性：量价齐升可能=融资驱动
        df["margin_change"] = (
            df["ret_5d"].clip(-0.2, 0.2) * 0.5
            + df["vol_change_5d"].clip(-1, 1) * 0.5
        )
        return self._make_series(df, "margin_change")


@register_factor("dragon_tiger_sentiment", "sentiment", "龙虎榜情绪")
class DragonTigerSentiment(BaseFactor):
    """龙虎榜情绪代理.

    精确值需龙虎榜数据（买卖席位、净买入额）.
    代理: 大涨且放量的股票可能受游资关注（龙虎榜常客）.
    """

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        df["ret_1d"] = df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(1)
        )
        df["volume_ratio_20"] = df["volume"] / (
            df.groupby("symbol")["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean()) + 1
        )
        # 大涨+放量 = 高龙虎榜概率
        df["dragon_tiger_sentiment"] = (
            df["ret_1d"].clip(-0.05, 0.10) * 0.6
            + np.log(df["volume_ratio_20"] + 0.5).clip(-1, 2) * 0.4
        )
        return self._make_series(df, "dragon_tiger_sentiment")
