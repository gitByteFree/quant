"""质量类因子（6个）.

ROE变动/稳定性、毛利率稳定性、应计利润、资产周转率、杠杆率.
"""

import numpy as np
import pandas as pd

from quant_system.factors.base import BaseFactor, FactorResult, register_factor


@register_factor("roe_change", "quality", "ROE变动（TTM同比变化）")
class ROEChange(BaseFactor):
    """ROE变动代理：利用价格和成交额估算ROE变化.

    精确值需财务数据.
    """

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        # ROE代理 = 成交额/市值 (粗略代理)
        df["roe_proxy"] = df["amount"] / (df["close"] * df["volume"].replace(0, np.nan))
        df["roe_avg_252"] = df.groupby("symbol")["roe_proxy"].transform(
            lambda x: x.rolling(252, min_periods=60).mean()
        )
        df["roe_avg_63"] = df.groupby("symbol")["roe_proxy"].transform(
            lambda x: x.rolling(63, min_periods=20).mean()
        )
        df["roe_change"] = (df["roe_avg_63"] - df["roe_avg_252"]) / (df["roe_avg_252"].abs() + 1e-8)
        return self._make_series(df, "roe_change")


@register_factor("roe_stability", "quality", "ROE稳定性（过去3年ROE标准差倒数）")
class ROEStability(BaseFactor):
    """ROE稳定性 = 1 / std(ROE_proxy, 252d).

    值越高表示盈利越稳定.
    """

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        df["roe_proxy"] = df["amount"] / (df["close"] * df["volume"].replace(0, np.nan))
        df["roe_std_252"] = df.groupby("symbol")["roe_proxy"].transform(
            lambda x: x.rolling(252, min_periods=120).std()
        )
        df["roe_stability"] = 1.0 / (df["roe_std_252"] + 1e-8)
        return self._make_series(df, "roe_stability")


@register_factor("gross_margin_stability", "quality", "毛利率稳定性")
class GrossMarginStability(BaseFactor):
    """毛利率稳定性代理.

    精确值需营业成本和营业收入数据. 这里用价格波动性作为反向代理.
    """

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        df["log_ret"] = df.groupby("symbol")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )
        df["ret_std_252"] = df.groupby("symbol")["log_ret"].transform(
            lambda x: x.rolling(252, min_periods=60).std()
        )
        # 低波动 = 高毛利率稳定性
        df["gross_margin_stability"] = 1.0 / (1.0 + df["ret_std_252"] * np.sqrt(252))
        return self._make_series(df, "gross_margin_stability")


@register_factor("accruals", "quality", "应计利润（反向指标，高应计=低质量）")
class Accruals(BaseFactor):
    """应计利润代理.

    应计利润 = (净利润 - 经营现金流) / 总资产.
    精确值需财务数据. 代理: (价格变化 - 成交量变化率) 表示非现金驱动的涨幅.
    """

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)

        df["price_change_252"] = df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(252)
        )
        df["vol_change_252"] = df.groupby("symbol")["volume"].transform(
            lambda x: x.pct_change(252)
        )

        # 若价格上涨但成交量下降，可能应计利润高（盈利质量差）
        df["accruals"] = df["price_change_252"] - df["vol_change_252"]
        # 高应计 = 低质量，所以取负数使其成为正向因子
        df["accruals"] = -df["accruals"]
        return self._make_series(df, "accruals")


@register_factor("asset_turnover", "quality", "资产周转率代理")
class AssetTurnover(BaseFactor):
    """资产周转率代理 = 成交额 / 市值.

    高换手率可能意味着高资产周转效率（或投机）.
    """

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        # 使用换手率作为资产周转率代理
        if "turnover" in df.columns:
            df["asset_turnover"] = df.groupby("symbol")["turnover"].transform(
                lambda x: x.rolling(60, min_periods=20).mean()
            )
        else:
            df["asset_turnover"] = df.groupby("symbol")["volume"].transform(
                lambda x: x.rolling(60, min_periods=20).mean()
            )
            df["asset_turnover"] = df["asset_turnover"] / df.groupby("symbol")["volume"].transform(
                lambda x: x.rolling(252, min_periods=60).mean()
            )
        return self._make_series(df, "asset_turnover")


@register_factor("leverage", "quality", "杠杆率（反向指标，高杠杆=高风险）")
class Leverage(BaseFactor):
    """杠杆率代理.

    精确值: 总负债/总资产. 代理: 用价格波动放大效应估计杠杆.
    高杠杆股票在下跌时波动更大.
    """

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        df["log_ret"] = df.groupby("symbol")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )
        # 下行波动率 / 上行波动率 比值越大说明杠杆越高
        df["down_std"] = df.groupby("symbol")["log_ret"].transform(
            lambda x: x.clip(upper=0).rolling(60, min_periods=30).std()
        )
        df["up_std"] = df.groupby("symbol")["log_ret"].transform(
            lambda x: x.clip(lower=0).rolling(60, min_periods=30).std()
        )
        df["leverage_ratio"] = df["down_std"] / (df["up_std"] + 1e-8)
        # 高杠杆 = 高风险 = 负面因子，取倒数或负数
        df["leverage"] = 1.0 / (1.0 + df["leverage_ratio"])
        return self._make_series(df, "leverage")
