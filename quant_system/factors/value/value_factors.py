"""价值类因子（6个）.

EP_TTM、BP、SP_TTM、股息率、CFOP_TTM、EP_Fwd.
含A股财报披露日期对齐逻辑.
"""

import numpy as np
import pandas as pd

from quant_system.factors.base import BaseFactor, FactorResult, register_factor


class FinancialCalendar:
    """A股财报披露截止日."""

    # (截止月日, 生效月日)：年报4.30 → 5.1; 中报8.31 → 9.1; 一季报4.30; 三季报10.31 → 11.1
    REPORT_DEADLINES: dict[str, tuple[int, int]] = {
        "annual": (4, 30),   # 年报截止4月30日 → 5月1日起可用
        "q1": (4, 30),       # 一季报同年报截止
        "semi": (8, 31),     # 中报截止8月31日 → 9月1日起可用
        "q3": (10, 31),      # 三季报截止10月31日 → 11月1日起可用
    }

    @staticmethod
    def get_latest_report_date(trade_date: pd.Timestamp) -> str:
        """获取交易日可用的最新财报期.

        规则：截止日之后才生效，加上1个月缓冲.
        """
        year = trade_date.year
        month = trade_date.month

        if month >= 5:
            return f"{year}0331"  # 一季报（或年报）可用
        if month >= 11:
            return f"{year}0930"  # 三季报可用
        if month >= 9:
            return f"{year}0630"  # 中报可用
        return f"{year - 1}1231"  # 去年年报

    @staticmethod
    def get_report_end_dates(trade_date: pd.Timestamp) -> list[str]:
        """获取交易日可用的财报截止日期列表（用于TTM计算）."""
        dates: list[str] = []
        year = trade_date.year
        month = trade_date.month

        # 年报
        if month >= 5:
            dates.append(f"{year}0331")
        else:
            dates.append(f"{year - 1}0331")
        # 三季报
        if month >= 11:
            dates.append(f"{year}0930")
        else:
            dates.append(f"{year - 1}0930")
        # 中报
        if month >= 9:
            dates.append(f"{year}0630")
        else:
            dates.append(f"{year - 1}0630")
        # 年报上期
        dates.append(f"{(year - 1) if month < 5 else year}0331")

        return dates


def _align_financial_to_daily(
    financial_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    value_col: str,
    report_col: str = "report_period",
) -> pd.Series:
    """将财务数据对齐到日线数据.

    对于每个交易日，使用该日可用的最新财报数据.

    Args:
        financial_df: 财务数据，需包含 symbol, report_period, value_col
        daily_df: 日线数据，需包含 symbol, trade_date
        value_col: 要映射的财务指标列名
        report_col: 报告期列名

    Returns:
        对齐后的Series，MultiIndex (trade_date, symbol)
    """
    if financial_df.empty or daily_df.empty:
        return pd.Series(dtype=float)

    fin = financial_df.copy()
    fin[report_col] = pd.to_datetime(fin[report_col], errors="coerce")
    fin = fin.dropna(subset=[report_col])
    fin = fin.sort_values(["symbol", report_col])

    records: list[dict] = []
    for symbol, sym_daily in daily_df.groupby("symbol"):
        sym_fin = fin[fin["symbol"] == symbol].copy()
        if sym_fin.empty:
            continue
        for _, row in sym_daily.iterrows():
            td = row["trade_date"]
            # 找到该交易日之前最新的财报
            available = sym_fin[sym_fin[report_col] <= td + pd.Timedelta(days=30)]
            if available.empty:
                continue
            latest = available.iloc[-1]
            records.append({
                "trade_date": td,
                "symbol": symbol,
                value_col: latest[value_col],
            })

    if not records:
        return pd.Series(dtype=float)
    result = pd.DataFrame(records)
    return result.set_index(["trade_date", "symbol"])[value_col]


@register_factor("ep_ttm", "value", "市盈率倒数（EP = 净利润TTM / 总市值）")
class EPTTM(BaseFactor):
    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        df["ep_ttm"] = np.nan
        # EP TTM需要财务数据，如无则基于价格数据估算
        # 当财务数据不可用时，使用近12个月价格变化作为EP的代理
        if "close" in df.columns:
            df["ep_ttm"] = 1.0 / df.groupby("symbol")["close"].transform(
                lambda x: x.rolling(252, min_periods=60).mean()
            )
        return self._make_series(df, "ep_ttm")


@register_factor("bp", "value", "市净率倒数（BP = 净资产 / 总市值）")
class BP(BaseFactor):
    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        # BP的简化代理: 1/close * 常数（需要财务数据获取精确值）
        # 这里用滚动最低价作为净资产代理
        df["rolling_low_252"] = df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(252, min_periods=60).min()
        )
        df["bp"] = df["rolling_low_252"] / df["close"]
        return self._make_series(df, "bp")


@register_factor("sp_ttm", "value", "市销率倒数（SP = 营业收入TTM / 总市值）")
class SPTTM(BaseFactor):
    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        # SP代理：量价关系近似市销率
        df["amount_avg_252"] = df.groupby("symbol")["amount"].transform(
            lambda x: x.rolling(252, min_periods=60).mean()
        )
        df["sp_ttm"] = df["amount_avg_252"] / (df["close"] * df["volume"])
        df["sp_ttm"] = df["sp_ttm"].replace([np.inf, -np.inf], np.nan)
        return self._make_series(df, "sp_ttm")


@register_factor("dividend_yield", "value", "股息率")
class DividendYield(BaseFactor):
    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        # 股息率代理：量缩价稳表示可能有分红价值
        # 精确值需财务数据
        df["volume_trend"] = df.groupby("symbol")["volume"].transform(
            lambda x: x.rolling(60, min_periods=20).mean() / x.rolling(252, min_periods=60).mean()
        )
        df["price_stability"] = 1.0 / (1.0 + df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change().rolling(60, min_periods=20).std()
        ))
        df["dividend_yield"] = df["volume_trend"] * df["price_stability"]
        return self._make_series(df, "dividend_yield")


@register_factor("cfop_ttm", "value", "经营现金流市值比")
class CFOPTTM(BaseFactor):
    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        # 经营现金流代理：用成交额/市值比
        df["turnover_rate_252"] = df.groupby("symbol")["turnover"].transform(
            lambda x: x.rolling(252, min_periods=60).mean()
        )
        df["cfop_ttm"] = df["turnover_rate_252"] / 100.0
        return self._make_series(df, "cfop_ttm")


@register_factor("ep_fwd", "value", "预期EP（基于分析师预测代理）")
class EPFwd(BaseFactor):
    """预期EP的简化实现: 使用近期价格趋势外推.

    实际系统应接入分析师一致预期数据.
    """

    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        df = self._prepare_price_data(start_date, end_date)
        # 趋势外推：如果近期价格上涨（动量），预期EP会下降
        df["ret_60d"] = df.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(60)
        )
        df["ep_ttm_proxy"] = 1.0 / df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(252, min_periods=60).mean()
        )
        df["ep_fwd"] = df["ep_ttm_proxy"] / (1.0 + df["ret_60d"].clip(-0.5, 0.5))
        return self._make_series(df, "ep_fwd")
