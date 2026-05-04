"""市场分段分析.

牛熊市分段表现评估（2015牛市/2018熊市/2020结构市）.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_system.evaluation.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    win_rate,
    compute_returns,
)
from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SegmentResult:
    """单段市场表现."""

    name: str
    start: str
    end: str
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    n_days: int


def analyze_segments(
    portfolio_values: pd.Series,
    segments: dict[str, tuple[str, str]],
) -> list[SegmentResult]:
    """分析多个时间段的表现.

    Args:
        portfolio_values: 组合净值序列（index为日期）
        segments: {segment_name: (start_date, end_date)}

    Returns:
        SegmentResult列表
    """
    returns = compute_returns(portfolio_values)
    results: list[SegmentResult] = []

    for name, (start, end) in segments.items():
        seg_ret = returns.loc[start:end]
        if seg_ret.empty:
            logger.warning("分段 %s (%s~%s) 无数据", name, start, end)
            continue
        seg_values = portfolio_values.loc[start:end]
        total_ret = seg_values.iloc[-1] / seg_values.iloc[0] - 1 if len(seg_values) > 1 else 0

        results.append(SegmentResult(
            name=name,
            start=start,
            end=end,
            total_return=float(total_ret),
            annual_return=annualized_return(seg_ret),
            annual_volatility=annualized_volatility(seg_ret),
            sharpe=sharpe_ratio(seg_ret),
            max_drawdown=max_drawdown(seg_ret),
            win_rate=win_rate(seg_ret),
            n_days=len(seg_ret),
        ))

    return results


def to_dataframe(results: list[SegmentResult]) -> pd.DataFrame:
    """将分段结果转换为DataFrame."""
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([{
        "市场阶段": r.name,
        "起始": r.start,
        "结束": r.end,
        "交易天数": r.n_days,
        "总收益率(%)": round(r.total_return * 100, 2),
        "年化收益率(%)": round(r.annual_return * 100, 2),
        "年化波动率(%)": round(r.annual_volatility * 100, 2),
        "夏普比率": round(r.sharpe, 2),
        "最大回撤(%)": round(r.max_drawdown * 100, 2),
        "日胜率(%)": round(r.win_rate * 100, 1),
    } for r in results])


# 预定义A股关键分段
A_SHARE_SEGMENTS: dict[str, tuple[str, str]] = {
    "2015上半年牛市": ("2015-01-05", "2015-06-12"),
    "2015下半年暴跌": ("2015-06-15", "2015-12-31"),
    "2016熔断修复": ("2016-01-04", "2016-12-30"),
    "2017白马蓝筹": ("2017-01-03", "2017-12-29"),
    "2018熊市": ("2018-01-02", "2018-12-28"),
    "2019反弹": ("2019-01-02", "2019-12-31"),
    "2020结构市": ("2020-01-02", "2020-12-31"),
    "2021震荡": ("2021-01-04", "2021-12-31"),
    "2022熊市": ("2022-01-04", "2022-12-30"),
    "2023震荡": ("2023-01-03", "2023-12-29"),
    "2024修复": ("2024-01-02", "2024-12-31"),
}


def rolling_analysis(
    portfolio_values: pd.Series,
    benchmark_values: pd.Series | None = None,
    window_days: int = 252,
) -> pd.DataFrame:
    """滚动窗口分析.

    计算滚动年化收益、波动率、夏普比率.

    Args:
        portfolio_values: 组合净值序列
        benchmark_values: 基准净值序列
        window_days: 滚动窗口（交易日）

    Returns:
        DataFrame with columns: rolling_ret, rolling_vol, rolling_sharpe, rolling_excess
    """
    returns = compute_returns(portfolio_values)

    rolling_ret = returns.rolling(window_days).apply(
        lambda x: (1 + x).prod() ** (252 / len(x)) - 1
    )
    rolling_vol = returns.rolling(window_days).std() * np.sqrt(252)
    rolling_sharpe = rolling_ret / rolling_vol

    result = pd.DataFrame({
        "rolling_ret": rolling_ret,
        "rolling_vol": rolling_vol,
        "rolling_sharpe": rolling_sharpe,
    })

    if benchmark_values is not None:
        bench_returns = compute_returns(benchmark_values)
        excess = returns - bench_returns.reindex(returns.index).fillna(0)
        rolling_excess = excess.rolling(window_days).mean() * 252
        result["rolling_excess"] = rolling_excess

    return result.dropna()


def bull_bear_analysis(
    portfolio_values: pd.Series,
    benchmark_values: pd.Series,
) -> dict[str, list[SegmentResult]]:
    """自动识别牛熊市并分析.

    简单规则：
    - 牛市：基准20日均线上行且组合累计收益为正
    - 熊市：基准20日均线下行且组合累计收益为负
    - 震荡：其余时期

    Returns:
        {"bull": [...], "bear": [...], "sideways": [...]}
    """
    bench_returns = compute_returns(benchmark_values)
    bench_ma20 = benchmark_values.rolling(20).mean()
    bench_trend = bench_ma20.pct_change(20)  # 20日均线的变化率

    returns = compute_returns(portfolio_values)
    cum_ret = (1 + returns).cumprod()

    bull_mask = (bench_trend > 0.005) & (cum_ret > cum_ret.shift(60))
    bear_mask = (bench_trend < -0.005) & (cum_ret < cum_ret.shift(60))
    sideways_mask = ~(bull_mask | bear_mask)

    segments: dict[str, pd.Series] = {
        "bull": returns[bull_mask],
        "bear": returns[bear_mask],
        "sideways": returns[sideways_mask],
    }

    results: dict[str, list[SegmentResult]] = {}
    for regime, seg_returns in segments.items():
        if seg_returns.empty:
            results[regime] = []
            continue
        start = str(seg_returns.index[0].date())
        end = str(seg_returns.index[-1].date())
        seg_values = portfolio_values.loc[start:end]
        total_ret = seg_values.iloc[-1] / seg_values.iloc[0] - 1 if len(seg_values) > 1 else 0
        results[regime] = [SegmentResult(
            name=regime,
            start=start,
            end=end,
            total_return=float(total_ret),
            annual_return=annualized_return(seg_returns),
            annual_volatility=annualized_volatility(seg_returns),
            sharpe=sharpe_ratio(seg_returns),
            max_drawdown=max_drawdown(seg_returns),
            win_rate=win_rate(seg_returns),
            n_days=len(seg_returns),
        )]

    return results
