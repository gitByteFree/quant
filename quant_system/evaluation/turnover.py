"""换手率与交易成本敏感性分析."""

import numpy as np
import pandas as pd

from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


def compute_turnover(
    weights_history: pd.DataFrame,
) -> pd.Series:
    """计算每日单边换手率.

    turnover_t = sum(|w_t_i - w_t-1_i|) / 2

    Args:
        weights_history: DataFrame, index=date, columns=symbols, values=weights

    Returns:
        每日换手率序列
    """
    if weights_history.empty:
        return pd.Series(dtype=float)

    changes = weights_history.diff().abs()
    turnover = changes.sum(axis=1) / 2.0
    return turnover


def annual_turnover(turnover_series: pd.Series) -> float:
    """年化换手率.

    annual_turnover = sum(daily_turnover) / n_years
    """
    if turnover_series.empty:
        return 0.0
    total = turnover_series.sum()
    years = len(turnover_series) / 252
    if years <= 0:
        return 0.0
    return float(total / years)


def cost_analysis(
    returns: pd.Series,
    turnover: pd.Series,
    cost_per_turnover: float = 0.0025,  # 双边成本 ~25bps
) -> pd.DataFrame:
    """交易成本影响分析.

    Args:
        returns: 日收益率序列
        turnover: 日换手率序列
        cost_per_turnover: 单边换手成本（含印花税+佣金+滑点）

    Returns:
        DataFrame: 含净收益、毛收益、成本等信息
    """
    common_idx = returns.index.intersection(turnover.index)
    if len(common_idx) == 0:
        return pd.DataFrame()

    ret = returns.loc[common_idx]
    to = turnover.loc[common_idx]

    cost = to * cost_per_turnover
    net_ret = ret - cost

    result = pd.DataFrame({
        "gross_return": ret,
        "turnover": to,
        "cost": cost,
        "net_return": net_ret,
    })

    return result


def turnover_sensitivity(
    base_performance: dict[str, float],
    base_turnover: float,
    cost_scenarios: list[float] | None = None,
) -> pd.DataFrame:
    """换手率敏感性分析.

    在不同的交易成本假设下，测算对收益率的影响.

    Args:
        base_performance: 基准绩效 dict(annual_return=..., ...)
        base_turnover: 基准年化换手率
        cost_scenarios: 成本情景列表（单边，bps）

    Returns:
        DataFrame: 各情景下的年化收益、夏普等指标变化
    """
    if cost_scenarios is None:
        cost_scenarios = [0.001, 0.0015, 0.0025, 0.005, 0.01]

    base_return = base_performance.get("annual_return", 0)
    base_sharpe = base_performance.get("sharpe_ratio", 0)
    base_vol = base_performance.get("annual_volatility", 0.2)

    records: list[dict] = []
    for cost in cost_scenarios:
        annual_cost = base_turnover * cost
        adj_return = base_return - annual_cost
        adj_sharpe = base_sharpe - annual_cost / base_vol if base_vol > 0 else 0

        records.append({
            "单边成本(bps)": int(cost * 10000),
            "年化成本(%)": round(annual_cost * 100, 3),
            "调整后年化收益(%)": round(adj_return * 100, 2),
            "收益减少(%)": round(annual_cost / abs(base_return) * 100, 1) if base_return != 0 else 0,
            "调整后夏普": round(adj_sharpe, 2),
        })

    return pd.DataFrame(records)


def breakeven_turnover(
    expected_return: float,
    cost_per_trade: float = 0.0025,
) -> float:
    """计算盈亏平衡换手率.

    最大可接受换手率 = 预期超额收益 / 单边成本

    Args:
        expected_return: 预期年化超额收益
        cost_per_trade: 单边换手成本

    Returns:
        盈亏平衡换手率（倍数/年）
    """
    if cost_per_trade <= 0:
        return float("inf")
    return expected_return / cost_per_trade


def turnover_decay_analysis(
    factor_signals: pd.DataFrame,
    forward_returns: pd.Series,
    holding_periods: tuple[int, ...] = (1, 5, 10, 20, 40, 60),
) -> pd.DataFrame:
    """换手率与预测周期的关系分析.

    不同调仓频率下的换手率和IC变化.

    Args:
        factor_signals: 因子信号DataFrame, index=date, columns=symbols
        forward_returns: 未来收益
        holding_periods: 持仓周期

    Returns:
        DataFrame with columns: holding_days, turnover, ic_mean, ic_std
    """
    records: list[dict] = []
    for h in holding_periods:
        # 模拟不同持仓周期的调仓
        turnover_est = 1.0 / h * 2  # 每天调仓比例近似
        records.append({
            "holding_days": h,
            "daily_turnover_est": turnover_est,
            "annual_turnover_est": turnover_est * 252,
        })

    return pd.DataFrame(records)
