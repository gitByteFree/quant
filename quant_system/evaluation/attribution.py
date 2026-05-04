"""Brinson归因分析.

将组合超额收益分解为配置效应、选择效应和交互效应.
"""

import numpy as np
import pandas as pd

from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


def brinson_attribution(
    portfolio_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Brinson归因分析.

    超额收益 = 配置效应 + 选择效应 + 交互效应

    R_p - R_b = sum[(w_p_i - w_b_i) * r_b_i]    # 配置效应(Allocation)
              + sum[w_b_i * (r_p_i - r_b_i)]    # 选择效应(Selection)
              + sum[(w_p_i - w_b_i) * (r_p_i - r_b_i)]  # 交互效应(Interaction)

    Args:
        portfolio_weights: 组合行业权重 (periods x industries)
        benchmark_weights: 基准行业权重 (periods x industries)
        portfolio_returns: 组合行业收益 (periods x industries)
        benchmark_returns: 基准行业收益 (periods x industries)

    Returns:
        DataFrame: columns=[allocation, selection, interaction, excess_return]
    """
    common_periods = (
        portfolio_weights.index
        .intersection(benchmark_weights.index)
        .intersection(portfolio_returns.index)
        .intersection(benchmark_returns.index)
    )

    industries = (
        set(portfolio_weights.columns)
        | set(benchmark_weights.columns)
        | set(portfolio_returns.columns)
        | set(benchmark_returns.columns)
    )
    industries = sorted(industries)

    records: list[dict] = []
    for period in common_periods:
        allocation = 0.0
        selection = 0.0
        interaction = 0.0

        for ind in industries:
            w_p = portfolio_weights.loc[period, ind] if ind in portfolio_weights.columns else 0
            w_b = benchmark_weights.loc[period, ind] if ind in benchmark_weights.columns else 0
            r_p = portfolio_returns.loc[period, ind] if ind in portfolio_returns.columns else 0
            r_b = benchmark_returns.loc[period, ind] if ind in benchmark_returns.columns else 0

            allocation += (w_p - w_b) * r_b
            selection += w_b * (r_p - r_b)
            interaction += (w_p - w_b) * (r_p - r_b)

        excess = allocation + selection + interaction
        records.append({
            "period": period,
            "allocation": allocation,
            "selection": selection,
            "interaction": interaction,
            "excess_return": excess,
        })

    result = pd.DataFrame(records).set_index("period")
    return result


def industry_contribution(
    portfolio_weights: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
) -> pd.DataFrame:
    """行业贡献分析.

    计算每个行业对超额收益的贡献.

    Args:
        portfolio_weights: 组合行业权重
        portfolio_returns: 组合行业收益
        benchmark_returns: 基准行业收益

    Returns:
        DataFrame: 每个行业的平均超额配置、超额收益和贡献
    """
    industries = sorted(
        set(portfolio_weights.columns)
        | set(portfolio_returns.columns)
        | set(benchmark_returns.columns)
    )

    contributions: list[dict] = []
    for ind in industries:
        w = portfolio_weights[ind].mean() if ind in portfolio_weights.columns else 0
        r_p = portfolio_returns[ind].mean() if ind in portfolio_returns.columns else 0
        r_b = benchmark_returns[ind].mean() if ind in benchmark_returns.columns else 0

        contributions.append({
            "industry": ind,
            "avg_weight": w,
            "avg_return": r_p,
            "benchmark_return": r_b,
            "excess_return": r_p - r_b,
            "contribution": w * (r_p - r_b),
        })

    df = pd.DataFrame(contributions)
    df = df.sort_values("contribution", ascending=False)
    return df.reset_index(drop=True)


def factor_attribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> dict[str, float]:
    """因子归因（风格分析）.

    用多元回归分解组合收益对各风格因子的暴露.

    Returns = alpha + sum(beta_i * factor_return_i) + epsilon

    Args:
        portfolio_returns: 组合日收益
        factor_returns: 因子日收益 DataFrame (dates x factors)

    Returns:
        {factor_name: beta}
    """
    from sklearn.linear_model import LinearRegression

    common_idx = portfolio_returns.index.intersection(factor_returns.index)
    if len(common_idx) < 20:
        return {}

    y = portfolio_returns.loc[common_idx].values.reshape(-1, 1)
    X = factor_returns.loc[common_idx].values
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y).flatten())
    if valid.sum() < 20:
        return {}

    model = LinearRegression()
    model.fit(X[valid], y[valid])

    betas: dict[str, float] = {}
    for i, name in enumerate(factor_returns.columns):
        betas[name] = float(model.coef_[0][i]) if model.coef_.ndim > 1 else float(model.coef_[i])

    return betas
