"""因子评估器.

IC分析、Fama-MacBeth回归、分层回测、因子衰减分析.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ICRsult:
    """IC分析结果."""

    factor_name: str
    ic_mean: float
    ic_std: float
    ic_ir: float  # IC信息比率 = ic_mean / ic_std
    ic_positive_ratio: float  # IC>0的天数占比
    ic_t_stat: float
    ic_p_value: float
    ic_series: pd.Series  # 每日IC序列


def compute_ic(
    factor: pd.Series,
    forward_return: pd.Series,
    method: str = "pearson",
) -> ICRsult:
    """计算因子IC（信息系数）.

    对每个交易日截面计算因子值与未来收益的相关性.

    Args:
        factor: MultiIndex (trade_date, symbol) 的因子值
        forward_return: MultiIndex (trade_date, symbol) 的未来收益
        method: pearson | spearman

    Returns:
        ICRsult 对象
    """
    factor_name = factor.name if hasattr(factor, "name") else "unknown"
    common_dates = factor.index.get_level_values("trade_date").intersection(
        forward_return.index.get_level_values("trade_date")
    )
    ic_records: list[dict] = []

    for date in sorted(common_dates.unique()):
        f_cross = factor.loc[date]
        r_cross = forward_return.loc[date]
        common = f_cross.index.intersection(r_cross.index)
        if len(common) < 10:
            continue
        f = f_cross.loc[common].astype(float)
        r = r_cross.loc[common].astype(float)
        valid = ~(f.isna() | r.isna() | np.isinf(f) | np.isinf(r))
        if valid.sum() < 10:
            continue
        f = f[valid]
        r = r[valid]

        if method == "spearman":
            ic_val = stats.spearmanr(f, r)[0]
        else:
            ic_val = np.corrcoef(f, r)[0, 1]
        ic_records.append({"trade_date": date, "ic": ic_val})

    ic_series = pd.DataFrame(ic_records).set_index("trade_date")["ic"]
    if ic_series.empty:
        return ICRsult(
            factor_name=factor_name,
            ic_mean=0, ic_std=0, ic_ir=0, ic_positive_ratio=0,
            ic_t_stat=0, ic_p_value=1, ic_series=ic_series,
        )

    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    n = len(ic_series)
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    ic_t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 and n > 0 else 0
    ic_p_value = 2 * (1 - stats.t.cdf(abs(ic_t_stat), n - 1)) if n > 1 else 1

    return ICRsult(
        factor_name=factor_name,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_ir=ic_ir,
        ic_positive_ratio=(ic_series > 0).mean(),
        ic_t_stat=ic_t_stat,
        ic_p_value=ic_p_value,
        ic_series=ic_series,
    )


@dataclass
class FamaMacBethResult:
    """Fama-MacBeth回归结果."""

    factor_names: list[str]
    risk_premiums: dict[str, float]       # 每个因子的风险溢价
    t_stats: dict[str, float]            # t统计量
    p_values: dict[str, float]           # p值
    mean_r_squared: float                # 平均R²


def fama_macbeth(
    factor_panel: pd.DataFrame,
    forward_return: pd.Series,
) -> FamaMacBethResult:
    """Fama-MacBeth截面回归.

    两步法：
    1. 每期截面回归：ret_i = alpha + sum(beta_j * factor_j_i) + eps_i
    2. 时间序列平均系数即为风险溢价

    Args:
        factor_panel: columns=因子名, index=MultiIndex (trade_date, symbol)
        forward_return: MultiIndex (trade_date, symbol)

    Returns:
        FamaMacBethResult
    """
    factor_names = list(factor_panel.columns)
    premiums: dict[str, list[float]] = {f: [] for f in factor_names}
    r2_list: list[float] = []

    common_dates = factor_panel.index.get_level_values("trade_date").intersection(
        forward_return.index.get_level_values("trade_date")
    )

    for date in sorted(common_dates.unique()):
        cross_x = factor_panel.loc[date]
        cross_y = forward_return.loc[date]
        common = cross_x.index.intersection(cross_y.index)
        if len(common) < len(factor_names) + 10:
            continue

        X = cross_x.loc[common].values.astype(float)
        y = cross_y.loc[common].values.astype(float)
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        if valid.sum() < len(factor_names) + 10:
            continue

        model = LinearRegression()
        model.fit(X[valid], y[valid])
        for i, name in enumerate(factor_names):
            premiums[name].append(model.coef_[i])
        r2_list.append(model.score(X[valid], y[valid]))

    result_premiums: dict[str, float] = {}
    result_t: dict[str, float] = {}
    result_p: dict[str, float] = {}

    for name in factor_names:
        series = pd.Series(premiums[name])
        result_premiums[name] = series.mean()
        if len(series) > 1:
            result_t[name] = series.mean() / (series.std() / np.sqrt(len(series)))
            result_p[name] = 2 * (1 - stats.t.cdf(abs(result_t[name]), len(series) - 1))
        else:
            result_t[name] = 0
            result_p[name] = 1

    return FamaMacBethResult(
        factor_names=factor_names,
        risk_premiums=result_premiums,
        t_stats=result_t,
        p_values=result_p,
        mean_r_squared=np.mean(r2_list) if r2_list else 0,
    )


def stratified_backtest(
    factor: pd.Series,
    forward_return: pd.Series,
    n_groups: int = 10,
) -> pd.DataFrame:
    """分层回测.

    每个交易日按因子值将股票分为n_groups组，计算每组的等权收益.

    Args:
        factor: MultiIndex (trade_date, symbol)
        forward_return: MultiIndex (trade_date, symbol)
        n_groups: 分组数量

    Returns:
        DataFrame: columns=group_i, index=trade_date, values=每组等权收益
    """
    common_dates = factor.index.get_level_values("trade_date").intersection(
        forward_return.index.get_level_values("trade_date")
    )
    results: dict[int, list[dict]] = {}

    for date in sorted(common_dates.unique()):
        f_cross = factor.loc[date].dropna()
        r_cross = forward_return.loc[date].dropna()
        common = f_cross.index.intersection(r_cross.index)
        if len(common) < n_groups * 3:
            continue
        f = f_cross.loc[common]
        r = r_cross.loc[common]

        # 按因子值排序分组
        labels = pd.qcut(f, n_groups, labels=False, duplicates="drop")
        for g in range(n_groups):
            mask = labels == g
            if mask.sum() == 0:
                continue
            group_ret = r[mask].mean()
            if g not in results:
                results[g] = []
            results[g].append({"trade_date": date, "return": group_ret})

    # 转换为DataFrame
    series_dict: dict[str, pd.Series] = {}
    for g in range(n_groups):
        if g in results:
            df_g = pd.DataFrame(results[g]).set_index("trade_date")["return"]
            series_dict[f"group_{g + 1}"] = df_g

    if not series_dict:
        return pd.DataFrame()
    result = pd.DataFrame(series_dict)
    result["long_short"] = result[f"group_{n_groups}"] - result["group_1"]
    return result


def factor_decay_analysis(
    factor: pd.Series,
    returns: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20, 40, 60),
) -> pd.DataFrame:
    """因子衰减分析.

    计算因子在不同预测周期上的IC衰减.

    Args:
        factor: MultiIndex (trade_date, symbol)
        returns: 每日收益率DataFrame，含 trade_date, symbol, ret 列
        horizons: 预测周期（交易日）

    Returns:
        DataFrame: index=horizon, columns=[ic_mean, ic_std, ic_ir]
    """
    records: list[dict] = []
    for horizon in horizons:
        # 构建未来h日收益
        if "symbol" in returns.columns and "trade_date" in returns.columns:
            ret_flat = returns.set_index(["trade_date", "symbol"])["ret"]
        else:
            ret_flat = returns

        # 计算未来h日累计收益
        forward_ret = ret_flat.groupby(level="symbol").transform(
            lambda x: x.shift(-horizon).rolling(horizon).sum()
        ).dropna()

        ic_result = compute_ic(factor, forward_ret)
        records.append({
            "horizon": horizon,
            "ic_mean": ic_result.ic_mean,
            "ic_std": ic_result.ic_std,
            "ic_ir": ic_result.ic_ir,
        })

    result = pd.DataFrame(records).set_index("horizon")
    # 计算半衰期：IC均值衰减到一半的周期
    if len(result) > 1 and abs(result["ic_mean"].iloc[0]) > 1e-8:
        initial_ic = result["ic_mean"].iloc[0]
        half_ic = initial_ic / 2
        for h in horizons:
            if abs(result.loc[h, "ic_mean"]) <= abs(half_ic):
                result.attrs["half_life"] = h
                break

    return result


def factor_summary(
    factor: pd.Series,
    forward_return: pd.Series,
    factor_name: str = "",
) -> dict[str, Any]:
    """生成因子综合评估摘要.

    Args:
        factor: MultiIndex (trade_date, symbol)
        forward_return: MultiIndex (trade_date, symbol)
        factor_name: 因子名称

    Returns:
        包含IC、分层回测、覆盖率等指标的字典
    """
    name = factor_name or (factor.name if hasattr(factor, "name") else "unknown")

    ic = compute_ic(factor, forward_return)
    strat = stratified_backtest(factor, forward_return, n_groups=10)

    coverage = factor.notna().sum() / len(factor) if len(factor) > 0 else 0
    n_stocks_per_day = factor.groupby(level="trade_date").size().mean()

    long_short_ret = 0.0
    long_short_std = 0.0
    if not strat.empty and "long_short" in strat.columns:
        long_short_ret = strat["long_short"].mean()
        long_short_std = strat["long_short"].std()

    return {
        "factor_name": name,
        "ic_mean": ic.ic_mean,
        "ic_std": ic.ic_std,
        "ic_ir": ic.ic_ir,
        "ic_positive_ratio": ic.ic_positive_ratio,
        "ic_t_stat": ic.ic_t_stat,
        "ic_p_value": ic.ic_p_value,
        "long_short_daily_ret": long_short_ret,
        "long_short_daily_std": long_short_std,
        "long_short_sharpe": long_short_ret / long_short_std if long_short_std > 0 else 0,
        "coverage": coverage,
        "avg_stocks_per_day": n_stocks_per_day,
        "top_group_ret": strat["group_10"].mean() if "group_10" in strat.columns else 0,
        "bottom_group_ret": strat["group_1"].mean() if "group_1" in strat.columns else 0,
    }
