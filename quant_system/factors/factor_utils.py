"""因子预处理工具函数.

MAD去极值、行业+市值中性化、Z-score标准化.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from quant_system.utils.constants import MAD_THRESHOLD, ZSCORE_EPS


def mad_clip(series: pd.Series, threshold: float = MAD_THRESHOLD) -> pd.Series:
    """MAD法去极值.

    value > median + threshold * MAD → 截断到上界
    value < median - threshold * MAD → 截断到下界

    Args:
        series: 因子值序列
        threshold: MAD倍数，默认3.0

    Returns:
        去极值后的序列
    """
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return series
    upper = median + threshold * mad
    lower = median - threshold * mad
    return series.clip(lower=lower, upper=upper)


def winsorize(series: pd.Series, limits: tuple[float, float] = (0.01, 0.01)) -> pd.Series:
    """分位数缩尾.

    Args:
        series: 因子值序列
        limits: (下分位数, 上分位数)

    Returns:
        缩尾后的序列
    """
    lower = series.quantile(limits[0])
    upper = series.quantile(1 - limits[1])
    return series.clip(lower=lower, upper=upper)


def standardize_zscore(series: pd.Series, eps: float = ZSCORE_EPS) -> pd.Series:
    """Z-score标准化（截面）.

    对每个交易日截面进行Z-score标准化.

    Args:
        series: MultiIndex (trade_date, symbol) 的因子值
        eps: 防止除零

    Returns:
        标准化后的序列
    """
    grouped = series.groupby(level="trade_date")
    return grouped.transform(lambda x: (x - x.mean()) / (x.std() + eps))


def standardize_rank(series: pd.Series) -> pd.Series:
    """Rank标准化（截面）.

    每个交易日截面内按因子值排序，转换为0-1之间的分位数.

    Args:
        series: MultiIndex (trade_date, symbol) 的因子值

    Returns:
        0-1分位数序列
    """
    grouped = series.groupby(level="trade_date")
    return grouped.transform(lambda x: x.rank(pct=True))


def minmax_scale(series: pd.Series) -> pd.Series:
    """Min-Max缩放（截面）."""
    grouped = series.groupby(level="trade_date")
    return grouped.transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + ZSCORE_EPS)
    )


def neutralize_industry(
    factor: pd.Series,
    industry_map: pd.Series,
) -> pd.Series:
    """行业中性化.

    因子值对行业哑变量做截面回归，取残差.

    Args:
        factor: MultiIndex (trade_date, symbol) 的因子值
        industry_map: index=symbol, 值为行业名称的Series

    Returns:
        行业中性化后的因子残差
    """
    factor = factor.copy()
    for date, _ in factor.groupby(level="trade_date"):
        cross = factor.loc[date]
        symbols = cross.index
        inds = industry_map.reindex(symbols).fillna("未知")
        dummies = pd.get_dummies(inds)
        if dummies.shape[1] <= 1:
            continue
        X = dummies.values.astype(float)
        y = cross.values.astype(float)
        valid = ~np.isnan(y)
        if valid.sum() < dummies.shape[1] + 5:
            continue
        model = LinearRegression()
        model.fit(X[valid], y[valid])
        pred = model.predict(X)
        factor.loc[date] = cross - pred
    return factor


def neutralize_market_cap(
    factor: pd.Series,
    market_cap: pd.Series,
) -> pd.Series:
    """市值中性化.

    因子值对log(市值)做截面回归，取残差.

    Args:
        factor: MultiIndex (trade_date, symbol) 的因子值
        market_cap: MultiIndex (trade_date, symbol) 的市值数据

    Returns:
        市值中性化后的因子残差
    """
    factor = factor.copy()
    common_dates = factor.index.get_level_values("trade_date").intersection(
        market_cap.index.get_level_values("trade_date")
    )
    for date in common_dates.unique():
        cross_f = factor.loc[date]
        cross_m = market_cap.loc[date]
        common_symbols = cross_f.index.intersection(cross_m.index)
        if len(common_symbols) < 10:
            continue
        y = cross_f.loc[common_symbols].values.astype(float)
        x = np.log(cross_m.loc[common_symbols].values.astype(float) + 1).reshape(-1, 1)
        valid = ~np.isnan(y)
        if valid.sum() < 5:
            continue
        model = LinearRegression()
        model.fit(x[valid], y[valid])
        pred = model.predict(x)
        factor.loc[date, common_symbols] = y - pred
    return factor


def neutralize_industry_market_cap(
    factor: pd.Series,
    industry_map: pd.Series,
    market_cap: pd.Series,
) -> pd.Series:
    """行业+市值双重中性化.

    因子值对行业哑变量+log(市值)做截面回归，取残差.

    Args:
        factor: MultiIndex (trade_date, symbol)
        industry_map: index=symbol, 值为行业
        market_cap: MultiIndex (trade_date, symbol)

    Returns:
        双重中性化后的因子残差
    """
    factor = factor.copy()
    common_dates = factor.index.get_level_values("trade_date").intersection(
        market_cap.index.get_level_values("trade_date")
    )
    for date in common_dates.unique():
        cross_f = factor.loc[date]
        cross_m = market_cap.loc[date]
        common = cross_f.index.intersection(cross_m.index)
        if len(common) < 15:
            continue
        y = cross_f.loc[common].values.astype(float)
        inds = industry_map.reindex(common).fillna("未知")
        dummies = pd.get_dummies(inds)
        log_mcap = np.log(cross_m.loc[common].values.astype(float) + 1).reshape(-1, 1)
        X = np.hstack([dummies.values.astype(float), log_mcap])
        valid = ~np.isnan(y)
        if valid.sum() < X.shape[1] + 5:
            continue
        model = LinearRegression()
        model.fit(X[valid], y[valid])
        pred = model.predict(X)
        factor.loc[date, common] = y - pred
    return factor


STANDARDIZE_METHODS = {
    "zscore": standardize_zscore,
    "rank": standardize_rank,
    "minmax": minmax_scale,
}


def preprocess_factor(
    factor: pd.Series,
    clip_method: str = "mad",
    clip_threshold: float = MAD_THRESHOLD,
    industry_map: pd.Series | None = None,
    market_cap: pd.Series | None = None,
    standardize: str = "zscore",
) -> pd.Series:
    """一站式因子预处理.

    Args:
        factor: MultiIndex (trade_date, symbol) 原始因子值
        clip_method: 去极值方法 mad | winsorize
        clip_threshold: 去极值阈值
        industry_map: 行业映射 Series (index=symbol)
        market_cap: 市值数据 Series (MultiIndex trade_date, symbol)
        standardize: 标准化方法 zscore | rank | minmax

    Returns:
        预处理后的因子值
    """
    result = factor.copy()

    # 1. 去极值
    if clip_method == "mad":
        grouped = result.groupby(level="trade_date")
        result = grouped.transform(lambda x: mad_clip(x, clip_threshold))
    elif clip_method == "winsorize":
        grouped = result.groupby(level="trade_date")
        result = grouped.transform(lambda x: winsorize(x))

    # 2. 中性化
    if industry_map is not None and market_cap is not None:
        result = neutralize_industry_market_cap(result, industry_map, market_cap)
    elif industry_map is not None:
        result = neutralize_industry(result, industry_map)
    elif market_cap is not None:
        result = neutralize_market_cap(result, market_cap)

    # 3. 标准化
    if standardize in STANDARDIZE_METHODS:
        result = STANDARDIZE_METHODS[standardize](result)

    return result
