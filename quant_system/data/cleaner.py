"""数据清洗与预处理.

处理停牌、ST标记、涨跌停检测、缺失值插补.
"""

import numpy as np
import pandas as pd

from quant_system.utils.constants import FORWARD_FILL_LIMIT, MAD_THRESHOLD
from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


def filter_st_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """过滤ST股票.

    Args:
        df: 包含 is_st 列的DataFrame

    Returns:
        过滤掉ST股票后的DataFrame
    """
    if "is_st" not in df.columns:
        logger.warning("缺少 is_st 列，跳过ST过滤")
        return df
    n_before = len(df)
    result = df[df["is_st"] != 1].copy()
    logger.debug("ST过滤: %d -> %d 条记录", n_before, len(result))
    return result


def detect_suspended_stocks(
    df: pd.DataFrame,
    threshold_days: int = 60,
) -> pd.DataFrame:
    """标记长期停牌股票.

    如果一只股票在一段时间内没有交易记录（连续停牌超过threshold_days），
    将其标记为长期停牌.

    Args:
        df: 日线DataFrame，需包含 symbol, trade_date, volume
        threshold_days: 停牌阈值天数

    Returns:
        添加 is_long_suspended 标记的DataFrame
    """
    df = df.sort_values(["symbol", "trade_date"]).copy()
    df["trade_gap"] = df.groupby("symbol")["trade_date"].diff().dt.days
    df["is_long_suspended"] = df["trade_gap"] > threshold_days
    return df


def detect_limit_up_down(
    df: pd.DataFrame,
    limit_pct: float = 0.10,
) -> pd.DataFrame:
    """检测涨跌停状态.

    通过比较当日价格与前一日收盘价判断涨跌停.

    Args:
        df: 日线DataFrame，需包含 symbol, trade_date, close, pre_close
        limit_pct: 涨跌停幅度（主板10%，科创/创业板20%）

    Returns:
        添加 is_limit_up, is_limit_down 标记的DataFrame
    """
    df = df.sort_values(["symbol", "trade_date"]).copy()
    df["pre_close_shift"] = df.groupby("symbol")["close"].shift(1)
    df["ret"] = df["close"] / df["pre_close_shift"] - 1
    df["is_limit_up"] = df["ret"] >= limit_pct * 0.99
    df["is_limit_down"] = df["ret"] <= -limit_pct * 0.99
    df.drop(columns=["pre_close_shift", "ret"], inplace=True)
    return df


def forward_fill_by_symbol(
    df: pd.DataFrame,
    columns: list[str],
    limit: int = FORWARD_FILL_LIMIT,
) -> pd.DataFrame:
    """按股票进行前向填充缺失值.

    Args:
        df: 日线DataFrame
        columns: 需要填充的列
        limit: 最大填充天数

    Returns:
        填充后的DataFrame
    """
    df = df.sort_values(["symbol", "trade_date"]).copy()
    for col in columns:
        if col in df.columns:
            df[col] = df.groupby("symbol")[col].ffill(limit=limit)
    return df


def impute_with_industry_median(
    df: pd.DataFrame,
    target_col: str,
    industry_col: str = "industry",
) -> pd.DataFrame:
    """用行业中位数填补剩余缺失值.

    Args:
        df: DataFrame
        target_col: 要填补的列
        industry_col: 行业分类列

    Returns:
        填补后的DataFrame
    """
    df = df.copy()
    if target_col not in df.columns:
        return df
    if industry_col in df.columns:
        industry_medians = df.groupby(industry_col)[target_col].transform("median")
        df[target_col] = df[target_col].fillna(industry_medians)
    # 最后用全局中位数填补
    global_median = df[target_col].median()
    df[target_col] = df[target_col].fillna(global_median)
    return df


def remove_mad_outliers(
    series: pd.Series,
    threshold: float = MAD_THRESHOLD,
) -> pd.Series:
    """MAD法去极值.

    将超出 median ± threshold * MAD 范围的值截断到边界.

    Args:
        series: 输入序列
        threshold: MAD倍数阈值

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


def clean_daily_data(
    df: pd.DataFrame,
    remove_st: bool = True,
    remove_suspend_days: int = 60,
    forward_fill_cols: list[str] | None = None,
) -> pd.DataFrame:
    """一站式日线数据清洗.

    Args:
        df: 原始日线DataFrame
        remove_st: 是否移除ST
        remove_suspend_days: 停牌阈值
        forward_fill_cols: 需要前向填充的列

    Returns:
        清洗后的DataFrame
    """
    logger.info("开始数据清洗: 输入 %d 条记录", len(df))

    if remove_st:
        df = filter_st_stocks(df)

    df = detect_suspended_stocks(df, threshold_days=remove_suspend_days)
    df = detect_limit_up_down(df)

    cols_to_fill = forward_fill_cols or ["volume", "amount", "turnover"]
    df = forward_fill_by_symbol(df, columns=cols_to_fill)

    if "turnover" in df.columns:
        df = impute_with_industry_median(df, target_col="turnover")

    logger.info("数据清洗完成: 输出 %d 条记录", len(df))
    return df
