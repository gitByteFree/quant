"""复权因子计算与价格复权."""

import numpy as np
import pandas as pd


def compute_adjustment_factor(df: pd.DataFrame) -> pd.DataFrame:
    """从日线数据计算复权因子.

    使用后复权收盘价和原始收盘价反推复权因子.

    Args:
        df: 日线DataFrame，需包含 close, hfq_close (后复权收盘价)

    Returns:
        添加了 adjust_factor 列的DataFrame
    """
    df = df.copy()
    df["adjust_factor"] = np.where(
        df["close"] > 0,
        df["hfq_close"] / df["close"],
        1.0,
    )
    return df


def apply_forward_adjustment(
    df: pd.DataFrame,
    adjust_factor_col: str = "adjust_factor",
    price_cols: tuple[str, ...] = ("open", "high", "low", "close"),
) -> pd.DataFrame:
    """应用前复权.

    Args:
        df: 日线DataFrame，按日期升序排列
        adjust_factor_col: 复权因子列名
        price_cols: 需要复权的价格列

    Returns:
        复权后的DataFrame
    """
    df = df.copy()
    latest_factor = df[adjust_factor_col].iloc[-1] if len(df) > 0 else 1.0
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col] * df[adjust_factor_col] / latest_factor
    return df


def apply_backward_adjustment(
    df: pd.DataFrame,
    adjust_factor_col: str = "adjust_factor",
    price_cols: tuple[str, ...] = ("open", "high", "low", "close"),
) -> pd.DataFrame:
    """应用后复权.

    Args:
        df: 日线DataFrame，按日期升序排列
        adjust_factor_col: 复权因子列名
        price_cols: 需要复权的价格列

    Returns:
        后复权后的DataFrame
    """
    df = df.copy()
    earliest_factor = df[adjust_factor_col].iloc[0] if len(df) > 0 else 1.0
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col] * df[adjust_factor_col] / earliest_factor
    return df
