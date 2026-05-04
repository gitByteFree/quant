"""因子基类与注册表系统."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from quant_system.utils.logger import get_logger

logger = get_logger(__name__)

# 全局因子注册表
_factor_registry: dict[str, type["BaseFactor"]] = {}


def register_factor(
    name: str | None = None,
    category: str | None = None,
    description: str = "",
) -> Callable:
    """因子注册装饰器.

    Args:
        name: 因子名称，默认使用类名
        category: 因子分类 momentum/volatility/value/quality/sentiment
        description: 因子描述

    Usage:
        @register_factor("momentum_20d", "momentum", "20日收益率")
        class Momentum20D(BaseFactor):
            ...
    """

    def decorator(cls: type["BaseFactor"]) -> type["BaseFactor"]:
        factor_name = name or cls.__name__
        cls.factor_name = factor_name
        cls.category = category or ""
        cls.description = description
        _factor_registry[factor_name] = cls
        return cls

    return decorator


@dataclass
class FactorResult:
    """因子计算结果."""

    factor_name: str
    category: str
    values: pd.Series  # index: (trade_date, symbol)
    metadata: dict = field(default_factory=dict)


class BaseFactor(ABC):
    """因子抽象基类.

    所有因子实现需继承此类并实现 compute 方法.
    """

    factor_name: str = ""
    category: str = ""
    description: str = ""

    def __init__(self, data: "pd.DataFrame | None" = None):
        """初始化因子.

        Args:
            data: 日线DataFrame，包含 symbol, trade_date, open, high, low,
                  close, volume, amount, turnover 等列
        """
        self._data = data

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            raise ValueError("因子未绑定数据，请先设置 self._data")
        return self._data

    @data.setter
    def data(self, df: pd.DataFrame) -> None:
        self._data = df

    @abstractmethod
    def compute(self, start_date: str | None = None, end_date: str | None = None) -> FactorResult:
        """计算因子值.

        Args:
            start_date: 起始日期
            end_date: 截止日期

        Returns:
            FactorResult 包含因子名称、分类和MultiIndex Series
        """
        ...

    def _prepare_price_data(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """准备价格数据，按 trading_date + symbol 排序并建立MultiIndex.

        Returns:
            带有 (symbol, trade_date) 层次索引且按日期排序的DataFrame
        """
        df = self.data.copy()
        if start_date:
            df = df[df["trade_date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["trade_date"] <= pd.Timestamp(end_date)]
        df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        return df

    def _make_series(
        self,
        df: pd.DataFrame,
        value_col: str,
        factor_name: str | None = None,
    ) -> FactorResult:
        """从DataFrame构建FactorResult."""
        name = factor_name or self.factor_name
        s = df.set_index(["trade_date", "symbol"])[value_col].dropna()
        return FactorResult(
            factor_name=name,
            category=self.category,
            values=s,
            metadata={"description": self.description},
        )


# ---- 便利函数 ----


def list_factors(category: str | None = None) -> list[str]:
    """列出所有已注册的因子.

    Args:
        category: 按分类过滤，None=全部

    Returns:
        因子名称列表
    """
    if category:
        return [n for n, c in _factor_registry.items() if c.category == category]
    return sorted(_factor_registry.keys())


def get_factor_class(name: str) -> type[BaseFactor]:
    """根据名称获取因子类."""
    if name not in _factor_registry:
        raise KeyError(f"因子 '{name}' 未注册. 可用: {list(_factor_registry)}")
    return _factor_registry[name]


def compute_all_factors(
    data: pd.DataFrame,
    factor_names: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, pd.Series]:
    """批量计算所有因子.

    Args:
        data: 日线DataFrame
        factor_names: 要计算的因子列表，None=全部
        start_date: 起始日期
        end_date: 截止日期

    Returns:
        {factor_name: pd.Series with MultiIndex (trade_date, symbol)}
    """
    names = factor_names or list_factors()
    results: dict[str, pd.Series] = {}
    for name in names:
        try:
            cls = get_factor_class(name)
            instance = cls(data.copy())
            result = instance.compute(start_date, end_date)
            results[name] = result.values
            logger.debug("因子 %s 计算完成: %d 条记录", name, len(result.values))
        except Exception as e:
            logger.error("因子 %s 计算失败: %s", name, e)
    return results


def build_factor_panel(factor_results: dict[str, pd.Series]) -> pd.DataFrame:
    """将多个因子结果合并为面板数据.

    Args:
        factor_results: {factor_name: Series with MultiIndex (trade_date, symbol)}

    Returns:
        DataFrame with columns = factor_names, index = (trade_date, symbol)
    """
    if not factor_results:
        return pd.DataFrame()
    panel = pd.DataFrame(factor_results)
    panel.index.names = ["trade_date", "symbol"]
    return panel.sort_index()
