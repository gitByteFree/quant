"""数据源抽象接口."""

from abc import ABC, abstractmethod
from typing import Protocol

import pandas as pd


class DataSource(ABC):
    """数据源抽象基类.

    定义所有数据源必须实现的接口，支持akshare/tushare/wind等多种数据源.
    """

    @abstractmethod
    def get_daily(
        self,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取日线行情.

        Args:
            symbols: 股票代码列表，None=全部A股
            start_date: 起始日期 "YYYYMMDD" 或 "YYYY-MM-DD"
            end_date: 截止日期
            adjust: 复权方式 qfq=前复权 hfq=后复权 None=不复权

        Returns:
            DataFrame with columns: symbol, trade_date, open, high, low, close,
            volume, amount, turnover, adjust_factor, is_st, is_suspended
        """
        ...

    @abstractmethod
    def get_minute(
        self,
        symbol: str,
        trade_date: str,
        freq: str = "1min",
    ) -> pd.DataFrame:
        """获取分钟线.

        Args:
            symbol: 股票代码
            trade_date: 交易日期
            freq: 频率 1min/5min/15min/30min/60min

        Returns:
            DataFrame with OHLCV columns
        """
        ...

    @abstractmethod
    def get_financial(
        self,
        symbols: list[str] | None = None,
        report_period: str | None = None,
    ) -> pd.DataFrame:
        """获取财务数据.

        Args:
            symbols: 股票代码列表
            report_period: 报告期 "20231231" 表示2023年报

        Returns:
            DataFrame with columns: symbol, report_period, ann_date,
            total_assets, total_equity, revenue, net_profit, operating_cf,
            roe, gross_margin, ...
        """
        ...

    @abstractmethod
    def get_fund_flow(
        self,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """获取资金流向.

        Returns:
            DataFrame: 主力/超大单/大单/中单/小单净流入
        """
        ...

    @abstractmethod
    def get_dragon_tiger(
        self,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        """获取龙虎榜数据.

        Returns:
            DataFrame: 上榜股票、买卖席位、净买入额等
        """
        ...

    @abstractmethod
    def get_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表.

        Returns:
            DataFrame with columns: symbol, name, industry, market_cap, list_date
        """
        ...

    @abstractmethod
    def get_index_daily(
        self,
        symbol: str = "000300",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """获取指数日线（基准比对用）.

        Args:
            symbol: 指数代码 000300=沪深300 000905=中证500
        """
        ...

    @abstractmethod
    def get_north_flow(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """获取北向资金流向."""
        ...

    @abstractmethod
    def get_margin_data(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """获取融资融券数据."""
        ...

    @abstractmethod
    def get_industry_classification(self) -> pd.DataFrame:
        """获取行业分类（申万一级）.

        Returns:
            DataFrame with columns: symbol, industry
        """
        ...


class DataSourceFactory:
    """数据源工厂."""

    _registry: dict[str, type[DataSource]] = {}

    @classmethod
    def register(cls, name: str, source_cls: "type[DataSource] | None" = None):
        """注册数据源.

        支持两种用法:
            DataSourceFactory.register("akshare", AkShareDataSource)  # 直接调用
            @DataSourceFactory.register("akshare")                    # 装饰器
        """
        if source_cls is not None:
            cls._registry[name] = source_cls
            return source_cls

        def decorator(sc: type[DataSource]) -> type[DataSource]:
            cls._registry[name] = sc
            return sc
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: object) -> DataSource:
        if name not in cls._registry:
            raise ValueError(f"未知数据源: {name}. 可用: {list(cls._registry)}")
        return cls._registry[name](**kwargs)
