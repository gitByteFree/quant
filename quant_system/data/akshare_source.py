"""akShare数据源实现.

提供A股日线、财务、资金流向、龙虎榜等数据获取.
"""

import time
from functools import lru_cache

import numpy as np
import pandas as pd

from quant_system.data.base import DataSource, DataSourceFactory
from quant_system.utils.calendar import TradingCalendar
from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


@DataSourceFactory.register("akshare")
class AkShareDataSource(DataSource):
    """akShare数据源.

    使用akshare免费API获取A股数据，内置限速和重试机制.
    """

    def __init__(
        self,
        cache_path: str = "data/parquet",
        rate_limit_per_minute: int = 30,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ):
        import akshare as ak

        self.ak = ak
        self._cache_path = cache_path
        self._rate_limit = 60.0 / rate_limit_per_minute
        self._max_retries = max_retries
        self._retry_delay = retry_delay_seconds
        self._last_request_time = 0.0
        self._calendar = TradingCalendar()

    def _rate_limit_wait(self) -> None:
        """限速等待."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_request_time = time.monotonic()

    def _retry_call(self, func, *args, **kwargs):
        """带重试的函数调用."""
        last_error = None
        for attempt in range(self._max_retries):
            try:
                self._rate_limit_wait()
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    "API调用失败 (第%d/%d次): %s",
                    attempt + 1,
                    self._max_retries,
                    e,
                )
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
        raise RuntimeError(f"API调用失败，已达最大重试次数: {last_error}")

    def _normalize_symbol(self, code: str) -> str:
        """标准化股票代码为6位."""
        return code.zfill(6)

    # ---- 日线数据 ----

    def get_daily(
        self,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取全A股日线数据."""
        start = start_date or "20100101"
        end = end_date or pd.Timestamp.today().strftime("%Y%m%d")
        start_clean = start.replace("-", "")
        end_clean = end.replace("-", "")

        logger.info("获取A股日线: %s ~ %s", start_clean, end_clean)

        def _fetch() -> pd.DataFrame:
            df = self.ak.stock_zh_a_hist(
                symbol="",  # 空字符串表示全市场
                period="daily",
                start_date=start_clean,
                end_date=end_clean,
                adjust=adjust,
            )
            if df is None or df.empty:
                return pd.DataFrame()
            return df

        df = self._retry_call(_fetch)
        if df.empty:
            logger.warning("未获取到日线数据")
            return df

        # 统一列名
        col_map = {
            "日期": "trade_date",
            "股票代码": "symbol",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
            "涨跌幅": "pct_change",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).apply(self._normalize_symbol)

        # 根据调整方式标记复权类型
        if adjust == "qfq":
            df["adjust_type"] = "qfq"
        elif adjust == "hfq":
            df["adjust_type"] = "hfq"

        if symbols:
            df = df[df["symbol"].isin([self._normalize_symbol(s) for s in symbols])]
        return df.reset_index(drop=True)

    # ---- 股票列表 ----

    @lru_cache(maxsize=1)
    def get_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表."""
        logger.info("获取A股股票列表")

        def _fetch() -> pd.DataFrame:
            df = self.ak.stock_info_a_code_name()
            if df is None or df.empty:
                return pd.DataFrame(columns=["symbol", "name"])
            return df

        df = self._retry_call(_fetch)
        if df.empty:
            return df
        df.columns = ["symbol", "name"]
        df["symbol"] = df["symbol"].astype(str).apply(self._normalize_symbol)
        return df

    # ---- 财务数据 ----

    def get_financial(
        self,
        symbols: list[str] | None = None,
        report_period: str | None = None,
    ) -> pd.DataFrame:
        """获取财务数据（利润表+资产负债表主要指标）."""
        logger.info("获取财务数据")
        target = symbols[0] if symbols and len(symbols) == 1 else None
        if target is None and symbols:
            # 多股票逐个获取
            frames = []
            for sym in symbols:
                try:
                    f = self._retry_call(
                        self.ak.stock_financial_abstract, symbol=sym
                    )
                    if f is not None and not f.empty:
                        frames.append(f)
                except Exception:
                    logger.warning("获取 %s 财务数据失败", sym)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        def _fetch():
            if target:
                return self.ak.stock_financial_abstract(symbol=target)
            return self.ak.stock_financial_abstract_ths()

        df = self._retry_call(_fetch)
        if df is None or df.empty:
            return pd.DataFrame()

        # 统一列名
        rename_map = {
            "股票代码": "symbol",
            "报告期": "report_period",
            "公告日期": "ann_date",
            "营业收入": "revenue",
            "净利润": "net_profit",
            "总资产": "total_assets",
            "净资产": "total_equity",
            "经营现金流": "operating_cf",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).apply(self._normalize_symbol)
        if report_period and "report_period" in df.columns:
            df = df[df["report_period"] == report_period]
        return df.reset_index(drop=True)

    # ---- 资金流向 ----

    def get_fund_flow(
        self,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """获取个股资金流向."""
        logger.info("获取资金流向: %s ~ %s", start_date, end_date)

        def _fetch():
            return self.ak.stock_fund_flow_individual(symbol=start_date)

        # 使用市场整体资金流向
        def _fetch_market():
            return self.ak.stock_market_fund_flow()

        try:
            df = self._retry_call(_fetch_market)
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception:
            pass
        return pd.DataFrame()

    # ---- 龙虎榜 ----

    def get_dragon_tiger(
        self,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        """获取龙虎榜数据."""
        date_str = trade_date or pd.Timestamp.today().strftime("%Y%m%d")
        logger.info("获取龙虎榜数据: %s", date_str)

        def _fetch():
            return self.ak.stock_lhb_detail_daily(
                date=date_str.replace("-", ""),
            )

        try:
            df = self._retry_call(_fetch)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        return pd.DataFrame()

    # ---- 指数数据 ----

    def get_index_daily(
        self,
        symbol: str = "000300",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """获取指数日线."""
        start = (start_date or "20100101").replace("-", "")
        end = (end_date or pd.Timestamp.today().strftime("%Y%m%d")).replace("-", "")

        logger.info("获取指数日线: %s (%s ~ %s)", symbol, start, end)

        def _fetch():
            return self.ak.stock_zh_index_daily(symbol=f"sh{symbol}")

        try:
            df = self._retry_call(_fetch)
            if df is not None and not df.empty:
                df["trade_date"] = pd.to_datetime(df["date"]) if "date" in df.columns else pd.NaT
                return df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        except Exception:
            pass

        # 回退: 用akshare指数日线接口
        def _fetch_index():
            return self.ak.index_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
            )

        return self._retry_call(_fetch_index)

    # ---- 北向资金 ----

    def get_north_flow(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """获取北向资金流向."""
        logger.info("获取北向资金数据")

        def _fetch():
            return self.ak.stock_hsgt_north_net_flow_in_em()

        try:
            df = self._retry_call(_fetch)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        return pd.DataFrame()

    # ---- 融资融券 ----

    def get_margin_data(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """获取融资融券数据."""
        logger.info("获取融资融券数据")

        def _fetch():
            return self.ak.stock_margin_detail_sse()

        try:
            df = self._retry_call(_fetch)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        return pd.DataFrame()

    # ---- 行业分类 ----

    @lru_cache(maxsize=1)
    def get_industry_classification(self) -> pd.DataFrame:
        """获取申万行业分类."""
        logger.info("获取申万行业分类")

        def _fetch():
            return self.ak.stock_info_sz_category_sw()

        try:
            df = self._retry_call(_fetch)
            if df is not None and not df.empty:
                # 尝试标准化列名
                cols_map = {}
                for c in df.columns:
                    cl = str(c).lower()
                    if "code" in cl or "代码" in str(c):
                        cols_map[c] = "symbol"
                    elif "name" in cl or "名称" in str(c):
                        cols_map[c] = "name"
                    elif "industry" in cl or "行业" in str(c):
                        cols_map[c] = "industry"
                if cols_map:
                    df = df.rename(columns=cols_map)
                if "symbol" in df.columns:
                    df["symbol"] = df["symbol"].astype(str).apply(self._normalize_symbol)
                return df
        except Exception:
            pass
        return pd.DataFrame(columns=["symbol", "name", "industry"])

    # ---- 分钟线 ----

    def get_minute(
        self,
        symbol: str,
        trade_date: str,
        freq: str = "1min",
    ) -> pd.DataFrame:
        """获取分钟线."""
        date_str = trade_date.replace("-", "")
        period_map = {"1min": "1", "5min": "5", "15min": "15", "30min": "30", "60min": "60"}
        period = period_map.get(freq, "1")
        logger.info("获取分钟线: %s %s (%s)", symbol, date_str, freq)

        def _fetch():
            return self.ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period=period,
                start_date=f"{date_str} 09:30:00",
                end_date=f"{date_str} 15:00:00",
            )

        try:
            df = self._retry_call(_fetch)
            return df if df is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    # ---- 批量获取 ----

    def fetch_daily(
        self,
        start_date: str = "2010-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """便捷方法：获取并返回日线数据（推荐用于初始数据获取）."""
        df = self.get_daily(start_date=start_date, end_date=end_date)
        if df.empty:
            return df

        # 统一列名
        if "日期" in df.columns:
            df = df.rename(columns={"日期": "trade_date"})
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        logger.info("获取到 %d 条日线记录, %d 只股票, %s ~ %s",
                     len(df),
                     df["symbol"].nunique() if "symbol" in df.columns else 0,
                     df["trade_date"].min().date(),
                     df["trade_date"].max().date())
        return df.reset_index(drop=True)
