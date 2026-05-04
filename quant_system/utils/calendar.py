"""A股交易日历."""

from datetime import date, datetime

import pandas as pd

from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


class TradingCalendar:
    """A股交易日历管理.

    优先使用akshare获取真实交易日历，获取失败时回退到基于规则的估算.
    """

    def __init__(self, start: str = "2010-01-01", end: str | None = None):
        self._start = pd.Timestamp(start)
        self._end = pd.Timestamp(end) if end else pd.Timestamp.today()
        self._trade_dates: pd.DatetimeIndex | None = None
        self._load_calendar()

    def _load_calendar(self) -> None:
        """加载交易日历."""
        try:
            import akshare as ak

            df = ak.tool_trade_date_hist_sina()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            self._trade_dates = pd.DatetimeIndex(
                df.loc[
                    (df["trade_date"] >= self._start)
                    & (df["trade_date"] <= self._end),
                    "trade_date",
                ]
            ).sort_values()
            logger.info(
                "交易日历加载成功: %d 个交易日 (%s ~ %s)",
                len(self._trade_dates),
                self._trade_dates[0].date(),
                self._trade_dates[-1].date(),
            )
        except Exception:
            logger.warning("akshare交易日历获取失败，使用简易规则生成")
            self._trade_dates = self._generate_rough_calendar()

    def _generate_rough_calendar(self) -> pd.DatetimeIndex:
        """生成简易交易日历（排除周末+元旦/国庆/春节附近）."""
        dates = pd.bdate_range(start=self._start, end=self._end, freq="C", holidays=[])
        return pd.DatetimeIndex(dates)

    @property
    def trade_dates(self) -> pd.DatetimeIndex:
        if self._trade_dates is None:
            self._load_calendar()
        return self._trade_dates  # type: ignore[return-value]

    def is_trade_date(self, dt: date | datetime | str) -> bool:
        """判断是否为交易日."""
        ts = pd.Timestamp(dt)
        return ts in self.trade_dates

    def next_trade_date(self, dt: date | datetime | str, offset: int = 1) -> pd.Timestamp:
        """获取dt之后第offset个交易日."""
        ts = pd.Timestamp(dt)
        idx = self.trade_dates.searchsorted(ts)
        target = idx + offset
        if target < 0:
            target = 0
        if target >= len(self.trade_dates):
            target = len(self.trade_dates) - 1
        return self.trade_dates[target]

    def prev_trade_date(self, dt: date | datetime | str, offset: int = 1) -> pd.Timestamp:
        """获取dt之前第offset个交易日."""
        return self.next_trade_date(dt, -offset)

    def trade_dates_between(self, start: str, end: str) -> pd.DatetimeIndex:
        """获取两个日期之间的所有交易日."""
        mask = (self.trade_dates >= pd.Timestamp(start)) & (self.trade_dates <= pd.Timestamp(end))
        return self.trade_dates[mask]

    def month_end_trade_dates(self, start: str, end: str) -> pd.DatetimeIndex:
        """获取每个月的最后一个交易日."""
        dates = self.trade_dates_between(start, end)
        monthly = dates.to_series().groupby(dates.to_period("M")).last()
        return pd.DatetimeIndex(monthly.values)
