"""Parquet分区存储层.

按日期分区存储数据，构建本地data lake.
"""

from pathlib import Path

import pandas as pd

from quant_system.utils.logger import get_logger

logger = get_logger(__name__)


class ParquetStore:
    """Parquet分区存储管理器.

    分区布局:
        data/parquet/daily/trade_date=2024-01-01/000001.parquet
        data/parquet/factors/factor_name=momentum_20d/trade_date=2024-01-01/data.parquet
    """

    def __init__(self, base_path: str = "data/parquet"):
        self._base = Path(base_path)

    @property
    def base_path(self) -> Path:
        return self._base

    def _ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    # ---- 日线数据 ----

    def write_daily(
        self,
        df: pd.DataFrame,
        partition_by: str = "trade_date",
    ) -> None:
        """按日期分区写入日线数据.

        Args:
            df: 包含 trade_date, symbol 列的DataFrame
            partition_by: 分区列
        """
        if df.empty:
            logger.warning("空DataFrame，跳过写入")
            return
        df = df.copy()
        df[partition_by] = pd.to_datetime(df[partition_by])
        for trade_date, group in df.groupby(partition_by):
            date_dir = self._base / "daily" / f"trade_date={trade_date.date()}"
            self._ensure_dir(date_dir)
            file_path = date_dir / "data.parquet"
            group.to_parquet(file_path, index=False)
        logger.info("日线数据写入完成: %d 个分区", df[partition_by].nunique())

    def read_daily(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """读取日线数据.

        Args:
            start_date: 起始日期
            end_date: 截止日期
            symbols: 股票代码过滤

        Returns:
            合并后的DataFrame
        """
        daily_dir = self._base / "daily"
        if not daily_dir.exists():
            logger.warning("日线数据目录不存在: %s", daily_dir)
            return pd.DataFrame()

        dfs: list[pd.DataFrame] = []
        for partition_dir in sorted(daily_dir.glob("trade_date=*")):
            date_str = partition_dir.name.split("=")[1]
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue
            file_path = partition_dir / "data.parquet"
            if file_path.exists():
                chunk = pd.read_parquet(file_path)
                dfs.append(chunk)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        if symbols:
            result = result[result["symbol"].isin(symbols)]
        return result.reset_index(drop=True)

    def get_unique_symbols(self) -> list[str]:
        """获取所有已存储的股票代码."""
        df = self.read_daily()
        if df.empty:
            return []
        return sorted(df["symbol"].unique().tolist())

    def get_date_range(self) -> tuple[str, str]:
        """获取已存储数据的日期范围."""
        daily_dir = self._base / "daily"
        if not daily_dir.exists():
            return ("", "")
        partitions = sorted(daily_dir.glob("trade_date=*"))
        if not partitions:
            return ("", "")
        start = partitions[0].name.split("=")[1]
        end = partitions[-1].name.split("=")[1]
        return (start, end)

    # ---- 因子数据 ----

    def write_factors(
        self,
        factor_name: str,
        df: pd.DataFrame,
    ) -> None:
        """写入因子数据.

        Args:
            factor_name: 因子名称
            df: 包含 trade_date, symbol, value 列的DataFrame
        """
        if df.empty:
            return
        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        for trade_date, group in df.groupby("trade_date"):
            factor_dir = (
                self._base
                / "factors"
                / f"factor_name={factor_name}"
                / f"trade_date={trade_date.date()}"
            )
            self._ensure_dir(factor_dir)
            group.to_parquet(factor_dir / "data.parquet", index=False)
        logger.info("因子 %s 写入完成: %d 个分区", factor_name, df["trade_date"].nunique())

    def read_factors(
        self,
        factor_name: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """读取因子数据.

        Args:
            factor_name: 因子名称
            start_date: 起始日期
            end_date: 截止日期

        Returns:
            因子DataFrame，包含 trade_date, symbol, value 列
        """
        factor_dir = self._base / "factors" / f"factor_name={factor_name}"
        if not factor_dir.exists():
            logger.warning("因子目录不存在: %s", factor_dir)
            return pd.DataFrame()

        dfs: list[pd.DataFrame] = []
        for partition_dir in sorted(factor_dir.glob("trade_date=*")):
            date_str = partition_dir.name.split("=")[1]
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue
            file_path = partition_dir / "data.parquet"
            if file_path.exists():
                chunk = pd.read_parquet(file_path)
                dfs.append(chunk)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    # ---- 财务数据 ----

    def write_financial(
        self,
        df: pd.DataFrame,
    ) -> None:
        """写入财务数据."""
        if df.empty:
            return
        financial_dir = self._base / "financial"
        self._ensure_dir(financial_dir)
        df.to_parquet(financial_dir / "data.parquet", index=False)
        logger.info("财务数据写入完成: %d 条记录", len(df))

    def read_financial(self) -> pd.DataFrame:
        """读取财务数据."""
        file_path = self._base / "financial" / "data.parquet"
        if not file_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(file_path)

    # ---- 工具方法 ----

    def list_factors(self) -> list[str]:
        """列出所有已存储的因子名称."""
        factors_dir = self._base / "factors"
        if not factors_dir.exists():
            return []
        return sorted(
            {d.name.split("=")[1] for d in factors_dir.glob("factor_name=*")}
        )

    def drop_partitions(
        self,
        dataset: str,
        before_date: str,
    ) -> int:
        """删除指定日期之前的分区.

        Args:
            dataset: 数据集名称 daily/factors
            before_date: 删除此日期之前的分区

        Returns:
            删除的分区数量
        """
        dataset_dir = self._base / dataset
        if not dataset_dir.exists():
            return 0
        count = 0
        for partition_dir in dataset_dir.glob("**/trade_date=*"):
            date_str = partition_dir.name.split("=")[1]
            if date_str < before_date:
                import shutil

                shutil.rmtree(partition_dir)
                count += 1
        logger.info("删除了 %d 个分区 (%s, before %s)", count, dataset, before_date)
        return count

    def get_stats(self) -> dict:
        """获取存储统计信息."""
        start, end = self.get_date_range()
        return {
            "base_path": str(self._base),
            "date_range": f"{start} ~ {end}",
            "factors": self.list_factors(),
            "symbols_count": len(self.get_unique_symbols()),
        }
