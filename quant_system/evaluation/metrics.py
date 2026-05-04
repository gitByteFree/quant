"""绩效指标计算.

年化收益、夏普比率、最大回撤、Calmar比率、信息比率等.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_system.utils.constants import DEFAULT_RISK_FREE_RATE

TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceMetrics:
    """绩效指标汇总."""

    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    avg_daily_return: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0  # 95% VaR
    cvar_95: float = 0.0  # 95% CVaR

    def summary(self) -> str:
        """生成文本摘要."""
        lines = [
            "=" * 50,
            "绩效指标汇总",
            "=" * 50,
            f"总收益率:     {self.total_return * 100:8.2f}%",
            f"年化收益率:   {self.annual_return * 100:8.2f}%",
            f"年化波动率:   {self.annual_volatility * 100:8.2f}%",
            f"夏普比率:     {self.sharpe_ratio:8.2f}",
            f"最大回撤:     {self.max_drawdown * 100:8.2f}%",
            f"Calmar比率:   {self.calmar_ratio:8.2f}",
            f"信息比率:     {self.information_ratio:8.2f}",
            f"Sortino比率:  {self.sortino_ratio:8.2f}",
            f"胜率:         {self.win_rate * 100:8.2f}%",
            f"盈亏比:       {self.profit_loss_ratio:8.2f}",
            f"日均收益:     {self.avg_daily_return * 100:8.4f}%",
            f"偏度:         {self.skewness:8.2f}",
            f"峰度:         {self.kurtosis:8.2f}",
            f"95% VaR:      {self.var_95 * 100:8.2f}%",
            f"95% CVaR:     {self.cvar_95 * 100:8.2f}%",
            "=" * 50,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, float]:
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "information_ratio": self.information_ratio,
            "sortino_ratio": self.sortino_ratio,
            "win_rate": self.win_rate,
            "profit_loss_ratio": self.profit_loss_ratio,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
        }


def compute_returns(
    portfolio_values: pd.Series | np.ndarray,
) -> pd.Series:
    """计算日收益率序列.

    Args:
        portfolio_values: 组合净值序列

    Returns:
        日收益率序列
    """
    if isinstance(portfolio_values, np.ndarray):
        portfolio_values = pd.Series(portfolio_values)
    return portfolio_values.pct_change().dropna()


def annualized_return(daily_returns: pd.Series) -> float:
    """年化收益率."""
    if len(daily_returns) == 0:
        return 0.0
    total = (1 + daily_returns).prod()
    years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def annualized_volatility(daily_returns: pd.Series) -> float:
    """年化波动率."""
    if len(daily_returns) == 0:
        return 0.0
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """夏普比率."""
    excess = daily_returns.mean() * TRADING_DAYS_PER_YEAR - risk_free_rate
    vol = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if vol == 0:
        return 0.0
    return float(excess / vol)


def max_drawdown(daily_returns: pd.Series) -> float:
    """最大回撤."""
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def calmar_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """Calmar比率 = 年化收益 / |最大回撤|."""
    ann_ret = annualized_return(daily_returns) - risk_free_rate
    mdd = abs(max_drawdown(daily_returns))
    if mdd == 0:
        return 0.0
    return float(ann_ret / mdd)


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """信息比率 = mean(超额收益) / std(超额收益) * sqrt(252)."""
    excess = portfolio_returns - benchmark_returns
    aligned = excess.dropna()
    if len(aligned) == 0 or aligned.std() == 0:
        return 0.0
    return float(aligned.mean() / aligned.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """Sortino比率 = 超额收益 / 下行波动率."""
    excess = daily_returns.mean() * TRADING_DAYS_PER_YEAR - risk_free_rate
    downside = daily_returns[daily_returns < 0].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside == 0 or np.isnan(downside):
        return 0.0
    return float(excess / downside)


def win_rate(daily_returns: pd.Series) -> float:
    """胜率（日收益>0的占比）."""
    if len(daily_returns) == 0:
        return 0.0
    return float((daily_returns > 0).mean())


def profit_loss_ratio(daily_returns: pd.Series) -> float:
    """盈亏比 = avg(正收益) / |avg(负收益)|."""
    positive = daily_returns[daily_returns > 0]
    negative = daily_returns[daily_returns < 0]
    if len(negative) == 0:
        return float("inf") if len(positive) > 0 else 0.0
    avg_win = positive.mean() if len(positive) > 0 else 0.0
    avg_loss = abs(negative.mean())
    if avg_loss == 0:
        return 0.0
    return float(avg_win / avg_loss)


def value_at_risk(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """VaR（历史模拟法）."""
    if len(daily_returns) == 0:
        return 0.0
    return float(daily_returns.quantile(1 - confidence))


def cvar(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR（条件在险价值）."""
    if len(daily_returns) == 0:
        return 0.0
    var = value_at_risk(daily_returns, confidence)
    return float(daily_returns[daily_returns <= var].mean())


def compute_all_metrics(
    portfolio_values: pd.Series,
    benchmark_values: pd.Series | None = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> PerformanceMetrics:
    """一站式计算所有绩效指标.

    Args:
        portfolio_values: 组合净值序列
        benchmark_values: 基准净值序列（用于信息比率）
        risk_free_rate: 无风险利率

    Returns:
        PerformanceMetrics 对象
    """
    returns = compute_returns(portfolio_values)
    if returns.empty:
        return PerformanceMetrics()

    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    mdd = max_drawdown(returns)

    ir_val = 0.0
    if benchmark_values is not None:
        bench_returns = compute_returns(benchmark_values)
        common_idx = returns.index.intersection(bench_returns.index)
        if len(common_idx) > 0:
            ir_val = information_ratio(returns.loc[common_idx], bench_returns.loc[common_idx])

    return PerformanceMetrics(
        total_return=float((1 + returns).prod() - 1),
        annual_return=ann_ret,
        annual_volatility=ann_vol,
        sharpe_ratio=sharpe_ratio(returns, risk_free_rate),
        max_drawdown=mdd,
        calmar_ratio=calmar_ratio(returns, risk_free_rate),
        information_ratio=ir_val,
        sortino_ratio=sortino_ratio(returns, risk_free_rate),
        win_rate=win_rate(returns),
        profit_loss_ratio=profit_loss_ratio(returns),
        avg_daily_return=float(returns.mean()),
        skewness=float(returns.skew()),
        kurtosis=float(returns.kurtosis()),
        var_95=value_at_risk(returns),
        cvar_95=cvar(returns),
    )
