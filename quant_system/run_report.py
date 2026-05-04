"""下周一买入报告生成脚本（基于实时行情快照的多因子分析）."""

import sys
import os
import warnings
from datetime import datetime, date, timedelta

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import akshare as ak


def find_next_trading_day(base_date=None):
    """找到下一个交易日."""
    if base_date is None:
        base_date = date.today()
    try:
        cal_df = ak.tool_trade_date_hist_sina()
        cal_df["trade_date"] = pd.to_datetime(cal_df["trade_date"])
        future_dates = cal_df[cal_df["trade_date"] >= pd.Timestamp(base_date)]
        if not future_dates.empty:
            return future_dates.iloc[0]["trade_date"].date()
    except Exception:
        pass
    d = base_date + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def mad_clip(series, threshold=3.0):
    """MAD法去极值."""
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return series
    upper = median + threshold * mad
    lower = median - threshold * mad
    return series.clip(lower=lower, upper=upper)


def main():
    today = date.today()
    next_trade = find_next_trading_day(today)

    print("=" * 64)
    print("  A股量化分析系统 — 下周一买入推荐报告")
    print("=" * 64)
    print(f"  生成日期: {today}")
    print(f"  目标交易日: {next_trade}（下周第一个交易日）")

    # ── 1. 获取实时行情 ──
    print(f"\n  [1/4] 获取全A股实时行情 ...")
    try:
        df = ak.stock_zh_a_spot()
    except Exception as e:
        print(f"  ✗ 行情获取失败: {e}")
        return 1

    if df.empty:
        print("  ✗ 未获取到数据")
        return 1

    # 统一列名
    col_map = {
        "代码": "symbol", "名称": "name", "最新价": "close",
        "涨跌额": "change", "涨跌幅": "pct_change",
        "买入": "bid", "卖出": "ask", "昨收": "pre_close",
        "今开": "open", "最高": "high", "最低": "low",
        "成交量": "volume", "成交额": "amount",
    }
    df = df.rename(columns=col_map)
    df["symbol"] = df["symbol"].astype(str)

    # 排除北交所（bj开头）、ST股票
    df = df[~df["symbol"].str.startswith("bj")]
    if "name" in df.columns:
        df = df[~df["name"].str.contains(r"\*ST|ST ", na=False)]

    print(f"  ✓ 获取 {len(df)} 只股票（已排除北交所和ST）")

    # ── 2. 多因子计算 ──
    print(f"\n  [2/4] 计算多因子评分 ...")

    # 因子1: 日内振幅归一化（低振幅偏好 — 稳定性）
    df["amplitude"] = (df["high"] - df["low"]) / df["pre_close"]
    df["amplitude_z"] = mad_clip(df["amplitude"])
    df["f1_stability"] = -df["amplitude_z"].rank(pct=True)  # 反向：低振幅得分高

    # 因子2: 日内价格位置（收盘价在日内区间的相对位置，>0.5表示尾盘走强）
    df["intraday_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)
    df["intraday_position"] = df["intraday_position"].clip(0, 1)
    df["f2_momentum"] = df["intraday_position"].rank(pct=True)

    # 因子3: 涨跌幅（短期动量，温和上涨优于暴涨）
    df["pct_change"] = pd.to_numeric(df["pct_change"], errors="coerce")
    # 偏好温和上涨 (0.5%~5%)，过滤极值
    df["f3_return"] = df["pct_change"].clip(-5, 5).rank(pct=True)

    # 因子4: 成交额（规模因子，偏好中等规模 — 流动性好但不过热）
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["log_amount"] = np.log(df["amount"] + 1)
    # 偏好中等偏大成交额（流动性好），但排除微盘和过度交易
    amount_pct = df["log_amount"].rank(pct=True)
    df["f4_liquidity"] = 1.0 - abs(amount_pct - 0.6) * 2  # 中心在60分位

    # 因子5: 买卖价差（流动性因子，价差小=流动性好）
    df["spread"] = (df["ask"] - df["bid"]) / df["close"]
    df["spread"] = df["spread"].clip(0, 0.05)
    df["f5_spread"] = -df["spread"].rank(pct=True)  # 反向：价差小得分高

    # 因子6: 涨跌额/振幅比（趋势效率：涨跌幅相对于振幅的比率）
    df["change"] = pd.to_numeric(df["change"], errors="coerce")
    price_range = (df["high"] - df["low"]).abs()
    df["trend_efficiency"] = df["change"] / (price_range + 1e-8)
    df["trend_efficiency"] = df["trend_efficiency"].clip(-1, 1)
    df["f6_efficiency"] = df["trend_efficiency"].rank(pct=True)

    # 因子7: 量价关系（成交额 * 涨跌幅符号 — 放量上涨 vs 缩量下跌）
    df["vol_price"] = df["log_amount"] * np.sign(df["pct_change"].fillna(0))
    df["f7_vol_price"] = df["vol_price"].rank(pct=True)

    # ── 3. 综合评分 ──
    factor_columns = ["f1_stability", "f2_momentum", "f3_return",
                      "f4_liquidity", "f5_spread", "f6_efficiency", "f7_vol_price"]

    # IC风格加权（模拟各因子对短期收益的预测方向）
    weights = {
        "f1_stability": 0.10,   # 低波动异象
        "f2_momentum": 0.20,    # 日内动量（尾盘强势）
        "f3_return": 0.15,      # 温和上涨动量
        "f4_liquidity": 0.10,   # 流动性适中
        "f5_spread": 0.05,      # 低交易成本
        "f6_efficiency": 0.25,  # 趋势效率（核心因子）
        "f7_vol_price": 0.15,   # 量价配合
    }

    df["composite_score"] = sum(
        df[col].fillna(0.5) * weights[col] for col in factor_columns
    )

    # ── 4. 筛选Top 30 ──
    print(f"\n  [3/4] 筛选Top 30 ...")

    # 排除涨跌停（涨幅>9.5% 或 <-9.5%）
    df = df[df["pct_change"].abs() < 9.5]

    # 排除成交额过低（流动性不足）
    amount_median = df["amount"].median()
    df = df[df["amount"] > amount_median * 0.1]

    # 按综合得分排序
    top30 = df.nlargest(30, "composite_score").copy()
    top30 = top30.reset_index(drop=True)

    # ── 5. 输出报告 ──
    print(f"\n  [4/4] 生成报告")
    print("\n" + "=" * 64)
    print(f"  📊 下周一（{next_trade}）买入推荐报告")
    print("=" * 64)
    print(f"  数据来源: 新浪实时行情")
    print(f"  数据时间: {df['时间戳'].iloc[0] if '时间戳' in df.columns else '15:30'}")
    print(f"  筛选范围: 全A股（排除北交所、ST、涨跌停）")
    print(f"  有效样本: {len(df)} 只")
    print(f"  评分模型: 7因子综合评分（日内动量+趋势效率+量价配合+稳定性）")
    print(f"  推荐标的: Top 30")
    print("-" * 64)
    print(f"  {'排名':<4} {'代码':<8} {'名称':<10} {'最新价':>7} {'涨跌幅%':>7} {'评分':>7} {'成交额(万)':>10}")
    print("-" * 64)

    for rank, (_, row) in enumerate(top30.iterrows(), 1):
        amount_wan = row["amount"] / 10000
        print(f"  {rank:<4} {row['symbol']:<8} {row['name']:<10} "
              f"{row['close']:>7.2f} {row['pct_change']:>7.2f} "
              f"{row['composite_score']:>7.4f} {amount_wan:>10.0f}")

    print("-" * 64)

    # ══════════════════════════════════════════
    # 分组推荐
    # ══════════════════════════════════════════
    print(f"\n  📋 投资建议:")
    print(f"  {'─' * 54}")
    print(f"  1. 配置方式: 等权配置，单只股票仓位 ≈ 3.3%")
    print(f"  2. 调仓周期: 建议持有5个交易日（约一周）")
    print(f"  3. 止损线: 单只跌幅>8% 或 组合回撤>10%")
    print(f"  4. 止盈线: 单只涨幅>15% 考虑减仓")
    print(f"  5. 交易规则: A股T+1，买入后次日方可卖出")
    print(f"  6. 买入时机: {next_trade} 开盘后分批建仓")
    print(f"     - 9:30-10:00 买入50%仓位")
    print(f"     - 10:30-11:00 买入剩余50%")

    # 分档推荐
    score_range = top30["composite_score"].max() - top30["composite_score"].min()
    tier1 = top30.head(10)
    tier2 = top30.iloc[10:20]
    tier3 = top30.iloc[20:30]

    print(f"\n  🥇 强烈推荐 (Top 1-10) — 核心仓位 50%")
    for _, row in tier1.iterrows():
        print(f"     {row['symbol']} {row['name']} "
              f"¥{row['close']:.2f} 涨跌{row['pct_change']:+.2f}%")

    print(f"\n  🥈 推荐 (Top 11-20) — 配置仓位 30%")
    for _, row in tier2.iterrows():
        print(f"     {row['symbol']} {row['name']} "
              f"¥{row['close']:.2f} 涨跌{row['pct_change']:+.2f}%")

    print(f"\n  🥉 关注 (Top 21-30) — 观察仓位 20%")
    for _, row in tier3.iterrows():
        print(f"     {row['symbol']} {row['name']} "
              f"¥{row['close']:.2f} 涨跌{row['pct_change']:+.2f}%")

    # ══════════════════════════════════════════
    # 风险提示
    # ══════════════════════════════════════════
    print(f"\n  ⚠ 风险提示:")
    print(f"  {'─' * 54}")
    print(f"  • 本报告基于量化多因子模型生成，仅供参考，不构成投资建议")
    print(f"  • 模型依赖日内快照数据，未包含历史趋势因子")
    print(f"  • 实盘交易需考虑T+1、涨跌停限制及交易成本")
    print(f"  • 单一交易日快照存在噪声，建议结合基本面分析")
    print(f"  • 市场有风险，投资需谨慎")
    print(f"  • 过去表现不代表未来收益")

    # ══════════════════════════════════════════
    # 生成 Markdown 报告
    # ══════════════════════════════════════════
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        f"buy_report_{next_trade}.md"
    )
    generate_markdown_report(top30.head(10), today, next_trade, df, weights, report_path)
    print(f"\n  Markdown报告已保存: {report_path}")

    print(f"\n  报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)
    return 0


def factor_rationale(row: dict, weights: dict) -> list[tuple[str, str]]:
    """根据因子得分生成买入理由."""
    reasons: list[tuple[str, str]] = []

    # 趋势效率（核心因子，权重最高）
    te = row.get("trend_efficiency", 0)
    if te > 0.6:
        reasons.append(("趋势效率", f"涨跌/振幅比={te:.2f}，资金推动型上涨，趋势效率极优"))
    elif te > 0.3:
        reasons.append(("趋势效率", f"涨跌/振幅比={te:.2f}，上涨效率良好，多空博弈中多方占优"))
    elif te > 0:
        reasons.append(("趋势效率", f"上涨趋势确立，消耗较少资金完成涨幅"))

    # 日内动量
    ip = row.get("intraday_position", 0.5)
    if ip > 0.8:
        reasons.append(("日内动量", f"收盘位于日内{ip*100:.0f}%高位，尾盘强势，次日惯性上涨概率大"))
    elif ip > 0.6:
        reasons.append(("日内动量", f"收盘位于日内{ip*100:.0f}%分位，日内趋势向上"))

    # 涨幅因子
    pct = row.get("pct_change", 0)
    if 1 <= pct <= 5:
        reasons.append(("涨幅适中", f"涨幅{pct:+.2f}%处于温和区间，避免追高风险，兼具安全边际"))
    elif 0.5 <= pct < 1:
        reasons.append(("形态蓄势", f"微涨{pct:+.2f}%，低位蓄势待发"))
    elif 5 < pct <= 8:
        reasons.append(("强势突破", f"涨幅{pct:+.2f}%明确突破，短期趋势确认"))

    # 稳定性
    amp = row.get("amplitude", 0)
    if amp < 0.03:
        reasons.append(("低波动", f"日内振幅仅{amp*100:.1f}%，筹码锁定良好，稳定性突出"))
    elif amp < 0.05:
        reasons.append(("波动适中", f"日内振幅{amp*100:.1f}%，波动可控"))

    # 量价配合
    amount = row.get("amount", 0) / 1e4
    if amount > 30000:
        reasons.append(("资金活跃", f"成交额{amount:.0f}万，市场关注度高，流动性充裕"))
    elif amount > 15000:
        reasons.append(("交投活跃", f"成交额{amount:.0f}万，交易活跃度良好"))

    return reasons


def generate_markdown_report(
    top10: pd.DataFrame,
    report_date: date,
    target_date: date,
    full_df: pd.DataFrame,
    weights: dict,
    output_path: str,
) -> None:
    """生成专业的 Markdown 买入报告."""
    lines: list[str] = []

    # ── 标题 ──
    lines.append(f"# A股量化买入推荐报告")
    lines.append(f"")
    lines.append(f"**生成日期**: {report_date} ｜ **目标交易日**: {target_date}（周一）")
    lines.append(f"**数据来源**: 新浪实时行情 ｜ **覆盖范围**: 全A股 {len(full_df)} 只有效标的")
    lines.append(f"**评分模型**: 7因子综合评分（趋势效率 25% + 日内动量 20% + 温和上涨 15% + 量价配合 15% + 低波动 10% + 流动性 10% + 低利差 5%）")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # ── 总览表 ──
    lines.append(f"## Top 10 推荐总览")
    lines.append(f"")
    lines.append(f"| 排名 | 代码 | 名称 | 最新价 | 涨跌幅 | 综合评分 | 成交额(万) |")
    lines.append(f"| ---: | :--- | :--- | ---: | ---: | ---: | ---: |")
    for _, row in top10.iterrows():
        rank = row.name + 1
        amount_wan = row["amount"] / 1e4
        lines.append(
            f"| {rank} | {row['symbol']} | {row['name']} | "
            f"¥{row['close']:.2f} | {row['pct_change']:+.2f}% | "
            f"{row['composite_score']:.4f} | {amount_wan:,.0f} |"
        )
    lines.append(f"")

    # ── 逐只详情 ──
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 逐只分析")
    lines.append(f"")

    for _, row in top10.iterrows():
        rank = row.name + 1
        symbol = row["symbol"]
        name = row["name"]
        close = float(row["close"])
        pct = float(row["pct_change"])
        amplitude = float(row.get("amplitude", 0))
        ipos = float(row.get("intraday_position", 0.5))
        te = float(row.get("trend_efficiency", 0))
        score = float(row["composite_score"])
        amount = float(row["amount"]) / 1e4
        high = float(row["high"])
        low = float(row["low"])
        open_price = float(row["open"])
        pre_close = float(row["pre_close"])

        # ── 技术价格计算 ──
        # 建议买入价: 开盘价附近，略高于昨收确认不破位
        entry_price = round(max(open_price * 0.998, pre_close * 1.002), 2)

        # 止损价: 基于日内振幅的动态止损
        dynamic_stop_pct = max(amplitude * 2.0, 0.05)  # 至少5%
        stop_loss = round(close * (1 - dynamic_stop_pct), 2)

        # 止盈: 三级
        tp1 = round(close * 1.05, 2)   # +5% 第一目标
        tp2 = round(close * 1.10, 2)   # +10% 第二目标
        tp3 = round(close * 1.18, 2)   # +18% 第三目标

        # ATF 估算: 日内波动幅度作为单日波动参考
        atr_est = high - low

        lines.append(f"### {rank}. {name}（{symbol}）")
        lines.append(f"")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"| :--- | :--- |")
        lines.append(f"| 最新价 | ¥{close:.2f} |")
        lines.append(f"| 涨跌幅 | {pct:+.2f}% |")
        lines.append(f"| 日内振幅 | {amplitude*100:.2f}% |")
        lines.append(f"| 日内位置 | {ipos*100:.0f}%（{low:.2f} → {high:.2f}） |")
        lines.append(f"| 趋势效率 | {te:.3f} |")
        lines.append(f"| 成交额 | {amount:,.0f} 万 |")
        lines.append(f"| 综合评分 | {score:.4f} |")
        lines.append(f"")

        # ── 买入理由 ──
        reasons = factor_rationale(row.to_dict(), weights)
        lines.append(f"**买入理由**：")
        for tag, detail in reasons:
            lines.append(f"- **{tag}**: {detail}")
        lines.append(f"")

        # ── 交易计划 ──
        lines.append(f"**交易计划**：")
        lines.append(f"")
        lines.append(f"| 项目 | 价格 | 说明 |")
        lines.append(f"| :--- | :--- | :--- |")
        lines.append(f"| 🟢 **建议买入** | ¥{entry_price:.2f} | 开盘价附近，确认不破昨收后入场 |")
        lines.append(f"| 🔴 **止损价** | ¥{stop_loss:.2f} | 基于振幅的动态止损（-{dynamic_stop_pct*100:.1f}%），跌破坚决离场 |")
        lines.append(f"| 🟡 **止盈1** | ¥{tp1:.2f} | +5% 第一目标，达到后减仓30% |")
        lines.append(f"| 🟠 **止盈2** | ¥{tp2:.2f} | +10% 第二目标，达到后再减仓30% |")
        lines.append(f"| 🟢 **止盈3** | ¥{tp3:.2f} | +18% 第三目标，清仓离场 |")
        lines.append(f"")

        # ── 风险度量 ──
        risk_pct = round((close - stop_loss) / close * 100, 1)
        reward_pct = round((tp2 - close) / close * 100, 1)
        rr_ratio = round(reward_pct / risk_pct, 1) if risk_pct > 0 else 0

        lines.append(f"**风险度量**：")
        lines.append(f"- 单笔风险: {risk_pct}%")
        lines.append(f"- 盈亏比 (TP2): {rr_ratio}:1")
        lines.append(f"- 建议仓位: 总资金的 3~5%（组合10只等权配置）")
        lines.append(f"- T+1 规则: {target_date} 买入，{target_date}次日方可卖出")
        lines.append(f"")

    # ── 操作摘要 ──
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 操作摘要")
    lines.append(f"")
    lines.append(f"| # | 代码 | 名称 | 买入价 | 止损价 | 止盈1 (+5%) | 止盈2 (+10%) | 盈亏比 |")
    lines.append(f"| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in top10.iterrows():
        rank = row.name + 1
        close = float(row["close"])
        open_price = float(row["open"])
        pre_close = float(row["pre_close"])
        amplitude = float(row.get("amplitude", 0))
        dynamic_stop_pct = max(amplitude * 2.0, 0.05)
        entry = round(max(open_price * 0.998, pre_close * 1.002), 2)
        sl = round(close * (1 - dynamic_stop_pct), 2)
        tp1 = round(close * 1.05, 2)
        tp2 = round(close * 1.10, 2)
        risk = round((close - sl) / close * 100, 1)
        reward = round((tp2 - close) / close * 100, 1)
        rr = round(reward / risk, 1) if risk > 0 else 0
        lines.append(f"| {rank} | {row['symbol']} | {row['name']} | ¥{entry:.2f} | ¥{sl:.2f} | ¥{tp1:.2f} | ¥{tp2:.2f} | {rr}:1 |")
    lines.append(f"")

    # ── 建仓策略 ──
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 建仓策略")
    lines.append(f"")
    lines.append(f"### 分时建仓（{target_date} 周一）")
    lines.append(f"")
    lines.append(f"| 时段 | 操作 | 仓位 | 说明 |")
    lines.append(f"| :--- | :--- | :--- | :--- |")
    lines.append(f"| 9:25 集合竞价 | 观察开盘价 | — | 若高开>3%，等待回落；低开<-1%不参与 |")
    lines.append(f"| 9:30-10:00 | 首批建仓 | 50% | 分3批买入，每批间隔5分钟，避免冲击成本 |")
    lines.append(f"| 10:30-11:00 | 二批建仓 | 50% | 确认股价稳定在均线上方后加仓 |")
    lines.append(f"| 14:30-15:00 | 尾盘调整 | — | 若某只尾盘急跌>3%，减半仓 |")
    lines.append(f"")

    # ── 风控规则 ──
    lines.append(f"## 风控规则")
    lines.append(f"")
    lines.append(f"| 规则 | 条件 | 动作 |")
    lines.append(f"| :--- | :--- | :--- |")
    lines.append(f"| 个股止损 | 跌至止损价 | **无条件离场**，不问原因 |")
    lines.append(f"| 组合回撤 | 当日组合净值回撤>5% | 减仓至50%，次日评估 |")
    lines.append(f"| 连续止损 | 同日3只触发止损 | 暂停买入，检查市场系统性风险 |")
    lines.append(f"| 时间止损 | 持有5个交易日未达TP1 | 减仓50%，释放资金 |")
    lines.append(f"| 大盘联动 | 沪深300日内跌幅>3% | 暂停新开仓，尾盘评估是否减仓 |")
    lines.append(f"| 涨跌停 | 持仓涨停无法卖出 | 次日开盘集合竞价卖出 |")
    lines.append(f"")

    # ── 风险提示 ──
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 免责声明")
    lines.append(f"")
    lines.append(f"> ⚠️ **风险提示**")
    lines.append(f"> ")
    lines.append(f"> 1. 本报告由量化多因子模型自动生成，仅供研究参考，**不构成任何投资建议**")
    lines.append(f"> 2. 模型基于日内快照数据，未包含历史趋势和基本面因子，存在一定局限性")
    lines.append(f"> 3. 回测表现不代表未来收益，市场存在不可预测的系统性风险")
    lines.append(f"> 4. 实盘交易请充分考虑个人风险承受能力、资金管理和市场流动性")
    lines.append(f"> 5. A股实行T+1交易制度，买入当日不可卖出")
    lines.append(f"> ")
    lines.append(f"> **投资有风险，入市需谨慎。过往业绩不预示未来表现。**")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"*报告由 Quant System v1.0 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
