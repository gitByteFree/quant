# 数据字典

## 1. 日线数据表 (daily)

**存储格式**: Parquet，按 trade_date 日期分区
**频率**: 日频
**覆盖范围**: 全A股（不含北交所），2010年至今

| 字段名 | 类型 | 说明 | 示例 |
|---|---|---|---|
| symbol | string | 股票代码（6位） | "000001" |
| trade_date | datetime | 交易日期 | 2024-01-15 |
| open | float64 | 开盘价（复权后） | 13.25 |
| high | float64 | 最高价（复权后） | 13.80 |
| low | float64 | 最低价（复权后） | 13.10 |
| close | float64 | 收盘价（复权后） | 13.50 |
| volume | float64 | 成交量（手） | 1234567.0 |
| amount | float64 | 成交额（元） | 166666545.0 |
| turnover | float64 | 换手率（%） | 2.35 |
| pct_change | float64 | 涨跌幅（%） | 1.50 |
| adjust_type | string | 复权方式 qfq/hfq | "qfq" |
| is_st | int8 | 是否ST (1=是) | 0 |
| is_limit_up | int8 | 是否涨停 (1=是) | 0 |
| is_limit_down | int8 | 是否跌停 (1=是) | 0 |
| is_long_suspended | int8 | 是否长期停牌 (1=是) | 0 |

**分区路径示例**:
```
data/parquet/daily/trade_date=2024-01-15/data.parquet
```

## 2. 财务数据表 (financial)

**存储格式**: Parquet（单文件，定期全量更新）
**频率**: 季频（每季度更新）
**覆盖范围**: 全A股

| 字段名 | 类型 | 说明 |
|---|---|---|
| symbol | string | 股票代码 |
| report_period | string | 报告期 "20231231" |
| ann_date | datetime | 公告日期 |
| revenue | float64 | 营业收入（元） |
| net_profit | float64 | 净利润（元） |
| total_assets | float64 | 总资产（元） |
| total_equity | float64 | 净资产（元） |
| operating_cf | float64 | 经营活动现金流（元） |
| roe | float64 | ROE（%） |
| gross_margin | float64 | 毛利率（%） |
| eps | float64 | 每股收益 |

**A股财报披露截止日**:

| 报告期 | 截止日期 | 生效日期 |
|---|---|---|
| 年报 (12/31) | 4月30日 | 5月1日 |
| 一季报 (3/31) | 4月30日 | 5月1日 |
| 中报 (6/30) | 8月31日 | 9月1日 |
| 三季报 (9/30) | 10月31日 | 11月1日 |

## 3. 因子数据表 (factors)

**存储格式**: Parquet，按因子名称 + trade_date 双层分区
**频率**: 日频
**覆盖范围**: 30+因子

| 字段名 | 类型 | 说明 |
|---|---|---|
| trade_date | datetime | 交易日期 |
| symbol | string | 股票代码 |
| value | float64 | 因子值（标准化后） |

**分区路径示例**:
```
data/parquet/factors/factor_name=momentum_20d/trade_date=2024-01-15/data.parquet
```

### 3.1 因子列表

#### 动量类 (momentum) - 8个因子

| 因子名 | 说明 | 计算方式 |
|---|---|---|
| momentum_5d | 5日收益率 | $\frac{P_t - P_{t-5}}{P_{t-5}}$ |
| momentum_20d | 20日收益率 | $\frac{P_t - P_{t-20}}{P_{t-20}}$ |
| momentum_60d | 60日收益率 | $\frac{P_t - P_{t-60}}{P_{t-60}}$ |
| momentum_120d | 120日收益率 | $\frac{P_t - P_{t-120}}{P_{t-120}}$ |
| rsi_14d | 相对强弱指标 | $100 - \frac{100}{1 + RS}, RS = \frac{avg\_gain}{avg\_loss}$ |
| macd_divergence | MACD背离度 | $\frac{DIF - DEA}{close}$ |
| weighted_momentum | 加权动量 | $0.4r_5 + 0.3r_{20} + 0.2r_{60} + 0.1r_{120}$ |
| momentum_stability | 动量稳定性 | $\frac{\mu_{ret,20}}{\sigma_{ret,20}}$ |

#### 波动类 (volatility) - 5个因子

| 因子名 | 说明 | 计算方式 |
|---|---|---|
| atr_14d | 平均真实波幅 | $\frac{EMA(TR, 14)}{close}$ |
| hv_20d | 20日历史波动率 | $\sigma_{log\_ret, 20} \times \sqrt{252}$ |
| hv_60d | 60日历史波动率 | $\sigma_{log\_ret, 60} \times \sqrt{252}$ |
| downside_volatility | 下行波动率 | $\sigma(\min(r_{log}, 0)) \times \sqrt{252}$ |
| max_drawdown_60d | 60日最大回撤 | $\min(\frac{P_t}{\max_{t-60:t} P} - 1)$ |

#### 价值类 (value) - 6个因子

| 因子名 | 说明 | 计算方式 |
|---|---|---|
| ep_ttm | 市盈率倒数 | $\frac{NetProfit_{TTM}}{MarketCap}$ |
| bp | 市净率倒数 | $\frac{BookValue}{MarketCap}$ |
| sp_ttm | 市销率倒数 | $\frac{Revenue_{TTM}}{MarketCap}$ |
| dividend_yield | 股息率 | $\frac{Dividend}{MarketCap}$ |
| cfop_ttm | 经营现金流市值比 | $\frac{OpCF_{TTM}}{MarketCap}$ |
| ep_fwd | 预期EP | $\frac{EP_{TTM}}{1 + momentum_{60d}}$ |

#### 质量类 (quality) - 6个因子

| 因子名 | 说明 | 计算方式 |
|---|---|---|
| roe_change | ROE变动 | $ROE_{近一季} - ROE_{一年前}$ |
| roe_stability | ROE稳定性 | $\frac{1}{\sigma_{ROE, 3年}}$ |
| gross_margin_stability | 毛利率稳定性 | $\frac{1}{1 + \sigma_{GM, 3年}}$ |
| accruals | 应计利润 | $-\frac{NetProfit - OpCF}{TotalAssets}$ |
| asset_turnover | 资产周转率 | $\frac{Revenue}{TotalAssets}$ |
| leverage | 杠杆率 | $\frac{1}{1 + Debt/Equity}$ |

#### 情绪类 (sentiment) - 5个因子

| 因子名 | 说明 | 计算方式 |
|---|---|---|
| abnormal_turnover | 异常换手率 | $\frac{turnover - MA_{20}(turnover)}{MA_{20}(turnover)}$ |
| volume_ratio | 量比 | $\log(\frac{volume}{MA_5(volume)} + 0.5)$ |
| north_flow_change | 北向资金变化 | 北向持仓变化率（需北向数据） |
| margin_change | 融资余额变化 | 融资余额变化率（需两融数据） |
| dragon_tiger_sentiment | 龙虎榜情绪 | 大涨+放量可能受游资关注 |

## 4. 指数数据表

**用途**: 基准比对

| 指数代码 | 名称 | 用途 |
|---|---|---|
| 000300 | 沪深300 | 大盘基准 |
| 000905 | 中证500 | 中盘基准 |
| 000852 | 中证1000 | 小盘基准 |

## 5. 行业分类

采用申万一级行业分类（28个行业），参见 `utils/constants.py:SW_INDUSTRY_LIST`。
