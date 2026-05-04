# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & Commands

```bash
# 首次安装
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 激活虚拟环境（每次使用前）
source venv/bin/activate

# 代码格式化
black quant_system/

# 类型检查
mypy quant_system/

# 运行测试
pytest tests/ --cov=quant_system --cov-report=term
pytest tests/test_factors.py -v  # 单个测试文件
pytest tests/ -k "test_ic"       # 按关键字筛选

# 验证导入
python -c "from quant_system.factors.base import list_factors; print(list_factors())"
```

## Architecture

**依赖方向（严格单向）**:
```
utils --> config --> data --> factors --> models --> backtest --> evaluation
```

每层只能依赖其左侧的模块。`utils/` 是最底层，无内部依赖。

## Core Patterns

### 因子注册表（`factors/base.py`）

新增因子 = 一个类 + `@register_factor` 装饰器，无需手动注册：

```python
@register_factor("momentum_20d", "momentum", "20日收益率")
class Momentum20D(BaseFactor):
    def compute(self, start_date, end_date) -> FactorResult:
        ...
```

所有因子存储在全局 `_factor_registry` 字典中。批量计算用 `compute_all_factors()`，结果合并用 `build_factor_panel()`，返回 MultiIndex `(trade_date, symbol)` 的 DataFrame。

### 模型接口（`models/base.py`）

所有模型遵循 sklearn 风格 `fit(X, y) / predict(X)`。`BaseModel` 同时定义了 `feature_importance()`, `save()`, `load()`。三个模型实现（LightGBM / TabNet / Transformer）可互换接入回测引擎。

### 回测引擎（`backtest/engine.py`）

事件驱动主循环：`MarketEvent → SignalEvent(模型预测) → OrderEvent(PortfolioManager) → FillEvent(ExecutionEngine)`。每步由风控(RiskManager)和成本(CostModel)过滤。

### 配置系统

YAML 文件支持 `${VAR:default}` 环境变量替换，通过 `utils/config_loader.py` 递归解析。`config/schema.py` 提供 dataclass 类型校验。

## Key Files

- `utils/constants.py` — 全局常量（费率、风控阈值、MAD参数、随机种子）
- `factors/base.py` — 因子基类 + 注册表 + `compute_all_factors()`
- `factors/factor_utils.py` — MAD去极值、行业/市值中性化、Z-score标准化
- `models/trainer.py` — 滚动/扩展窗口训练、时间衰减+波动率倒数样本权重
- `backtest/engine.py` — 回测主循环，整合所有模块
- `backtest/risk_manager.py` — 行业偏离度、60日回撤预警降仓
- `evaluation/metrics.py` — 年化收益/夏普/Calmar/信息比率等全套指标

## Data Flow

1. `data/akshare_source.py` 从 akShare 获取原始数据
2. `data/cleaner.py` 清洗（ST、停牌、涨跌停、缺失值）
3. `data/storage.py` 按 `trade_date` 分区写入 Parquet（`data/parquet/daily/trade_date=YYYY-MM-DD/data.parquet`）
4. 因子计算从 ParquetStore 读取日线，产出因子面板
5. 模型训练读取因子面板，产出预测信号
6. 回测引擎消费信号，产出组合净值曲线
7. 评估模块分析净值曲线，输出绩效报告

## A-Share Specific Rules

- T+1：当日买入次日才能卖出（`backtest/execution.py:_pending_sells`）
- 涨跌停不可交易（10%主板，`backtest/execution.py:can_trade()`）
- 印花税仅卖出单边 0.1%（`backtest/cost.py`）
- 财报截止日对齐：年报 4.30→5.1 生效，中报 8.31→9.1 生效（`factors/value/value_factors.py:FinancialCalendar`）
