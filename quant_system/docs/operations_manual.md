# 部署与运维手册

## 1. 环境配置

### 1.1 系统要求

- **操作系统**: Linux (Ubuntu 20.04+) / macOS 12+
- **Python**: 3.11+
- **内存**: ≥32GB RAM
- **磁盘**: ≥500GB SSD
- **GPU** (可选): NVIDIA GPU with CUDA 11.8+ (用于Transformer训练)

### 1.2 安装步骤

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证安装
python -c "
from quant_system.data.akshare_source import AkShareDataSource
from quant_system.factors.base import list_factors
from quant_system.models.lightgbm_model import LightGBMModel
print('安装成功!')
print(f'已注册因子: {len(list_factors())}个')
"
```

### 1.3 配置文件

修改 `quant_system/config/config.yaml`：

```yaml
data:
  provider: akshare  # 或 tushare (需要token)
  cache_path: data/parquet  # 数据存储路径
  date_range:
    start: "2010-01-01"
    end: null  # null表示至今

backtest:
  initial_capital: 10000000  # 初始资金
  top_k: 100  # 持仓数量
  weighting: equal  # 等权配置
```

环境变量覆盖：
```bash
export DATA_PROVIDER=akshare
export BACKTEST_INITIAL_CAPITAL=20000000
```

## 2. 数据更新流程

### 2.1 首次全量获取

```python
from quant_system.data.akshare_source import AkShareDataSource
from quant_system.data.cleaner import clean_daily_data
from quant_system.data.storage import ParquetStore

# 获取全A股日线数据
source = AkShareDataSource()
df = source.fetch_daily("2010-01-01", "2024-12-31")

# 清洗
df_clean = clean_daily_data(df)

# 存储
store = ParquetStore("data/parquet")
store.write_daily(df_clean)

print(store.get_stats())
```

### 2.2 每日增量更新

```bash
#!/bin/bash
# cron: 0 18 * * 1-5  (每个交易日下午6点执行)

source venv/bin/activate
python -c "
from datetime import datetime, timedelta
from quant_system.data.akshare_source import AkShareDataSource
from quant_system.data.cleaner import clean_daily_data
from quant_system.data.storage import ParquetStore

# 获取最近5个交易日的数据（覆盖模式）
end = datetime.now().strftime('%Y%m%d')
start = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')

source = AkShareDataSource()
df = source.fetch_daily(start, end)
df_clean = clean_daily_data(df)

store = ParquetStore('data/parquet')
store.write_daily(df_clean)

print(f'{datetime.now()}: 更新完成, {len(df_clean)}条记录')
"
```

### 2.3 数据完整性检查

```python
from quant_system.data.storage import ParquetStore
from quant_system.utils.calendar import TradingCalendar

store = ParquetStore("data/parquet")
cal = TradingCalendar()

data_start, data_end = store.get_date_range()
print(f"数据范围: {data_start} ~ {data_end}")

# 检查缺失交易日
expected = cal.trade_dates_between(data_start, data_end)
# ... 对比检查
```

## 3. 因子计算流程

### 3.1 全量因子计算

```python
from quant_system.data.storage import ParquetStore
from quant_system.factors.base import compute_all_factors, list_factors
from quant_system.factors.factor_utils import preprocess_factor

# 读取数据
store = ParquetStore("data/parquet")
df = store.read_daily("2010-01-01", "2024-12-31")

# 计算所有因子
factor_results = compute_all_factors(df, list_factors())

# 预处理（去极值+中性化+标准化）
for name, factor in factor_results.items():
    factor_clean = preprocess_factor(factor, standardize="zscore")
    store.write_factors(name, factor_clean.reset_index(name="value"))

print(f"已计算 {len(factor_results)} 个因子")
```

### 3.2 因子评估

```python
from quant_system.factors.evaluator import factor_summary, compute_ic

# 读取因子和收益率数据
factor = store.read_factors("momentum_20d")
factor_series = factor.set_index(["trade_date", "symbol"])["value"]

# 计算未来1日收益率
df["ret_1d"] = df.groupby("symbol")["close"].pct_change().shift(-1)
forward_ret = df.set_index(["trade_date", "symbol"])["ret_1d"]

# IC分析
summary = factor_summary(factor_series, forward_ret, "momentum_20d")
for k, v in summary.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
```

## 4. 模型训练流程

### 4.1 单模型训练

```python
from quant_system.factors.base import build_factor_panel
from quant_system.models.lightgbm_model import LightGBMModel
import numpy as np

# 构建因子面板
panel = build_factor_panel(factor_results)

# 构建标签：未来5日收益率
y = ...  # forward return

# 划分时间序列训练/验证集
split_date = "2023-01-01"
train_mask = panel.index.get_level_values("trade_date") < split_date
valid_mask = panel.index.get_level_values("trade_date") >= split_date

X_train = panel[train_mask].values.astype(np.float32)
y_train = y[train_mask].values.astype(np.float32)
X_valid = panel[valid_mask].values.astype(np.float32)
y_valid = y[valid_mask].values.astype(np.float32)

# 训练
model = LightGBMModel()
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
print(model.feature_importance().head(10))
model.save("models/lgb_model.pkl")
```

### 4.2 滚动窗口训练

```python
from quant_system.models.trainer import ModelTrainer
from quant_system.models.lightgbm_model import LightGBMModel

model = LightGBMModel()
trainer = ModelTrainer(
    model=model,
    window_type="rolling",
    train_window_years=5,
    valid_window_years=1,
    step_months=6,
)

trainer.fit(panel, y)
print(trainer.summary())

# 使用最新模型预测
predictions = trainer.predict(X_new, method="latest")
```

### 4.3 模型重训练机制

```
频率: 月度
触发条件:
  - 每月第一个交易日自动执行
  - 数据更新超过20个交易日后触发
  - 因子IC_IR月度变化超过±0.2时触发

流程:
  1. 获取最新数据
  2. 计算最新因子值
  3. 使用最新的5年数据重新训练
  4. 评估新旧模型差异
  5. 如果新模型显著优于旧模型（IC_IR提升>0.05），替换
```

## 5. 回测执行

### 5.1 运行回测

```python
from quant_system.backtest.engine import BacktestEngine
from quant_system.evaluation.metrics import compute_all_metrics, PerformanceMetrics
from quant_system.utils.config_loader import load_config

config = load_config("quant_system/config/config.yaml")

engine = BacktestEngine(model, df, config)
portfolio_df = engine.run("2022-01-01", "2024-12-31")

# 计算绩效指标
metrics = compute_all_metrics(portfolio_df["total_value"])
print(metrics.summary())
```

### 5.2 批量回测场景

```python
from quant_system.evaluation.segment import analyze_segments, A_SHARE_SEGMENTS

# 牛熊市分段分析
seg_results = analyze_segments(portfolio_df["total_value"], A_SHARE_SEGMENTS)
print(pd.DataFrame(seg_results))
```

## 6. 监控与告警

### 6.1 日志配置

```python
from quant_system.utils.logger import get_logger

logger = get_logger("monitor", log_file="logs/monitor.log")
```

### 6.2 数据质量监控

| 监控项 | 频率 | 告警条件 |
|---|---|---|
| 日线数据更新 | 每日18:30 | 缺失超过2个交易日 |
| 因子覆盖率 | 每日 | 任一因子覆盖率<80% |
| 财务数据更新 | 每月 | 财报截止日后30天未更新 |

### 6.3 模型监控

| 监控项 | 频率 | 告警条件 |
|---|---|---|
| 预测值分布 | 每日 | KS统计量>0.3（分布偏移） |
| 在线IC | 每日 | 20日滚动IC低于历史均值-2std |
| 预测极值比例 | 每日 | >5%的预测值超3个标准差 |

### 6.4 绩效监控

| 监控项 | 频率 | 告警条件 |
|---|---|---|
| 滚动60日收益率 | 每日 | 低于基准-5% |
| 滚动60日回撤 | 每日 | 超过15% |
| 行业偏离度 | 每日 | 超过5% |
| 个股权重 | 每日 | 超过3% |

## 7. 故障恢复

### 7.1 数据恢复

```python
# 从Parquet重新读取
store = ParquetStore("data/parquet")
df = store.read_daily("2010-01-01")
```

### 7.2 模型回滚

```python
# 加载上一版本模型
model = LightGBMModel()
model.load("models/lgb_model_v2.pkl")
```

### 7.3 回测验证

上线前必须通过回测验证：
- 前视偏差纯度测试
- 2018熊市回撤≤20%
- 换手率<50%（单边日频）

## 8. CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: black --check quant_system/
      - run: mypy quant_system/
      - run: pytest tests/ --cov=quant_system --cov-report=term --cov-fail-under=80
```
