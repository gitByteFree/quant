# Quant System - A股量化分析系统

面向A股市场的量化分析系统，实现从数据获取、因子工程、模型训练到回测评估的完整链路。

## 架构

```
utils --> config --> data --> factors --> models --> backtest --> evaluation
```

## 快速开始

```bash
pip install -r requirements.txt
python -c "from quant_system.utils.config_loader import load_config; print(load_config('quant_system/config/config.yaml'))"
```

## 目录结构

```
quant_system/
├── config/          # YAML配置文件
├── data/            # 数据获取、清洗、存储
│   ├── fetcher/     # 数据源适配器
│   ├── cleaner/     # 数据清洗
│   └── storage/     # Parquet分区存储
├── factors/         # 因子工程（30+因子）
│   ├── momentum/    # 动量类
│   ├── volatility/  # 波动类
│   ├── value/       # 价值类
│   ├── quality/     # 质量类
│   └── sentiment/   # 情绪类
├── models/          # 模型层（LGB/TabNet/Transformer）
├── backtest/        # 回测系统
├── evaluation/      # 评估体系
├── docs/            # 技术文档
├── tests/           # 单元测试
└── utils/           # 工具函数
```

## 模块使用

```python
from quant_system.data.akshare_source import AkShareDataSource
from quant_system.factors.base import compute_all_factors
from quant_system.models.lightgbm_model import LightGBMModel
from quant_system.backtest.engine import BacktestEngine

# 获取数据
data = AkShareDataSource()
data.fetch_daily("2020-01-01", "2024-12-31")

# 计算因子
factor_panel = compute_all_factors(data, list_factors(), "2020-01-01", "2024-12-31")

# 训练模型
model = LightGBMModel()
model.fit(X_train, y_train)

# 回测
engine = BacktestEngine(data, model, config)
result = engine.run("2022-01-01", "2024-12-31")
```
