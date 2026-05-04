"""配置数据类定义（dataclass + 类型校验）."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataSourceConfig:
    """数据源配置."""

    provider: str = "akshare"  # akshare | tushare | wind
    cache_path: str = "data/parquet"
    rate_limit_per_minute: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class DataCleanConfig:
    """数据清洗配置."""

    remove_st: bool = True
    remove_suspend_days: int = 60
    forward_fill_limit: int = 5
    mad_threshold: float = 3.0
    min_daily_volume: int = 0  # 最小日成交额，0=不过滤


@dataclass
class FactorConfig:
    """因子配置."""

    neutralization: bool = True
    neutralization_method: str = "regression"  # regression | simple
    industry_neutralize: bool = True
    market_cap_neutralize: bool = True
    standardization: str = "zscore"  # zscore | rank | minmax


@dataclass
class LightGBMConfig:
    """LightGBM模型配置."""

    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 8
    num_leaves: int = 128
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 100
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    random_state: int = 42
    objective: str = "regression"
    boosting_type: str = "gbdt"


@dataclass
class TabNetConfig:
    """TabNet模型配置."""

    n_d: int = 64
    n_a: int = 64
    n_steps: int = 5
    gamma: float = 1.5
    n_independent: int = 2
    n_shared: int = 2
    momentum: float = 0.3
    mask_type: str = "entmax"
    optimizer: str = "adam"
    learning_rate: float = 0.001
    scheduler_patience: int = 10


@dataclass
class TransformerConfig:
    """Transformer模型配置."""

    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 3
    d_feedforward: int = 512
    dropout: float = 0.1
    seq_len: int = 60
    learning_rate: float = 0.001
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 10


@dataclass
class EnsembleConfig:
    """Stacking集成配置."""

    n_folds: int = 5
    meta_model: str = "lightgbm"
    use_oof_features: bool = True


@dataclass
class TrainConfig:
    """训练配置."""

    window_type: str = "rolling"  # rolling | expanding
    train_window_years: int = 5
    valid_window_years: int = 1
    step_months: int = 6
    time_decay_alpha: float = 0.95
    use_volatility_weight: bool = True


@dataclass
class BacktestConfig:
    """回测配置."""

    initial_capital: float = 10_000_000.0
    rebalance_freq: str = "daily"  # daily | weekly
    top_k: int = 100
    weighting: str = "equal"  # equal | market_cap
    stamp_duty: float = 0.001
    commission: float = 0.00025
    slippage: float = 0.001
    max_single_weight: float = 0.03
    max_industry_deviation: float = 0.05
    max_drawdown_threshold: float = 0.15
    drawdown_reduction: float = 0.5
    benchmark: str = "000300"  # 沪深300


@dataclass
class SystemConfig:
    """系统总配置."""

    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    data_clean: DataCleanConfig = field(default_factory=DataCleanConfig)
    factor: FactorConfig = field(default_factory=FactorConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    tabnet: TabNetConfig = field(default_factory=TabNetConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    log_level: str = "INFO"
    log_file: str | None = None
    seed: int = 42
