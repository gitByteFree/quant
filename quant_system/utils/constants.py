"""全局常量定义."""

# A股行业分类（申万一级，28个行业）
SW_INDUSTRY_LIST: list[str] = [
    "农林牧渔", "采掘", "化工", "钢铁", "有色金属", "电子", "家用电器",
    "食品饮料", "纺织服装", "轻工制造", "医药生物", "公用事业", "交通运输",
    "房地产", "商业贸易", "休闲服务", "综合", "建筑材料", "建筑装饰",
    "电气设备", "国防军工", "计算机", "传媒", "通信", "银行", "非银金融",
    "汽车", "机械设备",
]

# 交易费率（A股）
STAMP_DUTY_RATE: float = 0.001       # 卖出印花税 0.1%
COMMISSION_RATE: float = 0.00025     # 佣金 0.025%
SLIPPAGE_BASIS_POINTS: float = 0.001 # 滑点 0.1%

# 风控参数
MAX_SINGLE_STOCK_WEIGHT: float = 0.03        # 个股最大权重 3%
MAX_INDUSTRY_DEVIATION: float = 0.05          # 行业偏离度上限 5%
MAX_ROLLING_DRAWDOWN: float = 0.15            # 60日滚动最大回撤阈值 15%
DRAWDOWN_POSITION_REDUCTION: float = 0.5      # 回撤触发后仓位降至 50%

# 因子相关
MAD_THRESHOLD: float = 3.0   # 去极值MAD倍数
ZSCORE_EPS: float = 1e-8     # Z-score标准化防零

# 缺失值处理
FORWARD_FILL_LIMIT: int = 5  # 前向填充最大天数

# 日期
TRADING_DAY_START: str = "2010-01-01"
DEFAULT_RISK_FREE_RATE: float = 0.025  # 无风险利率 2.5%

# 模型
DEFAULT_SEED: int = 42
DEFAULT_SEQ_LEN: int = 60  # Transformer序列长度
