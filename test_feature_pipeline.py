
from pathlib import Path
from feature.feature_builder import FeatureBuilder

# 实例化FeatureBuilder，指定输入输出CSV路径
builder = FeatureBuilder(
    raw_csv=Path("data/raw/raw_20251230.csv"),
    feature_csv=Path("data/feature/feature_20251230.csv"),
)

# 执行完整特征工程流程
builder.process()