"""
딥러닝 설정 모듈
하이퍼파라미터 및 경로 설정
"""

from pathlib import Path
from dataclasses import dataclass


# 기본 디렉토리
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results" / "DL"

# 데이터 경로
RAW_DATA_DIR = DATA_DIR / "actual"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 결과 저장 경로
MODEL_SAVE_DIR = MODEL_DIR
RESULTS_SAVE_DIR = RESULTS_DIR


@dataclass
class DataConfig:
    """데이터 처리 설정"""
    # 시간 정합
    time_freq: str = "1min"  # 1분 간격
    
    # 결측치 처리
    short_term_hours: int = 3      # 단기: 1-3시간 -> ffill
    medium_term_hours: int = 12    # 중기: 4-12시간 -> EWMA
    long_term_hours: int = 48      # 장기: 48시간+ -> NaN
    ewma_span: int = 6
    
    # 이상치 처리
    outlier_method: str = "iqr"
    iqr_threshold: float = 1.5
    zscore_threshold: float = 3.0
    require_both: bool = True  # 도메인 + 통계 둘 다 이상치여야 처리
    
    # 리샘플링
    resample_freq: str = "1h"  # 1시간 간격
    resample_agg: str = "mean"


@dataclass
class FeatureConfig:
    """특성 생성 설정"""
    # 시간 특성
    add_time_features: bool = True
    add_sin_cos: bool = True
    
    # Lag 특성
    lag_hours: list = None
    
    # Rolling 특성
    rolling_windows: list = None
    rolling_stats: list = None
    
    def __post_init__(self):
        if self.lag_hours is None:
            self.lag_hours = [1, 2, 3, 6, 12, 24]
        if self.rolling_windows is None:
            self.rolling_windows = [3, 6, 12, 24]
        if self.rolling_stats is None:
            self.rolling_stats = ["mean", "std"]


@dataclass
class WindowConfig:
    """슬라이딩 윈도우 설정"""
    window_size: int = 24  # 24시간
    horizon: int = 1       # 1시간 후 예측
    stride: int = 1        # 1시간씩 이동


@dataclass
class SplitConfig:
    """데이터 분할 설정"""
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class LSTMConfig:
    """LSTM 모델 설정"""
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False


@dataclass
class TrainingConfig:
    """학습 설정"""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    patience: int = 10
    grad_clip: float = 1.0
    weight_decay: float = 0.0001


@dataclass
class PipelineConfig:
    """전체 파이프라인 설정"""
    data: DataConfig = None
    feature: FeatureConfig = None
    window: WindowConfig = None
    split: SplitConfig = None
    lstm: LSTMConfig = None
    training: TrainingConfig = None
    
    random_seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.feature is None:
            self.feature = FeatureConfig()
        if self.window is None:
            self.window = WindowConfig()
        if self.split is None:
            self.split = SplitConfig()
        if self.lstm is None:
            self.lstm = LSTMConfig()
        if self.training is None:
            self.training = TrainingConfig()


def get_default_config() -> PipelineConfig:
    """기본 설정 반환"""
    return PipelineConfig()
