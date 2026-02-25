"""
데이터 분할 모듈
시간 기반 train/valid/test 분할
"""

import math
from typing import Dict, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class SplitConfig:
    """데이터 분할 비율 설정"""
    train_ratio: float = 0.6
    valid_ratio: float = 0.2
    test_ratio: float = 0.2
    
    def __post_init__(self):
        """비율 합계 검증"""
        if not math.isclose(self.train_ratio + self.valid_ratio + self.test_ratio, 1.0, rel_tol=1e-6):
            raise ValueError(f"train/valid/test 비율의 합이 1.0이어야 합니다. "
                           f"현재: {self.train_ratio + self.valid_ratio + self.test_ratio}")


def time_split(
    X: pd.DataFrame, 
    y: pd.DataFrame, 
    cfg: SplitConfig = SplitConfig()
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    시간 순서 기반 데이터 분할
    
    Parameters:
    -----------
    X : DataFrame
        입력 데이터
    y : DataFrame
        타겟 데이터
    cfg : SplitConfig
        분할 비율 설정
        
    Returns:
    --------
    dict : {"train": (X_train, y_train), "valid": (X_valid, y_valid), "test": (X_test, y_test)}
    """
    n = len(X)
    if n == 0:
        raise ValueError("전처리/피처 생성 후 데이터셋이 비어있습니다.")

    n_train = int(n * cfg.train_ratio)
    n_valid = int(n * cfg.valid_ratio)

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_valid, y_valid = X.iloc[n_train:n_train + n_valid], y.iloc[n_train:n_train + n_valid]
    X_test, y_test = X.iloc[n_train + n_valid:], y.iloc[n_train + n_valid:]

    return {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid),
        "test": (X_test, y_test),
    }
