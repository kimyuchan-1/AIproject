"""
슬라이딩 윈도우 모듈
시계열 데이터를 LSTM 입력 형태로 변환
"""

from typing import Tuple
import numpy as np
import pandas as pd


def create_sliding_windows(X: np.ndarray,
                          y: np.ndarray,
                          window_size: int,
                          horizon: int = 1,
                          stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    슬라이딩 윈도우 생성
    
    Parameters:
    -----------
    X : ndarray
        입력 특성 (shape: [n_samples, n_features])
    y : ndarray
        타겟 변수 (shape: [n_samples, n_targets])
    window_size : int
        과거 몇 개의 시간 스텝을 볼 것인지
    horizon : int
        미래 몇 스텝 후를 예측할 것인지
    stride : int
        윈도우 이동 간격
        
    Returns:
    --------
    tuple : (X_seq, y_seq)
        X_seq shape: [n_windows, window_size, n_features]
        y_seq shape: [n_windows, n_targets]
    """
    # y가 1차원이면 2차원으로 변환
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    X_seq = []
    y_seq = []
    
    for i in range(0, len(X) - window_size - horizon + 1, stride):
        # 입력: i부터 i+window_size까지의 과거 데이터
        X_window = X[i:i + window_size]
        
        # 타겟: i+window_size+horizon-1 시점의 값
        y_target = y[i + window_size + horizon - 1]
        
        X_seq.append(X_window)
        y_seq.append(y_target)
    
    return np.array(X_seq), np.array(y_seq)


def split_windowed_data(X_seq: np.ndarray,
                       y_seq: np.ndarray,
                       train_ratio: float = 0.7,
                       valid_ratio: float = 0.15,
                       test_ratio: float = 0.15) -> dict:
    """
    윈도우 데이터를 train/valid/test로 분할
    
    Parameters:
    -----------
    X_seq : ndarray
        윈도우 형태의 입력
    y_seq : ndarray
        대응하는 타겟
    train_ratio : float
        학습 데이터 비율
    valid_ratio : float
        검증 데이터 비율
    test_ratio : float
        테스트 데이터 비율
        
    Returns:
    --------
    dict : {"train": (X_train, y_train), "valid": (X_valid, y_valid), "test": (X_test, y_test)}
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
        "분할 비율의 합은 1.0이어야 합니다"
    
    n_samples = len(X_seq)
    train_size = int(n_samples * train_ratio)
    valid_size = int(n_samples * valid_ratio)
    
    # 순차적 분할 (시계열 순서 유지)
    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    
    X_valid = X_seq[train_size:train_size + valid_size]
    y_valid = y_seq[train_size:train_size + valid_size]
    
    X_test = X_seq[train_size + valid_size:]
    y_test = y_seq[train_size + valid_size:]
    
    return {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid),
        "test": (X_test, y_test)
    }
