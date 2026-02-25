"""
Sliding Window 모듈
시계열 데이터를 sliding window 방식으로 변환

Sliding Window란?
- 과거 N개의 시간 스텝을 입력으로 사용해서 미래 값을 예측
- 예: window_size=24이면 과거 24시간 데이터로 다음 시간 예측

사용 예시:
    X_train_seq, y_train_seq = create_sliding_windows(X_train, y_train, window_size=24)
    # X_train_seq shape: (samples, window_size, n_features)
    # y_train_seq shape: (samples, n_targets)
"""

from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd


def create_sliding_windows(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
    window_size: int,
    horizon: int = 1,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    시계열 데이터를 sliding window 방식으로 변환
    
    Parameters:
    -----------
    X : DataFrame or ndarray
        입력 특성 (shape: [n_samples, n_features])
    y : DataFrame, Series, or ndarray
        타겟 변수 (shape: [n_samples] or [n_samples, n_targets])
    window_size : int
        과거 몇 개의 시간 스텝을 볼 것인지 (예: 24 = 과거 24시간)
    horizon : int
        미래 몇 스텝 후를 예측할 것인지 (기본: 1 = 다음 시간)
    stride : int
        윈도우 이동 간격 (기본: 1 = 매 시간마다)
        
    Returns:
    --------
    X_seq : ndarray
        윈도우 형태의 입력 (shape: [n_windows, window_size, n_features])
    y_seq : ndarray
        대응하는 타겟 (shape: [n_windows, n_targets])
        
    Example:
    --------
    >>> X = np.random.rand(100, 5)  # 100 시간, 5개 특성
    >>> y = np.random.rand(100, 1)  # 100 시간, 1개 타겟
    >>> X_seq, y_seq = create_sliding_windows(X, y, window_size=24)
    >>> print(X_seq.shape)  # (76, 24, 5) - 과거 24시간씩 묶음
    >>> print(y_seq.shape)  # (76, 1) - 각 윈도우에 대응하는 타겟
    """
    # DataFrame을 numpy로 변환
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X
    
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y_values = y.values
    else:
        y_values = y
    
    # y가 1차원이면 2차원으로 변환
    if y_values.ndim == 1:
        y_values = y_values.reshape(-1, 1)
    
    # 윈도우 생성
    X_seq = []
    y_seq = []
    
    # 윈도우를 stride 간격으로 이동하면서 생성
    for i in range(0, len(X_values) - window_size - horizon + 1, stride):
        # 입력: i부터 i+window_size까지의 과거 데이터
        X_window = X_values[i:i + window_size]
        
        # 타겟: i+window_size+horizon-1 시점의 값
        y_target = y_values[i + window_size + horizon - 1]
        
        X_seq.append(X_window)
        y_seq.append(y_target)
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    return X_seq, y_seq


def create_sliding_windows_with_index(
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series],
    window_size: int,
    horizon: int = 1,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    인덱스 정보를 유지하면서 sliding window 생성
    
    Parameters:
    -----------
    X : DataFrame
        입력 특성 (DatetimeIndex 필요)
    y : DataFrame or Series
        타겟 변수
    window_size : int
        윈도우 크기
    horizon : int
        예측 horizon
    stride : int
        윈도우 이동 간격
        
    Returns:
    --------
    X_seq : ndarray
        윈도우 형태의 입력
    y_seq : ndarray
        대응하는 타겟
    target_index : Index
        각 윈도우의 타겟 시점 인덱스
    """
    X_seq, y_seq = create_sliding_windows(X, y, window_size, horizon, stride)
    
    # 타겟 시점의 인덱스 추출
    target_indices = []
    for i in range(0, len(X) - window_size - horizon + 1, stride):
        target_idx = X.index[i + window_size + horizon - 1]
        target_indices.append(target_idx)
    
    target_index = pd.Index(target_indices)
    
    return X_seq, y_seq, target_index


def flatten_windows_for_ml(X_seq: np.ndarray) -> np.ndarray:
    """
    3D 윈도우 데이터를 2D로 평탄화 (ML 모델용)
    
    일반 ML 모델(RF, XGBoost 등)은 2D 입력 (samples, features)이 필요합니다.
    
    이 함수는 (samples, window_size, n_features)를 
    (samples, window_size * n_features)로 변환합니다.
    
    Parameters:
    -----------
    X_seq : ndarray
        3D 윈도우 데이터 (shape: [n_samples, window_size, n_features])
        
    Returns:
    --------
    X_flat : ndarray
        2D 평탄화된 데이터 (shape: [n_samples, window_size * n_features])
        
    Example:
    --------
    >>> X_seq = np.random.rand(100, 24, 5)  # 100개 샘플, 24시간 윈도우, 5개 특성
    >>> X_flat = flatten_windows_for_ml(X_seq)
    >>> print(X_flat.shape)  # (100, 120) - 24*5=120개 특성
    """
    n_samples, window_size, n_features = X_seq.shape
    X_flat = X_seq.reshape(n_samples, window_size * n_features)
    return X_flat


def create_feature_names_for_flattened_windows(
    original_features: list,
    window_size: int
) -> list:
    """
    평탄화된 윈도우 데이터의 특성 이름 생성
    
    Parameters:
    -----------
    original_features : list
        원본 특성 이름 리스트
    window_size : int
        윈도우 크기
        
    Returns:
    --------
    feature_names : list
        평탄화된 특성 이름 리스트
        
    Example:
    --------
    >>> features = ['temp', 'humidity']
    >>> names = create_feature_names_for_flattened_windows(features, window_size=3)
    >>> print(names)
    ['temp_t-2', 'humidity_t-2', 'temp_t-1', 'humidity_t-1', 'temp_t0', 'humidity_t0']
    """
    feature_names = []
    
    # 시간 역순으로 생성 (t-window_size+1, ..., t-1, t0)
    for t in range(window_size - 1, -1, -1):
        time_label = f"t-{t}" if t > 0 else "t0"
        for feat in original_features:
            feature_names.append(f"{feat}_{time_label}")
    
    return feature_names


def split_windowed_data(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> dict:
    """
    윈도우 데이터를 train/valid/test로 분할
    
    시계열 데이터이므로 셔플하지 않고 순차적으로 분할합니다.
    
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
    
    print(f"\nSliding Window 데이터 분할:")
    print(f"  Train: {len(X_train)} 윈도우 ({train_ratio*100:.0f}%)")
    print(f"  Valid: {len(X_valid)} 윈도우 ({valid_ratio*100:.0f}%)")
    print(f"  Test:  {len(X_test)} 윈도우 ({test_ratio*100:.0f}%)")
    
    return {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid),
        "test": (X_test, y_test)
    }


def print_window_info(X_seq: np.ndarray, y_seq: np.ndarray, window_size: int):
    """
    윈도우 데이터 정보 출력
    
    Parameters:
    -----------
    X_seq : ndarray
        윈도우 형태의 입력
    y_seq : ndarray
        대응하는 타겟
    window_size : int
        윈도우 크기
    """
    # X_seq가 비어있거나 잘못된 형태인지 확인
    if X_seq is None or len(X_seq) == 0:
        print(f"\n{'='*60}")
        print("⚠️  경고: 윈도우가 생성되지 않았습니다!")
        print(f"{'='*60}")
        return
    
    # shape 확인
    if X_seq.ndim != 3:
        print(f"\n{'='*60}")
        print(f"⚠️  경고: X_seq의 차원이 잘못되었습니다!")
        print(f"예상: 3D (samples, window_size, features)")
        print(f"실제: {X_seq.ndim}D, shape={X_seq.shape}")
        print(f"{'='*60}")
        return
    
    n_samples, actual_window, n_features = X_seq.shape
    n_targets = y_seq.shape[1] if y_seq.ndim > 1 else 1
    
    print(f"\n{'='*60}")
    print("Sliding Window 정보")
    print(f"{'='*60}")
    print(f"윈도우 크기: {window_size} 시간 스텝")
    print(f"생성된 윈도우 수: {n_samples:,}개")
    print(f"입력 특성 수: {n_features}개")
    print(f"타겟 변수 수: {n_targets}개")
    print(f"입력 shape: {X_seq.shape}")
    print(f"타겟 shape: {y_seq.shape}")
    print(f"{'='*60}")



# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    """
    Sliding Window 모듈 사용 예시
    """
    print("Sliding Window 모듈 테스트\n")
    
    # 예시 데이터 생성 (100시간, 5개 특성, 2개 타겟)
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    n_targets = 2
    
    X = pd.DataFrame(
        np.random.rand(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)],
        index=pd.date_range('2024-01-01', periods=n_samples, freq='h')
    )
    y = pd.DataFrame(
        np.random.rand(n_samples, n_targets),
        columns=[f'target_{i}' for i in range(n_targets)],
        index=X.index
    )
    
    print(f"원본 데이터:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}\n")
    
    # 1. 기본 sliding window 생성
    window_size = 24  # 과거 24시간
    X_seq, y_seq = create_sliding_windows(X, y, window_size=window_size)
    print_window_info(X_seq, y_seq, window_size)
    
    # 2. 인덱스 정보 포함
    X_seq, y_seq, target_index = create_sliding_windows_with_index(
        X, y, window_size=window_size
    )
    print(f"\n타겟 시점 인덱스 (처음 5개):")
    print(target_index[:5])
    
    # 3. ML 모델용 평탄화
    X_flat = flatten_windows_for_ml(X_seq)
    print(f"\n평탄화된 데이터 shape: {X_flat.shape}")
    print(f"  원본: (samples={X_seq.shape[0]}, window={X_seq.shape[1]}, features={X_seq.shape[2]})")
    print(f"  평탄화: (samples={X_flat.shape[0]}, features={X_flat.shape[1]})")
    
    # 4. 특성 이름 생성
    feature_names = create_feature_names_for_flattened_windows(
        X.columns.tolist(), window_size=3
    )
    print(f"\n평탄화된 특성 이름 예시 (window_size=3):")
    print(feature_names[:6])
    
    # 5. 데이터 분할
    splits = split_windowed_data(X_seq, y_seq)
    
    print("\n✅ Sliding Window 모듈 테스트 완료!")
