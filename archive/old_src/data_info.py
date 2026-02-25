"""
데이터 정보 출력 유틸리티
각 전처리 단계에서 데이터 상태를 확인하기 위한 함수들
"""

import pandas as pd
import numpy as np


def print_data_info(df, stage_name="데이터", show_nan_details=True, show_sample=False):
    """
    데이터프레임의 상세 정보 출력
    
    Parameters:
    -----------
    df : pd.DataFrame
        확인할 데이터프레임
    stage_name : str
        현재 단계 이름
    show_nan_details : bool
        NaN 상세 정보 표시 여부
    show_sample : bool
        샘플 데이터 표시 여부
    """
    print(f"\n{'='*60}")
    print(f"📊 {stage_name} 정보")
    print(f"{'='*60}")
    
    # 기본 정보
    print(f"Shape: {df.shape} (행={df.shape[0]:,}, 열={df.shape[1]:,})")
    
    if isinstance(df.index, pd.DatetimeIndex):
        print(f"시간 범위: {df.index.min()} ~ {df.index.max()}")
        print(f"시간 간격: {df.index.freq if df.index.freq else '불규칙'}")
    
    # 메모리 사용량
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"메모리 사용량: {memory_mb:.2f} MB")
    
    # 데이터 타입
    dtype_counts = df.dtypes.value_counts()
    print(f"\n데이터 타입:")
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count}개 컬럼")
    
    # NaN 정보
    total_nan = df.isna().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    nan_ratio = (total_nan / total_cells * 100) if total_cells > 0 else 0
    
    print(f"\nNaN 정보:")
    print(f"  전체 NaN 수: {total_nan:,} / {total_cells:,} ({nan_ratio:.2f}%)")
    
    if show_nan_details and total_nan > 0:
        nan_by_col = df.isna().sum()
        cols_with_nan = nan_by_col[nan_by_col > 0].sort_values(ascending=False)
        
        if len(cols_with_nan) > 0:
            print(f"  NaN이 있는 컬럼: {len(cols_with_nan)}개")
            print(f"  상위 10개 컬럼:")
            for col, count in cols_with_nan.head(10).items():
                ratio = count / df.shape[0] * 100
                print(f"    {col}: {count:,} ({ratio:.1f}%)")
    
    # 숫자형 컬럼 통계
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n숫자형 컬럼: {len(numeric_cols)}개")
        
        # 무한대 값 확인
        inf_count = np.isinf(df[numeric_cols].select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            print(f"  ⚠️  무한대 값: {inf_count}개")
    
    # 샘플 데이터
    if show_sample and len(df) > 0:
        print(f"\n샘플 데이터 (처음 3행):")
        print(df.head(3))
    
    print(f"{'='*60}\n")


def print_array_info(arr, name="배열", y=None):
    """
    numpy 배열의 정보 출력
    
    Parameters:
    -----------
    arr : np.ndarray
        확인할 배열
    name : str
        배열 이름
    y : np.ndarray, optional
        타겟 배열 (함께 출력)
    """
    print(f"\n{'='*60}")
    print(f"📊 {name} 정보")
    print(f"{'='*60}")
    
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    
    # 메모리
    memory_mb = arr.nbytes / (1024**2)
    print(f"메모리: {memory_mb:.2f} MB")
    
    # NaN 정보
    if np.issubdtype(arr.dtype, np.floating):
        nan_count = np.isnan(arr).sum()
        total = arr.size
        nan_ratio = (nan_count / total * 100) if total > 0 else 0
        print(f"NaN: {nan_count:,} / {total:,} ({nan_ratio:.2f}%)")
        
        # 무한대
        inf_count = np.isinf(arr).sum()
        if inf_count > 0:
            print(f"⚠️  무한대: {inf_count:,}")
    
    # 통계
    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
        valid_data = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
        if len(valid_data) > 0:
            print(f"\n통계 (유효 데이터):")
            print(f"  Min: {valid_data.min():.4f}")
            print(f"  Max: {valid_data.max():.4f}")
            print(f"  Mean: {valid_data.mean():.4f}")
            print(f"  Std: {valid_data.std():.4f}")
    
    # y 정보
    if y is not None:
        print(f"\n타겟 (y) 정보:")
        print(f"  Shape: {y.shape}")
        print(f"  Dtype: {y.dtype}")
        
        if np.issubdtype(y.dtype, np.floating):
            y_nan = np.isnan(y).sum()
            y_total = y.size
            y_nan_ratio = (y_nan / y_total * 100) if y_total > 0 else 0
            print(f"  NaN: {y_nan:,} / {y_total:,} ({y_nan_ratio:.2f}%)")
    
    print(f"{'='*60}\n")


def print_split_info(splits, split_names=["train", "valid", "test"]):
    """
    데이터 분할 정보 출력
    
    Parameters:
    -----------
    splits : dict
        분할된 데이터 딕셔너리 {"train": (X, y), "valid": (X, y), "test": (X, y)}
    split_names : list
        분할 이름 리스트
    """
    print(f"\n{'='*60}")
    print(f"📊 데이터 분할 정보")
    print(f"{'='*60}")
    
    total_samples = sum(len(splits[name][0]) for name in split_names if name in splits)
    
    for name in split_names:
        if name not in splits:
            continue
        
        X, y = splits[name]
        n_samples = len(X)
        ratio = (n_samples / total_samples * 100) if total_samples > 0 else 0
        
        print(f"\n{name.upper()}:")
        print(f"  샘플 수: {n_samples:,} ({ratio:.1f}%)")
        
        if hasattr(X, 'shape'):
            print(f"  X shape: {X.shape}")
        if hasattr(y, 'shape'):
            print(f"  y shape: {y.shape}")
        
        # NaN 확인
        if hasattr(X, 'isna'):
            x_nan = X.isna().sum().sum()
            print(f"  X NaN: {x_nan:,}")
        elif isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.floating):
            x_nan = np.isnan(X).sum()
            print(f"  X NaN: {x_nan:,}")
        
        if hasattr(y, 'isna'):
            y_nan = y.isna().sum().sum()
            print(f"  y NaN: {y_nan:,}")
        elif isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.floating):
            y_nan = np.isnan(y).sum()
            print(f"  y NaN: {y_nan:,}")
    
    print(f"{'='*60}\n")
