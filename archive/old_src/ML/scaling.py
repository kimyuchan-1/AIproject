"""
데이터 스케일링 모듈
StandardScaler 적용
"""

from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def scale_data(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_valid: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    scaler: Optional[StandardScaler] = None
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], 
           Union[pd.DataFrame, np.ndarray], StandardScaler]:
    """
    학습/검증/테스트 데이터 스케일링
    
    Parameters:
    -----------
    X_train : DataFrame or ndarray
        학습 데이터
    X_valid : DataFrame or ndarray
        검증 데이터
    X_test : DataFrame or ndarray
        테스트 데이터
    scaler : StandardScaler, optional
        기존 스케일러 (None이면 새로 생성)
        
    Returns:
    --------
    tuple : (X_train_scaled, X_valid_scaled, X_test_scaled, scaler)
        스케일링된 데이터는 DataFrame으로 반환 (컬럼명 유지)
    """
    if scaler is None:
        scaler = StandardScaler()
    
    # 스케일링 수행
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_valid_scaled_array = scaler.transform(X_valid)
    X_test_scaled_array = scaler.transform(X_test)
    
    # DataFrame인 경우 컬럼명과 인덱스 유지
    if isinstance(X_train, pd.DataFrame):
        X_train_scaled = pd.DataFrame(
            X_train_scaled_array, 
            index=X_train.index, 
            columns=X_train.columns
        )
        X_valid_scaled = pd.DataFrame(
            X_valid_scaled_array, 
            index=X_valid.index, 
            columns=X_valid.columns
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled_array, 
            index=X_test.index, 
            columns=X_test.columns
        )
    else:
        # ndarray인 경우 그대로 반환
        X_train_scaled = X_train_scaled_array
        X_valid_scaled = X_valid_scaled_array
        X_test_scaled = X_test_scaled_array
    
    return X_train_scaled, X_valid_scaled, X_test_scaled, scaler
