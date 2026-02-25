"""
피처 선택 모듈
중요도 기반 피처 선택
"""

from typing import List, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


def select_top_features(
    X_train: pd.DataFrame,
    y_train: Union[pd.DataFrame, pd.Series],
    n_features: int = 50,
    random_state: int = 42
) -> List[str]:
    """
    RandomForest로 피처 중요도 계산 후 상위 n개 선택
    
    Parameters:
    -----------
    X_train : DataFrame
        학습 데이터
    y_train : DataFrame or Series
        타겟 데이터
    n_features : int
        선택할 피처 개수
    random_state : int
        랜덤 시드
        
    Returns:
    --------
    list : 선택된 피처 이름 리스트
    """
    print(f"\n피처 선택 중... (총 {X_train.shape[1]}개 → 상위 {n_features}개)")
    
    # 단일 타겟인 경우
    is_single_target = len(y_train.shape) == 1 or y_train.shape[1] == 1
    
    if is_single_target:
        rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        y_values = y_train.values.ravel() if hasattr(y_train, 'values') else y_train
        rf.fit(X_train, y_values)
        importances = rf.feature_importances_
    else:
        # 다중 타겟인 경우 평균 중요도 사용
        rf = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        )
        rf.fit(X_train, y_train)
        importances = np.mean([est.feature_importances_ for est in rf.estimators_], axis=0)
    
    # 상위 n개 피처 선택
    top_indices = np.argsort(importances)[-n_features:]
    top_features = X_train.columns[top_indices].tolist()
    
    print(f"선택된 상위 10개 피처: {top_features[-10:]}")
    
    return top_features
