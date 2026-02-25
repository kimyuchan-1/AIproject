"""
전처리 모듈
결측치 처리, 이상치 처리, 리샘플링
"""

from typing import Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class ImputationConfig:
    """결측치 보간 설정"""
    short_term_hours: int = 3
    medium_term_hours: int = 12
    long_term_hours: int = 48
    ewma_span: int = 6


@dataclass
class OutlierConfig:
    """이상치 탐지 설정"""
    method: str = "iqr"
    iqr_threshold: float = 1.5
    zscore_threshold: float = 3.0
    require_both: bool = False


def impute_missing(df: pd.DataFrame, 
                  freq: str = "1h",
                  config: ImputationConfig = ImputationConfig()) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    결측치 보간
    - 단기 (1-3시간): Forward Fill
    - 중기 (4-12시간): EWMA
    - 장기 (12시간+): EWMA (장기 span 사용)
    
    Parameters:
    -----------
    df : DataFrame
        입력 데이터프레임
    freq : str
        시간 간격
    config : ImputationConfig
        보간 설정
        
    Returns:
    --------
    tuple : (보간된 데이터프레임, 마스크 데이터프레임)
    """
    df_out = df.copy()
    
    # 시간 간격을 시간 단위로 변환
    freq_td = pd.Timedelta(freq)
    freq_hours = freq_td.total_seconds() / 3600
    
    # 마스크 데이터를 딕셔너리에 모아서 한 번에 생성 (성능 최적화)
    mask_dict = {}
    
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        series = df[col].copy()
        original_missing = series.isna()
        
        # 마스크 저장 (딕셔너리에 추가)
        mask_dict[f"{col}_is_missing"] = original_missing.astype(int)
        
        # 1단계: Forward Fill (단기)
        limit_short = int(config.short_term_hours / freq_hours)
        series_ffill = series.ffill(limit=limit_short)
        ffill_mask = original_missing & ~series_ffill.isna()
        mask_dict[f"{col}_imputed_ffill"] = ffill_mask.astype(int)
        
        # 2단계: EWMA (중기)
        still_missing = series_ffill.isna()
        if still_missing.sum() > 0:
            ewma_span = int(config.ewma_span / freq_hours)
            series_ewma = series_ffill.ewm(span=ewma_span, adjust=False).mean()
            
            limit_medium = int(config.medium_term_hours / freq_hours)
            missing_groups = (still_missing != still_missing.shift()).cumsum()
            missing_lengths = still_missing.groupby(missing_groups).transform("sum")
            
            medium_mask = still_missing & (missing_lengths > limit_short) & (missing_lengths <= limit_medium)
            series_ffill[medium_mask] = series_ewma[medium_mask]
            mask_dict[f"{col}_imputed_ewma"] = medium_mask.astype(int)
        else:
            mask_dict[f"{col}_imputed_ewma"] = pd.Series(0, index=df.index, dtype=int)
        
        # 3단계: 장기 결측도 EWMA로 채우기 (더 긴 span 사용)
        still_missing_long = series_ffill.isna()
        if still_missing_long.sum() > 0:
            # 장기 결측용 더 긴 span (기본 span의 4배)
            long_ewma_span = int(config.ewma_span * 4 / freq_hours)
            series_long_ewma = series_ffill.ewm(span=long_ewma_span, adjust=False).mean()
            
            long_mask = still_missing_long
            series_ffill[long_mask] = series_long_ewma[long_mask]
            mask_dict[f"{col}_imputed_long_ewma"] = long_mask.astype(int)
        else:
            mask_dict[f"{col}_imputed_long_ewma"] = pd.Series(0, index=df.index, dtype=int)
        
        df_out[col] = series_ffill
    
    # 마스크 DataFrame을 한 번에 생성 (성능 최적화)
    df_mask = pd.DataFrame(mask_dict, index=df.index)
    
    return df_out, df_mask


def detect_outliers_domain(series: pd.Series, col_name: str) -> pd.Series:
    """
    도메인 지식 기반 이상치 탐지
    
    Parameters:
    -----------
    series : Series
        검사할 시리즈
    col_name : str
        컬럼명
        
    Returns:
    --------
    Series : 이상치 마스크 (True = 이상치)
    """
    outliers = pd.Series([False] * len(series), index=series.index)
    
    if not pd.api.types.is_numeric_dtype(series):
        return outliers
    
    # 도메인별 임계값
    domain_rules = {
        "TOC_VU": (0, 100),
        "PH_VU": (0, 14),
        "SS_VU": (0, 500),
        "TN_VU": (0, 100),
        "TP_VU": (0, 100),
        "level_TankA": (0, 10),
        "level_TankB": (0, 10),
        "TA": (-30, 45),
        "HM": (0, 100),
        "TD": (-40, 35),
    }
    
    if col_name in domain_rules:
        lower, upper = domain_rules[col_name]
        outliers = (series < lower) | (series > upper)
    elif "RN_" in col_name:
        outliers = (series < 0) | (series > 300)
    elif "flow" in col_name.lower() or "flux" in col_name.lower():
        valid_values = series.dropna()
        if len(valid_values) > 0:
            outliers = (series < 0) | (series > valid_values.quantile(0.99) * 3)
    
    return outliers


def detect_outliers_statistical(series: pd.Series, 
                                method: str = "iqr",
                                iqr_threshold: float = 1.5,
                                zscore_threshold: float = 3.0) -> pd.Series:
    """
    통계적 방법 기반 이상치 탐지
    
    Parameters:
    -----------
    series : Series
        검사할 시리즈
    method : str
        "iqr" 또는 "zscore"
    iqr_threshold : float
        IQR 배수
    zscore_threshold : float
        Z-score 임계값
        
    Returns:
    --------
    Series : 이상치 마스크
    """
    outliers = pd.Series([False] * len(series), index=series.index)
    
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_threshold * IQR
        upper = Q3 + iqr_threshold * IQR
        outliers = (series < lower) | (series > upper)
    elif method == "zscore":
        from scipy.stats import zscore
        valid_mask = ~series.isna()
        if valid_mask.sum() > 0:
            z_scores = np.abs(zscore(series[valid_mask]))
            outliers[valid_mask] = z_scores > zscore_threshold
    
    return outliers


def handle_outliers(df: pd.DataFrame, 
                   config: OutlierConfig = OutlierConfig(),
                   ewma_span: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    이상치 탐지 및 처리 (EWMA로 대체)
    
    Parameters:
    -----------
    df : DataFrame
        입력 데이터프레임
    config : OutlierConfig
        이상치 탐지 설정
    ewma_span : int
        EWMA span (기본: 12시간)
        
    Returns:
    --------
    tuple : (처리된 데이터프레임, 마스크 데이터프레임)
    """
    df_out = df.copy()
    
    # 마스크 데이터를 딕셔너리에 모아서 한 번에 생성 (성능 최적화)
    mask_dict = {}
    
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        series = df[col].copy()
        
        # 도메인 기반 이상치
        domain_outliers = detect_outliers_domain(series, col)
        
        # 통계 기반 이상치
        statistical_outliers = detect_outliers_statistical(
            series,
            method=config.method,
            iqr_threshold=config.iqr_threshold,
            zscore_threshold=config.zscore_threshold
        )
        
        # 최종 이상치 결정
        if config.require_both:
            final_outliers = domain_outliers & statistical_outliers
        else:
            final_outliers = domain_outliers | statistical_outliers
        
        # 마스크 저장 (딕셔너리에 추가)
        mask_dict[f"{col}_outlier_domain"] = domain_outliers.astype(int)
        mask_dict[f"{col}_outlier_statistical"] = statistical_outliers.astype(int)
        mask_dict[f"{col}_outlier_final"] = final_outliers.astype(int)
        
        # 이상치를 NaN으로 변환
        series[final_outliers] = np.nan
        
        # EWMA로 이상치 대체
        if final_outliers.sum() > 0:
            series_ewma = series.ewm(span=ewma_span, adjust=False).mean()
            series[final_outliers] = series_ewma[final_outliers]
            mask_dict[f"{col}_outlier_replaced_ewma"] = final_outliers.astype(int)
        else:
            mask_dict[f"{col}_outlier_replaced_ewma"] = pd.Series(0, index=df.index, dtype=int)
        
        df_out[col] = series
    
    # 마스크 DataFrame을 한 번에 생성 (성능 최적화)
    df_mask = pd.DataFrame(mask_dict, index=df.index)
    
    return df_out, df_mask


def resample_data(df: pd.DataFrame, freq: str = "1h", agg: str = "mean") -> pd.DataFrame:
    """
    데이터 리샘플링
    
    Parameters:
    -----------
    df : DataFrame
        입력 데이터프레임
    freq : str
        리샘플링 주기
    agg : str
        집계 함수
        
    Returns:
    --------
    DataFrame : 리샘플링된 데이터프레임
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("리샘플링을 위해서는 DatetimeIndex가 필요합니다")
    
    # 숫자형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    
    return df_numeric.resample(freq).agg(agg)
