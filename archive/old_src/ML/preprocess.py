"""
데이터 전처리 모듈
결측치 처리, 리샘플링, 연속성 확인, 이상치 처리
"""

from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy.stats import zscore


def drop_missing_rows(df: pd.DataFrame, 
                     cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    결측치가 있는 행 제거
    
    Parameters:
    -----------
    df : DataFrame
        입력 데이터프레임
    cols : list, optional
        검사할 컬럼 리스트 (None이면 모든 컬럼)
        
    Returns:
    --------
    DataFrame : 결측치가 제거된 데이터프레임
    """
    out = df.copy()
    return out.dropna() if cols is None else out.dropna(subset=cols)


def resample_hourly(df: pd.DataFrame, 
                    rule: str = "1h", 
                    agg: Union[str, Dict] = "mean") -> pd.DataFrame:
    """
    시계열 데이터 리샘플링
    
    Parameters:
    -----------
    df : DataFrame
        입력 데이터프레임
    rule : str
        리샘플링 규칙 (예: '1h', '5min')
    agg : str or dict
        집계 함수
        
    Returns:
    --------
    DataFrame : 리샘플링된 데이터프레임
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("리샘플링을 위해서는 DatetimeIndex가 필요합니다.")
    return df.copy().resample(rule).agg(agg)


def check_continuity(df: pd.DataFrame, 
                    freq: str = "1h") -> Dict[str, Any]:
    """
    시계열 데이터의 연속성 확인
    
    Parameters:
    -----------
    df : DataFrame
        입력 데이터프레임
    freq : str
        예상 주기
        
    Returns:
    --------
    dict : 연속성 정보 (is_continuous, n_missing_timestamps, max_gap)
    """
    if len(df) == 0:
        return {
            "is_continuous": True, 
            "n_missing_timestamps": 0, 
            "max_gap": pd.Timedelta(0)
        }

    idx = df.index
    expected = pd.date_range(start=idx.min(), end=idx.max(), freq=freq, tz=idx.tz)
    missing = expected.difference(idx)
    
    # diff 기반 최대 간격 계산
    diffs = idx.to_series().diff().dropna()
    max_gap = diffs.max() if len(diffs) > 0 else pd.Timedelta(0)
    
    return {
        "is_continuous": len(missing) == 0,
        "n_missing_timestamps": len(missing),
        "max_gap": max_gap
    }


# ========================================
# 결측치 보간 전략
# ========================================

@dataclass
class ImputationConfig:
    """결측치 보간 설정"""
    short_term_hours: int = 3      # 단기 결측: 1-3시간
    medium_term_hours: int = 12    # 중기 결측: 4-12시간
    ewma_span: int = 6             # EWMA 스팬 (시간 단위)
    
    
def impute_missing_with_strategy(df: pd.DataFrame, 
                                freq: str = '1h', 
                                config: ImputationConfig = ImputationConfig(), 
                                add_mask: bool = True) -> pd.DataFrame:
    """
    전략적 결측치 보간
    
    전략:
    - 단기 결측 (1-3시간): Forward Fill
    - 중기 결측 (4-12시간): EWMA (시간 가중, 과거 데이터만 사용)
    - 장기 결측 (12시간+): NaN 유지
    
    Parameters:
    -----------
    df : DataFrame
        결측치가 있는 데이터프레임
    freq : str
        시간 간격 (예: '1h', '5min')
    config : ImputationConfig
        보간 설정
    add_mask : bool
        마스크 컬럼 추가 여부
        
    Returns:
    --------
    DataFrame : 보간된 데이터프레임 (마스크 포함)
    """
    df_out = df.copy()
    
    # 시간 간격을 시간 단위로 변환
    freq_td = pd.Timedelta(freq)
    freq_hours = freq_td.total_seconds() / 3600
    
    # 각 컬럼별 처리
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
            
        series = df[col].copy()
        original_missing = series.isna()
        
        if add_mask:
            df_out[f'{col}_is_missing'] = original_missing.astype(int)
        
        # 1단계: Forward Fill (단기 결측)
        limit_short = int(config.short_term_hours / freq_hours)
        series_ffill = series.ffill(limit=limit_short)
        ffill_mask = original_missing & ~series_ffill.isna()
        
        if add_mask:
            df_out[f'{col}_imputed_ffill'] = ffill_mask.astype(int)
        
        # 2단계: EWMA (중기 결측) - 과거 데이터만 사용
        still_missing = series_ffill.isna()
        if still_missing.sum() > 0:
            ewma_span = int(config.ewma_span / freq_hours)
            series_ewma = series_ffill.ewm(span=ewma_span, adjust=False).mean()
            
            # 중기 결측만 EWMA로 채우기
            limit_medium = int(config.medium_term_hours / freq_hours)
            
            # 연속된 결측 구간 찾기
            missing_groups = (still_missing != still_missing.shift()).cumsum()
            missing_lengths = still_missing.groupby(missing_groups).transform('sum')
            
            # 중기 결측 마스크 (단기 < 길이 <= 중기)
            medium_mask = still_missing & (missing_lengths > limit_short) & (missing_lengths <= limit_medium)
            series_ffill[medium_mask] = series_ewma[medium_mask]
            
            if add_mask:
                df_out[f'{col}_imputed_ewma'] = medium_mask.astype(int)
        
        # 3단계: 장기 결측은 NaN 유지
        long_missing = series_ffill.isna()
        if add_mask:
            df_out[f'{col}_imputed_nan'] = long_missing.astype(int)
        
        # 최종 결과 저장
        df_out[col] = series_ffill
    
    return df_out


def summarize_imputation(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    결측치 보간 요약 통계
    
    Parameters:
    -----------
    df : DataFrame
        보간된 데이터프레임 (마스크 포함)
        
    Returns:
    --------
    DataFrame : 보간 요약 테이블
    """
    # 마스크 컬럼 찾기
    mask_cols = [c for c in df.columns if '_is_missing' in c or '_imputed_' in c]
    
    # 원본 컬럼 추출
    original_cols = set()
    for mask_col in mask_cols:
        for suffix in ['_is_missing', '_imputed_ffill', '_imputed_ewma', '_imputed_nan']:
            if suffix in mask_col:
                original_cols.add(mask_col.replace(suffix, ''))
                break
    
    summary_rows = []
    for col in sorted(original_cols):
        if col not in df.columns:
            continue
            
        row = {
            'column': col,
            'original_missing': df.get(f'{col}_is_missing', pd.Series(0)).sum(),
            'ffill': df.get(f'{col}_imputed_ffill', pd.Series(0)).sum(),
            'ewma': df.get(f'{col}_imputed_ewma', pd.Series(0)).sum(),
            'still_nan': df.get(f'{col}_imputed_nan', pd.Series(0)).sum()
        }
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows) if summary_rows else None


# ========================================
# 이상치 탐지 및 처리
# ========================================

@dataclass
class OutlierConfig:
    """이상치 탐지 설정"""
    method: str = 'iqr'           # 'iqr' 또는 'zscore'
    iqr_threshold: float = 1.5    # IQR 배수
    zscore_threshold: float = 3.0 # Z-score 임계값
    require_both: bool = True     # True: 도메인+통계 둘 다 이상치여야 처리
                                  # False: 둘 중 하나만 이상치여도 처리


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
    
    # 숫자형이 아니면 모두 False 반환
    if not pd.api.types.is_numeric_dtype(series):
        return outliers
    
    # 도메인별 임계값 정의
    domain_rules = {
        # TMS 수질 데이터
        'TOC_VU': (0, 100),  # 총유기탄소 (mg/L)
        'PH_VU': (0, 14),    # pH
        'SS_VU': (0, 500),   # 부유물질 (mg/L)
        'TN_VU': (0, 100),   # 총질소 (mg/L)
        'TP_VU': (0, 100),   # 총인 (mg/L)
        # FLOW 데이터
        'level_TankA': (0, 10),  # 수위 (m)
        'level_TankB': (0, 10),
        # AWS 기상 데이터
        'TA': (-30, 45),     # 기온 (°C)
        'HM': (0, 100),      # 습도 (%)
        'TD': (-40, 35),     # 이슬점 온도 (°C)
    }
    
    # 강수량 컬럼 (동적 패턴 매칭)
    rain_cols = ['RN_15m', 'RN_60m', 'RN_12H', 'RN_DAY']
    
    if col_name in domain_rules:
        lower, upper = domain_rules[col_name]
        outliers = (series < lower) | (series > upper)
    elif any(rain_col in col_name for rain_col in rain_cols):
        # 강수량: 음수 또는 300mm 초과
        outliers = (series < 0) | (series > 300)
    elif col_name in ['FLUX_VU', 'flow_TankA', 'flow_TankB', 'Q_in']:
        # 유량: 음수 또는 99th percentile의 3배 초과
        valid_values = series.dropna()
        if len(valid_values) > 0:
            outliers = (series < 0) | (series > valid_values.quantile(0.99) * 3)
    else:
        # 기본값: 음수 또는 99.9th percentile의 2배 초과
        valid_values = series.dropna()
        if len(valid_values) > 0:
            outliers = (series < 0) | (series > valid_values.quantile(0.999) * 2)
    
    return outliers


def detect_outliers_statistical(series: pd.Series, 
                                method: str = 'iqr', 
                                iqr_threshold: float = 1.5, 
                                zscore_threshold: float = 3.0) -> pd.Series:
    """
    통계적 방법 기반 이상치 탐지
    
    Parameters:
    -----------
    series : Series
        검사할 시리즈
    method : str
        'iqr' 또는 'zscore'
    iqr_threshold : float
        IQR 배수
    zscore_threshold : float
        Z-score 임계값
        
    Returns:
    --------
    Series : 이상치 마스크 (True = 이상치)
    """
    outliers = pd.Series([False] * len(series), index=series.index)
    
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_threshold * IQR
        upper = Q3 + iqr_threshold * IQR
        outliers = (series < lower) | (series > upper)
    
    elif method == 'zscore':
        # NaN 제거 후 Z-score 계산
        valid_mask = ~series.isna()
        if valid_mask.sum() > 0:
            z_scores = np.abs(zscore(series[valid_mask]))
            outliers[valid_mask] = z_scores > zscore_threshold
    
    return outliers


def detect_and_handle_outliers(df: pd.DataFrame, 
                               config: OutlierConfig = OutlierConfig(), 
                               add_mask: bool = True) -> pd.DataFrame:
    """
    이상치 탐지 및 처리 (NaN으로 변환)
    
    전략:
    - 도메인 지식 + 통계적 방법 병행
    - require_both=True: 둘 다 이상치여야 처리 (보수적)
    - require_both=False: 둘 중 하나만 이상치여도 처리 (공격적)
    
    Parameters:
    -----------
    df : DataFrame
        데이터프레임
    config : OutlierConfig
        이상치 탐지 설정
    add_mask : bool
        마스크 컬럼 추가 여부
        
    Returns:
    --------
    DataFrame : 이상치가 NaN으로 변환된 데이터프레임
    """
    df_out = df.copy()
    
    # 마스크 패턴 (이미 처리된 컬럼 건너뛰기)
    skip_patterns = ['_is_missing', '_imputed_', '_outlier_']
    
    for col in df.columns:
        # 마스크 컬럼 및 비숫자형 컬럼 건너뛰기
        if any(pattern in col for pattern in skip_patterns):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        series = df[col].copy()
        valid_mask = ~series.isna()
        if valid_mask.sum() == 0:
            continue
        
        # 도메인 지식 기반 이상치
        domain_outliers = detect_outliers_domain(series, col)
        
        # 통계적 이상치
        statistical_outliers = detect_outliers_statistical(
            series, 
            method=config.method,
            iqr_threshold=config.iqr_threshold,
            zscore_threshold=config.zscore_threshold
        )
        
        # 최종 이상치 결정
        final_outliers = (domain_outliers & statistical_outliers) if config.require_both else (domain_outliers | statistical_outliers)
        
        # 마스크 추가
        if add_mask:
            df_out[f'{col}_outlier_domain'] = domain_outliers.astype(int)
            df_out[f'{col}_outlier_statistical'] = statistical_outliers.astype(int)
            df_out[f'{col}_outlier_final'] = final_outliers.astype(int)
        
        # 이상치를 NaN으로 변환
        df_out.loc[final_outliers, col] = np.nan
    
    return df_out


def summarize_outliers(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    이상치 탐지 요약 통계
    
    Parameters:
    -----------
    df : DataFrame
        이상치 처리된 데이터프레임 (마스크 포함)
        
    Returns:
    --------
    DataFrame : 이상치 요약 테이블
    """
    outlier_cols = [c for c in df.columns if '_outlier_final' in c]
    
    summary_rows = []
    for mask_col in outlier_cols:
        col = mask_col.replace('_outlier_final', '')
        
        if col not in df.columns:
            continue
        
        row = {
            'column': col,
            'domain': df.get(f'{col}_outlier_domain', pd.Series(0)).sum(),
            'statistical': df.get(f'{col}_outlier_statistical', pd.Series(0)).sum(),
            'final': df[mask_col].sum()
        }
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows) if summary_rows else None
