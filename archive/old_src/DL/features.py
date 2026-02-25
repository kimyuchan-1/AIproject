"""
특성 생성 모듈
시간 특성, Lag 특성, Rolling 특성 생성
모델별 도메인 특화 특성 생성
"""

import pandas as pd
import numpy as np
from typing import List, Optional


# ========================================
# 유틸리티 함수들
# ========================================

def calculate_rolling_std(df: pd.DataFrame, 
                         cols: List[str], 
                         windows: List[int], 
                         suffix_pattern: str = "{col}_std_{hours}H") -> pd.DataFrame:
    """
    여러 컬럼에 대해 rolling std 계산 (기상 안정성 등)
    
    Parameters:
    -----------
    df : DataFrame
    cols : list
        계산할 컬럼 리스트
    windows : list
        시간 단위 윈도우 (예: [3, 6])
    suffix_pattern : str
        출력 컬럼명 패턴
    
    Returns:
    --------
    DataFrame : rolling std가 추가된 데이터프레임
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        for hours in windows:
            col_name = suffix_pattern.format(col=col, hours=hours)
            out[col_name] = out[col].rolling(window=hours, min_periods=max(1, hours // 2)).std()
    return out


def calculate_spike_flags(df: pd.DataFrame, 
                         cols: List[str], 
                         window_hours: int = 24, 
                         threshold: float = 2.0, 
                         eps: float = 1e-6) -> pd.DataFrame:
    """
    여러 컬럼에 대해 spike flags 계산 (z-score > threshold)
    
    Parameters:
    -----------
    df : DataFrame
    cols : list
        계산할 컬럼 리스트
    window_hours : int
        rolling window 시간
    threshold : float
        z-score 임계값
    eps : float
        0으로 나누기 방지
    
    Returns:
    --------
    DataFrame : spike flags가 추가된 데이터프레임
    """
    out = df.copy()
    
    for col in cols:
        if col not in out.columns:
            continue
        col_mean = out[col].rolling(window=window_hours, min_periods=max(2, window_hours // 10)).mean()
        col_std = out[col].rolling(window=window_hours, min_periods=max(2, window_hours // 10)).std()
        z_score = (out[col] - col_mean) / (col_std + eps)
        out[f"{col}_spike_z2"] = (z_score > threshold).astype(np.int8)
    
    return out


def calculate_derivatives(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    여러 컬럼에 대해 변화율(derivative) 계산
    
    Parameters:
    -----------
    df : DataFrame
    cols : list
        계산할 컬럼 리스트
    
    Returns:
    --------
    DataFrame : 변화율이 추가된 데이터프레임
    """
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[f"d_{col}"] = out[col].diff()
    return out


def calculate_ari(rain_series, tau_steps: int, max_lookback: int = 96) -> np.ndarray:
    """
    선행강우지수 (ARI - Antecedent Rainfall Index) 계산
    
    ARI(t) = Σ rain(t-i) × exp(-i/τ)
    
    Parameters:
    -----------
    rain_series : Series or ndarray
        강수량 시계열
    tau_steps : int
        감쇠 시간 상수 (스텝 단위)
    max_lookback : int
        최대 과거 참조 스텝
    
    Returns:
    --------
    ndarray : ARI 값
    """
    rain = rain_series.to_numpy(dtype=np.float32) if isinstance(rain_series, pd.Series) else rain_series
    ari = np.full_like(rain, np.nan, dtype=np.float32)
    
    # tau_steps가 0이면 단순 합계 반환
    if tau_steps <= 0:
        for t in range(len(rain)):
            start = max(0, t - max_lookback)
            ari[t] = float(np.sum(rain[start:t+1]))
        return ari
    
    for t in range(len(rain)):
        start = max(0, t - max_lookback)
        seg = rain[start:t+1]
        if seg.size == 0:
            ari[t] = 0.0
        else:
            ages = np.arange(seg.size - 1, -1, -1, dtype=np.float32)
            weights = np.exp(-ages / tau_steps)
            ari[t] = float(np.sum(seg * weights))
    
    return ari


# ========================================
# 기본 피처 생성 함수들
# ========================================

def add_time_features(df: pd.DataFrame, add_sin_cos: bool = True) -> pd.DataFrame:
    """
    시간 관련 특성 추가
    
    Parameters:
    -----------
    df : DataFrame
        입력 데이터프레임 (DatetimeIndex 필요)
    add_sin_cos : bool
        주기성 특성 추가 여부
        
    Returns:
    --------
    DataFrame : 시간 특성이 추가된 데이터프레임
    """
    out = df.copy()
    idx = out.index
    
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("시간 특성 생성을 위해서는 DatetimeIndex가 필요합니다")
    
    # 기본 시간 특성
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    
    # 시간대 구간
    out["tod_bucket"] = pd.cut(
        out["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    # 계절
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 
                  6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    out["season"] = out["month"].map(season_map).astype(int)
    
    # 주기성 특성 (sin/cos)
    if add_sin_cos:
        out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
        out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)
        out["sin_dow"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
        out["cos_dow"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)
        out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12.0)
        out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12.0)
    
    return out


def add_lag_features(df: pd.DataFrame, 
                    base_cols: List[str], 
                    lags: List[int]) -> pd.DataFrame:
    """
    Lag 특성 추가
    
    Parameters:
    -----------
    df : DataFrame
        입력 데이터프레임
    base_cols : list
        Lag를 생성할 컬럼 리스트
    lags : list
        Lag 시간 리스트
        
    Returns:
    --------
    DataFrame : Lag 특성이 추가된 데이터프레임
    """
    out = df.copy()
    
    # 딕셔너리에 모아서 한 번에 추가 (성능 최적화)
    lag_dict = {}
    
    for col in base_cols:
        if col not in out.columns:
            continue
        for lag in lags:
            lag_dict[f"{col}_lag{lag}"] = out[col].shift(lag)
    
    # 한 번에 concat
    if lag_dict:
        lag_df = pd.DataFrame(lag_dict, index=out.index)
        out = pd.concat([out, lag_df], axis=1)
    
    return out


def add_rolling_features(df: pd.DataFrame,
                        base_cols: List[str],
                        windows: List[int],
                        stats: List[str] = ["mean"]) -> pd.DataFrame:
    """
    Rolling 통계 특성 추가
    
    Parameters:
    -----------
    df : DataFrame
        입력 데이터프레임
    base_cols : list
        Rolling을 계산할 컬럼 리스트
    windows : list
        Window 크기 리스트
    stats : list
        통계량 종류 ["mean", "std", "min", "max"]
        
    Returns:
    --------
    DataFrame : Rolling 특성이 추가된 데이터프레임
    """
    out = df.copy()
    
    # 딕셔너리에 모아서 한 번에 추가 (성능 최적화)
    rolling_dict = {}
    
    for col in base_cols:
        if col not in out.columns:
            continue
        
        for w in windows:
            r = out[col].rolling(window=w, min_periods=max(1, w // 2))
            
            if "mean" in stats:
                rolling_dict[f"{col}_r{w}_mean"] = r.mean()
            if "std" in stats:
                rolling_dict[f"{col}_r{w}_std"] = r.std()
            if "min" in stats:
                rolling_dict[f"{col}_r{w}_min"] = r.min()
            if "max" in stats:
                rolling_dict[f"{col}_r{w}_max"] = r.max()
    
    # 한 번에 concat
    if rolling_dict:
        rolling_df = pd.DataFrame(rolling_dict, index=out.index)
        out = pd.concat([out, rolling_df], axis=1)
    
    return out


def add_target_history_features(df: pd.DataFrame, 
                                target_cols: List[str], 
                                lags: List[int] = None, 
                                windows: List[int] = None, 
                                stats: List[str] = None) -> pd.DataFrame:
    """
    target의 과거 정보를 특성으로 추가 (데이터 누수 방지)
    
    예측 대상 변수의 과거 값을 특성으로 사용하되, shift를 적용하여
    현재 시점의 target 값이 특성에 포함되지 않도록 함
    
    Parameters:
    -----------
    df : DataFrame
        데이터프레임
    target_cols : list
        타겟 컬럼 리스트
    lags : list, optional
        lag 시간 리스트 (기본값: [1, 2, 3, 6, 12, 24])
    windows : list, optional
        rolling window 시간 리스트 (기본값: [3, 6, 12, 24])
    stats : list, optional
        rolling 통계량 종류 (기본값: ["mean", "std"])
        
    Returns:
    --------
    DataFrame : target 과거 정보 특성이 추가된 데이터프레임
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24]
    if windows is None:
        windows = [3, 6, 12, 24]
    if stats is None:
        stats = ["mean", "std"]
    
    out = df.copy()
    
    # 딕셔너리에 모아서 한 번에 추가 (성능 최적화)
    target_dict = {}
    
    for target_col in target_cols:
        if target_col not in out.columns:
            continue
        
        # 1) Lag 특성 (shift 적용하여 데이터 누수 방지)
        for lag in lags:
            target_dict[f"{target_col}_target_lag{lag}"] = out[target_col].shift(lag)
        
        # 2) Rolling 통계 특성 (shift(1) 후 계산)
        target_shifted = out[target_col].shift(1)
        
        for w in windows:
            r = target_shifted.rolling(window=w, min_periods=max(1, w // 2))
            
            if "mean" in stats:
                target_dict[f"{target_col}_target_r{w}_mean"] = r.mean()
            
            if "std" in stats:
                target_dict[f"{target_col}_target_r{w}_std"] = r.std()
            
            if "min" in stats:
                target_dict[f"{target_col}_target_r{w}_min"] = r.min()
            
            if "max" in stats:
                target_dict[f"{target_col}_target_r{w}_max"] = r.max()
        
        # 3) 추가 유용한 특성
        # 변화율 (1시간 전 대비)
        target_dict[f"{target_col}_target_delta1"] = out[target_col].diff(1)
        
        # 변화율 (2시간 전 대비)
        target_dict[f"{target_col}_target_delta2"] = out[target_col].diff(2)
        
        # 변화 가속도 (2차 차분)
        target_dict[f"{target_col}_target_accel"] = out[target_col].diff(1).diff(1)
    
    # 한 번에 concat
    if target_dict:
        target_df = pd.DataFrame(target_dict, index=out.index)
        out = pd.concat([out, target_df], axis=1)
    
    return out


# ========================================
# 도메인 특화 피처 생성 함수들
# ========================================

def add_rain_features(df: pd.DataFrame, 
                     station_ids: List[str] = ["368", "541", "569"], 
                     eps: float = 1e-6, 
                     mode: str = "flow") -> pd.DataFrame:
    """
    강수 관련 도메인 특화 피처 생성
    - 강수량 변화율 (Delta)
    - 선행 강수량 (Antecedent rainfall)
    - 건조 기간 (Dry duration)
    - 강수 강도 지표
    - ModelA/B/C 전용: 단기 집중도, API 지수, rain_start/end 플래그
    """
    out = df.copy()
    
    for sid in station_ids:
        rn15_col = f"RN_15m_{sid}"
        rn60_col = f"RN_60m_{sid}"
        rn12h_col = f"RN_12H_{sid}"
        
        # 강수량 변화율
        for rc in ["RN_15m", "RN_60m", "RN_12H"]:
            col = f"{rc}_{sid}"
            if col in out.columns:
                out[f"d_{col}"] = out[col].diff()
        
        # ModelA/B/C 전용: 단기 집중도 (RN_15m / RN_60m)
        if mode in ["modela", "modelb", "modelc"] and rn15_col in out.columns and rn60_col in out.columns:
            out[f"RN_15m_div_RN_60m_{sid}"] = out[rn15_col] / (out[rn60_col] + eps)
        
        # 선행 강수량 (Antecedent Rainfall)
        if rn15_col in out.columns:
            # AR_3H, AR_6H, AR_12H, AR_24H
            for hours in [3, 6, 12, 24]:
                ar_col = out[rn15_col].rolling(window=hours, min_periods=1).sum()
                out[f"AR_{hours}H_{sid}"] = ar_col
                
                # ModelA/B/C 전용: log(1 + AR_x)
                if mode in ["modela", "modelb", "modelc"]:
                    out[f"log1p_AR_{hours}H_{sid}"] = np.log1p(ar_col)
        
        # 건조 기간 (강수가 없는 시간)
        if rn15_col in out.columns:
            wet_thr = 0.1
            is_wet = (out[rn15_col] >= wet_thr).astype(int)
            is_dry = (is_wet == 0).astype(int)
            
            # 건조 지속 시간 계산
            dry_groups = (is_wet != is_wet.shift()).cumsum()
            dry_duration = is_dry.groupby(dry_groups).cumsum()
            out[f"dry_duration_hr_{sid}"] = dry_duration  # 시간 단위
            
            # ModelA/B/C 전용: rain_start, rain_end, post_rain_6H 플래그
            if mode in ["modela", "modelb", "modelc"]:
                rain_start = is_wet & (~is_wet.shift(1, fill_value=0).astype(bool))
                rain_end = (~is_wet.astype(bool)) & (is_wet.shift(1, fill_value=0).astype(bool))
                
                out[f"rain_start_{sid}"] = rain_start.astype(np.int8)
                out[f"rain_end_{sid}"] = rain_end.astype(np.int8)
                
                # 종료 후 6시간 잔류 효과
                post_win = 6  # 6시간
                out[f"post_rain_6H_{sid}"] = (
                    pd.Series(rain_end.values, index=out.index)
                    .rolling(post_win, min_periods=1).max()
                    .fillna(0).astype(np.int8)
                )
        
        # ModelA/B/C 전용: API 지수 (감쇠 누적)
        if mode in ["modela", "modelb", "modelc"] and rn15_col in out.columns:
            # API(t) = Σ RN_15m(t-i) * exp(-k*i)
            tau_steps = 25  # tau = 25 스텝 (약 25시간)
            api = calculate_ari(out[rn15_col], tau_steps=tau_steps, max_lookback=96)
            out[f"API_RN_15m_{sid}"] = api
    
    # 전체 관측소 통계
    rn15_cols = [f"RN_15m_{sid}" for sid in station_ids if f"RN_15m_{sid}" in out.columns]
    if rn15_cols:
        out["RN_15m_max_all"] = out[rn15_cols].max(axis=1)
        out["RN_15m_mean_all"] = out[rn15_cols].mean(axis=1)
    
    ar12_cols = [f"AR_12H_{sid}" for sid in station_ids if f"AR_12H_{sid}" in out.columns]
    if ar12_cols:
        out["AR_12H_sum_all"] = out[ar12_cols].sum(axis=1)
        out["AR_12H_mean_all"] = out[ar12_cols].mean(axis=1)
    
    return out


def add_weather_features(df: pd.DataFrame, 
                        station_ids: List[str] = ["368", "541", "569"], 
                        eps: float = 1e-6, 
                        mode: str = "flow") -> pd.DataFrame:
    """
    기상 관련 도메인 특화 피처 생성
    - 이슬점 강하 (Dew point depression)
    - 온도-습도 상호작용
    - 기상 변화율
    - ModelA/B/C 전용: VPD, 기상 안정성 (rolling_std)
    """
    out = df.copy()
    
    for sid in station_ids:
        ta_col = f"TA_{sid}"
        td_col = f"TD_{sid}"
        hm_col = f"HM_{sid}"
        
        # 이슬점 강하 (Dew point depression)
        if ta_col in out.columns and td_col in out.columns:
            out[f"TA_minus_TD_{sid}"] = out[ta_col] - out[td_col]
        
        # 온도-습도 상호작용
        if ta_col in out.columns and hm_col in out.columns:
            out[f"TAxHM_{sid}"] = out[ta_col] * out[hm_col]
        
        # ModelA/B/C 전용: VPD (Vapor Pressure Deficit)
        if mode in ["modela", "modelb", "modelc"] and ta_col in out.columns and hm_col in out.columns:
            T = out[ta_col]
            RH = out[hm_col].clip(0, 100)
            e_s = 0.6108 * np.exp((17.27 * T) / (T + 237.3))
            out[f"VPD_kPa_{sid}"] = e_s * (1 - RH / 100.0)
        
        # 온도 변화율
        if ta_col in out.columns:
            out[f"d_TA_{sid}"] = out[ta_col].diff()
        
        # ModelA/B/C 전용: 기상 안정성 (rolling_std)
        if mode in ["modela", "modelb", "modelc"]:
            weather_cols = [c for c in [ta_col, hm_col] if c in out.columns]
            out = calculate_rolling_std(out, weather_cols, windows=[3, 6])
    
    # 전체 관측소 통계
    ta_cols = [f"TA_{sid}" for sid in station_ids if f"TA_{sid}" in out.columns]
    if ta_cols:
        out["TA_mean_all"] = out[ta_cols].mean(axis=1)
        out["TA_std_all"] = out[ta_cols].std(axis=1)
    
    return out


def add_tms_interaction_features(df: pd.DataFrame, 
                                 available_tms_cols: List[str], 
                                 mode: str = "flow", 
                                 eps: float = 1e-6) -> pd.DataFrame:
    """
    TMS 지표 간 상호작용 피처 생성
    (예측 대상이 아닌 TMS 지표들 간의 상호작용)
    
    ModelA 전용 특성:
    - 부하 관련: TOC_proxy_load, SS_proxy_load
    - 영양염 비율: TN/TP, log(TN+TP), PH×TN, PH×TP
    - 공정 상태 플래그: PH_zone, TN_high_flag, TP_spike_flag
    - 변화율: ΔPH, ΔFLUX, ΔTN, ΔTP, |ΔFLUX|
    
    ModelB 전용 특성:
    - 부하 관련: SS_load, TOC_load, FLUX×(SS+TOC)
    - 상호작용: PH×TOC, SS×FLUX
    - 비율: TOC/SS
    - 변화율: ΔPH, ΔFLUX, ΔSS, ΔTOC, |ΔFLUX|
    
    ModelC 전용 특성:
    - 조성/비율: TOC/SS, SS/TOC, TN/TP, TP/TN, TOC/TN, TN/TOC
    - 상호결합: TOC×SS, TN×TP
    - Spike flags: TN/TP/SS/TOC_spike_z2
    - 변화율: ΔTN, ΔTP, ΔSS, ΔTOC
    - 온도 상호작용: TA×TN, TA×TOC
    """
    out = df.copy()
    
    # 기본 상호작용 (모든 TMS 모델)
    # pH x FLUX 상호작용
    if "PH_VU" in available_tms_cols and "FLUX_VU" in available_tms_cols:
        out["PHxFLUX"] = out["PH_VU"] * out["FLUX_VU"]
    
    # 영양염 부하 (FLUX x (TN + TP))
    if all(c in available_tms_cols for c in ["FLUX_VU", "TN_VU", "TP_VU"]):
        out["load_proxy_NP"] = out["FLUX_VU"] * (out["TN_VU"] + out["TP_VU"])
    
    # 유기물 부하 (FLUX x TOC)
    if "FLUX_VU" in available_tms_cols and "TOC_VU" in available_tms_cols:
        out["load_proxy_TOC"] = out["FLUX_VU"] * out["TOC_VU"]
    
    # TN/TP 비율
    if "TN_VU" in available_tms_cols and "TP_VU" in available_tms_cols:
        out["TN_TP_ratio"] = out["TN_VU"] / (out["TP_VU"] + eps)
    
    # ModelA 전용 특성
    if mode == "modela":
        # (6) 부하 관련 특성 (아주 중요)
        if "FLUX_VU" in available_tms_cols and "PH_VU" in available_tms_cols:
            out["TOC_proxy_load"] = out["FLUX_VU"] * out["PH_VU"]
        
        if all(c in available_tms_cols for c in ["FLUX_VU", "TN_VU", "TP_VU"]):
            out["SS_proxy_load"] = out["FLUX_VU"] * (out["TN_VU"] + out["TP_VU"])
        
        # (7) 영양염 비율
        if "TN_VU" in available_tms_cols and "TP_VU" in available_tms_cols:
            out["log1p_TN_TP"] = np.log1p(out["TN_VU"] + out["TP_VU"])
            
            if "PH_VU" in available_tms_cols:
                out["PH_x_TN"] = out["PH_VU"] * out["TN_VU"]
                out["PH_x_TP"] = out["PH_VU"] * out["TP_VU"]
        
        # (8) 공정 상태 플래그
        if "PH_VU" in available_tms_cols:
            ph = out["PH_VU"]
            # PH_zone: 산성(0) / 중성(1) / 염기성(2)
            out["pH_acid"] = (ph < 6.5).astype(np.int8)
            out["pH_neutral"] = ((ph >= 6.5) & (ph <= 8.5)).astype(np.int8)
            out["pH_basic"] = (ph > 8.5).astype(np.int8)
        
        if "TN_VU" in available_tms_cols:
            # TN_high_flag (상위 20%)
            tn_80th = out["TN_VU"].quantile(0.80)
            out["TN_high_flag"] = (out["TN_VU"] > tn_80th).astype(np.int8)
        
        if "TP_VU" in available_tms_cols:
            # TP_spike_flag (z-score > 2)
            window = 24  # 24시간
            tp_mean = out["TP_VU"].rolling(window, min_periods=max(5, window // 10)).mean()
            tp_std = out["TP_VU"].rolling(window, min_periods=max(5, window // 10)).std()
            z_score = (out["TP_VU"] - tp_mean) / (tp_std + eps)
            out["TP_spike_flag"] = (z_score > 2.0).astype(np.int8)
        
        # (10) 변화율
        derivative_cols = [c for c in ["PH_VU", "FLUX_VU", "TN_VU", "TP_VU"] if c in available_tms_cols]
        out = calculate_derivatives(out, derivative_cols)
        
        # |ΔFLUX| (급변 여부)
        if "FLUX_VU" in available_tms_cols:
            out["abs_d_FLUX_VU"] = out["FLUX_VU"].diff().abs()
    
    # ModelB 전용 특성
    elif mode == "modelb":
        # 부하(load) 계열 (아주 중요)
        if "FLUX_VU" in available_tms_cols and "SS_VU" in available_tms_cols:
            out["SS_load"] = out["FLUX_VU"] * out["SS_VU"]
        
        if "FLUX_VU" in available_tms_cols and "TOC_VU" in available_tms_cols:
            out["TOC_load"] = out["FLUX_VU"] * out["TOC_VU"]
        
        if all(c in available_tms_cols for c in ["FLUX_VU", "SS_VU", "TOC_VU"]):
            out["FLUX_x_SS_TOC"] = out["FLUX_VU"] * (out["SS_VU"] + out["TOC_VU"])
        
        # 상호작용(interaction)
        if "PH_VU" in available_tms_cols and "TOC_VU" in available_tms_cols:
            out["PH_x_TOC"] = out["PH_VU"] * out["TOC_VU"]
        
        if "SS_VU" in available_tms_cols and "FLUX_VU" in available_tms_cols:
            out["SS_x_FLUX"] = out["SS_VU"] * out["FLUX_VU"]
        
        # 비율(ratio)
        if "TOC_VU" in available_tms_cols and "SS_VU" in available_tms_cols:
            out["TOC_div_SS"] = out["TOC_VU"] / (out["SS_VU"] + eps)
        
        # 변화율(derivative)
        derivative_cols = [c for c in ["PH_VU", "FLUX_VU", "SS_VU", "TOC_VU"] if c in available_tms_cols]
        out = calculate_derivatives(out, derivative_cols)
        
        # |ΔFLUX| (급변 여부)
        if "FLUX_VU" in available_tms_cols:
            out["abs_d_FLUX_VU"] = out["FLUX_VU"].diff().abs()
        
        # Spike flags (공정 이상 감지)
        spike_cols = [c for c in ["SS_VU", "TOC_VU", "PH_VU", "FLUX_VU"] if c in available_tms_cols]
        out = calculate_spike_flags(out, spike_cols, window_hours=24, threshold=2.0, eps=eps)
    
    # ModelC 전용 특성
    elif mode == "modelc":
        # 조성/비율 (Composition ratios) - 아주 중요
        if "TOC_VU" in available_tms_cols and "SS_VU" in available_tms_cols:
            out["TOC_div_SS"] = out["TOC_VU"] / (out["SS_VU"] + eps)
            out["SS_div_TOC"] = out["SS_VU"] / (out["TOC_VU"] + eps)
        
        if "TN_VU" in available_tms_cols and "TP_VU" in available_tms_cols:
            out["TN_div_TP"] = out["TN_VU"] / (out["TP_VU"] + eps)
            out["TP_div_TN"] = out["TP_VU"] / (out["TN_VU"] + eps)
        
        if "TOC_VU" in available_tms_cols and "TN_VU" in available_tms_cols:
            out["TOC_div_TN"] = out["TOC_VU"] / (out["TN_VU"] + eps)
            out["TN_div_TOC"] = out["TN_VU"] / (out["TOC_VU"] + eps)
        
        # 상호결합 (비선형)
        if "TOC_VU" in available_tms_cols and "SS_VU" in available_tms_cols:
            out["TOC_x_SS"] = out["TOC_VU"] * out["SS_VU"]
        
        if "TN_VU" in available_tms_cols and "TP_VU" in available_tms_cols:
            out["TN_x_TP"] = out["TN_VU"] * out["TP_VU"]
        
        # 변화율 (derivative)
        derivative_cols = [c for c in ["TN_VU", "TP_VU", "SS_VU", "TOC_VU"] if c in available_tms_cols]
        out = calculate_derivatives(out, derivative_cols)
        
        # Spike flags (공정 이상 감지)
        spike_cols = [c for c in ["TN_VU", "TP_VU", "SS_VU", "TOC_VU"] if c in available_tms_cols]
        out = calculate_spike_flags(out, spike_cols, window_hours=24, threshold=2.0, eps=eps)
    
    return out


def add_rain_tms_interaction_features(df: pd.DataFrame, 
                                      station_ids: List[str] = ["368", "541", "569"], 
                                      mode: str = "flow") -> pd.DataFrame:
    """
    강수-TMS 상호작용 피처 생성
    (강우 이벤트가 수질에 미치는 영향)
    
    ModelA 전용:
    - RN_15m × FLUX
    - RN_60m × SS(t-1) - lag 사용
    - (TN/TP) × PH
    - dry_duration × RN_15m
    
    ModelB 전용:
    - RN_15m × FLUX_VU
    - dry_duration_h × RN_15m
    - RN_60m × SS_lag (예: SS_lag_10m)
    
    ModelC 전용:
    - RN_15m × SS, RN_15m × TOC (희석/충격 효과)
    - RN_60m × SS, RN_60m × TOC
    - 온도 × TN, 온도 × TOC (생물학적 반응)
    
    ModelFLOW 전용:
    - rain × level_sum_lag1 (강우 × 수위 상호작용)
    """
    out = df.copy()
    
    for sid in station_ids:
        rn15 = f"RN_15m_{sid}"
        rn60 = f"RN_60m_{sid}"
        dry_hr = f"dry_duration_hr_{sid}"
        
        # 강수 x FLUX 상호작용 (공통)
        if rn15 in out.columns and "FLUX_VU" in out.columns:
            out[f"RN15xFLUX_{sid}"] = out[rn15] * out["FLUX_VU"]
        
        # 선행 강수 x TMS 지표
        ar12 = f"AR_12H_{sid}"
        if ar12 in out.columns:
            if "TN_VU" in out.columns:
                out[f"AR12xTN_{sid}"] = out[ar12] * out["TN_VU"]
            if "TP_VU" in out.columns:
                out[f"AR12xTP_{sid}"] = out[ar12] * out["TP_VU"]
        
        # ModelA 전용 상호작용
        if mode == "modela":
            # RN_60m × SS(t-1)
            if rn60 in out.columns and "SS_VU" in out.columns:
                ss_lag1 = out["SS_VU"].shift(1)
                out[f"RN60xSS_lag1_{sid}"] = out[rn60] * ss_lag1
            
            # dry_duration × RN_15m
            if dry_hr in out.columns and rn15 in out.columns:
                out[f"dry_x_RN15_{sid}"] = out[dry_hr] * out[rn15]
            
            # (TN/TP) × PH
            if all(c in out.columns for c in ["TN_VU", "TP_VU", "PH_VU"]):
                tn_tp_ratio = out["TN_VU"] / (out["TP_VU"] + 1e-6)
                out[f"TN_TP_ratio_x_PH_{sid}"] = tn_tp_ratio * out["PH_VU"]
        
        # ModelB 전용 상호작용
        elif mode == "modelb":
            # dry_duration × RN_15m
            if dry_hr in out.columns and rn15 in out.columns:
                out[f"dry_x_RN15_{sid}"] = out[dry_hr] * out[rn15]
            
            # RN_60m × SS_lag (1시간 lag)
            if rn60 in out.columns and "SS_VU" in out.columns:
                ss_lag_1h = out["SS_VU"].shift(1)
                out[f"RN60xSS_lag1h_{sid}"] = out[rn60] * ss_lag_1h
        
        # ModelC 전용 상호작용
        elif mode == "modelc":
            # RN_15m × SS, RN_15m × TOC (희석/충격 효과)
            if rn15 in out.columns:
                if "SS_VU" in out.columns:
                    out[f"RN15xSS_{sid}"] = out[rn15] * out["SS_VU"]
                if "TOC_VU" in out.columns:
                    out[f"RN15xTOC_{sid}"] = out[rn15] * out["TOC_VU"]
            
            # RN_60m × SS, RN_60m × TOC
            if rn60 in out.columns:
                if "SS_VU" in out.columns:
                    out[f"RN60xSS_{sid}"] = out[rn60] * out["SS_VU"]
                if "TOC_VU" in out.columns:
                    out[f"RN60xTOC_{sid}"] = out[rn60] * out["TOC_VU"]
    
    # ModelC 전용: 온도 상호작용 (생물학적 반응)
    if mode == "modelc":
        for sid in station_ids:
            ta_col = f"TA_{sid}"
            if ta_col in out.columns:
                if "TN_VU" in out.columns:
                    out[f"TAxTN_{sid}"] = out[ta_col] * out["TN_VU"]
                if "TOC_VU" in out.columns:
                    out[f"TAxTOC_{sid}"] = out[ta_col] * out["TOC_VU"]
    
    # ModelFLOW 전용: 강우 × 수위 상호작용
    if mode == "flow":
        # level_sum 생성 (있으면)
        if "level_TankA" in out.columns and "level_TankB" in out.columns:
            level_sum = out["level_TankA"] + out["level_TankB"]
            level_sum_lag1 = level_sum.shift(1)
            
            # 강우 × 수위 상호작용
            for sid in station_ids:
                rn60 = f"RN_60m_{sid}"
                if rn60 in out.columns:
                    out[f"rain_x_levelsum_lag1_{sid}"] = out[rn60] * level_sum_lag1
    
    return out


def add_level_flow_features(df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """
    수위-유량 기반 특성 생성 (FLOW 모드 전용)
    
    특성:
    - level_sum, level_diff: 수위 합계 및 차이
    - lag 특성: 1, 2, 3, 6, 12, 36 시간
    - rolling 특성 (shift(1) 후 계산):
      * 평균, 표준편차, 최소, 최대
      * IQR (Q90 - Q10)
      * 추세 (선형회귀 기울기)
    """
    out = df.copy()
    
    # 수위 데이터 확인
    if "level_TankA" not in out.columns or "level_TankB" not in out.columns:
        return out
    
    # 딕셔너리에 모아서 한 번에 추가 (성능 최적화)
    level_dict = {}
    
    # 기본 수위 특성
    level_dict["level_sum"] = out["level_TankA"] + out["level_TankB"]
    level_dict["level_diff"] = out["level_TankA"] - out["level_TankB"]
    
    # 임시로 추가 (rolling 계산용)
    out["level_sum"] = level_dict["level_sum"]
    out["level_diff"] = level_dict["level_diff"]
    
    # 수위 관련 컬럼
    level_cols = ["level_TankA", "level_TankB", "level_sum", "level_diff"]
    
    # Lag 특성 (1, 2, 3, 6, 12, 36 시간)
    lags = [1, 2, 3, 6, 12, 36]
    for col in level_cols:
        if col in out.columns:
            for lag in lags:
                level_dict[f"{col}_lag{lag}"] = out[col].shift(lag)
    
    # Rolling 특성 (shift(1) 후 계산하여 미래 정보 누수 방지)
    windows = [3, 6, 12, 24]  # 3, 6, 12, 24 시간
    
    # 추세 계산 함수
    def calc_slope(x):
        if len(x) < 2 or x.isna().all():
            return np.nan
        y = x.values
        if np.isnan(y).all():
            return np.nan
        # 결측치 제거
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        x_idx = np.arange(len(y))[mask]
        y_clean = y[mask]
        # 선형회귀
        if len(x_idx) < 2:
            return np.nan
        slope = np.polyfit(x_idx, y_clean, 1)[0]
        return slope
    
    for col in level_cols:
        if col not in out.columns:
            continue
        
        # shift(1)로 미래 정보 누수 방지
        col_shifted = out[col].shift(1)
        
        for w in windows:
            # 평균
            level_dict[f"{col}_rmean{w}"] = col_shifted.rolling(window=w, min_periods=max(1, w // 2)).mean()
            
            # 표준편차
            level_dict[f"{col}_rstd{w}"] = col_shifted.rolling(window=w, min_periods=max(2, w // 2)).std()
            
            # 최소/최대
            level_dict[f"{col}_rmin{w}"] = col_shifted.rolling(window=w, min_periods=max(1, w // 2)).min()
            level_dict[f"{col}_rmax{w}"] = col_shifted.rolling(window=w, min_periods=max(1, w // 2)).max()
            
            # IQR (Q90 - Q10)
            q90 = col_shifted.rolling(window=w, min_periods=max(1, w // 2)).quantile(0.9)
            q10 = col_shifted.rolling(window=w, min_periods=max(1, w // 2)).quantile(0.1)
            level_dict[f"{col}_rIQR{w}"] = q90 - q10
            
            # 추세 (선형회귀 기울기)
            level_dict[f"{col}_rslope{w}"] = col_shifted.rolling(window=w, min_periods=max(2, w // 2)).apply(calc_slope, raw=False)
    
    # 한 번에 concat
    if level_dict:
        level_df = pd.DataFrame(level_dict, index=out.index)
        # level_sum, level_diff는 이미 out에 있으므로 제거
        out = out.drop(columns=["level_sum", "level_diff"], errors='ignore')
        out = pd.concat([out, level_df], axis=1)
    
    return out


def add_rain_spatial_features(df: pd.DataFrame, 
                              station_ids: List[str] = ["368", "541", "569"], 
                              eps: float = 1e-6) -> pd.DataFrame:
    """
    강우 공간 통합 특성 생성 (FLOW 모드 전용)
    
    특성:
    - 공간 통계: mean, max, min, std, spread
    - 강우 형태: 단기 집중도 (RN15_div_RN60)
    - 선행강우지수 (ARI): tau=0.5, 1, 2 시간
    - 건조/습윤 상태: wet_flag, dry_spell_minutes
    """
    out = df.copy()
    
    # 강수량 컬럼 그룹
    rain_groups = {
        "RN_15m": [f"RN_15m_{sid}" for sid in station_ids],
        "RN_60m": [f"RN_60m_{sid}" for sid in station_ids],
        "RN_12H": [f"RN_12H_{sid}" for sid in station_ids],
    }
    
    # RN_DAY 생성 (24시간 누적)
    for sid in station_ids:
        rn15_col = f"RN_15m_{sid}"
        if rn15_col in out.columns:
            window = 24  # 24시간
            out[f"RN_DAY_{sid}"] = out[rn15_col].rolling(window=window, min_periods=1).sum()
    
    rain_groups["RN_DAY"] = [f"RN_DAY_{sid}" for sid in station_ids]
    
    # 공간 통합 통계
    for group_name, cols in rain_groups.items():
        available_cols = [c for c in cols if c in out.columns]
        if not available_cols:
            continue
        
        out[f"{group_name}_mean"] = out[available_cols].mean(axis=1)
        out[f"{group_name}_max"] = out[available_cols].max(axis=1)
        out[f"{group_name}_min"] = out[available_cols].min(axis=1)
        out[f"{group_name}_std"] = out[available_cols].std(axis=1)
        out[f"{group_name}_spread"] = out[available_cols].max(axis=1) - out[available_cols].min(axis=1)
    
    # 강우 형태: 단기 집중도
    if "RN_15m_mean" in out.columns and "RN_60m_mean" in out.columns:
        out["RN15_div_RN60"] = out["RN_15m_mean"] / (out["RN_60m_mean"] + eps)
    
    # 선행강우지수 (ARI) - 지수 감쇠 누적
    if "RN_15m_mean" in out.columns:
        rain = out["RN_15m_mean"]
        
        # ARI with different tau values
        for tau_hours in [0.5, 1, 2]:  # 30분, 60분, 120분
            tau_steps = int(tau_hours)  # 시간 단위
            ari_name = f"ARI_tau{int(tau_hours * 2)}"  # tau1, tau2, tau4
            ari = calculate_ari(rain, tau_steps=tau_steps, max_lookback=96)
            out[ari_name] = ari
    
    # 건조/습윤 상태
    if "RN_15m_mean" in out.columns:
        wet_thr = 0.1
        is_wet = (out["RN_15m_mean"] >= wet_thr).astype(int)
        out["wet_flag"] = is_wet
        
        # dry_spell_minutes: 마지막 강우 이후 경과 시간 (시간)
        dry_spell = np.zeros(len(out), dtype=np.float32)
        hours_since_rain = 0
        
        for i in range(len(out)):
            if is_wet.iloc[i]:
                hours_since_rain = 0
            else:
                hours_since_rain += 1  # 1시간 단위
            dry_spell[i] = hours_since_rain
        
        out["dry_spell_hours"] = dry_spell
    
    return out


# ========================================
# 통합 특성 생성 함수
# ========================================

def create_features(df: pd.DataFrame,
                   model_mode: str = "flow",
                   target_cols: List[str] = None,
                   add_time: bool = True,
                   add_sin_cos: bool = True,
                   lag_cols: List[str] = None,
                   lag_hours: List[int] = None,
                   rolling_cols: List[str] = None,
                   rolling_windows: List[int] = None,
                   rolling_stats: List[str] = None) -> pd.DataFrame:
    """
    전체 특성 생성 파이프라인
    
    Parameters:
    -----------
    df : DataFrame
        입력 데이터프레임
    model_mode : str
        모델 모드 (flow, modela, modelb, modelc)
    target_cols : list
        타겟 컬럼 리스트 (특성 생성에서 제외)
    add_time : bool
        시간 특성 추가 여부
    add_sin_cos : bool
        주기성 특성 추가 여부
    lag_cols : list
        Lag 특성을 생성할 컬럼 리스트
    lag_hours : list
        Lag 시간 리스트
    rolling_cols : list
        Rolling 특성을 생성할 컬럼 리스트
    rolling_windows : list
        Rolling window 크기 리스트
    rolling_stats : list
        Rolling 통계량 종류
        
    Returns:
    --------
    DataFrame : 모든 특성이 추가된 데이터프레임
    """
    out = df.copy()
    
    # 타겟 컬럼 기본값
    if target_cols is None:
        target_cols = []
    
    # 사용 가능한 컬럼 확인
    available_cols = set(out.columns)
    
    # TMS 컬럼 확인
    tms_cols = ["TOC_VU", "PH_VU", "SS_VU", "FLUX_VU", "TN_VU", "TP_VU"]
    has_tms = any(col in available_cols for col in tms_cols)
    
    # FLOW 컬럼 확인
    flow_cols = ["level_TankA", "level_TankB"]
    has_flow = all(col in available_cols for col in flow_cols)
    
    # 강수 컬럼 확인
    rain_cols = [f"RN_15m_{sid}" for sid in ["368", "541", "569"]]
    has_rain = any(col in available_cols for col in rain_cols)
    
    print(f"  컬럼 확인: TMS={has_tms}, FLOW={has_flow}, RAIN={has_rain}")
    
    # 시간 특성
    if add_time:
        out = add_time_features(out, add_sin_cos=add_sin_cos)
    
    # 도메인 특화 특성 (모델별)
    # 사용 가능한 TMS 컬럼 (예측 대상 제외)
    all_tms_cols = ["TOC_VU", "PH_VU", "SS_VU", "FLUX_VU", "TN_VU", "TP_VU"]
    available_tms_cols = [c for c in all_tms_cols if c in available_cols and c not in target_cols]
    
    print(f"  사용 가능한 TMS 컬럼 (예측 대상 제외): {available_tms_cols}")
    
    # 강수 및 기상 피처 (강수 데이터가 있을 때만)
    if has_rain:
        out = add_rain_features(out, mode=model_mode)
        out = add_weather_features(out, mode=model_mode)
    
    # FLOW 모델: 수위-유량 및 강우 공간 특화 특성
    if model_mode == "flow":
        if has_flow:
            out = add_level_flow_features(out)
        if has_rain:
            out = add_rain_spatial_features(out)
    
    # TMS 모델들만: TMS 상호작용 피처 추가
    if model_mode in ["modela", "modelb", "modelc"] and has_tms and len(available_tms_cols) > 0:
        out = add_tms_interaction_features(out, available_tms_cols, mode=model_mode)
    
    # 강수-TMS 상호작용 (해당 데이터가 있을 때만)
    if has_rain:
        if model_mode == "flow" and has_flow:
            out = add_rain_tms_interaction_features(out, mode=model_mode)
        elif model_mode in ["modela", "modelb", "modelc"] and has_tms:
            out = add_rain_tms_interaction_features(out, mode=model_mode)
    
    # 타겟 과거 정보 특성 추가 (데이터 누수 방지)
    # 예측 대상 변수의 과거 값을 특성으로 사용
    if target_cols:
        print(f"  타겟 과거 정보 특성 생성: {target_cols}")
        target_lags = [1, 2, 3, 6, 12, 24]
        target_windows = [3, 6, 12, 24]
        out = add_target_history_features(
            out, 
            target_cols=target_cols,
            lags=target_lags,
            windows=target_windows,
            stats=["mean", "std"]
        )
    
    # 숫자형 컬럼 선택 (타겟 제외)
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    
    # 마스크 컬럼 패턴 제외
    mask_patterns = ['_is_missing', '_imputed_', '_outlier_']
    
    # 기본 특성 컬럼 (타겟 및 마스크 제외)
    base_cols = [
        c for c in numeric_cols 
        if c not in target_cols and not any(pattern in c for pattern in mask_patterns)
    ]
    
    # Lag 특성
    if lag_cols is None:
        lag_cols = base_cols
    if lag_hours:
        out = add_lag_features(out, lag_cols, lag_hours)
    
    # Rolling 특성
    if rolling_cols is None:
        rolling_cols = base_cols
    if rolling_windows:
        if rolling_stats is None:
            rolling_stats = ["mean"]
        out = add_rolling_features(out, rolling_cols, rolling_windows, rolling_stats)
    
    # 타겟 컬럼은 유지 (step7에서 X, y 분리 시 필요)
    # 타겟의 과거 정보(_target_lag, _target_r, _target_delta 등)도 유지
    print(f"  최종 특성 수: {len(out.columns)} (타겟 포함)")
    
    return out
