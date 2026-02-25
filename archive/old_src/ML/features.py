"""
피처 엔지니어링 모듈
시간 피처, 지연 피처, 롤링 통계 피처 생성
모델별 도메인 특화 피처 생성
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


# ========================================
# 유틸리티 함수들
# ========================================

def calculate_rolling_std(df, cols, windows, suffix_pattern="{col}_std_{hours}H"):
    """
    여러 컬럼에 대해 rolling std 계산 (기상 안정성 등)
    
    Parameters:
    -----------
    df : pd.DataFrame
    cols : list
        계산할 컬럼 리스트
    windows : list
        시간 단위 윈도우 (예: [3, 6])
    suffix_pattern : str
        출력 컬럼명 패턴 (기본: "{col}_std_{hours}H")
    
    Returns:
    --------
    pd.DataFrame : rolling std가 추가된 데이터프레임
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        for hours in windows:
            window = hours * 4  # 15분 단위
            col_name = suffix_pattern.format(col=col, hours=hours)
            out[col_name] = out[col].rolling(window, min_periods=max(2, window // 10)).std()
    return out


def calculate_spike_flags(df, cols, window_hours=24, threshold=2.0, eps=1e-6):
    """
    여러 컬럼에 대해 spike flags 계산 (z-score > threshold)
    
    Parameters:
    -----------
    df : pd.DataFrame
    cols : list
        계산할 컬럼 리스트
    window_hours : int
        rolling window 시간 (기본: 24시간)
    threshold : float
        z-score 임계값 (기본: 2.0)
    eps : float
        0으로 나누기 방지
    
    Returns:
    --------
    pd.DataFrame : spike flags가 추가된 데이터프레임
    """
    out = df.copy()
    window = window_hours * 4  # 15분 단위
    
    for col in cols:
        if col not in out.columns:
            continue
        col_mean = out[col].rolling(window, min_periods=max(5, window // 10)).mean()
        col_std = out[col].rolling(window, min_periods=max(5, window // 10)).std()
        z_score = (out[col] - col_mean) / (col_std + eps)
        out[f"{col}_spike_z2"] = (z_score > threshold).astype(np.int8)
    
    return out


def calculate_derivatives(df, cols):
    """
    여러 컬럼에 대해 변화율(derivative) 계산
    
    Parameters:
    -----------
    df : pd.DataFrame
    cols : list
        계산할 컬럼 리스트
    
    Returns:
    --------
    pd.DataFrame : 변화율이 추가된 데이터프레임
    """
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[f"d_{col}"] = out[col].diff()
    return out


def calculate_ari(rain_series, tau_steps, max_lookback=96):
    """
    선행강우지수 (ARI - Antecedent Rainfall Index) 계산
    
    ARI(t) = Σ rain(t-i) × exp(-i/τ)
    
    Parameters:
    -----------
    rain_series : pd.Series or np.ndarray
        강수량 시계열
    tau_steps : int
        감쇠 시간 상수 (스텝 단위)
    max_lookback : int
        최대 과거 참조 스텝 (기본: 96 = 24시간)
    
    Returns:
    --------
    np.ndarray : ARI 값
    """
    rain = rain_series.to_numpy(dtype=np.float32) if isinstance(rain_series, pd.Series) else rain_series
    ari = np.full_like(rain, np.nan, dtype=np.float32)
    
    for t in range(len(rain)):
        start = max(0, t - max_lookback)
        seg = rain[start:t+1]
        if seg.size == 0:
            ari[t] = 0.0
        else:
            # 지수 감쇠 가중치
            ages = np.arange(seg.size - 1, -1, -1, dtype=np.float32)
            weights = np.exp(-ages / tau_steps)
            ari[t] = float(np.sum(seg * weights))
    
    return ari


# ========================================
# 기본 피처 생성 함수들
# ========================================

def add_time_features(df, add_sin_cos=True):
    """시간 관련 피처 추가 (시간, 요일, 월, 주말, 시간대, 계절)"""
    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("시간 피처 생성을 위해서는 DatetimeIndex가 필요합니다.")

    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek  # 월요일=0
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # 시간대 구간 (필요시 커스터마이징 가능)
    # 0: 밤(0-5), 1: 아침(6-11), 2: 오후(12-17), 3: 저녁(18-23)
    out["tod_bucket"] = pd.cut(
        out["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # 계절
    m = out["month"]
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    out["season"] = m.map(season_map).astype(int)

    if add_sin_cos:
        # 시간 주기 (24시간)
        out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
        out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)

        # 요일 주기 (7일)
        out["sin_dow"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
        out["cos_dow"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)

        # 월 주기 (12개월)
        out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12.0)
        out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12.0)

    return out


def add_lag_features(df, base_cols, lags):
    """지연(lag) 피처 추가"""
    out = df.copy()
    for c in base_cols:
        if c not in out.columns:
            continue
        for k in lags:
            out[f"{c}_lag{k}"] = out[c].shift(k)
    return out


def add_rolling_features(df, base_cols, windows, stats=["mean"]):
    """
    롤링 통계 피처 추가
    stats: ["mean", "std", "min", "max"] - 통계량 종류
    """
    out = df.copy()
    for c in base_cols:
        if c not in out.columns:
            continue
        for w in windows:
            r = out[c].rolling(window=w, min_periods=w)
            if "mean" in stats:
                out[f"{c}_r{w}_mean"] = r.mean()
            if "std" in stats:
                out[f"{c}_r{w}_std"] = r.std()
            if "min" in stats:
                out[f"{c}_r{w}_min"] = r.min()
            if "max" in stats:
                out[f"{c}_r{w}_max"] = r.max()
    return out


# ========================================
# 도메인 특화 피처 생성 함수들
# ========================================

def add_rain_features(df, station_ids=["368", "541", "569"], eps=1e-6, mode="flow"):
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
                window = hours * 4  # 15분 단위 (1시간 = 4개)
                ar_col = out[rn15_col].rolling(window=window, min_periods=1).sum()
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
            out[f"dry_duration_hr_{sid}"] = dry_duration * 0.25  # 15분 단위를 시간으로 변환
            
            # ModelA/B/C 전용: rain_start, rain_end, post_rain_6H 플래그
            if mode in ["modela", "modelb", "modelc"]:
                rain_start = is_wet & (~is_wet.shift(1, fill_value=0).astype(bool))
                rain_end = (~is_wet.astype(bool)) & (is_wet.shift(1, fill_value=0).astype(bool))
                
                out[f"rain_start_{sid}"] = rain_start.astype(np.int8)
                out[f"rain_end_{sid}"] = rain_end.astype(np.int8)
                
                # 종료 후 6시간 잔류 효과
                post_win = 6 * 4  # 6시간 = 24개 (15분 단위)
                out[f"post_rain_6H_{sid}"] = (
                    pd.Series(rain_end.values, index=out.index)
                    .rolling(post_win, min_periods=1).max()
                    .fillna(0).astype(np.int8)
                )
        
        # ModelA/B/C 전용: API 지수 (감쇠 누적)
        if mode in ["modela", "modelb", "modelc"] and rn15_col in out.columns:
            # API(t) = Σ RN_15m(t-i) * exp(-k*i)
            tau_steps = 100  # tau = 100 스텝 (약 25시간)
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


def add_weather_features(df, station_ids=["368", "541", "569"], eps=1e-6, mode="flow"):
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


def add_tms_interaction_features(df, available_tms_cols, mode="flow", eps=1e-6):
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
            window = 24 * 4  # 24시간 (15분 단위)
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


def add_rain_tms_interaction_features(df, station_ids=["368", "541", "569"], mode="flow"):
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
            
            # RN_60m × SS_lag (10분 lag)
            if rn60 in out.columns and "SS_VU" in out.columns:
                # 10분 lag = 2 스텝 (5분 단위)
                ss_lag_10m = out["SS_VU"].shift(2)
                out[f"RN60xSS_lag10m_{sid}"] = out[rn60] * ss_lag_10m
        
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


def add_level_flow_features(df, eps=1e-6):
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
    
    # 기본 수위 특성
    out["level_sum"] = out["level_TankA"] + out["level_TankB"]
    out["level_diff"] = out["level_TankA"] - out["level_TankB"]
    
    # 수위 관련 컬럼
    level_cols = ["level_TankA", "level_TankB", "level_sum", "level_diff"]
    
    # Lag 특성 (1, 2, 3, 6, 12, 36 시간)
    lags = [1, 2, 3, 6, 12, 36]
    for col in level_cols:
        if col in out.columns:
            for lag in lags:
                out[f"{col}_lag{lag}"] = out[col].shift(lag)
    
    # Rolling 특성 (shift(1) 후 계산하여 미래 정보 누수 방지)
    windows = [3, 6, 12, 24]  # 3, 6, 12, 24 시간
    
    for col in level_cols:
        if col not in out.columns:
            continue
        
        # shift(1)로 미래 정보 누수 방지
        col_shifted = out[col].shift(1)
        
        for w in windows:
            # 평균
            out[f"{col}_rmean{w}"] = col_shifted.rolling(window=w, min_periods=max(1, w // 2)).mean()
            
            # 표준편차
            out[f"{col}_rstd{w}"] = col_shifted.rolling(window=w, min_periods=max(2, w // 2)).std()
            
            # 최소/최대
            out[f"{col}_rmin{w}"] = col_shifted.rolling(window=w, min_periods=max(1, w // 2)).min()
            out[f"{col}_rmax{w}"] = col_shifted.rolling(window=w, min_periods=max(1, w // 2)).max()
            
            # IQR (Q90 - Q10)
            q90 = col_shifted.rolling(window=w, min_periods=max(1, w // 2)).quantile(0.9)
            q10 = col_shifted.rolling(window=w, min_periods=max(1, w // 2)).quantile(0.1)
            out[f"{col}_rIQR{w}"] = q90 - q10
            
            # 추세 (선형회귀 기울기)
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
            
            out[f"{col}_rslope{w}"] = col_shifted.rolling(window=w, min_periods=max(2, w // 2)).apply(calc_slope, raw=False)
    
    return out


def add_rain_spatial_features(df, station_ids=["368", "541", "569"], eps=1e-6):
    """
    강우 공간 통합 특성 생성 (FLOW 모드 전용)
    
    특성:
    - 공간 통계: mean, max, min, std, spread
    - 강우 형태: 단기 집중도 (RN15_div_RN60)
    - 선행강우지수 (ARI): tau=6, 12, 24
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
            window = 24 * 4  # 24시간 (15분 단위)
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
            tau_steps = int(tau_hours * 4)  # 15분 단위로 변환
            ari_name = f"ARI_tau{int(tau_hours * 2)}"  # tau6, tau12, tau24
            ari = calculate_ari(rain, tau_steps=tau_steps, max_lookback=96)
            out[ari_name] = ari
    
    # 건조/습윤 상태
    if "RN_15m_mean" in out.columns:
        wet_thr = 0.1
        is_wet = (out["RN_15m_mean"] >= wet_thr).astype(int)
        out["wet_flag"] = is_wet
        
        # dry_spell_minutes: 마지막 강우 이후 경과 시간 (분)
        dry_spell = np.zeros(len(out), dtype=np.float32)
        minutes_since_rain = 0
        
        for i in range(len(out)):
            if is_wet.iloc[i]:
                minutes_since_rain = 0
            else:
                minutes_since_rain += 15  # 15분 단위
            dry_spell[i] = minutes_since_rain
        
        out["dry_spell_minutes"] = dry_spell
    
    return out


# ========================================
# 모델별 피처 생성 함수
# ========================================

def add_model_specific_features(df, mode, available_tms_cols):
    """
    모델별 특화 피처 생성
    
    각 모델은 노트북에 정의된 대로 다른 특성 엔지니어링을 사용:
    - ModelA (TOC+SS): FLUX를 입력으로 사용, 강수/기상/TMS 상호작용 포함
      * 강수: 단기 집중도, API 지수, rain_start/end, post_rain_6H
      * 기상: VPD, 기상 안정성 (rolling_std)
      * TMS: 부하 특성, 영양염 비율, 공정 상태 플래그, 변화율
      * 상호작용: RN_60m×SS(t-1), (TN/TP)×PH, dry×RN_15m
    - ModelB (TN+TP): FLUX를 입력으로 사용, TMS 상호작용 포함  
    - ModelC (FLUX+PH): FLUX 제외 (예측 대상), TMS 상호작용 포함
    - ModelFLOW: TMS 데이터 전혀 사용 안 함, 수위-유량 및 강우 특화 특성
      * 수위-유량: level_sum/diff, lag, rolling (평균/표준편차/IQR/추세)
      * 강우: 공간 통합, ARI 지수, 건조/습윤 상태
      * 상호작용: rain × level_sum_lag1
    
    Parameters:
    -----------
    df : pd.DataFrame
        데이터프레임
    mode : str
        모델 모드 (flow, modelA, modelB, modelC, tms)
    available_tms_cols : list
        사용 가능한 TMS 컬럼 리스트 (예측 대상 제외)
        
    Returns:
    --------
    pd.DataFrame : 피처가 추가된 데이터프레임
    """
    out = df.copy()
    
    # 모든 모델에 공통: 강수 및 기상 피처 (mode 전달)
    out = add_rain_features(out, mode=mode)
    out = add_weather_features(out, mode=mode)
    
    # FLOW 모델: 수위-유량 및 강우 공간 특화 특성
    if mode == "flow":
        out = add_level_flow_features(out)
        out = add_rain_spatial_features(out)
    
    # TMS 모델들만: TMS 상호작용 피처 추가
    # ModelFLOW는 TMS 데이터를 전혀 사용하지 않음
    if mode in ["modela", "modelb", "modelc", "tms"]:
        out = add_tms_interaction_features(out, available_tms_cols, mode=mode)
        out = add_rain_tms_interaction_features(out, mode=mode)
    
    # FLOW 모델: 강우-수위 상호작용 (TMS 없이)
    if mode == "flow":
        out = add_rain_tms_interaction_features(out, mode=mode)
    
    return out


# ========================================
# 데이터셋 생성
# ========================================

def make_supervised_dataset(df, target_cols, exclude_cols=None, dropna=True):
    """
    지도학습용 X, y 데이터셋 생성
    
    Parameters:
    -----------
    df : pd.DataFrame
        피처가 포함된 데이터프레임
    target_cols : list
        타겟 컬럼 리스트
    exclude_cols : list, optional
        X에서 제외할 컬럼 리스트 (기본값: target_cols)
        TMS 모델의 경우 예측 대상만 제외하고 나머지 TMS 지표는 입력으로 사용
    dropna : bool
        결측치가 있는 행 제거 여부
        
    Returns:
    --------
    tuple : (X, y)
    """
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"타겟 컬럼을 찾을 수 없습니다: {missing}")

    y = df[target_cols].copy()
    
    # 제외할 컬럼 결정
    if exclude_cols is None:
        exclude_cols = target_cols
    
    # X 생성: exclude_cols에 있는 컬럼만 제외
    X = df.drop(columns=exclude_cols, errors='ignore').copy()
    
    # 숫자형 컬럼만 선택
    X = X.select_dtypes(include=[np.number])

    if dropna:
        keep = X.notna().all(axis=1) & y.notna().all(axis=1)
        return X.loc[keep], y.loc[keep]
    
    return X, y


# ========================================
# 설정 및 메인 함수
# ========================================

@dataclass
class FeatureConfig:
    """피처 생성 설정"""
    add_time: bool = True
    add_sin_cos: bool = True
    add_domain_features: bool = True  # 도메인 특화 피처 추가 여부
    lag_hours: list = None
    roll_hours: list = None

    def __post_init__(self):
        if self.lag_hours is None:
            self.lag_hours = [1, 2, 3, 6, 12, 24]
        if self.roll_hours is None:
            self.roll_hours = [1, 2, 24]


def build_features(df_hourly, target_cols, exclude_cols=None, feature_base_cols=None, 
                   mode="flow", cfg=FeatureConfig()):
    """
    전체 피처 생성 파이프라인
    
    Parameters:
    -----------
    df_hourly : pd.DataFrame
        리샘플링된 데이터프레임
    target_cols : list
        타겟 컬럼 리스트
    exclude_cols : list, optional
        피처 생성에서 제외할 컬럼 (기본값: target_cols)
        이 컬럼들의 lag/rolling 피처는 생성되지 않음 (데이터 누수 방지)
    feature_base_cols : list, optional
        피처 생성에 사용할 기본 컬럼
    mode : str
        모델 모드 (flow, modelA, modelB, modelC, tms)
    cfg : FeatureConfig
        피처 생성 설정
        
    Returns:
    --------
    pd.DataFrame : 피처가 추가된 데이터프레임
    """
    out = df_hourly.copy()

    # 1) 시간 피처
    if cfg.add_time:
        out = add_time_features(out, add_sin_cos=cfg.add_sin_cos)

    # 제외할 컬럼 결정 (데이터 누수 방지)
    if exclude_cols is None:
        exclude_cols = target_cols
    
    # 2) 도메인 특화 피처 (모델별)
    if cfg.add_domain_features:
        # 사용 가능한 TMS 컬럼 (예측 대상 제외)
        all_tms_cols = ["TOC_VU", "PH_VU", "SS_VU", "FLUX_VU", "TN_VU", "TP_VU"]
        available_tms_cols = [c for c in all_tms_cols if c in out.columns and c not in exclude_cols]
        
        out = add_model_specific_features(out, mode, available_tms_cols)
    
    # 3) 기본 컬럼 결정
    if feature_base_cols is None:
        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        
        # 제외할 컬럼 패턴 정의
        # - exclude_cols: 예측 대상 변수 (데이터 누수 방지)
        # - 마스크 컬럼: is_missing, imputed_*, outlier_* (메타데이터이므로 lag/rolling 불필요)
        mask_patterns = ['_is_missing', '_imputed_', '_outlier_']
        
        feature_base_cols = [
            c for c in numeric_cols 
            if c not in exclude_cols and not any(pattern in c for pattern in mask_patterns)
        ]
    
    # 4) lag/rolling 피처 생성 (예측 대상 및 마스크 제외)
    out = add_lag_features(out, base_cols=feature_base_cols, lags=cfg.lag_hours)
    out = add_rolling_features(out, base_cols=feature_base_cols, windows=cfg.roll_hours)
    
    return out
