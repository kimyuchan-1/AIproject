"""
Feature Engineering Module for WWTP Time Series Prediction
===========================================================

This module contains feature engineering functions for water treatment plant
time series prediction. Each function adds specific types of features to the
input DataFrame.

Author: WWTP Project
Date: 2026-02-06
"""

import numpy as np
import pandas as pd


# ==============================================================================
# Configuration: Data Leakage Prevention
# ==============================================================================

DATA_LEAKAGE_CONFIG = {
    "flow": {
        "target": ["Q_in"],
        "safe_process_features": ["level_TankA", "level_TankB"]
    },
    "toc": {
        "target": ["TOC_VU"],
        "safe_process_features": ["FLUX_VU", "SS_VU", "TN_VU", "TP_VU", "PH_VU"]
    },
    "ss": {
        "target": ["SS_VU"],
        "safe_process_features": ["FLUX_VU", "TOC_VU", "TN_VU", "TP_VU", "PH_VU"]
    },
    "tn": {
        "target": ["TN_VU"],
        "safe_process_features": ["FLUX_VU", "TOC_VU", "SS_VU", "TP_VU", "PH_VU"]
    },
    "tp": {
        "target": ["TP_VU"],
        "safe_process_features": ["FLUX_VU", "TOC_VU", "SS_VU", "TN_VU", "PH_VU"]
    },
    "flux": {
        "target": ["FLUX_VU"],
        "safe_process_features": ["TOC_VU", "SS_VU", "TN_VU", "TP_VU", "PH_VU"] 
    },
    "ph": {
        "target": ["PH_VU"],
        "safe_process_features": ["FLUX_VU", "TOC_VU", "SS_VU", "TN_VU", "TP_VU"]
    },
}


# ==============================================================================
# Feature Engineering Functions
# ==============================================================================

def add_rain_features(df):
    """
    강수 + 기상-강수 상호작용 통합 (PerformanceWarning 최소화 버전)
    - 새 컬럼은 new_cols dict에 누적 후 마지막에 한 번에 concat

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime index

    Returns
    -------
    pd.DataFrame
        DataFrame with added rain features
    """
    df_out = df.copy()
    new_cols = {}

    station_ids = ["368", "541", "569"]
    rain_cols = [c for c in df_out.columns if c.startswith("RN_")]
    antecedent_hours = (3, 6, 12, 24)
    wet_thr_mm = 0.1
    ari_taus = (6, 12, 24)
    eps = 1e-9

    # ====== 1. 지점별 기본 강수 특성 ======
    for sid in station_ids:
        for rc in rain_cols[:3]:
            col = f"{rc}_{sid}"
            if col in df_out.columns:
                new_cols[f"d_{col}"] = df_out[col].diff()

    # ====== 2. 지점별 기상-강수 상호작용 ======
    for sid in station_ids:
        ta = f"TA_{sid}"
        td = f"TD_{sid}"
        hm = f"HM_{sid}"
        rn15 = f"RN_15m_{sid}"
        rn60 = f"RN_60m_{sid}"

        if ta in df_out.columns and td in df_out.columns:
            tadtd = df_out[ta] - df_out[td]
            new_cols[f"TA_minus_TD_{sid}"] = tadtd

            if hm in df_out.columns:
                new_cols[f"TAxHM_{sid}"] = df_out[ta] * df_out[hm]
                new_cols[f"TA_minus_TD_x_HM_{sid}"] = tadtd * df_out[hm]

        if rn15 in df_out.columns and rn60 in df_out.columns:
            new_cols[f"RN15_div_RN60_{sid}"] = df_out[rn15] / (df_out[rn60] + eps)

    # ====== 3. 공간 통합 강수/기상 특성 ======
    def _spatial_stats(base_name):
        cols = [f"{base_name}_{sid}" for sid in station_ids if f"{base_name}_{sid}" in df_out.columns]
        if not cols:
            return {}

        arr = df_out[cols]
        mean_ = arr.mean(axis=1)
        max_ = arr.max(axis=1)
        min_ = arr.min(axis=1)
        std_ = arr.std(axis=1, ddof=0)

        out = {
            f"{base_name}_mean": mean_,
            f"{base_name}_max": max_,
            f"{base_name}_min": min_,
            f"{base_name}_std": std_,
            f"{base_name}_spread": max_ - min_,
        }
        return out

    for rc in rain_cols:
        new_cols.update(_spatial_stats(rc))

    for mc in ["TA", "TD", "HM"]:
        new_cols.update(_spatial_stats(mc))

    # 이슬점 감차 공간 통계 (TA_minus_TD는 위에서 new_cols로 만들었을 수 있음)
    if any(f"TA_minus_TD_{sid}" in new_cols or f"TA_minus_TD_{sid}" in df_out.columns for sid in station_ids):
        df_out_tmp = df_out
        if new_cols:
            df_out_tmp = pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)
        # tmp에서 계산해 다시 new_cols에 추가(중복키는 덮어씀)
        cols_tadtd = [f"TA_minus_TD_{sid}" for sid in station_ids if f"TA_minus_TD_{sid}" in df_out_tmp.columns]
        if cols_tadtd:
            arr = df_out_tmp[cols_tadtd]
            mean_ = arr.mean(axis=1)
            max_ = arr.max(axis=1)
            min_ = arr.min(axis=1)
            std_ = arr.std(axis=1, ddof=0)
            new_cols.update({
                "TA_minus_TD_mean": mean_,
                "TA_minus_TD_max": max_,
                "TA_minus_TD_min": min_,
                "TA_minus_TD_std": std_,
                "TA_minus_TD_spread": max_ - min_,
            })

    # ====== 4. 강수 비율 ======
    for sid in station_ids:
        rn15 = f"RN_15m_{sid}"
        rn60 = f"RN_60m_{sid}"
        if rn15 in df_out.columns and rn60 in df_out.columns:
            new_cols[f"rain_ratio_15m_60m_{sid}"] = df_out[rn15] / (df_out[rn60] + eps)

    # ====== 5. 누적강수 (선행강우) ======
    steps_per_hour = 2  # 30분 리샘플링: 1시간 = 2 steps
    for sid in station_ids:
        rn_col = f"RN_60m_{sid}"
        if rn_col in df_out.columns:
            s = df_out[rn_col]
            for h in antecedent_hours:
                win = h * steps_per_hour
                ar = s.rolling(win, min_periods=1).sum()
                new_cols[f"AR_{h}h_{sid}"] = ar
                new_cols[f"log1p_AR_{h}h_{sid}"] = np.log1p(ar.clip(lower=0))

    # ====== 6. Wet/Dry 상태 ======
    for sid in station_ids:
        col = f"RN_15m_{sid}"
        if col not in df_out.columns:
            continue

        wet = (df_out[col].fillna(0.0) >= wet_thr_mm)

        # last_wet_ts (datetimeindex) -> wet인 시점 index를 ffill
        idx = df_out.index
        last_wet_ts = pd.Series(idx.where(wet), index=idx).ffill()

        dry_td = idx - last_wet_ts
        dry_min = (dry_td / pd.Timedelta("1min")).astype("float64")
        dry_hr = (dry_td / pd.Timedelta("1h")).astype("float64")

        new_cols[f"dry_minutes_{sid}"] = pd.Series(dry_min, index=idx).fillna(0.0)
        new_cols[f"dry_hours_{sid}"] = pd.Series(dry_hr, index=idx).fillna(0.0)
        new_cols[f"is_wet_{sid}"] = wet.astype(np.int8)

        rain_start = wet & (~wet.shift(1, fill_value=False))
        rain_end = (~wet) & (wet.shift(1, fill_value=False))
        new_cols[f"rain_start_{sid}"] = rain_start.astype(np.int8)
        new_cols[f"rain_end_{sid}"] = rain_end.astype(np.int8)

        post_win = 6 * steps_per_hour
        new_cols[f"post_rain_6H_{sid}"] = (
            pd.Series(rain_end.to_numpy(), index=idx)
            .rolling(post_win, min_periods=1).max()
            .fillna(0).astype(np.int8)
        )

    # ====== 7. ARI (선행강우지수) ======
    # RN_60m_mean이 df_out에 있거나 new_cols에서 만들어졌을 수 있음
    rn60_mean = None
    if "RN_60m_mean" in new_cols:
        rn60_mean = new_cols["RN_60m_mean"]
    elif "RN_60m_mean" in df_out.columns:
        rn60_mean = df_out["RN_60m_mean"]

    if rn60_mean is not None:
        rain_for_ari = rn60_mean.shift(1)  # 현재 강우 제외

        for tau in ari_taus:
            klen = int(6 * tau)  # (원 코드 유지)
            w = np.exp(-np.arange(klen) / float(tau))
            w = w / (w.sum() + eps)

            def _ari(x):
                x = np.asarray(x, dtype=float)
                if np.any(~np.isfinite(x)):
                    return np.nan
                ww = w[-len(x):]
                return float((x * ww).sum())

            new_cols[f"ARI_tau{tau}"] = rain_for_ari.rolling(klen, min_periods=klen).apply(_ari, raw=True)

    # ====== 8. 면적강수 통합 ======
    ar12_cols = [f"AR_12h_{sid}" for sid in station_ids]
    # ar12가 new_cols에 있을 수도 있으니 tmp에서 계산
    df_tmp = df_out
    if new_cols:
        df_tmp = pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)

    ar12_exist = [c for c in ar12_cols if c in df_tmp.columns]
    if ar12_exist:
        new_cols["AR_12h_sum_all"] = df_tmp[ar12_exist].sum(axis=1)
        new_cols["AR_12h_mean_all"] = df_tmp[ar12_exist].mean(axis=1)

    rn15_mean = df_tmp["RN_15m_mean"] if "RN_15m_mean" in df_tmp.columns else None
    rn60_mean2 = df_tmp["RN_60m_mean"] if "RN_60m_mean" in df_tmp.columns else None
    if rn15_mean is not None and rn60_mean2 is not None:
        new_cols["RN_15m_div_RN_60m_mean"] = rn15_mean / (rn60_mean2 + eps)

    # ====== 마지막: 한 번에 합치기 ======
    if new_cols:
        df_out = pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)

    return df_out


def add_process_features(df):
    """
    공정 변수 기반 특성 추가
    - Load 계산 (FLUX × 농도)
    - 변수 간 상호작용
    - pH zone 플래그

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with process variables

    Returns
    -------
    pd.DataFrame
        DataFrame with added process features
    """
    ph_low, ph_high = (5.8, 8.5)
    eps = 1e-6

    df_out = df.copy()
    new_cols = {}

    cols = df_out.columns

    # 자주 쓰는 컬럼은 미리 뽑아두기(있을 때만)
    FLUX = df_out["FLUX_VU"] if "FLUX_VU" in cols else None
    PH   = df_out["PH_VU"]   if "PH_VU"   in cols else None
    SS   = df_out["SS_VU"]   if "SS_VU"   in cols else None
    TOC  = df_out["TOC_VU"]  if "TOC_VU"  in cols else None
    TN   = df_out["TN_VU"]   if "TN_VU"   in cols else None
    TP   = df_out["TP_VU"]   if "TP_VU"   in cols else None

    # Loads
    if FLUX is not None and SS is not None:
        new_cols["SS_load"] = FLUX * SS
        new_cols["SS_x_FLUX"] = SS * FLUX

    if FLUX is not None and TOC is not None:
        new_cols["TOC_load"] = FLUX * TOC

    if FLUX is not None and TN is not None and TP is not None:
        new_cols["load_proxy_NP"] = FLUX * (TN + TP)

    # Interactions
    if PH is not None and FLUX is not None:
        new_cols["PH_x_FLUX"] = PH * FLUX

    if PH is not None and TOC is not None:
        new_cols["PH_x_TOC"] = PH * TOC

    if TOC is not None and SS is not None:
        new_cols["TOC_x_SS"] = TOC * SS
        new_cols["TOC_div_SS"] = TOC / (SS + eps)

    if TN is not None and TP is not None:
        new_cols["TN_x_TP"] = TN * TP
        new_cols["TN_div_TP"] = TN / (TP + eps)
        new_cols["log1p_TN_TP"] = np.log1p((TN + TP).clip(lower=0))

    if TOC is not None and TN is not None:
        new_cols["TOC_div_TN"] = TOC / (TN + eps)

    # pH zone flags
    if PH is not None:
        phv = PH.astype("float64")
        new_cols["pH_acid"] = (phv < ph_low).astype(np.int8)
        new_cols["pH_neutral"] = ((phv >= ph_low) & (phv <= ph_high)).astype(np.int8)
        new_cols["pH_basic"] = (phv > ph_high).astype(np.int8)

    if new_cols:
        df_out = pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)

    return df_out


def add_temporal_features(df):
    """
    시간 특성 생성 (성능 개선 버전)
    - Lag features
    - Rolling 통계 (mean, std, max, min, IQR=Q90-Q10)
    - Rolling slope (closed-form)
    - Difference features

    주기: 30분 리샘플링 기준 (1시간 = 2 steps)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime index

    Returns
    -------
    pd.DataFrame
        DataFrame with added temporal features
    """
    station_ids = ["368", "541", "569"]
    weather_cols = ["TA", "TD", "HM"]
    process_cols = ["PH_VU", "FLUX_VU", "TN_VU", "TP_VU", "SS_VU", "TOC_VU"]

    # 30분 리샘플링: 1시간 = 2 steps
    roll_windows = [6, 12, 24, 72]      # 3h, 6h, 12h, 36h
    lags = [2, 4, 6, 12, 24, 48, 72]    # 1h, 2h, 3h, 6h, 12h, 24h, 36h
    slope_windows = [12, 24, 48]        # 6h, 12h, 24h

    df_out = df.copy()
    new_cols = {}

    # ====== 1) 대상 컬럼(존재하는 것만) ======
    roll_targets = (
        list(process_cols)
        + [f"{wc}_{sid}" for sid in station_ids for wc in weather_cols]
        + [f"RN_15m_{sid}" for sid in station_ids]
        + [f"RN_60m_{sid}" for sid in station_ids]
    )
    roll_targets = [c for c in roll_targets if c in df_out.columns]

    lag_targets = (
        list(process_cols)
        + [f"{col}_{sid}" for sid in station_ids for col in (["RN_15m", "RN_60m", "RN_12H"] + weather_cols)]
    )
    lag_targets = [c for c in lag_targets if c in df_out.columns]

    # ====== 2) Rolling 통계 (DataFrame 단위) ======
    if roll_targets:
        X = df_out[roll_targets].shift(1)  # 미래정보 방지

        for w in roll_windows:
            win_steps = int(w)                 # 30분 단위 steps
            minp = max(1, win_steps // 2)

            r = X.rolling(win_steps, min_periods=minp)

            mean_df = r.mean()
            std_df  = r.std(ddof=0)
            max_df  = r.max()
            min_df  = r.min()

            q90_df = r.quantile(0.90)
            q10_df = r.quantile(0.10)
            iqr_df = q90_df - q10_df

            for c in roll_targets:
                new_cols[f"{c}_roll_mean_{w}"] = mean_df[c]
                new_cols[f"{c}_roll_std_{w}"]  = std_df[c]
                new_cols[f"{c}_roll_max_{w}"]  = max_df[c]
                new_cols[f"{c}_roll_min_{w}"]  = min_df[c]
                new_cols[f"{c}_roll_IQR_{w}"]  = iqr_df[c]

    # ====== 3) Rolling slope (closed-form; stats.linregress 제거) ======
    def _rolling_slope_fast(s: pd.Series, window: int) -> pd.Series:
        """
        slope = Σ((t- t̄)(x- x̄)) / Σ((t- t̄)²)
        raw=True rolling.apply에서 x만 들어오므로 t는 고정 벡터로 처리
        window: 30분 단위 steps
        """
        t = np.arange(window, dtype=float)
        t0 = t - t.mean()
        denom = float((t0 * t0).sum())
        if denom <= 0:
            return s.rolling(window, min_periods=window).mean() * np.nan

        def _slope(x):
            x = np.asarray(x, dtype=float)
            if np.any(~np.isfinite(x)):
                return np.nan
            x0 = x - x.mean()
            return float((t0 * x0).sum() / denom)

        return s.rolling(window, min_periods=window).apply(_slope, raw=True)

    if roll_targets:
        for w in slope_windows:
            win_steps = int(w)
            for c in roll_targets:
                # 이미 shift(1)한 X가 있으면 재사용
                s = df_out[c].shift(1)
                new_cols[f"{c}_roll_slope_{w}"] = _rolling_slope_fast(s, win_steps)

    # ====== 4) Lag ======
    for lag in lags:
        lag_steps = int(lag)
        for c in lag_targets:
            new_cols[f"{c}_lag_{lag}steps"] = df_out[c].shift(lag_steps)

    # ====== 5) Difference ======
    for c in lag_targets:
        new_cols[f"d_{c}"] = df_out[c].diff()

    if "FLUX_VU" in df_out.columns:
        new_cols["abs_d_FLUX_VU"] = df_out["FLUX_VU"].diff().abs()

    # ====== 6) 한 번에 합치기 ======
    if new_cols:
        df_out = pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)

    return df_out


def add_interaction_features(df):
    """
    상호작용 특성 (성능 개선: new_cols 누적 후 1회 concat)
    - 강우 × 수위
    - 강우 × 공정변수
    - 기상 × 공정변수
    - 건조기간 × 강우
    - 누적강수 × 공정변수
    - 공정변수 간 상호작용
    - ARI × 공정변수

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with multiple feature types

    Returns
    -------
    pd.DataFrame
        DataFrame with added interaction features
    """
    df_out = df.copy()
    new_cols = {}

    station_ids = ["368", "541", "569"]
    eps = 1e-9

    # ====== 1) 공정 컬럼 캐시 ======
    proc_cols = ["TOC_VU", "SS_VU", "TN_VU", "TP_VU", "FLUX_VU", "PH_VU"]
    cols = df_out.columns
    has_proc = {c: (c in cols) for c in proc_cols}

    TOC  = df_out["TOC_VU"]  if has_proc["TOC_VU"]  else None
    SS   = df_out["SS_VU"]   if has_proc["SS_VU"]   else None
    TN   = df_out["TN_VU"]   if has_proc["TN_VU"]   else None
    TP   = df_out["TP_VU"]   if has_proc["TP_VU"]   else None
    FLUX = df_out["FLUX_VU"] if has_proc["FLUX_VU"] else None

    # ====== 2) 수위/플래그 존재 ======
    level_sum_exists  = "level_sum" in cols
    level_diff_exists = "level_diff" in cols

    # ====== 3) 강우×수위 (lag1) ======
    # shift(1) 재사용
    if level_sum_exists:
        level_sum_l1 = df_out["level_sum"].shift(1)
    else:
        level_sum_l1 = None

    if level_diff_exists:
        level_diff_l1 = df_out["level_diff"].shift(1)
    else:
        level_diff_l1 = None

    rn60_mean = df_out["RN_60m_mean"] if "RN_60m_mean" in cols else None
    rn15_mean = df_out["RN_15m_mean"] if "RN_15m_mean" in cols else None

    if rn60_mean is not None and level_sum_l1 is not None:
        new_cols["rain_x_levelsum_lag1"] = rn60_mean.shift(1) * level_sum_l1

    if rn15_mean is not None and level_sum_l1 is not None:
        new_cols["rain15_x_levelsum_lag1"] = rn15_mean.shift(1) * level_sum_l1

    if rn60_mean is not None and level_diff_l1 is not None:
        new_cols["rain_x_leveldiff_lag1"] = rn60_mean.shift(1) * level_diff_l1

    if "wet_flag" in cols and level_sum_l1 is not None:
        new_cols["wet_x_levelsum"] = df_out["wet_flag"].shift(1) * level_sum_l1

    # ====== 4) 지점별 강우 × 공정 ======
    for sid in station_ids:
        rn15 = f"RN_15m_{sid}"
        rn60 = f"RN_60m_{sid}"

        if rn15 in cols:
            s15 = df_out[rn15]
            if FLUX is not None:
                new_cols[f"RN15m_x_FLUX_{sid}"] = s15 * FLUX
            if SS is not None:
                new_cols[f"RN15_x_SS_{sid}"] = s15 * SS
            if TOC is not None:
                new_cols[f"RN15_x_TOC_{sid}"] = s15 * TOC

        if rn60 in cols:
            s60 = df_out[rn60]
            if FLUX is not None:
                new_cols[f"RN60m_x_FLUX_{sid}"] = s60 * FLUX
            if SS is not None:
                new_cols[f"RN60_x_SS_{sid}"] = s60 * SS
            if TOC is not None:
                new_cols[f"RN60_x_TOC_{sid}"] = s60 * TOC

    # ====== 5) 공간 평균 강우 × 공정 ======
    if rn15_mean is not None:
        if SS is not None:
            new_cols["RN15_mean_x_SS"] = rn15_mean * SS
        if TOC is not None:
            new_cols["RN15_mean_x_TOC"] = rn15_mean * TOC
        if TN is not None:
            new_cols["RN15_mean_x_TN"] = rn15_mean * TN
        if TP is not None:
            new_cols["RN15_mean_x_TP"] = rn15_mean * TP

    if rn60_mean is not None:
        if SS is not None:
            new_cols["RN60_mean_x_SS"] = rn60_mean * SS
        if TOC is not None:
            new_cols["RN60_mean_x_TOC"] = rn60_mean * TOC

    # ====== 6) 기상 × 공정 ======
    for sid in station_ids:
        ta = f"TA_{sid}"
        hm = f"HM_{sid}"
        td = f"TD_{sid}"

        if ta in cols:
            sTA = df_out[ta]
            if TN is not None:
                new_cols[f"TA_x_TN_{sid}"] = sTA * TN
            if TOC is not None:
                new_cols[f"TA_x_TOC_{sid}"] = sTA * TOC
            if FLUX is not None:
                new_cols[f"TA_x_FLUX_{sid}"] = sTA * FLUX

        if hm in cols and SS is not None:
            new_cols[f"HM_x_SS_{sid}"] = df_out[hm] * SS

        # (TA-TD) x TN : TA_minus_TD_{sid}가 이미 만들어져 있을 때만
        if (f"TA_{sid}" in cols) and (f"TD_{sid}" in cols) and (TN is not None):
            tadtd = df_out[f"TA_{sid}"] - df_out[f"TD_{sid}"]
            new_cols[f"(TA-TD)_x_TN_{sid}"] = tadtd * TN

    # ====== 7) 건조기간 × 강우 ======
    for sid in station_ids:
        dry_hr = f"dry_hours_{sid}"
        rn15 = f"RN_15m_{sid}"
        if dry_hr in cols and rn15 in cols:
            new_cols[f"dry_hours_x_RN15m_{sid}"] = df_out[dry_hr] * df_out[rn15]

    if "dry_hours_mean" in cols and rn15_mean is not None:
        new_cols["dry_hours_mean_x_RN15_mean"] = df_out["dry_hours_mean"] * rn15_mean

    # ====== 8) 누적강수 × 공정 ======
    for sid in station_ids:
        ar12 = f"AR_12h_{sid}"
        if ar12 in cols:
            sAR = df_out[ar12]
            if SS is not None:
                new_cols[f"AR12h_x_SS_{sid}"] = sAR * SS
            if TOC is not None:
                new_cols[f"AR12h_x_TOC_{sid}"] = sAR * TOC

    if "AR_12h_mean_all" in cols:
        ar_mean = df_out["AR_12h_mean_all"]
        if SS is not None:
            new_cols["AR12h_mean_x_SS"] = ar_mean * SS
        if TOC is not None:
            new_cols["AR12h_mean_x_TOC"] = ar_mean * TOC

    # ====== 9) 공정 지표 간 상호작용 ======
    if SS is not None and FLUX is not None:
        new_cols["SS_x_FLUX"] = SS * FLUX
    if TOC is not None and FLUX is not None:
        new_cols["TOC_x_FLUX"] = TOC * FLUX
    if TN is not None and TP is not None:
        new_cols["TN_x_TP"] = TN * TP

    # ====== 10) ARI × 공정 ======
    if "ARI_tau12" in cols:
        ari12 = df_out["ARI_tau12"]
        if SS is not None:
            new_cols["ARI_tau12_x_SS"] = ari12 * SS
        if TOC is not None:
            new_cols["ARI_tau12_x_TOC"] = ari12 * TOC

    # ====== 마지막: 한 번에 합치기 ======
    if new_cols:
        df_out = pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)

    return df_out


def add_station_agg_rain_features(df):
    """
    지점 통합 강수 특성
    - 공간 통계 (평균, 최대, 표준편차)
    - 지점 간 변동성

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with station-wise rainfall data

    Returns
    -------
    pd.DataFrame
        DataFrame with added aggregated rain features
    """
    station_ids = ["368", "541", "569"]
    rain_cols = [c for c in df.columns if c.startswith("RN_")]
    df_out = df.copy()

    new_cols = {}
    eps = 1e-6

    # 1) RN_* 지점 통합
    for rc in rain_cols:
        cols = [f"{rc}_{sid}" for sid in station_ids if f"{rc}_{sid}" in df_out.columns]
        if not cols:
            continue

        arr = df_out[cols]  # 한 번만 뽑아 재사용

        areal_mean = arr.mean(axis=1)
        areal_max  = arr.max(axis=1)
        areal_std  = arr.std(axis=1, ddof=0)

        new_cols[f"{rc}_areal_mean"] = areal_mean
        new_cols[f"{rc}_areal_max"]  = areal_max
        new_cols[f"{rc}_areal_max_minus_mean"] = areal_max - areal_mean
        new_cols[f"{rc}_areal_std"]  = areal_std

    # 2) AR_12h 통합 (원본 df_out 기준 + 혹시 new_cols에 있을 수도 있으니 tmp로 안전 처리)
    df_tmp = df_out
    if new_cols:
        df_tmp = pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)

    ar12_cols = [f"AR_12h_{sid}" for sid in station_ids if f"AR_12h_{sid}" in df_tmp.columns]
    if ar12_cols:
        new_cols["AR_12h_sum_all"] = df_tmp[ar12_cols].sum(axis=1)
        new_cols["AR_12h_mean_all"] = df_tmp[ar12_cols].mean(axis=1)

    # 3) 비율 특성
    rn15_mean = f"RN_15m_areal_mean"
    rn60_mean = f"RN_60m_areal_mean"
    if rn15_mean in df_tmp.columns and rn60_mean in df_tmp.columns:
        new_cols["RN_15m_div_RN_60m_areal_mean"] = df_tmp[rn15_mean] / (df_tmp[rn60_mean] + eps)

    # 4) 마지막에 한 번에 합치기
    if new_cols:
        df_out = pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)

    return df_out


def add_weather_features(df):
    """
    기상 특성 추가
    - Dewpoint depression (TA - TD)
    - Vapor Pressure Deficit (VPD)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with weather data

    Returns
    -------
    pd.DataFrame
        DataFrame with added weather features
    """
    df_out = df.copy()
    station_ids = ["368", "541", "569"]

    new_cols = {}

    for sid in station_ids:
        ta = f"TA_{sid}"
        td = f"TD_{sid}"
        hm = f"HM_{sid}"

        # Dewpoint depression
        if ta in df_out.columns and td in df_out.columns:
            new_cols[f"TA_minus_TD_{sid}"] = df_out[ta] - df_out[td]

        # VPD (kPa)
        if ta in df_out.columns and hm in df_out.columns:
            T = df_out[ta].astype("float64")
            RH = df_out[hm].astype("float64").clip(0.0, 100.0)

            e_s = 0.6108 * np.exp((17.27 * T) / (T + 237.3))  # saturation vapor pressure
            new_cols[f"VPD_{sid}"] = e_s * (1.0 - RH / 100.0)

    if new_cols:
        df_out = pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)

    return df_out


def add_target_lag_features(df, target_cols, min_lag=1):
    """
    타겟 변수의 시차(lag) 특성 추가
    - Lag features (shift)
    - Rolling 통계 (mean, std, max, min)
    - Difference / 변화율
    - EWMA (지수가중이동평균)

    타겟 분리 전에 호출하여, lag 특성은 base에 남기고 raw 타겟만 y로 분리.
    shift(min_lag) 적용으로 미래 정보 누수 방지.

    주기: 30분 리샘플링 기준 (1시간 = 2 steps)

    Parameters
    ----------
    df : pd.DataFrame
        타겟 컬럼을 포함한 DataFrame (datetime index)
    target_cols : list of str
        타겟 컬럼명 리스트
    min_lag : int, default=1
        최소 시차 (미래 정보 누수 방지, 최소 1 이상 권장)

    Returns
    -------
    pd.DataFrame
        타겟 lag 특성이 추가된 DataFrame
    """
    df_out = df.copy()
    new_cols = {}

    # 30분 리샘플링 기준
    lags = [2, 4, 6, 12, 24, 48, 72]   # 1h, 2h, 3h, 6h, 12h, 24h, 36h
    roll_windows = [6, 12, 24, 72]         # 3h, 6h, 12h, 36h
    ewma_spans = [6, 12, 24]               # 3h, 6h, 12h

    for col in target_cols:
        if col not in df_out.columns:
            continue

        series = df_out[col]

        # ====== 1) Lag features ======
        for lag in lags:
            actual_lag = max(lag, min_lag)
            new_cols[f"{col}_tlag_{actual_lag}"] = series.shift(actual_lag)

        # ====== 2) Rolling 통계 (shift(min_lag)로 미래 정보 방지) ======
        shifted = series.shift(min_lag)
        for w in roll_windows:
            minp = max(1, w // 2)
            r = shifted.rolling(w, min_periods=minp)
            new_cols[f"{col}_troll_mean_{w}"] = r.mean()
            new_cols[f"{col}_troll_std_{w}"] = r.std(ddof=0)
            new_cols[f"{col}_troll_max_{w}"] = r.max()
            new_cols[f"{col}_troll_min_{w}"] = r.min()

        # ====== 3) Difference features ======
        new_cols[f"{col}_tdiff_1"] = series.diff(1)
        new_cols[f"{col}_tdiff_2"] = series.diff(2)
        new_cols[f"{col}_tdiff_6"] = series.diff(6)

        # ====== 4) 변화율 (percent change) ======
        eps = 1e-8
        new_cols[f"{col}_tpct_1"] = series.pct_change(1).clip(-10, 10)
        new_cols[f"{col}_tpct_6"] = series.pct_change(6).clip(-10, 10)

        # ====== 5) EWMA (지수가중이동평균) ======
        for span in ewma_spans:
            new_cols[f"{col}_tewma_{span}"] = shifted.ewm(span=span, adjust=False).mean()

    if new_cols:
        df_out = pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)

    return df_out


def add_time_features(df):
    """
    시간 기반 특성 추가
    - 시간/요일/월 정보
    - 주말 플래그
    - 주기적 인코딩 (sin/cos)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime index

    Returns
    -------
    pd.DataFrame
        DataFrame with added time features
    """
    df_out = df.copy()
    idx = df_out.index

    # DatetimeIndex 전제(아니면 여기서 에러 나게 두는 게 디버깅에 유리)
    hour = idx.hour
    dow = idx.dayofweek
    month = idx.month
    doy = idx.dayofyear

    new_cols = {
        "hour": hour,
        "dayofweek": dow,
        "month": month,
        "is_weekend": np.isin(dow, [5, 6]).astype(np.int8),
        "hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
        "doy_sin": np.sin(2.0 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2.0 * np.pi * doy / 365.25),
    }

    df_out = pd.concat([df_out, pd.DataFrame(new_cols, index=idx)], axis=1)
    return df_out


# ==============================================================================
# Helper Function: Apply All Feature Engineering
# ==============================================================================

def apply_all_features(df, mode=None, exclude_unsafe=True):
    """
    모든 특성 엔지니어링 함수를 순차적으로 적용

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    mode : str, optional
        Prediction mode for data leakage prevention
        (flow, toc, ss, tn, tp, flux, ph)
    exclude_unsafe : bool, default=True
        If True and mode is specified, removes unsafe process features

    Returns
    -------
    pd.DataFrame
        DataFrame with all features added
    """
    result = df.copy()

    # 데이터 누수 방지: 안전하지 않은 프로세스 변수 제거
    if exclude_unsafe and mode and mode in DATA_LEAKAGE_CONFIG:
        config = DATA_LEAKAGE_CONFIG[mode]
        target_vars = set(config["target"])
        safe_vars = set(config["safe_process_features"])

        all_process_vars = ["TOC_VU", "SS_VU", "TN_VU", "TP_VU", "FLUX_VU", "PH_VU"]
        unsafe_vars = [
            v for v in all_process_vars
            if v not in target_vars and v not in safe_vars and v in result.columns
        ]

        if unsafe_vars:
            print(f"⚠️ 데이터 누수 방지 ({mode}): {unsafe_vars} 제외")
            result = result.drop(columns=unsafe_vars)

    # 특성 엔지니어링 적용
    result = add_rain_features(result)
    result = add_station_agg_rain_features(result)
    result = add_weather_features(result)
    result = add_process_features(result)
    result = add_temporal_features(result)
    # result = add_interaction_features(result)  # 필요시 주석 해제
    result = add_time_features(result)

    return result
