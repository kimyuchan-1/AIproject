"""
FLOW 하이퍼파라미터 그리드 탐색 실험
====================================
타겟: Q_in (유입유량, 30분 평균)
DATA_LEAKAGE_CONFIG: safe_process_features = [level_TankA, level_TankB]
(TMS 지표 전체 미사용 - 유입 단계 예측이므로 출구 지표는 누수)

실험 전략:
  - Phase 1: 핵심 파라미터 탐색 (hidden 256~512, layers 1~2, lr 5e-4~2e-3) - 18 조합
  - Phase 2: 최적 핵심 파라미터 기반 보조 파라미터 미세 조정 - 18 조합
  - window_size=48 고정 (24시간)
  - LC_SPLIT_RATIOS (80/10/10) 사용 (유량 데이터 기간이 짧아 훈련 데이터 최대화)
  - 특성 선택: stability_ratio=0.3
  - 누적 인사이트: layers=1이 대체로 우수, attention 제외, batch=2048 선호
"""
import sys
import os
import time
import json
import itertools
from pathlib import Path
from datetime import datetime

# ====== 프로젝트 경로 설정 ======
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from scipy.stats import zscore
from tqdm import tqdm
import pickle

import notebook.feature.WF_feature_selection as wf_fs
import notebook.feature.feature_engineering as feat_eng

# ====== 글로벌 설정 ======
MODE = "flow"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIME_COL = "SYS_TIME"
RANDOM_SEED = 42
CURRENT_BEST_R2 = 0.8166  # 현재 최고 (노트북 탐색 결과)

# 경로 설정
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model"
RESULTS_DIR = PROJECT_DIR / "results" / "DL"
FEATURE_DIR = DATA_DIR / "features"

# 타겟 정의
FLOW_TARGET = "Q_in"
TOC_TARGET = "TOC_VU"
SS_TARGET = "SS_VU"
TN_TARGET = "TN_VU"
TP_TARGET = "TP_VU"
FLUX_TARGET = "FLUX_VU"
PH_TARGET = "PH_VU"

SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
LC_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

# ====== Phase 1: 핵심 파라미터 탐색 ======
# 누적 인사이트:
#   - TP/TOC/PH: layers=1이 압도적, FLUX/SS: layers=2가 약간 우수
#   - lr=2e-4는 너무 느려 제외, 5e-4~2e-3 탐색
#   - hidden=128 성능 한계 -> 256~512 탐색
# 총 18 조합
PHASE1_GRID = {
    "hidden_size": [256, 384, 512],
    "num_layers": [1, 2],
    "learning_rate": [5e-4, 1e-3, 2e-3],
    # 고정
    "dropout": [0.2],
    "batch_size": [2048],
    "window_size": [48],
    "use_attention": [False],
    "weight_decay": [0],
    "split_type": ["lc"],  # FLOW도 LC_SPLIT_RATIOS (80/10/10) - 데이터 기간이 짧아 훈련 최대화
    "shuffle": [True],
}


# ======================================================================
# 데이터 로드 및 전처리 함수들
# ======================================================================

def load_data(data_dir):
    dfs = {}
    dfs['flow'] = pd.read_csv(data_dir / "actual/FLOW_Actual.csv")
    dfs['flow']['Q_in'] = dfs['flow']["flow_TankA"] + dfs['flow']['flow_TankB']
    dfs['flow']['level_sum'] = dfs['flow']['level_TankA'] + dfs['flow']['level_TankB']
    dfs['flow'] = dfs['flow'].drop(columns=["data_save_dt"])
    dfs['tms'] = pd.read_csv(data_dir / "actual/TMS_Actual.csv")
    for station_id in ["368", "541", "569"]:
        aws_path = data_dir / f"actual/AWS_{station_id}.csv"
        df = pd.read_csv(aws_path)
        if "datetime" in df.columns:
            time_col = df["datetime"]
            df = df.drop(columns=["datetime", "YYMMDDHHMI", "STN"], errors="ignore")
            df = df.add_suffix(f"_{station_id}")
            df["SYS_TIME"] = time_col
        else:
            df = df.drop(columns=["YYMMDDHHMI", "STN"], errors="ignore")
            df = df.add_suffix(f"_{station_id}")
        dfs[f"aws{station_id}"] = df
    return dfs


def set_datetime_index(df, time_col):
    out = df.copy()
    if time_col not in out.columns:
        raise ValueError(f"시간 컬럼 '{time_col}'이 데이터프레임에 없습니다.")
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col])
    out = out.set_index(time_col).sort_index()
    return out


def align_data(dfs):
    aligned_dfs = {}
    for name, df in dfs.items():
        df_aligned = set_datetime_index(df, TIME_COL)
        df_aligned = df_aligned.resample("1min").ffill()
        aligned_dfs[name] = df_aligned
    return aligned_dfs


def merge_data(dfs):
    valid = {}
    merged_dfs = {}
    for name, df in dfs.items():
        df2 = df.sort_index()
        if df2.index.has_duplicates:
            df2 = df2[~df2.index.duplicated(keep="last")]
        valid[name] = df2
    for name, df in valid.items():
        if name == "flow":
            merged_dfs[name] = pd.concat([df, valid["aws368"], valid["aws541"], valid["aws569"]], axis=1, join="inner")
        if name == "tms":
            merged_dfs[name] = pd.concat([df, valid["aws368"], valid["aws541"], valid["aws569"]], axis=1, join="inner")
    return merged_dfs


@dataclass
class ImputationConfig:
    short_term_hours: int = 3
    medium_term_hours: int = 12
    long_term_hours: int = 48
    ewma_span: int = 6


@dataclass
class OutlierConfig:
    method: str = "iqr"
    iqr_threshold: float = 1.5
    zscore_threshold: float = 3.0
    require_both: bool = True


def impute_missing(df, freq="1h", config=ImputationConfig()):
    df_out = df.copy()
    freq_td = pd.Timedelta(freq)
    freq_hours = freq_td.total_seconds() / 3600
    mask_dict = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].copy()
        original_missing = series.isna()
        mask_dict[f"{col}_is_missing"] = original_missing.astype(int)
        limit_short = max(1, int(config.short_term_hours / freq_hours))
        series_ffill = series.ffill(limit=limit_short)
        ffill_mask = original_missing & ~series_ffill.isna()
        mask_dict[f"{col}_imputed_ffill"] = ffill_mask.astype(int)
        still_missing = series_ffill.isna()
        if still_missing.sum() > 0:
            ewma_span = max(1, int(config.ewma_span / freq_hours))
            series_ewma = series_ffill.ewm(span=ewma_span, adjust=False).mean()
            limit_medium = max(1, int(config.medium_term_hours / freq_hours))
            missing_groups = (still_missing != still_missing.shift()).cumsum()
            missing_lengths = still_missing.groupby(missing_groups).transform("sum")
            medium_mask = still_missing & (missing_lengths > limit_short) & (missing_lengths <= limit_medium)
            series_ffill[medium_mask] = series_ewma[medium_mask]
            mask_dict[f"{col}_imputed_ewma"] = medium_mask.astype(int)
        else:
            mask_dict[f"{col}_imputed_ewma"] = pd.Series(0, index=df.index, dtype=int)
        still_missing_long = series_ffill.isna()
        if still_missing_long.sum() > 0:
            long_ewma_span = max(1, int(config.ewma_span * 4 / freq_hours))
            series_long_ewma = series_ffill.ewm(span=long_ewma_span, adjust=False).mean()
            long_mask = still_missing_long
            series_ffill[long_mask] = series_long_ewma[long_mask]
            mask_dict[f"{col}_imputed_long_ewma"] = long_mask.astype(int)
        else:
            mask_dict[f"{col}_imputed_long_ewma"] = pd.Series(0, index=df.index, dtype=int)
        df_out[col] = series_ffill
    df_mask = pd.DataFrame(mask_dict, index=df.index)
    return df_out, df_mask


def imputate_data(dfs):
    config_impute = ImputationConfig()
    imputed_dfs = {}
    mask_imputed_dfs = {}
    for name, df in dfs.items():
        df_imputed, mask_imputed = impute_missing(df, freq="1min", config=config_impute)
        imputed_dfs[name] = df_imputed
        mask_imputed_dfs[name] = mask_imputed
    return imputed_dfs, mask_imputed_dfs


def outliers_domain(series, col_name):
    outliers = pd.Series([False] * len(series), index=series.index)
    if not pd.api.types.is_numeric_dtype(series):
        return outliers
    regulatory_rules = {
        "TOC_VU": (0, 30), "SS_VU": (0, 20), "PH_VU": (4.3, 10.0),
        "TN_VU": (0, 20), "TP_VU": (0, 1.0), "FLUX_VU": (0, 12.0),
    }
    if col_name in regulatory_rules:
        lower, upper = regulatory_rules[col_name]
        outliers = (series < lower) | (series > upper)
        return outliers
    physical_rules = {
        "level_TankA": (0, 10), "level_TankB": (0, 10),
        "TA": (-30, 45), "HM": (0, 100), "TD": (-40, 35),
    }
    if col_name in physical_rules:
        lower, upper = physical_rules[col_name]
        outliers = (series < lower) | (series > upper)
    elif "RN_" in col_name:
        outliers = (series < 0) | (series > 300)
    elif "flow" in col_name.lower() or "flux" in col_name.lower():
        valid_values = series.dropna()
        if len(valid_values) > 0:
            outliers = (series < 0) | (series > valid_values.quantile(0.99) * 3)
    else:
        valid_values = series.dropna()
        if len(valid_values) > 0:
            outliers = (series < 0) | (series > valid_values.quantile(0.999) * 2)
    return outliers


def outliers_statistical(series, method='iqr', iqr_threshold=1.5, zscore_threshold=3.0):
    outliers = pd.Series([False] * len(series), index=series.index)
    if not pd.api.types.is_numeric_dtype(series):
        return outliers
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_threshold * IQR
        upper = Q3 + iqr_threshold * IQR
        outliers = (series < lower) | (series > upper)
    elif method == 'zscore':
        valid_mask = ~series.isna()
        if valid_mask.sum() > 0:
            z_scores = np.abs(zscore(series[valid_mask]))
            outliers[valid_mask] = z_scores > zscore_threshold
    return outliers


def process_outliers(df, config=OutlierConfig(), ewma_span=12):
    df_out = df.copy()
    mask_dict = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].copy()
        domain_outliers = outliers_domain(series, col)
        stats_outliers = outliers_statistical(series, method=config.method, iqr_threshold=config.iqr_threshold, zscore_threshold=config.zscore_threshold)
        if config.require_both:
            final_outliers = domain_outliers & stats_outliers
        else:
            final_outliers = domain_outliers | stats_outliers
        mask_dict[f"{col}_outlier_domain"] = domain_outliers.astype(int)
        mask_dict[f"{col}_outlier_stats"] = stats_outliers.astype(int)
        mask_dict[f"{col}_outlier_final"] = final_outliers.astype(int)
        if final_outliers.sum() > 0:
            series_clean = series.copy()
            series_clean[final_outliers] = np.nan
            series_ewma = series_clean.ewm(span=ewma_span, adjust=False).mean()
            series[final_outliers] = series_ewma[final_outliers]
            mask_dict[f"{col}_outlier_replaced_ewma"] = final_outliers.astype(int)
        else:
            mask_dict[f"{col}_outlier_replaced_ewma"] = pd.Series(0, index=df.index, dtype=int)
        df_out[col] = series
    df_mask = pd.DataFrame(mask_dict, index=df.index)
    return df_out, df_mask


def handle_outliers(dfs):
    config_outlier = OutlierConfig()
    processed_dfs = {}
    mask_processed_dfs = {}
    for name, df in dfs.items():
        df_outlier, mask_outlier = process_outliers(df, config=config_outlier, ewma_span=12)
        processed_dfs[name] = df_outlier
        mask_processed_dfs[name] = mask_outlier
    return processed_dfs, mask_processed_dfs


def resample_data(df, freq="30min"):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("리샘플링을 위해서는 DatetimeIndex가 필요합니다")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    agg_dict = {}
    for col in numeric_cols:
        if col.startswith("RN_") or col.startswith("AR_") or col == "FLUX_VU":
            agg_dict[col] = "sum"
        else:
            agg_dict[col] = "mean"
    return df_numeric.resample(freq).agg(agg_dict)


def preprocess_data(dfs):
    """전처리 + 특성 선택 (WF) -> X, y 반환"""
    aligned_dfs = align_data(dfs)

    # FLUX_VU 차분 처리 (tms 데이터에만 적용, flow 모드에서는 source로 사용하지 않음)
    if 'tms' in aligned_dfs and 'FLUX_VU' in aligned_dfs['tms'].columns:
        flux = aligned_dfs['tms']['FLUX_VU'].copy()
        flux_diff = flux.diff()
        reset_mask = flux_diff < 0
        flux_diff[reset_mask] = flux[reset_mask]
        flux_diff.iloc[0] = 0
        flux_diff = flux_diff.clip(lower=0)
        aligned_dfs['tms']['FLUX_VU'] = flux_diff

    merged_dfs = merge_data(aligned_dfs)
    imputed_dfs, _ = imputate_data(merged_dfs)
    processed_dfs, _ = handle_outliers(imputed_dfs)

    resample_dfs = {}
    for name, df in processed_dfs.items():
        resample_dfs[name] = resample_data(df, freq="30min")

    mode = MODE
    MODE_CFG = {
        "toc": ("tms", [TOC_TARGET]),
        "ss": ("tms", [SS_TARGET]),
        "tn": ("tms", [TN_TARGET]),
        "tp": ("tms", [TP_TARGET]),
        "flux": ("tms", [FLUX_TARGET]),
        "ph": ("tms", [PH_TARGET]),
        "flow": ("flow", [FLOW_TARGET]),
    }
    source_key, target_col = MODE_CFG[mode]

    for name, df0 in resample_dfs.items():
        df = df0.loc[:, ~df0.columns.duplicated()].copy()
        if name == "flow":
            df = df.drop(columns=["flow_TankA", "flow_TankB"], errors="ignore")

        tgt_in_df = [c for c in target_col if c in df.columns]
        if tgt_in_df:
            df = feat_eng.add_target_lag_features(df, tgt_in_df, min_lag=2)

        if tgt_in_df:
            y_part = df[tgt_in_df].copy()
            base = df.drop(columns=tgt_in_df)
        else:
            y_part = None
            base = df

        if mode in feat_eng.DATA_LEAKAGE_CONFIG:
            all_process_vars = ["TOC_VU", "SS_VU", "TN_VU", "TP_VU", "FLUX_VU", "PH_VU"]
            target_vars = set(feat_eng.DATA_LEAKAGE_CONFIG[mode]["target"])
            safe_vars = set(feat_eng.DATA_LEAKAGE_CONFIG[mode]["safe_process_features"])
            unsafe_vars = [v for v in all_process_vars if v not in target_vars and v not in safe_vars and v in base.columns]
            if unsafe_vars:
                base = base.drop(columns=unsafe_vars)

        base = feat_eng.add_rain_features(base)
        base = feat_eng.add_station_agg_rain_features(base)
        base = feat_eng.add_weather_features(base)
        base = feat_eng.add_process_features(base)
        base = feat_eng.add_temporal_features(base)
        base = feat_eng.add_time_features(base)

        if "weekday" not in base.columns and "dayofweek" in base.columns:
            base["weekday"] = base["dayofweek"]
        if "iso_week" not in base.columns and isinstance(base.index, pd.DatetimeIndex):
            base["iso_week"] = base.index.isocalendar().week.astype(int).to_numpy()
        if "hour_x_weekday" not in base.columns:
            if "hour" in base.columns and "weekday" in base.columns:
                base["hour_x_weekday"] = base["hour"] * base["weekday"]
            elif "hour" in base.columns and "dayofweek" in base.columns:
                base["hour_x_weekday"] = base["hour"] * base["dayofweek"]

        if y_part is not None:
            df_fe = base.join(y_part, how="left")
        else:
            df_fe = base
        resample_dfs[name] = df_fe.dropna()

    source_data = resample_dfs[source_key]

    X_all = source_data.drop(columns=target_col).values
    y = source_data[target_col]
    feature_names_all = source_data.drop(columns=target_col).columns.tolist()

    print(f"\n전체 특성 수: {len(feature_names_all)}, 샘플 수: {len(X_all)}")

    # Walk-Forward Feature Selection
    wf_selector = wf_fs.WalkForwardFeatureSelector(
        X=X_all, y=y, feature_names=feature_names_all,
        n_splits=5, train_size=700, val_size=200, test_size=200, window_step=200,
    )
    results = wf_selector.run(model_type="rf", verbose=True)

    # FLOW: stability_ratio=0.3
    sr = 0.3
    recommended_idx = wf_selector.get_recommended_features(stability_ratio=sr)
    print(f"\nFLOW 모드: stability_ratio={sr}")

    MIN_FEATURES = 10
    if len(recommended_idx) < MIN_FEATURES:
        importance_mean = np.mean(wf_selector.feature_importance_folds, axis=0)
        top_idx = np.argsort(importance_mean)[::-1][:MIN_FEATURES]
        recommended_idx = np.sort(np.unique(np.concatenate([recommended_idx, top_idx])))
        print(f"  -> 특성 수 부족 -> 중요도 상위 {MIN_FEATURES}개 보충 -> 총 {len(recommended_idx)}개")

    recommended_names = [feature_names_all[i] for i in recommended_idx]
    X = X_all[:, recommended_idx]

    print(f"선택된 특성: {len(recommended_idx)}/{len(feature_names_all)}")
    for i, name in enumerate(recommended_names, 1):
        print(f"  {i}. {name}")

    # 저장
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": recommended_idx, "feature_name": recommended_names}).to_csv(
        FEATURE_DIR / f"{MODE}_recommended_features.csv", index=False
    )

    return X, y, recommended_idx, feature_names_all


# ======================================================================
# 스케일러, 데이터셋, 모델
# ======================================================================

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    def fit(self, x):
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True) + 1e-8
        return self
    def transform(self, x):
        return (x - self.mean_) / self.std_
    def inverse_transform(self, x):
        return x * self.std_ + self.mean_


class TimeSeriesWindowDataset(Dataset):
    def __init__(self, X, y, window_size, horizon):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.window_size = window_size
        self.horizon = horizon
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(1)
        self.max_start = len(self.X) - self.window_size - self.horizon + 1
        if self.max_start <= 0:
            raise ValueError("데이터 길이가 window_size + horizon 보다 짧습니다.")
    def __len__(self):
        return self.max_start
    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.window_size]
        y_t = self.y[idx + self.window_size + self.horizon - 1]
        return x_seq, y_t.reshape(-1)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2, out_size=1, use_attention=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_size = hidden_size

        if self.use_attention:
            self.layer_norm1 = nn.LayerNorm(lstm_out_size)
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_out_size, num_heads=4, dropout=dropout, batch_first=True
            )
            self.layer_norm2 = nn.LayerNorm(lstm_out_size)

        # Prediction Head (hidden_size에 따라 자동 결정)
        if hidden_size >= 256:
            self.head = nn.Sequential(
                nn.Linear(lstm_out_size, lstm_out_size // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(lstm_out_size // 2, lstm_out_size // 4), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(lstm_out_size // 4, lstm_out_size // 8), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(lstm_out_size // 8, out_size),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(lstm_out_size, lstm_out_size // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(lstm_out_size // 2, out_size),
            )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if self.use_attention:
            lstm_out_normed = self.layer_norm1(lstm_out)
            attn_out, _ = self.attention(lstm_out_normed, lstm_out_normed, lstm_out_normed)
            attn_out = attn_out + lstm_out
            attn_out = self.layer_norm2(attn_out)
            last = attn_out[:, -1, :]
        else:
            last = lstm_out[:, -1, :]
        return self.head(last)


# ======================================================================
# 유틸리티 함수
# ======================================================================

def report_and_fix(name, arr):
    if hasattr(arr, "values"):
        arr = arr.values
    t = torch.as_tensor(arr, dtype=torch.float32)
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    if n_nan > 0 or n_inf > 0:
        if t.ndim == 1:
            col_mean = torch.nanmean(t)
            if torch.isnan(col_mean):
                col_mean = torch.tensor(0.0)
            t = torch.where(torch.isfinite(t), t, col_mean.expand_as(t))
        else:
            col_mean = torch.nanmean(t, dim=0)
            col_mean = torch.where(torch.isnan(col_mean), torch.zeros_like(col_mean), col_mean)
            t = torch.where(torch.isfinite(t), t, col_mean.unsqueeze(0).expand_as(t))
    return t.numpy()


def split_timewise(X, y, ratio):
    T = len(X)
    n_train = int(T * ratio["train"])
    n_val = int(T * ratio["val"])
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train: n_train + n_val], y[n_train: n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def ensure_2d_y(y):
    y = torch.as_tensor(y, dtype=torch.float32)
    if y.ndim == 1:
        y = y.unsqueeze(1)
    return y


# ======================================================================
# 학습 & 평가
# ======================================================================

def train_and_evaluate(X, y, config, verbose=False):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    # 분할 - FLOW는 LC_SPLIT_RATIOS 사용 (데이터 기간이 짧아 훈련 데이터 최대화)
    ratio = LC_SPLIT_RATIOS if config["split_type"] == "lc" else SPLIT_RATIOS
    X_train, y_train, X_val, y_val, X_test, y_test = split_timewise(X, y, ratio)

    X_train = report_and_fix("X_train", X_train)
    X_val = report_and_fix("X_val", X_val)
    X_test = report_and_fix("X_test", X_test)
    y_train = report_and_fix("y_train", y_train)
    y_val = report_and_fix("y_val", y_val)
    y_test = report_and_fix("y_test", y_test)

    # 스케일링
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    X_tr_s = x_scaler.transform(X_train)
    X_va_s = x_scaler.transform(X_val)
    X_te_s = x_scaler.transform(X_test)
    y_tr_s = y_scaler.transform(y_train)
    y_va_s = y_scaler.transform(y_val)
    y_te_s = y_scaler.transform(y_test)

    y_tr_s = ensure_2d_y(y_tr_s)
    y_va_s = ensure_2d_y(y_va_s)
    y_te_s = ensure_2d_y(y_te_s)

    window_size = config["window_size"]
    horizon = 1

    train_ds = TimeSeriesWindowDataset(X_tr_s, y_tr_s, window_size, horizon)
    val_ds = TimeSeriesWindowDataset(X_va_s, y_va_s, window_size, horizon)
    test_ds = TimeSeriesWindowDataset(X_te_s, y_te_s, window_size, horizon)

    batch_size = config["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=config["shuffle"], drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    n_features = X_tr_s.shape[1]
    out_size = y_tr_s.shape[1]

    model = LSTMRegressor(
        n_features=n_features,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        out_size=out_size,
        use_attention=config["use_attention"],
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 학습
    num_epochs = 200
    patience = 20
    MIN_EPOCHS = 30
    best_val_rmse = float("inf")
    best_model_state = None
    patience_counter = 0
    prev_val_loss = None
    val_loss_increasing_streak = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        train_total = 0

        for X_batch, y_batch in train_dl:
            X_batch = X_batch.to(DEVICE).float()
            y_batch = y_batch.to(DEVICE).float()

            optimizer.zero_grad()
            preds = model(X_batch)
            if preds.dim() > 2:
                preds = preds.squeeze(1)
            if y_batch.dim() > 2:
                y_batch = y_batch.squeeze(1)

            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = y_batch.size(0)
            train_total += bs
            epoch_loss += loss.item() * bs
            with torch.no_grad():
                epoch_mse += ((preds - y_batch) ** 2).sum().item()

        avg_loss = epoch_loss / train_total
        train_rmse = (epoch_mse / train_total) ** 0.5

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_mse_sum = 0.0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch = X_batch.to(DEVICE).float()
                y_batch = y_batch.to(DEVICE).float()
                preds = model(X_batch)
                if preds.dim() > 2:
                    preds = preds.squeeze(1)
                if y_batch.dim() > 2:
                    y_batch = y_batch.squeeze(1)
                loss = criterion(preds, y_batch)
                bs = y_batch.size(0)
                val_total += bs
                val_loss_sum += loss.item() * bs
                val_mse_sum += ((preds - y_batch) ** 2).sum().item()

        v_loss = val_loss_sum / val_total
        v_rmse = (val_mse_sum / val_total) ** 0.5

        scheduler.step(v_rmse)

        # Early stopping logic
        if epoch >= MIN_EPOCHS:
            if prev_val_loss is not None:
                if v_loss < prev_val_loss:
                    val_loss_increasing_streak = 0
                else:
                    val_loss_increasing_streak += 1
            if val_loss_increasing_streak >= 5 or v_loss > avg_loss * 2.0:
                if verbose:
                    print(f"  Early stop at epoch {epoch+1}")
                break
        prev_val_loss = v_loss

        if v_rmse < best_val_rmse:
            best_val_rmse = v_rmse
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if epoch >= MIN_EPOCHS and patience_counter >= patience:
                if verbose:
                    print(f"  Patience stop at epoch {epoch+1}")
                break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train_rmse={train_rmse:.4f}, val_rmse={v_rmse:.4f}")

    elapsed = time.time() - start_time
    epochs_trained = epoch + 1

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Test evaluation
    model.eval()
    preds_all = []
    y_all = []
    with torch.no_grad():
        for X_batch, y_batch in test_dl:
            X_batch = X_batch.to(DEVICE).float()
            y_batch = y_batch.to(DEVICE).float()
            preds = model(X_batch)
            if preds.dim() > 2:
                preds = preds.squeeze(1)
            if y_batch.dim() > 2:
                y_batch = y_batch.squeeze(1)
            preds_all.append(preds.cpu())
            y_all.append(y_batch.cpu())

    y_cat = torch.cat(y_all)
    p_cat = torch.cat(preds_all)

    if y_cat.ndim > 1:
        r2_list = []
        for i in range(y_cat.shape[1]):
            ss_res = ((y_cat[:, i] - p_cat[:, i]) ** 2).sum()
            ss_tot = ((y_cat[:, i] - y_cat[:, i].mean()) ** 2).sum()
            r2_i = (1.0 - ss_res / ss_tot).item() if ss_tot.item() > 0 else None
            r2_list.append(r2_i)
        r2 = r2_list[0] if len(r2_list) == 1 else r2_list
    else:
        ss_res = ((y_cat - p_cat) ** 2).sum()
        ss_tot = ((y_cat - y_cat.mean()) ** 2).sum()
        r2 = (1.0 - ss_res / ss_tot).item() if ss_tot.item() > 0 else None

    test_rmse = ((y_cat - p_cat) ** 2).mean().sqrt().item()
    test_mae = (y_cat - p_cat).abs().mean().item()

    return {
        "r2": r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "best_val_rmse": best_val_rmse,
        "epochs_trained": epochs_trained,
        "time_seconds": elapsed,
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }


# ======================================================================
# 메인 실험 루프
# ======================================================================

RESULTS_FILE = "flow_experiment_results.csv"

def run_experiments(X, y, grid, top_n=5):
    """그리드의 모든 조합을 실행하고 결과를 정리"""
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    combinations = list(itertools.product(*values))

    total = len(combinations)
    print(f"\n{'='*70}")
    print(f"FLOW 하이퍼파라미터 실험 시작")
    print(f"{'='*70}")
    print(f"총 실험 수: {total}")
    if CURRENT_BEST_R2 is not None:
        print(f"현재 최고 R2: {CURRENT_BEST_R2}")
    else:
        print(f"현재 최고 R2: 미탐색 (첫 실험)")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*70}\n")

    results = []
    best_r2 = -float("inf")

    for i, combo in enumerate(combinations, 1):
        config = dict(zip(keys, combo))

        print(f"\n[{i}/{total}] hidden={config['hidden_size']}, layers={config['num_layers']}, "
              f"lr={config['learning_rate']}, dropout={config['dropout']}, "
              f"batch={config['batch_size']}, wd={config['weight_decay']}")

        try:
            result = train_and_evaluate(X, y, config, verbose=False)
            r2 = result["r2"]

            record = {
                **config,
                "r2": r2,
                "test_rmse": result["test_rmse"],
                "test_mae": result["test_mae"],
                "best_val_rmse": result["best_val_rmse"],
                "epochs": result["epochs_trained"],
                "time_sec": round(result["time_seconds"], 1),
            }
            results.append(record)

            marker = ""
            if r2 is not None and r2 > best_r2:
                best_r2 = r2
                marker = " * NEW BEST!"

            print(f"  -> R2={r2:.4f}, RMSE={result['test_rmse']:.4f}, "
                  f"epochs={result['epochs_trained']}, time={result['time_seconds']:.0f}s{marker}")

            # 중간 결과 저장 (매 실험마다)
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values("r2", ascending=False)
            save_path = RESULTS_DIR / RESULTS_FILE
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            df_results.to_csv(save_path, index=False)

            # GPU 메모리 정리
            del result["model"]
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append({**config, "r2": None, "error": str(e)})

    # 최종 결과 정리
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("r2", ascending=False)

    print(f"\n{'='*70}")
    print(f"실험 완료!")
    print(f"{'='*70}")
    print(f"\nTop {top_n} 결과:")
    print("-" * 120)

    top = df_results.head(top_n)
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        print(f"  #{rank} R2={row.get('r2', 'N/A'):.4f} | "
              f"hidden={row['hidden_size']}, layers={row['num_layers']}, "
              f"lr={row['learning_rate']}, dropout={row['dropout']}, "
              f"batch={row['batch_size']}, wd={row['weight_decay']}, "
              f"epochs={row.get('epochs', 'N/A')}, time={row.get('time_sec', 'N/A')}s")

    if CURRENT_BEST_R2 is not None:
        print(f"\n기존 최고 R2: {CURRENT_BEST_R2}")
        if best_r2 > CURRENT_BEST_R2:
            print(f"* 새로운 최고 R2: {best_r2:.4f} (개선: +{(best_r2 - CURRENT_BEST_R2):.4f})")
        else:
            print(f"현재 실험 최고 R2: {best_r2:.4f}")
    else:
        print(f"\n이번 실험 최고 R2: {best_r2:.4f}")

    save_path = RESULTS_DIR / RESULTS_FILE
    df_results.to_csv(save_path, index=False)
    print(f"\n결과 저장: {save_path}")

    return df_results


def run_phase2(X, y, best_config, top_n=10):
    """Phase 1에서 찾은 최적 hidden/layers/lr 기반으로 보조 파라미터 미세 조정"""
    print(f"\n{'='*70}")
    print(f"Phase 2: 보조 파라미터 미세 조정")
    print(f"기반 config: hidden={best_config['hidden_size']}, "
          f"layers={best_config['num_layers']}, lr={best_config['learning_rate']}")
    print(f"{'='*70}")

    # window_size=48 고정 (사용자 요청)
    # attention=True는 이 데이터셋에서 일관적으로 성능 하락 -> False만 사용
    # 3 dropout x 2 batch x 3 wd = 18 조합
    phase2_grid = {
        "hidden_size": [best_config["hidden_size"]],
        "num_layers": [best_config["num_layers"]],
        "learning_rate": [best_config["learning_rate"]],
        "dropout": [0.1, 0.15, 0.2],
        "batch_size": [1024, 2048],
        "window_size": [48],  # 고정
        "use_attention": [False],  # attention은 이 데이터셋에서 일관적으로 성능 하락
        "weight_decay": [0, 1e-4, 1e-3],
        "split_type": ["lc"],  # FLOW는 LC_SPLIT_RATIOS
        "shuffle": [True],
    }

    return run_experiments(X, y, phase2_grid, top_n=top_n)


# ======================================================================
# 엔트리 포인트
# ======================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FLOW Hyperparameter Search")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                       help="Phase 1: 핵심 파라미터 탐색, Phase 2: 미세 조정")
    parser.add_argument("--best-hidden", type=int, default=384)
    parser.add_argument("--best-layers", type=int, default=1)
    parser.add_argument("--best-lr", type=float, default=1e-3)
    args = parser.parse_args()

    print("=" * 70)
    print("FLOW 하이퍼파라미터 그리드 탐색")
    print(f"Phase: {args.phase}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # 1. 데이터 로드 & 전처리 (한 번만)
    print("\n[Step 1] 데이터 로드 및 전처리...")
    dfs = load_data(DATA_DIR)
    X, y, rec_idx, feat_names = preprocess_data(dfs)
    print(f"전처리 완료: X shape = {X.shape}")

    if args.phase == 1:
        # Phase 1: 핵심 파라미터 탐색
        results = run_experiments(X, y, PHASE1_GRID, top_n=10)
    else:
        # Phase 2: 보조 파라미터 미세 조정
        best_config = {
            "hidden_size": args.best_hidden,
            "num_layers": args.best_layers,
            "learning_rate": args.best_lr,
        }
        results = run_phase2(X, y, best_config, top_n=10)
