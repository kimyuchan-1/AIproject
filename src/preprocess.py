import numpy as np
import pandas as pd
import torch

from notebook.feature import feature_engineering as feat_eng

from .config import TMS_TARGETS
from .loader import (
    tms_features,
    tms_x_scalers,
    FLOW_FEATURE_NAMES,
    flow_x_scaler,
)


# ====== 공통 유틸 ======
def extract_input_lists(x):
    data_list = [item.model_dump() for item in x.input.dataList]
    aws368 = [item.model_dump() for item in (x.input.awsList.get("stn_368") or x.input.awsList.get("STN_368", []))]
    aws541 = [item.model_dump() for item in (x.input.awsList.get("stn_541") or x.input.awsList.get("STN_541", []))]
    aws569 = [item.model_dump() for item in (x.input.awsList.get("stn_569") or x.input.awsList.get("STN_569", []))]
    return data_list, aws368, aws541, aws569


def validate_input_lengths(data_list, aws368, aws541, aws569):
    if not data_list or not aws368 or not aws541 or not aws569:
        raise ValueError("input.dataList and input.awsList (stn_368, stn_541, stn_569) are required")
    if len(data_list) != 1440 or len(aws368) != 1440 or len(aws541) != 1440 or len(aws569) != 1440:
        raise ValueError(
            f"dataList and awsList require exactly 1440 records (24 hours), "
            f"got {len(data_list)}, {len(aws368)}, {len(aws541)}, {len(aws569)}"
        )


def build_hourly_predictions(pred_12h: list[float]) -> dict:
    return {f"{i * 0.5:.1f}h": pred_12h[i - 1] for i in range(1, len(pred_12h) + 1)}


# ====== 공통 데이터 전처리 함수 ======
def resample_to_30min(df: pd.DataFrame) -> pd.DataFrame:
    """1분 단위 데이터를 30분 단위로 리샘플링"""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    agg_dict = {}
    for col in numeric_cols:
        if col.startswith("RN_") or col.startswith("AR_") or col == "FLUX_VU":
            agg_dict[col] = "sum"
        else:
            agg_dict[col] = "mean"

    return df[numeric_cols].resample("30min").agg(agg_dict)


def merge_input_data(data_list, aws368, aws541, aws569) -> pd.DataFrame:
    """
    백엔드 입력 데이터를 병합하여 30분 리샘플링된 DataFrame 반환

    Args:
        data_list: 1분 단위 24시간 데이터 (1440개 레코드)
        aws368, aws541, aws569: AWS 데이터

    Returns:
        df_resampled: 30분 리샘플링된 DataFrame (DatetimeIndex)
    """
    if not data_list or not aws368 or not aws541 or not aws569:
        raise ValueError("dataList and awsList are empty")

    df = pd.DataFrame(data_list)
    aws368 = pd.DataFrame(aws368)
    aws541 = pd.DataFrame(aws541)
    aws569 = pd.DataFrame(aws569)

    # 시간 컬럼 처리
    for name, frame in [("dataList", df), ("aws368", aws368), ("aws541", aws541), ("aws569", aws569)]:
        if "SYS_TIME" not in frame.columns:
            raise ValueError(f"SYS_TIME column is required in {name}")
        frame["SYS_TIME"] = pd.to_datetime(frame["SYS_TIME"], errors="coerce")

    df = df.set_index("SYS_TIME").sort_index()
    aws368 = aws368.set_index("SYS_TIME").sort_index()
    aws541 = aws541.set_index("SYS_TIME").sort_index()
    aws569 = aws569.set_index("SYS_TIME").sort_index()

    # AWS 데이터 병합 (outer + ffill: 타임스탬프 불일치 허용)
    aws = aws368.add_suffix("_368").join(
        aws541.add_suffix("_541"), how="outer"
    ).join(
        aws569.add_suffix("_569"), how="outer"
    ).ffill()

    df = df.join(aws, how="left")

    if df.shape[0] == 0:
        raise ValueError("merge_input_data: join 결과가 비어 있습니다. 타임스탬프를 확인하세요.")

    # FLUX_VU: 누적값 → 증분값 변환 (TMS 노트북과 동일, 리샘플링 전에 처리)
    if "FLUX_VU" in df.columns:
        flux = df["FLUX_VU"].copy()
        flux_diff = flux.diff()
        reset_mask = flux_diff < 0      # 일 초기화 지점
        flux_diff[reset_mask] = flux[reset_mask]
        flux_diff.iloc[0] = 0
        flux_diff = flux_diff.clip(lower=0)
        df["FLUX_VU"] = flux_diff

    # 30분 리샘플링
    df_resampled = resample_to_30min(df)

    return df_resampled


# ====== 공통 특성 엔지니어링 파이프라인 ======
def _apply_common_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for feature engineering")

    df_features = feat_eng.add_rain_features(df)
    df_features = feat_eng.add_station_agg_rain_features(df_features)
    df_features = feat_eng.add_weather_features(df_features)
    df_features = feat_eng.add_process_features(df_features)
    df_features = feat_eng.add_temporal_features(df_features)
    df_features = feat_eng.add_time_features(df_features)

    # 노트북과 동일: weekday/iso_week/hour_x_weekday 보장
    if "weekday" not in df_features.columns and "dayofweek" in df_features.columns:
        df_features["weekday"] = df_features["dayofweek"]
    if "iso_week" not in df_features.columns and isinstance(df_features.index, pd.DatetimeIndex):
        df_features["iso_week"] = df_features.index.isocalendar().week.astype(int).to_numpy()
    if "hour_x_weekday" not in df_features.columns:
        if "hour" in df_features.columns and "weekday" in df_features.columns:
            df_features["hour_x_weekday"] = df_features["hour"] * df_features["weekday"]
        elif "hour" in df_features.columns and "dayofweek" in df_features.columns:
            df_features["hour_x_weekday"] = df_features["hour"] * df_features["dayofweek"]

    df_features = df_features.ffill().fillna(0)
    return df_features


# ====== Flow 전처리 함수 ======
def apply_flow_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flow 모델용 특성 엔지니어링 (LSTM_FLOW.ipynb와 동일)

    1. flow_TankA/B lag 특성 추가 (min_lag=2)
    2. raw flow_TankA/B 제거 (lag 특성은 유지)
    3. raw Q_in(타겟) 제거
    4. 특성 엔지니어링 파이프라인 적용
    """
    # 1. flow_TankA/B lag 특성 추가 (제거 전에 실행)
    flow_lag_cols = [c for c in ["flow_TankA", "flow_TankB"] if c in df.columns]
    if flow_lag_cols:
        df = feat_eng.add_target_lag_features(df, flow_lag_cols, min_lag=2)

    # 2. raw flow_TankA/B 제거 (lag 특성은 유지)
    df = df.drop(columns=["flow_TankA", "flow_TankB"], errors='ignore')

    # 3. raw Q_in(타겟) 제거
    df_base = df.drop(columns=["Q_in"], errors='ignore')

    # 4. 특성 엔지니어링 파이프라인
    return _apply_common_feature_pipeline(df_base)


def preprocess_flow_input(data_list, aws368, aws541, aws569) -> torch.Tensor:
    """Flow 모델 입력 전처리: 병합 → 리샘플링 → 특성 엔지니어링 → 텐서"""
    df_resampled = merge_input_data(data_list, aws368, aws541, aws569)
    df_features = apply_flow_feature_engineering(df_resampled)

    for feat in FLOW_FEATURE_NAMES:
        if feat not in df_features.columns:
            df_features[feat] = 0.0

    X = df_features[FLOW_FEATURE_NAMES].values
    X_scaled = flow_x_scaler.transform(X)
    tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
    return tensor


# ====== TMS 전처리 함수 ======
def apply_tms_feature_engineering(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """
    TMS 모델용 타겟별 특성 엔지니어링 (LSTM_TMS.ipynb와 동일)

    1. 타겟 lag 특성 추가 (min_lag=2)
    2. raw 타겟 컬럼 제거
    3. 특성 엔지니어링 파이프라인 적용
    """
    cfg = TMS_TARGETS[target_name]
    target_col = cfg["target_col"]

    df_work = df.copy()

    # 1. 타겟 lag 특성 추가 (분리 전에 실행)
    if target_col in df_work.columns:
        df_work = feat_eng.add_target_lag_features(df_work, [target_col], min_lag=2)

    # 2. raw 타겟 컬럼 제거 (lag 특성은 유지)
    if target_col in df_work.columns:
        df_work = df_work.drop(columns=[target_col])

    # 3. 특성 엔지니어링 파이프라인
    return _apply_common_feature_pipeline(df_work)


def make_tms_tensor(df_resampled: pd.DataFrame, target_name: str) -> torch.Tensor:
    """타겟별 특성 엔지니어링 → 추천 특성 선택 → 스케일링 → 텐서"""
    df_features = apply_tms_feature_engineering(df_resampled, target_name)
    feature_names = tms_features[target_name]

    for feat in feature_names:
        if feat not in df_features.columns:
            df_features[feat] = 0.0

    X = df_features[feature_names].values
    X_scaled = tms_x_scalers[target_name].transform(X)
    tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
    return tensor
