"""baseline.py

- 모든 로직은 함수/클래스로 구성되어 있습니다.
- main()은 최소화되어 있습니다: load -> prep -> run_pipeline.

사용법:
  python baseline.py --mode flow --data-root ../../../data/actual --resample 5min

"""

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =========================
# 0) 타겟 설정
# =========================
TARGETS_FLOW = ["Q_in"]
TARGETS_TMS  = ["TOC_VU", "PH_VU", "SS_VU", "FLUX_VU", "TN_VU", "TP_VU"]
TARGETS_ALL  = TARGETS_FLOW + TARGETS_TMS

def get_target_cols(mode):
    mode = mode.lower().strip()
    if mode == "flow":
        return TARGETS_FLOW
    if mode == "tms":
        return TARGETS_TMS
    if mode == "all":
        return TARGETS_ALL
    raise ValueError("mode must be one of: 'flow', 'tms', 'all'")


# =========================
# 1) 시간 인덱스 & 병합
# =========================
def set_datetime_index(df, time_col, tz= None):
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).set_index(time_col).sort_index()
    return out

def summarize_available_period(dfs):
    rows = []
    for name, df in dfs.items():
        if df is None or len(df) == 0:
            rows.append([name, None, None, 0])
        else:
            rows.append([name, df.index.min(), df.index.max(), len(df)])
    return pd.DataFrame(rows, columns=["source", "start", "end", "n_rows"])


def merge_sources_on_time(dfs, how = "outer"):
    items = [df.copy() for df in dfs.values() if df is not None and len(df) > 0]
    if not items:
        raise ValueError("No non-empty dataframes to merge.")
    out = items[0]
    for nxt in items[1:]:
        out = out.join(nxt, how=how)
    out = out.sort_index()
    return out

# =========================
# 2) 데이터 정제 & 리샘플링
# =========================
def drop_missing_rows(df, cols = None):
    out = df.copy()
    if cols is None:
        return out.dropna()
    return out.dropna(subset=cols)

def resample_hourly(df, rule = "1h", agg = "mean"):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex for resampling.")
    out = df.copy()
    if isinstance(agg, str):
        return out.resample(rule).agg(agg)
    return out.resample(rule).agg(agg)

def check_continuity(df, freq = "1h"):
    if len(df) == 0:
        return {"is_continuous": True, "n_missing_timestamps": 0, "max_gap": pd.Timedelta(0)}

    idx = df.index
    expected = pd.date_range(start=idx.min(), end=idx.max(), freq=freq, tz=idx.tz)
    missing = expected.difference(idx)
    # diff 기반 최대 간격 계산
    diffs = idx.to_series().diff().dropna()
    max_gap = diffs.max() if len(diffs) else pd.Timedelta(0)
    return {
        "is_continuous": len(missing) == 0,
        "n_missing_timestamps": len(missing),
        "max_gap": max_gap
    }

# =========================
# 3) 피처 엔지니어링
# =========================
def add_time_features(df, add_sin_cos = True):
    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex for time features.")

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
    season_map = {12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}
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

    out = df.copy()
    for c in base_cols:
        if c not in out.columns:
            continue
        for k in lags:
            out[f"{c}_lag{k}"] = out[c].shift(k)
    return out

def add_rolling_features(df, base_cols, windows, stats = ["mean"]):
    """
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


def make_supervised_dataset(df, target_cols, dropna = True):
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"target_cols not found in df: {missing}")

    y = df[target_cols].copy()
    X = df.drop(columns=target_cols).copy()

    X = X.select_dtypes(include=[np.number])

    keep = X.notna().all(axis=1) & y.notna().all(axis=1)
    return X.loc[keep], y.loc[keep]

@dataclass
class FeatureConfig:
    add_time = True
    add_sin_cos = True
    lag_hours = None
    roll_hours = None

    def __post_init__(self):
        if self.lag_hours is None:
            self.lag_hours = [1, 2, 3, 6, 12, 24]
        if self.roll_hours is None:
            self.roll_hours = [1, 2, 24]

def build_features(df_hourly, target_cols, feature_base_cols = None, cfg = FeatureConfig()):
    out = df_hourly.copy()

    if cfg.add_time:
        out = add_time_features(out, add_sin_cos=cfg.add_sin_cos)

    if feature_base_cols is None:
        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        feature_base_cols = [c for c in numeric_cols if c not in target_cols]

    out = add_lag_features(out, base_cols=feature_base_cols, lags=cfg.lag_hours)
    out = add_rolling_features(out, base_cols=feature_base_cols, windows=cfg.roll_hours)
    return out

# =========================
# 4) 데이터 분할 (시간 기반)
# =========================
@dataclass
class SplitConfig:
    train_ratio = 0.6
    valid_ratio = 0.2
    test_ratio = 0.2


def time_split(X, y, cfg = SplitConfig()):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset after preprocessing/feature generation.")
    if not math.isclose(cfg.train_ratio + cfg.valid_ratio + cfg.test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train/valid/test ratios must sum to 1.0")

    n_train = int(n * cfg.train_ratio)
    n_valid = int(n * cfg.valid_ratio)

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_valid, y_valid = X.iloc[n_train:n_train+n_valid], y.iloc[n_train:n_train+n_valid]
    X_test,  y_test  = X.iloc[n_train+n_valid:], y.iloc[n_train+n_valid:]

    return {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid),
        "test":  (X_test,  y_test),
    }

# =========================
# 5) 모델
# =========================
def build_model_zoo(random_state = 42):

    zoo = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.001, random_state=random_state, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state, max_iter=5000),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "HistGBR": HistGradientBoostingRegressor(
            random_state=random_state, learning_rate=0.05, max_iter=500
        ),
    }
    return zoo

def wrap_multioutput_if_needed(model, y):

    if y.shape[1] <= 1:
        return model

    # HistGBR은 래퍼가 필요함
    if isinstance(model, HistGradientBoostingRegressor):
        return MultiOutputRegressor(model)
    return model

# =========================
# 6) 평가 지표 & 평가
# =========================
def compute_metrics(y_true, y_pred):
    # 2D 형태로 변환
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)

    r2s, rmses, mapes = [], [], []
    for j in range(yt.shape[1]):
        r2s.append(r2_score(yt[:, j], yp[:, j]))
        rmses.append(math.sqrt(mean_squared_error(yt[:, j], yp[:, j])))
        mapes.append(mean_absolute_percentage_error(yt[:, j], yp[:, j]) * 100.0)

    return {
        "R2_mean": float(np.mean(r2s)),
        "RMSE_mean": float(np.mean(rmses)),
        "MAPE_mean(%)": float(np.mean(mapes)),
        "R2_by_target": r2s,
        "RMSE_by_target": rmses,
        "MAPE_by_target(%)": mapes,
    }

def fit_and_evaluate(model, splits):

    X_train, y_train = splits["train"]

    model.fit(X_train, y_train)

    out = {}
    for name, (X_, y_) in splits.items():
        pred = model.predict(X_)
        out[name] = compute_metrics(y_.to_numpy(), pred)
    return out

# =========================
# 7) 시각화
# =========================
def plot_predictions(y_true, y_pred, title, n_points = 500):
    yt = y_true.copy()
    yp = np.asarray(y_pred)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)

    # 마지막 n_points 정렬
    yt = yt.iloc[-n_points:]
    yp = yp[-len(yt):, :]

    for j, col in enumerate(yt.columns):
        plt.figure(figsize=(12, 4))
        plt.plot(yt.index, yt[col].values, label="true")
        plt.plot(yt.index, yp[:, j], label="pred")
        plt.title(f"{title} | target={col}")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_metric_table(result_by_model, split = "test"):
    rows = []
    for model_name, res in result_by_model.items():
        m = res[split]
        rows.append([model_name, m["R2_mean"], m["RMSE_mean"], m["MAPE_mean(%)"]])
    tbl = pd.DataFrame(rows, columns=["model", "R2_mean", "RMSE_mean", "MAPE_mean(%)"])
    return tbl.sort_values(by="RMSE_mean", ascending=True)

# =========================
# 8) 파이프라인 실행 (flow/tms/all)
# =========================
def run_pipeline(
    dfs,
    mode,
    time_col_map = None,
    tz = None,
    dropna_cols_before_resample = None,
    resample_rule = "1h",
    resample_agg = "mean",
    feature_base_cols = None,
    feature_cfg = FeatureConfig(),
    split_cfg = SplitConfig(),
    random_state = 42
):
    
    target_cols = get_target_cols(mode)

    dfs_indexed = {}
    for name, df in dfs.items():
        if df is None or len(df) == 0:
            dfs_indexed[name] = df
            continue

        if isinstance(df.index, pd.DatetimeIndex):
            dfs_indexed[name] = df.sort_index()
        else:
            if time_col_map is None or name not in time_col_map:
                raise ValueError(f"{name} has no DatetimeIndex and no time_col_map provided.")
            dfs_indexed[name] = set_datetime_index(df, time_col=time_col_map[name], tz=tz)

    period_summary = summarize_available_period(dfs_indexed)

    # 병합
    df_all = merge_sources_on_time(dfs_indexed, how="outer")

    # --- 단계 2) 결측치 행 제거 (원본)
    df_all_clean = drop_missing_rows(df_all, cols=dropna_cols_before_resample)

    # --- 단계 3) 1시간 단위로 리샘플링
    df_hourly = resample_hourly(df_all_clean, rule=resample_rule, agg=resample_agg)

    # --- 단계 4) 피처 생성
    df_feat = build_features(
        df_hourly=df_hourly,
        target_cols=target_cols,
        feature_base_cols=feature_base_cols,
        cfg=feature_cfg
    )

    # --- 단계 5) 연속성 확인
    continuity = check_continuity(df_hourly.dropna(how="all"), freq=resample_rule)

    # 지도학습 데이터셋 X, y 생성
    X, y = make_supervised_dataset(df_feat, target_cols=target_cols, dropna=True)

    # --- 단계 6) 데이터 분할
    splits = time_split(X, y, cfg=split_cfg)

    # --- 단계 7) 모델 생성
    zoo = build_model_zoo(random_state=random_state)

    # --- 단계 8) 평가
    results = {}
    fitted_models = {}
    for model_name, base_model in zoo.items():
        model = wrap_multioutput_if_needed(base_model, y)
        res = fit_and_evaluate(model, splits)
        results[model_name] = res
        fitted_models[model_name] = model

    metric_table = plot_metric_table(results, split="test")

    return {
        "mode": mode,
        "target_cols": target_cols,
        "period_summary": period_summary,
        "df_merged": df_all,
        "df_hourly": df_hourly,
        "df_features": df_feat,
        "continuity": continuity,
        "X": X, "y": y,
        "splits": splits,
        "results": results,
        "metric_table": metric_table,
        "fitted_models": fitted_models
    }


# =========================
# 9) 입출력 헬퍼 + CLI
# =========================
from pathlib import Path
import argparse

def load_csvs(data_root: str):
    """루트 디렉토리에서 FLOW/TMS/AWS CSV 파일들을 로드합니다."""
    root = Path(data_root)
    df_flow = pd.read_csv(root / "FLOW_Actual.csv")
    df_tms  = pd.read_csv(root / "TMS_Actual.csv")
    df_aws_368 = pd.read_csv(root / "AWS_368.csv")
    df_aws_541 = pd.read_csv(root / "AWS_541.csv")
    df_aws_569 = pd.read_csv(root / "AWS_569.csv")
    return df_flow, df_tms, df_aws_368, df_aws_541, df_aws_569

def prep_flow(df_flow: pd.DataFrame) -> pd.DataFrame:
    out = df_flow.copy()
    for c in ["flow_TankA", "flow_TankB"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "flow_TankA" in out.columns and "flow_TankB" in out.columns:
        out["Q_in"] = out["flow_TankA"] + out["flow_TankB"]
        # 레벨 컬럼은 유지; 원본 flow 컬럼은 타겟 유출 방지를 위해 제거
        out = out.drop(columns=["flow_TankA", "flow_TankB"])
    return out

def prep_aws_station(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    return out.add_suffix(suffix)

def prep_aws(df_aws_368, df_aws_541, df_aws_569) -> pd.DataFrame:
    aws368 = prep_aws_station(df_aws_368, "_368")
    aws541 = prep_aws_station(df_aws_541, "_541")
    aws569 = prep_aws_station(df_aws_569, "_569")
    df_aws = aws368.merge(aws541, left_index=True, right_index=True, how="inner").merge(
        aws569, left_index=True, right_index=True, how="inner"
    )
    df_aws["datetime"] = df_aws.index
    return df_aws

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["flow", "tms", "all"], default="flow")
    p.add_argument("--data-root", default="../../../data/actual", help="*_Actual.csv 및 AWS_*.csv 파일이 있는 디렉토리")
    p.add_argument("--resample", default="5min", help="Pandas 리샘플링 규칙, 예: 5min, 1h")
    p.add_argument("--how", default="inner", choices=["inner","outer","left","right"])
    p.add_argument("--agg", default="mean", help="집계 방법: mean 또는 'auto' (RN*는 sum, 나머지는 mean)")
    p.add_argument("--random-state", type=int, default=42)
    return p

def main():
    args = build_argparser().parse_args()

    df_flow, df_tms, df_aws_368, df_aws_541, df_aws_569 = load_csvs(args.data_root)
    df_flow = prep_flow(df_flow)
    df_aws  = prep_aws(df_aws_368, df_aws_541, df_aws_569)

    dfs = {"flow": df_flow, "tms": df_tms, "aws": df_aws}
    time_col_map = {"flow": "SYS_TIME", "tms": "SYS_TIME", "aws": "datetime"}

    if args.agg == "auto":
        # RN*는 리샘플링 윈도우에서 합산되어야 하고, 나머지는 평균
        merged_tmp = merge_sources_on_time({k: set_datetime_index(v, time_col_map[k]) for k,v in dfs.items()}, how=args.how)
        num_cols = merged_tmp.select_dtypes(include=[np.number]).columns
        resample_agg = {c: ("sum" if str(c).startswith("RN") else "mean") for c in num_cols}
    else:
        resample_agg = args.agg

    out = run_pipeline(
        dfs,
        mode=args.mode,
        time_col_map=time_col_map,
        tz=None,
        resample_rule=args.resample,
        resample_agg=resample_agg,
        random_state=args.random_state,
    )

    # 콘솔 출력
    print(out["period_summary"])
    print(out["metric_table"].to_string(index=False))

    # 선택사항: 테스트 데이터에서 최고 성능 모델 시각화
    best_model_name = out["metric_table"].iloc[0]["model"]
    best_model = out["fitted_models"][best_model_name]
    X_test, y_test = out["splits"]["test"]
    y_pred = best_model.predict(X_test)
    plot_predictions(y_test, y_pred, title=f"TEST | {best_model_name}")

if __name__ == "__main__":
    main()
