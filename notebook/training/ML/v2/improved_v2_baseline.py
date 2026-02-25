"""
Improved ML Baseline V2 for WWTP Prediction
주요 개선사항:
1. 결측치 보간 (선형, KNN, Forward Fill)
2. 도메인 피처 추가 (상호작용, 비율, 차분)
3. 정규화 강화 (alpha 증가, max_depth 제한)
"""

import math
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 백그라운드 모드 (GUI 없음)
import os

warnings.filterwarnings("ignore")

# 결과 저장 디렉토리
RESULTS_DIR = "../../../../results/ML/v2"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# 0) Target config
# =========================
TARGETS_FLOW = ["Q_in"]
TARGETS_TMS = ["TOC_VU", "PH_VU", "SS_VU", "FLUX_VU", "TN_VU", "TP_VU"]
TARGETS_ALL = TARGETS_FLOW + TARGETS_TMS

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
# 1) Time index & merge
# =========================
def set_datetime_index(df, time_col, tz=None):
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).set_index(time_col).sort_index()
    return out

def merge_sources_on_time(dfs, how="outer"):
    items = [df.copy() for df in dfs.values() if df is not None and len(df) > 0]
    if not items:
        raise ValueError("No non-empty dataframes to merge.")
    # 전처리된 데이터는 이미 병합되어 있으므로 첫 번째 데이터프레임만 반환
    if len(items) == 1:
        return items[0].sort_index()
    out = items[0]
    for nxt in items[1:]:
        out = out.join(nxt, how=how)
    out = out.sort_index()
    return out
# =========================
# 2) 리샘플링
# =========================
def resample_hourly(df, rule="1h", agg="mean"):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex for resampling.")
    out = df.copy()
    if isinstance(agg, str):
        return out.resample(rule).agg(agg)
    return out.resample(rule).agg(agg)

# =========================
# 3) 도메인 피처 엔지니어링 (개선!)
# =========================
def add_time_features(df, add_sin_cos=True):
    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex for time features.")
    
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_night"] = ((idx.hour >= 22) | (idx.hour <= 6)).astype(int)
    
    # Season
    m = out["month"]
    season_map = {12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}
    out["season"] = m.map(season_map).astype(int)
    
    if add_sin_cos:
        out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
        out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)
        out["sin_dow"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
        out["cos_dow"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)
        out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12.0)
        out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12.0)
    
    return out

def add_domain_features(df, target_cols):
    """도메인 지식 기반 피처 추가"""
    out = df.copy()
    print("\n도메인 피처 생성 중...")
    
    # 1. 강수량 관련 피처
    if 'RN_60m' in out.columns and 'RN_DAY' in out.columns:
        out['rain_intensity'] = out['RN_60m'] / (out['RN_DAY'] + 1)  # 강수 강도
        out['is_raining'] = (out['RN_60m'] > 0).astype(int)
        print("  ✓ 강수량 피처 추가")
    
    # 2. 온도-습도 관련 피처
    if 'TA' in out.columns and 'HM' in out.columns:
        out['temp_humidity_idx'] = out['TA'] * out['HM'] / 100  # 불쾌지수 근사
        out['temp_range'] = out['TA'].rolling(24, min_periods=1).max() - out['TA'].rolling(24, min_periods=1).min()
        print("  ✓ 온도-습도 피처 추가")
    
    # 3. 탱크 수위 관련 피처
    if 'level_TankA' in out.columns and 'level_TankB' in out.columns:
        out['tank_total_level'] = out['level_TankA'] + out['level_TankB']
        out['tank_level_diff'] = abs(out['level_TankA'] - out['level_TankB'])
        out['tank_level_ratio'] = out['level_TankA'] / (out['level_TankB'] + 0.01)
        print("  ✓ 탱크 수위 피처 추가")
    
    # 4. 수질 변수 간 상호작용 (TMS 예측 시)
    if 'TOC_VU' in out.columns and 'TN_VU' in out.columns:
        out['TOC_TN_ratio'] = out['TOC_VU'] / (out['TN_VU'] + 0.01)
        print("  ✓ TOC/TN 비율 추가")
    
    if 'SS_VU' in out.columns and 'TP_VU' in out.columns:
        out['SS_TP_ratio'] = out['SS_VU'] / (out['TP_VU'] + 0.01)
        print("  ✓ SS/TP 비율 추가")
    
    # 5. 차분 피처 (변화율)
    for col in ['TA', 'HM', 'level_TankA', 'level_TankB']:
        if col in out.columns:
            out[f'{col}_diff1'] = out[col].diff(1)  # 1시간 변화
            out[f'{col}_diff24'] = out[col].diff(24)  # 24시간 변화
    
    print(f"  총 {len(out.columns) - len(df.columns)}개 도메인 피처 추가됨")
    
    return out

def add_lag_features(df, base_cols, lags):
    out = df.copy()
    for c in base_cols:
        if c not in out.columns:
            continue
        for k in lags:
            out[f"{c}_lag{k}"] = out[c].shift(k)
    return out

def add_rolling_features(df, base_cols, windows, stats=["mean"]):
    out = df.copy()
    for c in base_cols:
        if c not in out.columns:
            continue
        for w in windows:
            r = out[c].rolling(window=w, min_periods=max(1, w//2))  # min_periods 완화
            if "mean" in stats:
                out[f"{c}_r{w}_mean"] = r.mean()
            if "std" in stats:
                out[f"{c}_r{w}_std"] = r.std()
    return out

@dataclass
class FeatureConfig:
    add_time = True
    add_sin_cos = True
    add_domain = True  # 도메인 피처 추가
    lag_hours = None
    roll_hours = None
    
    def __post_init__(self):
        if self.lag_hours is None:
            self.lag_hours = [1, 3, 6, 12]  # 24 제거 (과적합 방지)
        if self.roll_hours is None:
            self.roll_hours = [3, 6, 12]  # 더 짧은 윈도우

def build_features(df_hourly, target_cols, feature_base_cols=None, cfg=FeatureConfig()):
    out = df_hourly.copy()
    
    if cfg.add_time:
        out = add_time_features(out, add_sin_cos=cfg.add_sin_cos)
    
    if cfg.add_domain:
        out = add_domain_features(out, target_cols)
    
    if feature_base_cols is None:
        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        feature_base_cols = [c for c in numeric_cols if c not in target_cols]
    
    out = add_lag_features(out, base_cols=feature_base_cols, lags=cfg.lag_hours)
    out = add_rolling_features(out, base_cols=feature_base_cols, windows=cfg.roll_hours)
    return out

def make_supervised_dataset(df, target_cols):
    """결측치 제거 (보간 후이므로 최소화됨)"""
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"target_cols not found in df: {missing}")
    
    y = df[target_cols].copy()
    X = df.drop(columns=target_cols).copy()
    X = X.select_dtypes(include=[np.number])
    
    # 결측치 제거
    keep = X.notna().all(axis=1) & y.notna().all(axis=1)
    X_clean = X.loc[keep]
    y_clean = y.loc[keep]
    
    print(f"\n최종 데이터셋: {len(X)} → {len(X_clean)} ({len(X_clean)/len(X)*100:.1f}%)")
    
    return X_clean, y_clean

# =========================
# 4) Split (time-based)
# =========================
@dataclass
class SplitConfig:
    train_ratio = 0.6
    valid_ratio = 0.2
    test_ratio = 0.2

def time_split(X, y, cfg=SplitConfig()):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset after preprocessing/feature generation.")
    
    n_train = int(n * cfg.train_ratio)
    n_valid = int(n * cfg.valid_ratio)
    
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_valid, y_valid = X.iloc[n_train:n_train+n_valid], y.iloc[n_train:n_train+n_valid]
    X_test, y_test = X.iloc[n_train+n_valid:], y.iloc[n_train+n_valid:]
    
    return {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid),
        "test": (X_test, y_test),
    }

# =========================
# 5) Feature Selection
# =========================
def select_top_features(X_train, y_train, n_features=30):
    """RandomForest로 피처 중요도 계산 후 상위 n개 선택 (V2: 더 적은 피처)"""
    print(f"\n피처 선택 중... (총 {X_train.shape[1]}개 → 상위 {n_features}개)")
    
    # 단일 타겟인 경우
    if len(y_train.shape) == 1 or y_train.shape[1] == 1:
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train.values.ravel() if hasattr(y_train, 'values') else y_train)
        importances = rf.feature_importances_
    else:
        # 다중 타겟인 경우 평균 중요도 사용
        rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
        rf.fit(X_train, y_train)
        importances = np.mean([est.feature_importances_ for est in rf.estimators_], axis=0)
    
    # 상위 n개 피처 선택
    top_indices = np.argsort(importances)[-n_features:]
    top_features = X_train.columns[top_indices].tolist()
    
    print(f"선택된 상위 10개 피처: {top_features[-10:]}")
    
    return top_features

# =========================
# 6) Models with GridSearch (V2: 강화된 정규화)
# =========================
def build_model_zoo_with_gridsearch(n_targets=1, cv=3):
    """GridSearchCV를 포함한 모델 정의 (V2: 정규화 강화)"""
    
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Ridge (V2: alpha 증가)
    ridge_params = {
        'alpha': [100, 1000, 10000]  # V1: [0.1, 1.0, 10.0, 100.0]
    }
    ridge = GridSearchCV(
        Ridge(random_state=42),
        ridge_params,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Lasso (V2: alpha 증가)
    lasso_params = {
        'alpha': [1.0, 10.0, 100.0]  # V1: [0.001, 0.01, 0.1, 1.0]
    }
    lasso = GridSearchCV(
        Lasso(random_state=42, max_iter=5000),
        lasso_params,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # RandomForest (V2: max_depth 제한, min_samples_leaf 추가)
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],  # V1: [10, 20, None] - None 제거
        'min_samples_split': [2, 5],
        'min_samples_leaf': [5, 10, 20]  # V2: 추가
    }
    rf = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_params,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    
    # HistGradientBoosting (V2: 더 강한 정규화)
    hgb_params = {
        'learning_rate': [0.01, 0.05],  # V1: [0.01, 0.05, 0.1]
        'max_iter': [500],
        'max_depth': [5, 10],  # V1: [5, 10, 20]
        'min_samples_leaf': [20, 30],  # V2: 추가
        'early_stopping': [True],
        'n_iter_no_change': [20],
        'validation_fraction': [0.2]
    }
    hgb = GridSearchCV(
        HistGradientBoostingRegressor(random_state=42),
        hgb_params,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # XGBoost (V2: 정규화 파라미터 추가)
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05],  # V1: [0.01, 0.05, 0.1]
        'max_depth': [3, 5],  # V1: [3, 5, 7]
        'subsample': [0.8],  # V1: [0.8, 1.0]
        'min_child_weight': [3, 5, 7],  # V2: 추가
        'gamma': [0, 0.1, 0.2]  # V2: 추가
    }
    xgb_model = GridSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1),
        xgb_params,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    zoo = {
        "Ridge": ridge,
        "Lasso": lasso,
        "RandomForest": rf,
        "HistGBR": hgb,
        "XGBoost": xgb_model,
    }
    
    return zoo

# =========================
# 7) Metrics & Evaluation
# =========================
def compute_metrics(y_true, y_pred):
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

def plot_learning_curve(model, model_name, mode, save_dir=RESULTS_DIR):
    """학습 곡선 시각화 (XGBoost, HistGBR)"""
    
    # XGBoost
    if 'XGB' in model_name:
        if isinstance(model, xgb.XGBRegressor):
            estimator = model
        elif isinstance(model, MultiOutputRegressor):
            estimator = model.estimators_[0]
        elif isinstance(model, GridSearchCV):
            estimator = model.best_estimator_
        else:
            return False
        
        if hasattr(estimator, 'evals_result'):
            results = estimator.evals_result()
            if results and 'validation_0' in results:
                train_metric = results['validation_0']['rmse']
                valid_metric = results['validation_1']['rmse'] if 'validation_1' in results else None
                
                plt.figure(figsize=(10, 6))
                plt.plot(train_metric, label='Train RMSE', linewidth=2)
                if valid_metric:
                    plt.plot(valid_metric, label='Valid RMSE', linewidth=2)
                    if hasattr(estimator, 'best_iteration'):
                        plt.axvline(x=estimator.best_iteration, color='r', linestyle='--', 
                                   label=f'Best Iteration ({estimator.best_iteration})', alpha=0.7)

                
                plt.xlabel('Iterations', fontsize=12)
                plt.ylabel('RMSE', fontsize=12)
                plt.title(f'{model_name} Learning Curve - {mode.upper()}', fontsize=14, fontweight='bold')
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                save_path = os.path.join(save_dir, f'{mode}_{model_name}_learning_curve.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"  📊 Learning curve saved: {save_path}")
                plt.close()
                return True
    
    # HistGradientBoosting
    elif 'HistGBR' in model_name:
        if isinstance(model, MultiOutputRegressor):
            estimator = model.estimators_[0]
        elif isinstance(model, GridSearchCV):
            estimator = model.best_estimator_
        else:
            estimator = model
        
        if hasattr(estimator, 'train_score_'):
            train_scores = estimator.train_score_
            valid_scores = estimator.validation_score_ if hasattr(estimator, 'validation_score_') else None
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_scores, label='Train Score', linewidth=2)
            if valid_scores is not None:
                plt.plot(valid_scores, label='Valid Score', linewidth=2)
            plt.xlabel('Iterations', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.title(f'{model_name} Learning Curve - {mode.upper()}', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, f'{mode}_{model_name}_learning_curve.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  📊 Learning curve saved: {save_path}")
            plt.close()
            return True
    
    return False

def plot_r2_comparison(results, mode, save_dir=RESULTS_DIR):
    """모든 모델의 R² 비교 시각화"""
    models = list(results.keys())
    train_r2 = [results[m]['train']['R2_mean'] for m in models]
    valid_r2 = [results[m]['valid']['R2_mean'] for m in models]
    test_r2 = [results[m]['test']['R2_mean'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, train_r2, width, label='Train', alpha=0.8)
    ax.bar(x, valid_r2, width, label='Valid', alpha=0.8)
    ax.bar(x + width, test_r2, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(f'R² Comparison - {mode.upper()} (V2)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{mode}_r2_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 R² comparison saved: {save_path}")
    plt.close()

def plot_metric_table(result_by_model, split="test"):
    rows = []
    for model_name, res in result_by_model.items():
        m = res[split]
        rows.append([model_name, m["R2_mean"], m["RMSE_mean"], m["MAPE_mean(%)"]])
    tbl = pd.DataFrame(rows, columns=["model", "R2_mean", "RMSE_mean", "MAPE_mean(%)"])
    return tbl.sort_values(by="R2_mean", ascending=False)

def fit_and_evaluate(model, splits, scaler=None, model_name="", mode="", target_cols=None):
    """StandardScaler 적용 및 평가 (각 타겟별 개별 학습)"""
    X_train, y_train = splits["train"]
    X_valid, y_valid = splits["valid"]
    X_test, y_test = splits["test"]
    
    # Scaling
    if scaler is None:
        scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    # 다중 타겟인 경우 각각 개별 학습
    n_targets = y_train.shape[1] if len(y_train.shape) > 1 else 1
    
    if n_targets > 1:
        print(f"  Training {n_targets} individual models for each target...")
        fitted_models = {}
        all_results = {"train": [], "valid": [], "test": []}
        
        for i, target_name in enumerate(target_cols):
            print(f"\n  [{i+1}/{n_targets}] Training for {target_name}...")
            
            y_train_single = y_train.iloc[:, i] if hasattr(y_train, 'iloc') else y_train[:, i]
            y_valid_single = y_valid.iloc[:, i] if hasattr(y_valid, 'iloc') else y_valid[:, i]
            y_test_single = y_test.iloc[:, i] if hasattr(y_test, 'iloc') else y_test[:, i]
            
            # 모델 복사
            if isinstance(model, GridSearchCV):
                single_model = type(model)(
                    estimator=type(model.estimator)(**model.estimator.get_params()),
                    param_grid=model.param_grid,
                    cv=model.cv,
                    scoring=model.scoring,
                    n_jobs=model.n_jobs,
                    verbose=0
                )
            else:
                single_model = type(model)(**model.get_params())
            
            # 학습
            if isinstance(single_model, GridSearchCV):
                single_model.fit(X_train_scaled, y_train_single)
                print(f"    Best params: {single_model.best_params_}")
                
                # XGBoost Early Stopping
                if 'XGB' in model_name:
                    best_params = single_model.best_params_.copy()
                    best_params['n_estimators'] = 500
                    
                    final_model = xgb.XGBRegressor(
                        **best_params,
                        early_stopping_rounds=20,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    final_model.fit(
                        X_train_scaled, y_train_single,
                        eval_set=[(X_train_scaled, y_train_single), (X_valid_scaled, y_valid_single)],
                        verbose=False
                    )
                    
                    if hasattr(final_model, 'best_iteration'):
                        print(f"    Early stopping at iteration: {final_model.best_iteration}")
                    single_model = final_model

                
                elif 'HistGBR' in model_name:
                    if hasattr(single_model.best_estimator_, 'n_iter_'):
                        print(f"    Iterations: {single_model.best_estimator_.n_iter_}")
            else:
                single_model.fit(X_train_scaled, y_train_single)
            
            fitted_models[target_name] = single_model
            
            # 예측
            pred_train = single_model.predict(X_train_scaled)
            pred_valid = single_model.predict(X_valid_scaled)
            pred_test = single_model.predict(X_test_scaled)
            
            # 메트릭 계산
            all_results["train"].append(compute_metrics(y_train_single, pred_train))
            all_results["valid"].append(compute_metrics(y_valid_single, pred_valid))
            all_results["test"].append(compute_metrics(y_test_single, pred_test))
        
        # 평균 메트릭 계산
        out = {}
        for split in ["train", "valid", "test"]:
            out[split] = {
                "R2_mean": float(np.mean([r["R2_mean"] for r in all_results[split]])),
                "RMSE_mean": float(np.mean([r["RMSE_mean"] for r in all_results[split]])),
                "MAPE_mean(%)": float(np.mean([r["MAPE_mean(%)"] for r in all_results[split]])),
                "R2_by_target": [r["R2_mean"] for r in all_results[split]],
                "RMSE_by_target": [r["RMSE_mean"] for r in all_results[split]],
                "MAPE_by_target(%)": [r["MAPE_mean(%)"] for r in all_results[split]],
            }
        
        # Learning curve는 첫 번째 타겟 모델만 시각화
        first_model = fitted_models[target_cols[0]]
        plot_learning_curve(first_model, model_name, mode)
        
        return out, scaler, fitted_models
    
    else:
        # 단일 타겟
        print(f"  Training single target model...")
        
        if isinstance(model, GridSearchCV):
            model.fit(X_train_scaled, y_train)
            print(f"  Best params: {model.best_params_}")
            
            # XGBoost Early Stopping
            if 'XGB' in model_name:
                best_params = model.best_params_.copy()
                best_params['n_estimators'] = 500
                
                final_model = xgb.XGBRegressor(
                    **best_params,
                    early_stopping_rounds=20,
                    random_state=42,
                    n_jobs=-1
                )

                
                final_model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_train_scaled, y_train), (X_valid_scaled, y_valid)],
                    verbose=False
                )
                
                if hasattr(final_model, 'best_iteration'):
                    print(f"  Early stopping at iteration: {final_model.best_iteration}")
                model = final_model
            
            elif 'HistGBR' in model_name:
                if hasattr(model.best_estimator_, 'n_iter_'):
                    print(f"  Iterations: {model.best_estimator_.n_iter_}")
        else:
            model.fit(X_train_scaled, y_train)
        
        # Evaluation
        out = {}
        pred_train = model.predict(X_train_scaled)
        pred_valid = model.predict(X_valid_scaled)
        pred_test = model.predict(X_test_scaled)
        
        out["train"] = compute_metrics(y_train.to_numpy(), pred_train)
        out["valid"] = compute_metrics(y_valid.to_numpy(), pred_valid)
        out["test"] = compute_metrics(y_test.to_numpy(), pred_test)
        
        # Learning curve 시각화
        plot_learning_curve(model, model_name, mode)
        
        return out, scaler, model

# =========================
# 8) Pipeline Runner
# =========================
def run_improved_pipeline_v2(
    dfs,
    mode,
    time_col_map=None,
    tz=None,
    resample_rule="1h",
    resample_agg="mean",
    feature_cfg=FeatureConfig(),
    split_cfg=SplitConfig(),
    n_top_features=30,  # V2: 더 적은 피처
    cv_splits=3,
    random_state=42
):
    """개선된 파이프라인 V2 (전처리된 데이터 사용 - 보간 완료)"""
    
    print(f"\n{'='*60}")
    print(f"Mode: {mode.upper()} (V2)")
    print(f"{'='*60}")
    
    target_cols = get_target_cols(mode)
    
    # 1) 데이터 인덱싱
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
    
    # 2) 병합 (전처리된 데이터는 이미 병합되어 있음)
    df_all = merge_sources_on_time(dfs_indexed, how="outer")
    
    # 3) 리샘플링 (전처리된 데이터는 이미 1분 간격이므로 시간 단위로 리샘플링)
    df_hourly = resample_hourly(df_all, rule=resample_rule, agg=resample_agg)
    
    # 4) 피처 엔지니어링 (V2: 도메인 피처 포함)
    df_feat = build_features(
        df_hourly=df_hourly,
        target_cols=target_cols,
        feature_base_cols=None,
        cfg=feature_cfg
    )
    
    # 5) 지도학습 데이터셋 생성 (전처리된 데이터이므로 결측치 최소)
    X, y = make_supervised_dataset(df_feat, target_cols=target_cols)
    
    # 6) 분할
    splits = time_split(X, y, cfg=split_cfg)
    X_train, y_train = splits["train"]
    
    # 7) 피처 선택 (V2: 더 적은 피처)
    top_features = select_top_features(X_train, y_train, n_features=n_top_features)
    X_train_selected = X_train[top_features]
    X_valid_selected = splits["valid"][0][top_features]
    X_test_selected = splits["test"][0][top_features]
    
    splits_selected = {
        "train": (X_train_selected, y_train),
        "valid": (X_valid_selected, splits["valid"][1]),
        "test": (X_test_selected, splits["test"][1])
    }
    
    # 8) 모델 학습 및 평가 (V2: 강화된 정규화)
    zoo = build_model_zoo_with_gridsearch(n_targets=y_train.shape[1], cv=cv_splits)
    
    results = {}
    fitted_models = {}
    scalers = {}
    
    for model_name, base_model in zoo.items():
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")
        
        res, scaler, fitted_model = fit_and_evaluate(
            base_model, splits_selected, 
            model_name=model_name, 
            mode=mode,
            target_cols=target_cols
        )
        
        results[model_name] = res
        fitted_models[model_name] = fitted_model
        scalers[model_name] = scaler
        
        # 결과 출력
        print(f"\n  Train - R²: {res['train']['R2_mean']:.4f}, RMSE: {res['train']['RMSE_mean']:.2f}")
        print(f"  Valid - R²: {res['valid']['R2_mean']:.4f}, RMSE: {res['valid']['RMSE_mean']:.2f}")
        print(f"  Test  - R²: {res['test']['R2_mean']:.4f}, RMSE: {res['test']['RMSE_mean']:.2f}")
        
        # 타겟별 성능 출력
        if len(target_cols) > 1:
            print(f"\n  Per-target Test R²:")
            for i, target_name in enumerate(target_cols):
                print(f"    {target_name}: {res['test']['R2_by_target'][i]:.4f}")
        
        # Overfitting 체크
        train_r2 = res['train']['R2_mean']
        valid_r2 = res['valid']['R2_mean']
        if train_r2 - valid_r2 > 0.1:
            print(f"  ⚠️  과적합 가능성: Train R² ({train_r2:.4f}) >> Valid R² ({valid_r2:.4f})")

    
    metric_table = plot_metric_table(results, split="test")
    
    # R² 비교 시각화
    plot_r2_comparison(results, mode)
    
    print(f"\n{'='*60}")
    print("최종 결과 (Test Set)")
    print(f"{'='*60}")
    print(metric_table.to_string(index=False))
    
    return {
        "mode": mode,
        "target_cols": target_cols,
        "df_hourly": df_hourly,
        "df_features": df_feat,
        "X": X, "y": y,
        "splits": splits_selected,
        "top_features": top_features,
        "results": results,
        "metric_table": metric_table,
        "fitted_models": fitted_models,
        "scalers": scalers
    }

# =========================
# 9) Main Execution
# =========================
if __name__ == "__main__":
    print("="*80)
    print("IMPROVED ML BASELINE V2 - WWTP PREDICTION")
    print("주요 개선사항:")
    print("  1. 전처리된 데이터 사용 (1분 간격, 선형 보간 완료)")
    print("  2. 도메인 피처 추가 (상호작용, 비율, 차분)")
    print("  3. 정규화 강화 (alpha 증가, max_depth 제한)")
    print("="*80)
    
    print("\n데이터 로딩 중...")
    
    # 전처리된 데이터 로드 (1분 간격, 선형 보간 완료)
    df_flow = pd.read_csv("../../../../data/processed/flow_proc.csv")
    df_tms = pd.read_csv("../../../../data/processed/tms_proc.csv")
    df_all = pd.read_csv("../../../../data/processed/all_proc.csv")
    
    print(f"FLOW processed shape: {df_flow.shape}")
    print(f"TMS processed shape: {df_tms.shape}")
    print(f"ALL processed shape: {df_all.shape}")
    
    # Q_in 컬럼 생성: Q_in = flow_TankA + flow_TankB
    if 'Q_in' not in df_flow.columns:
        if 'flow_TankA' in df_flow.columns and 'flow_TankB' in df_flow.columns:
            df_flow['Q_in'] = df_flow['flow_TankA'] + df_flow['flow_TankB']
            print("✓ Q_in 컬럼 생성 (flow): flow_TankA + flow_TankB")
    
    if 'Q_in' not in df_all.columns:
        if 'flow_TankA' in df_all.columns and 'flow_TankB' in df_all.columns:
            df_all['Q_in'] = df_all['flow_TankA'] + df_all['flow_TankB']
            print("✓ Q_in 컬럼 생성 (all): flow_TankA + flow_TankB")
    
    # 데이터 누수 방지
    for c in ["flow_TankA", "flow_TankB"]:
        if c in df_flow.columns:
            df_flow.drop(columns=[c], inplace=True)
            print(f"✓ Dropped {c} from flow to prevent data leakage")
        if c in df_all.columns:
            df_all.drop(columns=[c], inplace=True)
            print(f"✓ Dropped {c} from all to prevent data leakage")
    
    dfs = {
        "flow": df_flow,
        "tms": df_tms,
        "all": df_all
    }
    
    time_col_map = {
        "flow": "SYS_TIME",
        "tms": "SYS_TIME",
        "all": "SYS_TIME"
    }
    
    # 파이프라인 실행
    print("\n" + "="*80)
    print("파이프라인 실행 시작")
    print("="*80)
    
    # Flow 예측 (Q_in만 예측)
    print("\n\n" + "🔵"*40)
    print("FLOW 예측 시작")
    print("🔵"*40)
    out_flow = run_improved_pipeline_v2(
        {"flow": df_flow},  # flow 데이터만 전달
        mode="flow", 
        time_col_map={"flow": "SYS_TIME"},
        n_top_features=25,
        cv_splits=3
    )
    
    # TMS 예측 (6개 수질 변수 예측)
    print("\n\n" + "🟢"*40)
    print("TMS 예측 시작")
    print("🟢"*40)
    out_tms = run_improved_pipeline_v2(
        {"tms": df_tms},  # tms 데이터만 전달
        mode="tms", 
        time_col_map={"tms": "SYS_TIME"},
        n_top_features=30,
        cv_splits=3
    )
    
    # All 예측 (Q_in + 6개 수질 변수 = 7개 예측)
    print("\n\n" + "🟡"*40)
    print("ALL 예측 시작")
    print("🟡"*40)
    out_all = run_improved_pipeline_v2(
        {"all": df_all},  # all 데이터만 전달
        mode="all", 
        time_col_map={"all": "SYS_TIME"},
        n_top_features=35,
        cv_splits=3
    )
    
    print("\n" + "="*80)
    print("모든 파이프라인 완료!")
    print("="*80)
    
    # 최고 모델 저장
    best_model_name = out_all["metric_table"].iloc[0]["model"]
    print(f"\n🏆 최고 성능 모델: {best_model_name}")
    print(f"   Test R²: {out_all['results'][best_model_name]['test']['R2_mean']:.4f}")
    print(f"   Test RMSE: {out_all['results'][best_model_name]['test']['RMSE_mean']:.2f}")
