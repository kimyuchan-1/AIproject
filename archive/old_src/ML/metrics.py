"""
평가 지표 모듈
모델 성능 평가 및 시각화
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib
matplotlib.use('Agg')  # GUI 없는 백엔드 사용 (멀티스레드 안전)
import matplotlib.pyplot as plt


def _ensure_2d(arr: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
    """배열을 2D 형태로 변환"""
    arr_np = np.asarray(arr)
    return arr_np.reshape(-1, 1) if arr_np.ndim == 1 else arr_np


def compute_metrics(y_true: Union[np.ndarray, pd.DataFrame, pd.Series], 
                   y_pred: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Dict[str, Any]:
    """
    예측 결과에 대한 평가 지표 계산
    
    Parameters:
    -----------
    y_true : array-like
        실제 값
    y_pred : array-like
        예측 값
        
    Returns:
    --------
    dict : 평가 지표 딕셔너리
    """
    yt = _ensure_2d(y_true)
    yp = _ensure_2d(y_pred)

    r2s = [r2_score(yt[:, j], yp[:, j]) for j in range(yt.shape[1])]
    rmses = [np.sqrt(mean_squared_error(yt[:, j], yp[:, j])) for j in range(yt.shape[1])]
    mapes = [mean_absolute_percentage_error(yt[:, j], yp[:, j]) * 100.0 for j in range(yt.shape[1])]

    return {
        "R2_mean": float(np.mean(r2s)),
        "RMSE_mean": float(np.mean(rmses)),
        "MAPE_mean(%)": float(np.mean(mapes)),
        "R2_by_target": r2s,
        "RMSE_by_target": rmses,
        "MAPE_by_target(%)": mapes,
    }


def fit_and_evaluate(model: Any, splits: Dict[str, tuple]) -> Dict[str, Dict[str, Any]]:
    """
    모델 학습 및 평가
    
    Parameters:
    -----------
    model : sklearn model
        학습할 모델
    splits : dict
        데이터 분할 딕셔너리 (train, valid, test)
        
    Returns:
    --------
    dict : 분할별 평가 지표
    """
    X_train, y_train = splits["train"]
    model.fit(X_train, y_train)

    out = {}
    for name, (X_, y_) in splits.items():
        pred = model.predict(X_)
        out[name] = compute_metrics(y_.to_numpy(), pred)
    return out


def plot_predictions(y_true: pd.DataFrame, 
                    y_pred: np.ndarray, 
                    title: str, 
                    n_points: int = 500) -> None:
    """
    예측 결과 시각화
    
    Parameters:
    -----------
    y_true : DataFrame
        실제 값
    y_pred : ndarray
        예측 값
    title : str
        그래프 제목
    n_points : int
        표시할 데이터 포인트 수
    """
    yt = y_true.copy()
    yp = _ensure_2d(y_pred)

    # 마지막 n_points만 표시
    yt = yt.iloc[-n_points:]
    yp = yp[-len(yt):, :]

    for j, col in enumerate(yt.columns):
        plt.figure(figsize=(12, 4))
        plt.plot(yt.index, yt[col].values, label="true", linewidth=1.5)
        plt.plot(yt.index, yp[:, j], label="pred", linewidth=1.5, alpha=0.8)
        plt.title(f"{title} | target={col}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_metric_table(result_by_model: Dict[str, Dict[str, Any]], 
                     split: str = "test") -> pd.DataFrame:
    """
    모델별 평가 지표 테이블 생성
    
    Parameters:
    -----------
    result_by_model : dict
        모델별 결과 딕셔너리
    split : str
        표시할 데이터 분할 (train/valid/test)
        
    Returns:
    --------
    DataFrame : 정렬된 평가 지표 테이블
    """
    rows = []
    for model_name, res in result_by_model.items():
        m = res[split]
        rows.append([model_name, m["R2_mean"], m["RMSE_mean"], m["MAPE_mean(%)"]])
    
    tbl = pd.DataFrame(rows, columns=["model", "R2_mean", "RMSE_mean", "MAPE_mean(%)"])
    return tbl.sort_values(by="RMSE_mean", ascending=True)
