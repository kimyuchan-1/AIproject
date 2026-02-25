"""
모델 평가 모듈
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from typing import Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    평가 지표 계산
    
    Parameters:
    -----------
    y_true : ndarray
        실제 값
    y_pred : ndarray
        예측 값
        
    Returns:
    --------
    dict : 평가 지표
    """
    # 1차원으로 변환
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE 계산 (0으로 나누기 방지)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
    else:
        mape = np.nan
    
    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "MAPE(%)": float(mape)
    }


def evaluate_model(predictions: Dict[str, tuple]) -> Dict[str, Dict[str, float]]:
    """
    모델 평가
    
    Parameters:
    -----------
    predictions : dict
        {"train": (y_true, y_pred), "valid": (y_true, y_pred), "test": (y_true, y_pred)}
        
    Returns:
    --------
    dict : 분할별 평가 지표
    """
    results = {}
    
    for split_name, (y_true, y_pred) in predictions.items():
        results[split_name] = compute_metrics(y_true, y_pred)
    
    return results


def print_metrics(results: Dict[str, Dict[str, float]]):
    """
    평가 지표 출력
    
    Parameters:
    -----------
    results : dict
        분할별 평가 지표
    """
    print(f"\n{'='*60}")
    print("평가 결과")
    print(f"{'='*60}")
    
    for split_name, metrics in results.items():
        print(f"\n{split_name.upper()} 세트:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    print(f"{'='*60}\n")


def save_metrics(results: Dict[str, Dict[str, float]], save_path: Path):
    """
    평가 지표 저장
    
    Parameters:
    -----------
    results : dict
        분할별 평가 지표
    save_path : Path
        저장 경로
    """
    import json
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"평가 지표 저장 완료: {save_path}")


def save_predictions(y_true: np.ndarray, 
                    y_pred: np.ndarray,
                    save_path: Path,
                    index: pd.Index = None):
    """
    예측값 저장
    
    Parameters:
    -----------
    y_true : ndarray
        실제 값
    y_pred : ndarray
        예측 값
    save_path : Path
        저장 경로
    index : Index, optional
        시간 인덱스
    """
    df = pd.DataFrame({
        "y_true": y_true.flatten(),
        "y_pred": y_pred.flatten()
    }, index=index)
    
    df.to_csv(save_path)
    print(f"예측값 저장 완료: {save_path}")


def plot_training_history(train_losses: list, 
                         valid_losses: list,
                         save_path: Path):
    """
    학습 히스토리 플롯
    
    Parameters:
    -----------
    train_losses : list
        학습 손실 리스트
    valid_losses : list
        검증 손실 리스트
    save_path : Path
        저장 경로
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"학습 히스토리 플롯 저장 완료: {save_path}")


def plot_predictions(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    save_path: Path,
                    title: str = "Predictions",
                    n_points: int = 500):
    """
    예측 결과 플롯
    
    Parameters:
    -----------
    y_true : ndarray
        실제 값
    y_pred : ndarray
        예측 값
    save_path : Path
        저장 경로
    title : str
        그래프 제목
    n_points : int
        표시할 데이터 포인트 수
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # 마지막 n_points만 표시
    if len(y_true) > n_points:
        y_true = y_true[-n_points:]
        y_pred = y_pred[-n_points:]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 시계열 플롯
    axes[0].plot(y_true, label="Actual", alpha=0.7)
    axes[0].plot(y_pred, label="Predicted", alpha=0.7)
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Value")
    axes[0].set_title(f"{title} - Time Series")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 산점도
    axes[1].scatter(y_true, y_pred, alpha=0.5)
    axes[1].plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 "r--", lw=2)
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"{title} - Scatter Plot")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"예측 플롯 저장 완료: {save_path}")
