"""
시각화 모듈
Learning Curve, R² 비교 등
"""

import os
from typing import Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 백엔드 사용 (멀티스레드 안전)
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor


def _extract_xgb_estimator(model: Any) -> Optional[xgb.XGBRegressor]:
    """XGBoost 모델에서 estimator 추출"""
    if isinstance(model, xgb.XGBRegressor):
        return model
    elif isinstance(model, MultiOutputRegressor):
        return model.estimators_[0]
    elif isinstance(model, GridSearchCV):
        return model.best_estimator_
    return None


def _extract_histgbr_estimator(model: Any) -> Optional[Any]:
    """HistGradientBoosting 모델에서 estimator 추출"""
    if isinstance(model, MultiOutputRegressor):
        return model.estimators_[0]
    elif isinstance(model, GridSearchCV):
        return model.best_estimator_
    return model


def _plot_xgb_learning_curve(estimator: xgb.XGBRegressor, 
                             model_name: str, 
                             mode: str, 
                             save_path: str) -> bool:
    """XGBoost 학습 곡선 시각화"""
    if not hasattr(estimator, 'evals_result'):
        return False
    
    results = estimator.evals_result()
    if not results or 'validation_0' not in results:
        return False
    
    train_metric = results['validation_0']['rmse']
    valid_metric = results.get('validation_1', {}).get('rmse')
    
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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  📊 Learning curve saved: {save_path}")
    plt.close()
    return True


def _plot_histgbr_learning_curve(estimator: Any, 
                                 model_name: str, 
                                 mode: str, 
                                 save_path: str) -> bool:
    """HistGradientBoosting 학습 곡선 시각화"""
    if not hasattr(estimator, 'train_score_'):
        return False
    
    train_scores = estimator.train_score_
    valid_scores = getattr(estimator, 'validation_score_', None)
    
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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  📊 Learning curve saved: {save_path}")
    plt.close()
    return True


def plot_learning_curve(model: Any, 
                       model_name: str, 
                       mode: str, 
                       save_dir: str = "results/ML") -> bool:
    """
    학습 곡선 시각화 (XGBoost, HistGBR)
    
    Parameters:
    -----------
    model : 학습된 모델
    model_name : str
        모델 이름
    mode : str
        예측 모드 (flow/tms/all)
    save_dir : str
        저장 디렉토리
        
    Returns:
    --------
    bool : 시각화 성공 여부
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{mode}_{model_name}_learning_curve.png')
    
    # XGBoost
    if 'XGB' in model_name:
        estimator = _extract_xgb_estimator(model)
        if estimator:
            return _plot_xgb_learning_curve(estimator, model_name, mode, save_path)
    
    # HistGradientBoosting
    elif 'HistGBR' in model_name:
        estimator = _extract_histgbr_estimator(model)
        if estimator:
            return _plot_histgbr_learning_curve(estimator, model_name, mode, save_path)
    
    return False


def plot_r2_comparison(results: Dict[str, Dict[str, Any]], 
                      mode: str, 
                      save_dir: str = "results/ML") -> None:
    """
    모든 모델의 R² 비교 시각화
    
    Parameters:
    -----------
    results : dict
        모델별 결과 딕셔너리
    mode : str
        예측 모드
    save_dir : str
        저장 디렉토리
    """
    os.makedirs(save_dir, exist_ok=True)
    
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
    ax.set_title(f'R² Comparison - {mode.upper()}', fontsize=14, fontweight='bold')
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
