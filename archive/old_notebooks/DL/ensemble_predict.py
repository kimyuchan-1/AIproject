"""
앙상블 예측 스크립트
여러 모델의 예측을 평균하여 안정성 향상
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

def ensemble_predict(models, test_loader, device="cpu"):
    """
    여러 모델의 예측을 평균

    Args:
        models: 학습된 모델 리스트
        test_loader: 테스트 데이터 로더
        device: 디바이스

    Returns:
        평균 예측값
    """
    all_predictions = []

    for model in models:
        model.eval()
        predictions = []

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device).float()
                preds = model(X)

                if preds.dim() > 2:
                    preds = preds.squeeze(1)

                predictions.append(preds.cpu().numpy())

        all_predictions.append(np.concatenate(predictions))

    # 평균 예측
    ensemble_pred = np.mean(all_predictions, axis=0)

    return ensemble_pred

def train_ensemble(num_models=5, seed_start=42):
    """
    여러 개의 모델을 다른 seed로 학습

    Returns:
        학습된 모델 리스트
    """
    models = []

    for i in range(num_models):
        seed = seed_start + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n{'='*70}")
        print(f"앙상블 모델 {i+1}/{num_models} 학습 (seed={seed})")
        print(f"{'='*70}")

        # 모델 학습 (main() 함수 호출)
        model, hist, metrics = main()
        models.append(model)

        print(f"\n모델 {i+1} 성능: R²={metrics['r2']:.4f}")

    return models

# 사용 예시:
# models = train_ensemble(num_models=3)
# ensemble_pred = ensemble_predict(models, test_dl)
