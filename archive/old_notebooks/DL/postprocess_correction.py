"""
후처리 보정 스크립트
예측값을 구간별로 보정하여 정확도 향상
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class PostProcessCorrector:
    """
    구간별 선형 보정 모델
    """
    def __init__(self, n_bins=5):
        self.n_bins = n_bins
        self.correctors = {}
        self.bin_edges = None

    def fit(self, predictions, actuals):
        """
        검증 세트에서 보정 모델 학습

        Args:
            predictions: 모델 예측값
            actuals: 실제값
        """
        # 실제값 기준으로 구간 나누기
        self.bin_edges = np.percentile(actuals,
                                       np.linspace(0, 100, self.n_bins + 1))

        # 각 구간별 보정 모델 학습
        for i in range(self.n_bins):
            lower = self.bin_edges[i]
            upper = self.bin_edges[i + 1]

            # 구간 내 데이터 선택
            mask = (actuals >= lower) & (actuals < upper)
            if i == self.n_bins - 1:  # 마지막 구간은 upper 포함
                mask = (actuals >= lower) & (actuals <= upper)

            if mask.sum() > 10:  # 최소 10개 샘플 필요
                X_bin = predictions[mask].reshape(-1, 1)
                y_bin = actuals[mask]

                # 선형 보정 모델
                corrector = LinearRegression()
                corrector.fit(X_bin, y_bin)
                self.correctors[i] = corrector

                print(f"구간 {i+1}: [{lower:.1f}, {upper:.1f}] "
                      f"- 샘플 {mask.sum()}개, "
                      f"보정 계수={corrector.coef_[0]:.3f}")
            else:
                print(f"구간 {i+1}: 샘플 부족 (보정 생략)")

    def transform(self, predictions):
        """
        예측값 보정

        Args:
            predictions: 모델 예측값

        Returns:
            보정된 예측값
        """
        corrected = predictions.copy()

        for i in range(self.n_bins):
            if i not in self.correctors:
                continue

            lower = self.bin_edges[i]
            upper = self.bin_edges[i + 1]

            # 구간 내 예측값 선택
            mask = (predictions >= lower) & (predictions < upper)
            if i == self.n_bins - 1:
                mask = (predictions >= lower) & (predictions <= upper)

            if mask.sum() > 0:
                X_bin = predictions[mask].reshape(-1, 1)
                corrected[mask] = self.correctors[i].predict(X_bin)

        return corrected


# 사용 예시:
"""
# 1. 검증 세트에서 보정 모델 학습
corrector = PostProcessCorrector(n_bins=5)
corrector.fit(val_predictions, val_actuals)

# 2. 테스트 세트 예측값 보정
test_predictions_corrected = corrector.transform(test_predictions)

# 3. 성능 개선 확인
from sklearn.metrics import r2_score, mean_absolute_error

print("보정 전:")
print(f"  R² = {r2_score(test_actuals, test_predictions):.4f}")
print(f"  MAE = {mean_absolute_error(test_actuals, test_predictions):.2f}")

print("보정 후:")
print(f"  R² = {r2_score(test_actuals, test_predictions_corrected):.4f}")
print(f"  MAE = {mean_absolute_error(test_actuals, test_predictions_corrected):.2f}")
"""
