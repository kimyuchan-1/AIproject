import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

mode = "ss" # flow, toc, ss, tn, tp, ph, flux

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
BASE_DIR = Path("C:/project_WWTP/python")
RESULTS_DIR = BASE_DIR / "data" / "output" / "save"

# 예측 결과 로드
df = pd.read_csv(RESULTS_DIR / f"{mode}_predictions.csv")

print("="*70)
print("예측 성능 진단 보고서")
print("="*70)

# 기본 통계
print("\n1. 기본 통계량")
print("-"*70)
print(df.describe())

# 에러 계산
df['error'] = df['predicted'] - df['actual']
df['abs_error'] = df['error'].abs()
df['pct_error'] = (df['abs_error'] / df['actual'].abs()) * 100

# 성능 지표
mae = df['abs_error'].mean()
rmse = np.sqrt((df['error']**2).mean())
mape = df['pct_error'].mean()
r2 = 1 - (df['error']**2).sum() / ((df['actual'] - df['actual'].mean())**2).sum()

print(f"\n2. 성능 지표")
print("-"*70)
print(f"MAE (평균 절대 오차):    {mae:.2f} m³/h")
print(f"RMSE (평균 제곱근 오차): {rmse:.2f} m³/h")
print(f"MAPE (평균 절대 비율 오차): {mape:.2f}%")
print(f"R² (결정계수):          {r2:.4f}")

# 에러 분석
print(f"\n3. 에러 분석")
print("-"*70)
print(f"평균 에러 (bias):        {df['error'].mean():.2f} m³/h")
print(f"에러 표준편차:           {df['error'].std():.2f} m³/h")
print(f"최대 과대예측:           {df['error'].max():.2f} m³/h")
print(f"최대 과소예측:           {df['error'].min():.2f} m³/h")

# 예측 범위 vs 실제 범위
print(f"\n4. 값의 범위")
print("-"*70)
print(f"실제값 범위:    [{df['actual'].min():.2f}, {df['actual'].max():.2f}] m³/h")
print(f"예측값 범위:    [{df['predicted'].min():.2f}, {df['predicted'].max():.2f}] m³/h")
print(f"실제값 평균:    {df['actual'].mean():.2f} m³/h")
print(f"예측값 평균:    {df['predicted'].mean():.2f} m³/h")
print(f"실제값 표준편차: {df['actual'].std():.2f} m³/h")
print(f"예측값 표준편차: {df['predicted'].std():.2f} m³/h")

# 분위수별 에러
print(f"\n5. 실제값 구간별 성능")
print("-"*70)
# 실제값 중복이 많으면 qcut 경계가 중복될 수 있어 duplicates='drop'으로 처리
quartile_binned = pd.qcut(df['actual'], q=4, duplicates='drop')
n_bins = len(quartile_binned.cat.categories)

quartile_label_pool = ['Q1(낮음)', 'Q2(중하)', 'Q3(중상)', 'Q4(높음)']
quartile_labels = quartile_label_pool[:n_bins]
df['actual_quartile'] = quartile_binned.cat.rename_categories(quartile_labels)

quartile_performance = df.groupby('actual_quartile', observed=False).agg({
    'actual': ['mean', 'count'],
    'abs_error': 'mean',
    'pct_error': 'mean'
}).round(2)
print(quartile_performance)

# 시각화
fig = plt.figure(figsize=(20, 12))

# 1. 실제 vs 예측 산점도
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(df['actual'], df['predicted'], alpha=0.5, s=20)
min_val = min(df['actual'].min(), df['predicted'].min())
max_val = max(df['actual'].max(), df['predicted'].max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('실제값 (m³/h)', fontsize=12)
ax1.set_ylabel('예측값 (m³/h)', fontsize=12)
ax1.set_title(f'실제 vs 예측 (R²={r2:.4f})', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 시계열 플롯 (전체)
ax2 = plt.subplot(2, 3, 2)
ax2.plot(df['actual'].values, label='실제값', alpha=0.7, linewidth=1.5)
ax2.plot(df['predicted'].values, label='예측값', alpha=0.7, linewidth=1.5)
ax2.set_xlabel('시간 (샘플)', fontsize=12)
ax2.set_ylabel('Flow (m³/h)', fontsize=12)
ax2.set_title('시계열 비교 (전체)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 시계열 플롯 (첫 200개)
ax3 = plt.subplot(2, 3, 3)
n_samples = min(200, len(df))
ax3.plot(df['actual'].values[:n_samples], label='실제값', alpha=0.7, linewidth=2)
ax3.plot(df['predicted'].values[:n_samples], label='예측값', alpha=0.7, linewidth=2)
ax3.set_xlabel('시간 (샘플)', fontsize=12)
ax3.set_ylabel('Flow (m³/h)', fontsize=12)
ax3.set_title(f'시계열 비교 (첫 {n_samples}개)', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 에러 분포 (히스토그램)
ax4 = plt.subplot(2, 3, 4)
ax4.hist(df['error'], bins=50, alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax4.axvline(x=df['error'].mean(), color='green', linestyle='--', linewidth=2,
            label=f"Bias={df['error'].mean():.2f}")
ax4.set_xlabel('예측 에러 (m³/h)', fontsize=12)
ax4.set_ylabel('빈도', fontsize=12)
ax4.set_title('에러 분포', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. 절대 에러 vs 실제값
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(df['actual'], df['abs_error'], alpha=0.5, s=20)
ax5.set_xlabel('실제값 (m³/h)', fontsize=12)
ax5.set_ylabel('절대 에러 (m³/h)', fontsize=12)
ax5.set_title('실제값 구간별 에러', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. 분위수별 성능 박스플롯
ax6 = plt.subplot(2, 3, 6)
df.boxplot(column='abs_error', by='actual_quartile', ax=ax6)
ax6.set_xlabel('실제값 구간', fontsize=12)
ax6.set_ylabel('절대 에러 (m³/h)', fontsize=12)
ax6.set_title('구간별 에러 분포', fontsize=14, fontweight='bold')
plt.suptitle('')  # 기본 제목 제거

plt.tight_layout()
plt.savefig(RESULTS_DIR / f"{mode}_diagnosis.png", dpi=300, bbox_inches='tight')
print(f"\n[OK] 진단 플롯 저장: {RESULTS_DIR / f'{mode}_diagnosis.png'}")

# 문제점 식별
print("\n" + "="*70)
print("6. 진단 결과 및 문제점")
print("="*70)

issues = []

# 1. R² 낮음
if r2 < 0.7:
    issues.append(f"- R²={r2:.4f} < 0.7: 모델이 분산의 {r2*100:.1f}%만 설명")

# 2. Bias 확인
if abs(df['error'].mean()) > df['actual'].std() * 0.1:
    if df['error'].mean() > 0:
        issues.append(f"- 과대예측 편향: 평균 {df['error'].mean():.2f} m³/h 높게 예측")
    else:
        issues.append(f"- 과소예측 편향: 평균 {abs(df['error'].mean()):.2f} m³/h 낮게 예측")

# 3. 예측 범위 축소
actual_range = df['actual'].max() - df['actual'].min()
pred_range = df['predicted'].max() - df['predicted'].min()
if pred_range < actual_range * 0.7:
    issues.append(f"- 예측 범위 축소: 예측값이 실제값 변동의 {pred_range/actual_range*100:.1f}%만 포착")

# 4. MAPE 높음
if mape > 20:
    issues.append(f"- MAPE={mape:.1f}% > 20%: 평균적으로 {mape:.1f}% 오차 발생")

# 5. 구간별 성능 차이
quartile_errors = df.groupby('actual_quartile', observed=False)['abs_error'].mean()
quartile_min = quartile_errors.min()
if quartile_min > 0 and quartile_errors.max() / quartile_min > 2:
    issues.append(f"- 구간별 성능 불균형: 특정 구간에서 에러가 {quartile_errors.max() / quartile_min:.1f}배 높음")

if issues:
    print("\n발견된 문제점:")
    for issue in issues:
        print(issue)
else:
    print("\n특별한 문제점 없음 (성능 양호)")

print("\n" + "="*70)
print("진단 완료")
print("="*70)
