import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

mode = "flux" # flow, toc, ss, tn, tp, ph, flux

# 데이터 로드
df = pd.read_csv(f"C:/project_WWTP/python/data/output/save/{mode}_predictions.csv")

print("="*70)
print("예측 결과 상세 분석")
print("="*70)

# 기본 통계
print("\n1. 기본 통계")
print("-"*70)
print(f"샘플 수: {len(df)}")
print(f"\n실제값:")
print(f"  평균: {df['actual'].mean():.2f} m³/h")
print(f"  범위: [{df['actual'].min():.2f}, {df['actual'].max():.2f}] m³/h")
print(f"  표준편차: {df['actual'].std():.2f} m³/h")
print(f"\n예측값:")
print(f"  평균: {df['predicted'].mean():.2f} m³/h")
print(f"  범위: [{df['predicted'].min():.2f}, {df['predicted'].max():.2f}] m³/h")
print(f"  표준편차: {df['predicted'].std():.2f} m³/h")

# 에러 분석
df['error'] = df['predicted'] - df['actual']
df['abs_error'] = df['error'].abs()
df['pct_error'] = (df['abs_error'] / df['actual'].abs()) * 100

print(f"\n2. 에러 통계")
print("-"*70)
print(f"MAE (평균 절대 오차): {df['abs_error'].mean():.2f} m³/h")
print(f"RMSE: {np.sqrt((df['error']**2).mean()):.2f} m³/h")
print(f"MAPE: {df['pct_error'].mean():.2f}%")
print(f"R²: {1 - (df['error']**2).sum() / ((df['actual'] - df['actual'].mean())**2).sum():.4f}")

print(f"\n편향 (Bias):")
print(f"  평균 에러: {df['error'].mean():.2f} m³/h")
if df['error'].mean() > 0:
    print(f"  → 모델이 평균적으로 과대예측")
else:
    print(f"  → 모델이 평균적으로 과소예측")

# 극값 분석
print(f"\n3. 극값 예측 성능")
print("-"*70)
low_threshold = df['actual'].quantile(0.25)
high_threshold = df['actual'].quantile(0.75)

low = df[df['actual'] <= low_threshold]
mid = df[(df['actual'] > low_threshold) & (df['actual'] < high_threshold)]
high = df[df['actual'] >= high_threshold]

print(f"낮은 예측값 (<{low_threshold:.1f}):")
print(f"  샘플 수: {len(low)}")
print(f"  평균 실제값: {low['actual'].mean():.2f}")
print(f"  평균 예측값: {low['predicted'].mean():.2f}")
print(f"  MAE: {low['abs_error'].mean():.2f}")
print(f"  MAPE: {low['pct_error'].mean():.2f}%")

print(f"\n중간 예측값 ({low_threshold:.1f}-{high_threshold:.1f}):")
print(f"  샘플 수: {len(mid)}")
print(f"  평균 실제값: {mid['actual'].mean():.2f}")
print(f"  평균 예측값: {mid['predicted'].mean():.2f}")
print(f"  MAE: {mid['abs_error'].mean():.2f}")
print(f"  MAPE: {mid['pct_error'].mean():.2f}%")

print(f"\n높은 예측값 (>{high_threshold:.1f}):")
print(f"  샘플 수: {len(high)}")
print(f"  평균 실제값: {high['actual'].mean():.2f}")
print(f"  평균 예측값: {high['predicted'].mean():.2f}")
print(f"  MAE: {high['abs_error'].mean():.2f}")
print(f"  MAPE: {high['pct_error'].mean():.2f}%")

# 최악의 예측 사례
print(f"\n4. 최악의 예측 TOP 10")
print("-"*70)
worst = df.nlargest(10, 'abs_error')[['actual', 'predicted', 'abs_error', 'pct_error']]
for idx, row in worst.iterrows():
    print(f"  [{idx+1:3d}] 실제={row['actual']:6.1f}, 예측={row['predicted']:6.1f}, "
          f"오차={row['abs_error']:5.1f} ({row['pct_error']:5.1f}%)")

# 시간대별 패턴
print(f"\n5. 시간대별 패턴 (100 샘플 단위)")
print("-"*70)
chunk_size = 100
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:min(i+chunk_size, len(df))]
    print(f"  [{i:3d}-{min(i+chunk_size-1, len(df)-1):3d}] "
          f"실제 평균={chunk['actual'].mean():6.1f}, "
          f"예측 평균={chunk['predicted'].mean():6.1f}, "
          f"MAE={chunk['abs_error'].mean():5.1f}")

print("\n" + "="*70)
print("분석 완료")
print("="*70)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. 시계열 비교
ax1 = axes[0, 0]
ax1.plot(df.index, df['actual'], label='실제값', alpha=0.7, linewidth=1.5)
ax1.plot(df.index, df['predicted'], label='예측값', alpha=0.7, linewidth=1.5)
ax1.set_xlabel('시간 (샘플 인덱스)')
ax1.set_ylabel('종속 변수')
ax1.set_title('시계열 비교: 실제 vs 예측')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 산점도
ax2 = axes[0, 1]
ax2.scatter(df['actual'], df['predicted'], alpha=0.5, s=20)
min_val = min(df['actual'].min(), df['predicted'].min())
max_val = max(df['actual'].max(), df['predicted'].max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('실제값')
ax2.set_ylabel('예측값')
ax2.set_title('실제 vs 예측 산점도')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 에러 시계열
ax3 = axes[1, 0]
ax3.plot(df.index, df['error'], alpha=0.7, linewidth=1)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
ax3.axhline(y=df['error'].mean(), color='g', linestyle='--', linewidth=2,
            label=f"Bias={df['error'].mean():.1f}")
ax3.fill_between(df.index, 0, df['error'], alpha=0.3)
ax3.set_xlabel('시간 (샘플 인덱스)')
ax3.set_ylabel('예측 에러 (m³/h)')
ax3.set_title('예측 에러 시계열')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 구간별 MAPE
ax4 = axes[1, 1]
bins = sorted(set([df['actual'].min(), low_threshold, high_threshold, df['actual'].max()]))
all_labels = ['낮음', '중간', '높음']
labels = all_labels[-len(bins)+1:]
df['category'] = pd.cut(df['actual'], bins=bins, labels=labels, include_lowest=True)
category_mape = df.groupby('category')['pct_error'].mean()
bars = ax4.bar(range(len(category_mape)), category_mape.values)
ax4.set_xticks(range(len(category_mape)))
ax4.set_xticklabels(category_mape.index)
ax4.set_ylabel('MAPE (%)')
ax4.set_title('구간별 예측 정확도 (MAPE)')
ax4.grid(True, alpha=0.3, axis='y')

# 색상 코딩 (MAPE 높을수록 빨간색)
for i, bar in enumerate(bars):
    mape_val = category_mape.values[i]
    if mape_val > 20:
        bar.set_color('red')
    elif mape_val > 10:
        bar.set_color('orange')
    else:
        bar.set_color('green')

plt.tight_layout()
plt.savefig(f"C:/project_WWTP/python/results/DL/prediction_analysis_{mode}.png", dpi=300, bbox_inches='tight')
print(f"\n✓ 분석 플롯 저장: C:/project_WWTP/python/results/DL/prediction_analysis_{mode}.png")
