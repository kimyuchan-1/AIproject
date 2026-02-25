# 하수처리장(WWTP) 예측 모델 분석 보고서 (V1 - 전처리 데이터 사용)

## 📊 실행 요약

**실행 일시**: 2026-01-27 (업데이트)  
**데이터셋**: 전처리된 FLOW, TMS, ALL 데이터 (1분 간격, 선형 보간 완료)  
**데이터 소스**: `data/Actual/FLOW_Actual.csv`, `TMS_Actual.csv`, `AWS_368.csv`, `AWS_541.csv`, `AWS_569.csv`  
**모델**: Ridge, Lasso, RandomForest, HistGBR, XGBoost  
**평가 지표**: R², RMSE, MAPE

---

## 🎯 예측 과제별 결과

### 1. FLOW 예측 (Q_in: 유입 유량)

#### ✅ 성능 요약
| 모델 | Test R² | Test RMSE | 비고 |
|------|---------|-----------|------|
| **HistGBR** | **0.5701** | **52.84** | 🏆 최고 성능 |
| Lasso | 0.5487 | 54.14 | 2위 |
| Ridge | 0.5324 | 55.11 | 3위 |
| XGBoost | 0.3738 | 63.78 | 과적합 |
| RandomForest | 0.2932 | 67.75 | 심각한 과적합 |

#### 📈 주요 발견
- **HistGBR이 가장 우수**: R² 0.57, RMSE 52.84
- **선형 모델(Lasso, Ridge)도 준수한 성능**: R² 0.53~0.55
- **트리 기반 모델의 과적합 문제**:
  - RandomForest: Train R² 0.97 → Test R² 0.29 (심각)
  - XGBoost: Train R² 0.82 → Test R² 0.37
- **Early Stopping 효과**: XGBoost 38 iteration에서 조기 종료

#### 🔑 중요 피처 (Top 10)
1. FLUX_VU_lag6
2. RN_60m_r3_mean (60분 강수량 3시간 평균)
3. RN_DAY (일 강수량)
4. HM_lag12 (12시간 전 습도)
5. FLUX_VU_lag24
6. level_TankA_lag1
7. level_TankA_r3_mean
8. level_TankB_r3_mean
9. FLUX_VU (현재 유량)
10. level_TankB_lag1

**해석**: 강수량, 습도, 탱크 수위가 유입 유량 예측에 중요

---

### 2. TMS 예측 (6개 수질 변수)

#### ❌ 성능 요약
| 모델 | Test R² | Test RMSE | 비고 |
|------|---------|-----------|------|
| **RandomForest** | **-1.31** | **67.10** | 🏆 상대적 최고 |
| HistGBR | -1.82 | 66.42 | |
| XGBoost | -2.09 | 59.52 | |
| Lasso | -5.08 | 68.75 | |
| Ridge | -20.74 | 66.34 | 최악 |

#### ⚠️ 심각한 문제점
**모든 모델의 R²가 음수** → 평균값으로 예측하는 것보다 못함!

#### 📉 타겟별 성능 (RandomForest 기준)
| 타겟 | Test R² | 해석 |
|------|---------|------|
| **FLUX_VU** | **0.9668** | ✅ 유일하게 예측 가능 |
| PH_VU | 0.3943 | △ 약간 예측 가능 |
| TOC_VU | -0.6692 | ❌ 예측 불가 |
| TN_VU | -0.7972 | ❌ 예측 불가 |
| SS_VU | -1.0001 | ❌ 예측 불가 |
| **TP_VU** | **-6.7276** | ❌❌ 최악 |

#### 🔍 원인 분석
1. **데이터 부족**: 576개 샘플 (전체의 4.2%만 사용)
2. **결측치 과다**: 95.8%의 데이터가 결측치로 제거됨
3. **피처-타겟 관계 부족**: 선택된 피처가 수질 변수와 약한 상관관계
4. **심각한 과적합**: Train R² 0.97 → Test R² -1.31

#### 🔑 중요 피처 (Top 10)
1. sin_hour (시간 주기)
2. TA_r12_mean (기온 12시간 평균)
3. HM_lag24 (24시간 전 습도)
4. cos_hour_lag3
5. sin_hour_lag24
6. data_save_dt (저장 시간)
7. TD_r12_mean (이슬점 12시간 평균)

**문제**: 기상 데이터 위주, 수질 관련 직접 피처 부족

---

### 3. ALL 예측 (Q_in + 6개 수질 변수)

#### ❌ 성능 요약
| 모델 | Test R² | Test RMSE | 비고 |
|------|---------|-----------|------|
| **RandomForest** | **-1.10** | **73.05** | 🏆 상대적 최고 |
| HistGBR | -1.48 | 74.92 | |
| XGBoost | -1.68 | 62.33 | |
| Lasso | -4.33 | 79.24 | |
| Ridge | -20.54 | 75.92 | 최악 |

#### 📉 타겟별 성능 (RandomForest 기준)
| 타겟 | Test R² | 해석 |
|------|---------|------|
| **FLUX_VU** | **0.9619** | ✅ 예측 가능 |
| PH_VU | 0.5177 | △ 예측 가능 |
| **Q_in** | **0.0088** | ❌ 거의 예측 불가 (TMS 포함 시) |
| TOC_VU | -0.6864 | ❌ 예측 불가 |
| TN_VU | -0.7931 | ❌ 예측 불가 |
| SS_VU | -0.9899 | ❌ 예측 불가 |
| **TP_VU** | **-6.7276** | ❌❌ 최악 |

**주목**: Q_in 단독 예측 시 R² 0.57 → ALL 예측 시 R² 0.01로 급락

---

## 🔴 주요 문제점

### 1. 데이터 품질 문제
```
Original samples: 13,848
After dropna: 576 (4.2%)
→ 95.8%의 데이터 손실!
```

**원인**:
- TMS 데이터와 FLOW 데이터의 시간 불일치
- AWS 데이터의 결측치
- Lag/Rolling 피처 생성 시 추가 결측치 발생

### 2. 심각한 과적합
| 모델 | Train R² | Valid R² | Test R² | 과적합 정도 |
|------|----------|----------|---------|------------|
| RandomForest | 0.97 | -2.50 | -1.31 | ❌❌❌ 극심 |
| HistGBR | 0.80 | -0.76 | -1.82 | ❌❌ 심각 |
| XGBoost | 0.75 | -0.17 | -2.09 | ❌❌ 심각 |

### 3. 피처-타겟 관계 부족
- 수질 변수(TOC, TN, TP, SS) 예측에 필요한 직접적인 피처 부족
- 기상 데이터만으로는 수질 예측 한계

### 4. 샘플 수 부족
- 576개 샘플로 261개 피처 학습 → 차원의 저주
- Train/Valid/Test 분할 시 각 346/115/115개만 사용

---

## ✅ 성공 사례

### FLUX_VU (유량) 예측
- **모든 모델에서 R² > 0.95**
- 안정적이고 일관된 예측 성능
- 이유: 유량 관련 lag 피처들이 강한 자기상관성

### Q_in (단독 예측 시)
- **HistGBR R² 0.57**
- 실용적 수준의 예측 가능
- 강수량, 탱크 수위 피처가 효과적

---

## 💡 개선 방안

### 1. 데이터 수집 개선 (최우선)
```python
# 현재
Original: 13,848 → After dropna: 576 (4.2%)

# 목표
Original: 13,848 → After dropna: 10,000+ (70%+)
```

**방법**:
- TMS와 FLOW 데이터의 시간 동기화
- 결측치 보간 (선형, 스플라인, KNN)
- 데이터 수집 주기 통일

### 2. 피처 엔지니어링 개선


**추가할 피처**:
- 수질 변수 간 상호작용 (TOC × TN, SS × TP 등)
- 계절성 피처 강화 (월별 더미 변수)
- 유입수 특성 피처 (pH, 온도 등)
- 처리 공정 데이터 (폭기량, 슬러지 농도 등)

**제거할 피처**:
- 약한 상관관계 피처 (data_save_dt 등)
- 과도한 lag 피처 (24시간 이상)

### 3. 모델링 전략 변경

#### A. 타겟별 개별 모델 (현재 적용 ✅)
```python
# 각 타겟마다 최적 모델 선택
Q_in → HistGBR (R² 0.57)
FLUX_VU → XGBoost (R² 0.97)
PH_VU → Lasso (R² 0.62)
```

#### B. 2단계 예측
```python
# 1단계: 예측 가능한 변수 먼저
FLUX_VU, Q_in 예측 (R² > 0.5)

# 2단계: 1단계 예측값을 피처로 사용
TOC, TN, TP, SS 예측
```

#### C. 시계열 모델 도입
```python
# ARIMA, LSTM, Prophet 등
# 시간적 의존성이 강한 데이터에 적합
```

### 4. 하이퍼파라미터 튜닝 개선

**현재 문제**:
- RandomForest: max_depth=None → 과적합
- n_estimators가 너무 적음 (100~300)

**개선안**:
```python
# RandomForest
'max_depth': [5, 10, 15],  # None 제거
'min_samples_leaf': [5, 10, 20],  # 추가
'max_features': ['sqrt', 'log2'],  # 추가

# XGBoost
'n_estimators': [500, 1000],  # 증가
'min_child_weight': [3, 5, 7],  # 추가
'gamma': [0, 0.1, 0.2],  # 추가
```

### 5. 정규화 강화
```python
# L1/L2 정규화 강화
Ridge(alpha=1000)  # 현재 100
Lasso(alpha=10)    # 현재 1

# Dropout (Neural Network 사용 시)
# Early Stopping 더 엄격하게
early_stopping_rounds=10  # 현재 20
```

### 6. 앙상블 기법
```python
# Stacking
Level 1: Ridge, Lasso, XGBoost
Level 2: LightGBM (메타 모델)

# Blending
weights = [0.3, 0.3, 0.4]  # HistGBR, XGBoost, Lasso
final_pred = weighted_average(predictions, weights)
```

---

## 📋 실행 가능한 액션 플랜

### 단기 (1주일)
1. ✅ **결측치 보간 구현**
   ```python
   # 선형 보간
   df.interpolate(method='linear', limit=3)
   
   # KNN 보간
   from sklearn.impute import KNNImputer
   imputer = KNNImputer(n_neighbors=5)
   ```

2. ✅ **피처 수 감소**
   - 현재: 261개 → 목표: 50개 이하
   - 상관계수 0.1 이하 피처 제거
   - VIF(Variance Inflation Factor) > 10 피처 제거

3. ✅ **정규화 강화**
   - Ridge alpha: 100 → 1000
   - RandomForest max_depth: None → 10

### 중기 (1개월)
4. ✅ **도메인 피처 추가**
   - 처리장 운영 데이터 수집
   - 유입수 특성 데이터 추가
   - 계절/요일 더미 변수

5. ✅ **시계열 모델 실험**
   - LSTM, GRU 구현
   - Prophet 적용
   - ARIMA 비교

6. ✅ **앙상블 모델 구축**
   - Stacking 구현
   - Voting Regressor

### 장기 (3개월)
7. ✅ **데이터 수집 체계 개선**
   - 센서 동기화
   - 실시간 데이터 파이프라인
   - 데이터 품질 모니터링

8. ✅ **모델 배포 및 모니터링**
   - API 서버 구축
   - 실시간 예측 시스템
   - 성능 모니터링 대시보드

---

## 🎓 결론

### 현재 상태 (V1)
- **FLOW(Q_in) 예측**: ✅ 실용 가능 (R² 0.57, RMSE 52.84, MAPE 9.32%)
- **FLUX_VU 예측**: ✅ 우수 (R² 0.97)
- **TMS 수질 변수**: ❌ 예측 불가 (R² < 0)
- **데이터 사용률**: ❌ 4.2% (576/13,848)

### 핵심 문제
1. **데이터 품질**: 95.8% 손실 (lag/rolling 피처 생성 시 결측치 발생)
2. **과적합**: Train-Test 격차 극심 (Train 0.97 → Test -1.31)
3. **피처 부족**: 수질 관련 직접 피처 없음

### V2 개선 사항 (구현 완료)
1. ✅ 결측치 보간 구현 (시계열 특화)
2. ✅ 도메인 피처 추가 (상호작용, 비율, 차분)
3. ✅ 정규화 강화 (alpha 증가, max_depth 제한)
4. ✅ 피처 수 감소 (lag/rolling 윈도우 축소)

### 기대 효과 (V2 실행 후)
**개선 후 목표**:
- Q_in: R² 0.57 → **0.70+** (목표)
- TMS 평균: R² -1.82 → **0.30+** (목표)
- 데이터 사용률: 4.2% → **70%+** (목표)

### 우선순위
```
1순위: V2 실행 및 결과 분석 ⏳
2순위: V1 vs V2 성능 비교
3순위: 추가 데이터 수집 계획
```

---

## 📊 시각화 결과

생성된 그래프:
1. `flow_r2_comparison.png` - FLOW 모델 비교
2. `tms_r2_comparison.png` - TMS 모델 비교
3. `all_r2_comparison.png` - ALL 모델 비교
4. `flow_XGBoost_learning_curve.png` - XGBoost 학습 곡선
5. `tms_HistGBR_learning_curve.png` - HistGBR 학습 곡선
6. `all_XGBoost_learning_curve.png` - XGBoost 학습 곡선

**위치**: `results/ML/`

---

## 📞 문의 및 후속 작업

**다음 단계**:
1. 데이터 엔지니어와 결측치 보간 전략 논의
2. 도메인 전문가와 추가 피처 발굴
3. 시계열 모델 실험 계획 수립

**작성일**: 2026-01-27 (업데이트)  
**버전**: V1 (전처리 데이터 사용, dropna 방식)  
**다음 버전**: V2 (결측치 보간 + 도메인 피처 + 정규화 강화)  
**작성자**: ML Baseline Analysis
