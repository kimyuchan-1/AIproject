# Improved ML Baseline

## 개선사항

### 1. ✅ 결측치 처리
- `make_supervised_dataset()` 함수에서 결측치가 있는 행 완전 제거
- 제거 전후 샘플 수 출력

### 2. ✅ StandardScaler 적용
- 모든 모델 학습 전 피처 스케일링
- Train set으로 fit, Valid/Test set은 transform만 수행

### 3. ✅ GridSearchCV 하이퍼파라미터 튜닝
**Ridge:**
- alpha: [0.1, 1.0, 10.0, 100.0]

**Lasso:**
- alpha: [0.001, 0.01, 0.1, 1.0]

**RandomForest:**
- n_estimators: [100, 200, 300]
- max_depth: [10, 20, None]
- min_samples_split: [2, 5]

**HistGradientBoosting (Early Stopping):**
- learning_rate: [0.01, 0.05, 0.1]
- max_iter: [500]
- max_depth: [5, 10, 20]
- early_stopping: True
- n_iter_no_change: 20

**XGBoost (Early Stopping):**
- n_estimators: [500]
- learning_rate: [0.01, 0.05, 0.1]
- max_depth: [3, 5, 7]
- subsample: [0.8, 1.0]
- early_stopping_rounds: 20

### 4. ✅ 피처 선택
- RandomForest로 피처 중요도 계산
- 상위 N개 피처만 선택 (Flow: 30, TMS: 40, All: 50)
- Lag/Rolling 피처 수 감소 (과적합 방지)

### 5. ✅ TimeSeriesSplit 교차 검증
- 시계열 데이터 특성 고려
- 3-fold TimeSeriesSplit 사용
- GridSearchCV 내부에서 자동 수행

### 6. ✅ XGBoost 추가
- 고성능 그래디언트 부스팅 모델
- GridSearch로 최적 파라미터 탐색

### 7. ✅ Validation Set & Early Stopping
- Train/Valid/Test 분할 (60/20/20)
- HistGBR과 XGBoost에 Early Stopping 적용
- Valid 성능이 20번 개선되지 않으면 학습 중단
- 과적합 자동 감지 및 경고

## 실행 방법

```bash
# 필요한 패키지 설치
pip install xgboost scikit-learn pandas numpy matplotlib

# 스크립트 실행
python notebook/ML/improved/v1/improved_baseline.py
```

## 예상 실행 시간
- Flow 예측: ~5-10분
- TMS 예측: ~10-15분  
- All 예측: ~15-20분
- **총 소요 시간: 약 30-45분** (GridSearch 때문)

## 출력 결과
각 모드별로:
1. 결측치 제거 통계
2. 선택된 상위 피처 목록
3. 각 모델의 최적 하이퍼파라미터
4. Train/Valid/Test R², RMSE, MAPE
5. 과적합 경고 (Train R² >> Valid R²인 경우)
6. Early Stopping 반복 횟수
7. 최종 성능 비교 테이블

## 시각화 결과 (자동 저장)
`results/ML/` 디렉토리에 자동 저장:

### 1. Learning Curves (학습 곡선)
- `flow_XGBoost_learning_curve.png`
- `flow_HistGBR_learning_curve.png`
- `tms_XGBoost_learning_curve.png`
- `tms_HistGBR_learning_curve.png`
- `all_XGBoost_learning_curve.png`
- `all_HistGBR_learning_curve.png`

**내용**: Train/Valid 성능 지표가 반복(iteration)에 따라 어떻게 변하는지 시각화
- X축: Iterations (학습 반복 횟수)
- Y축: RMSE 또는 Score
- 파란선: Train 성능
- 주황선: Valid 성능
- Early Stopping 지점 확인 가능

### 2. R² Comparison (모델 비교)
- `flow_r2_comparison.png`
- `tms_r2_comparison.png`
- `all_r2_comparison.png`

**내용**: 모든 모델의 Train/Valid/Test R² 점수를 막대 그래프로 비교
- 과적합 여부를 한눈에 확인 가능
- 최고 성능 모델 선택에 활용

## 기대 효과
- **성능 향상**: 스케일링 + 피처 선택 + 튜닝으로 R² 개선
- **과적합 방지**: Early Stopping + Validation Set + 교차 검증
- **안정성**: 결측치 완전 제거로 예측 안정성 향상
- **효율성**: Early Stopping으로 불필요한 학습 시간 단축
