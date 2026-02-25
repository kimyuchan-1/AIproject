# 하수처리장 유입량 및 수질 예측 AI 서비스

> WWTP (Wastewater Treatment Plant) Prediction & Diagnosis AI Service

---

## Slide 1. 프로젝트 개요

### 배경

- 하수처리장은 유입 하수의 수질 변동에 따라 처리 공정을 실시간 조정해야 함
- 수질 기준 초과 시 **법적 제재** 및 **환경 오염** 발생
- 기존 운영: 과거 경험에 의존 → **선제적 대응 불가**

### 목표

- 하수 **유입량(Flow)** 및 **수질 6개 지표(TMS)** 를 12시간 전에 예측
- 이상 징후 실시간 탐지 및 알림
- REST API 서비스로 기존 운영 시스템과 연동

### 핵심 가치

```
실시간 데이터 수집 → AI 예측 (12시간) → 선제적 공정 제어 → 방류 기준 준수
```

---

## Slide 2. 시스템 아키텍처

### 전체 구성

```
┌─────────────────────────────────────────────────────────┐
│                    Data Sources                         │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐   │
│  │ 유입유량  │  │ 수질(TMS)│  │ 기상관측소(AWS) ×3  │   │
│  │ 1분 단위  │  │ 1분 단위 │  │ 1.0km / 1.2km /     │   │
│  │ Tank A/B  │  │ 6개 지표 │  │ 4.6km 거리          │   │
│  └────┬─────┘  └────┬─────┘  └──────────┬──────────┘   │
│       └──────────────┼──────────────────┘               │
│                      ▼                                  │
│  ┌───────────────────────────────────────────┐          │
│  │          전처리 & 특성 엔지니어링          │          │
│  │  리샘플링(30분) · 결측치 보간 · 이상치 제거│          │
│  │  40+ 도메인 특화 피처 생성                 │          │
│  └──────────────────┬────────────────────────┘          │
│                     ▼                                   │
│  ┌───────────────────────────────────────────┐          │
│  │          LSTM + Attention 모델 ×7          │          │
│  │  Flow(1) + TMS(6): TOC·SS·TN·TP·FLUX·PH  │          │
│  └──────────────────┬────────────────────────┘          │
│                     ▼                                   │
│  ┌───────────────────────────────────────────┐          │
│  │          FastAPI REST API Server           │          │
│  │  /predict/flow  ·  /predict/tms           │          │
│  │  /health  ·  /ready                       │          │
│  └──────────────────┬────────────────────────┘          │
│                     ▼                                   │
│  ┌───────────────────────────────────────────┐          │
│  │       Frontend Dashboard (연동)            │          │
│  │  12시간 예측 시각화 · 이상 탐지 알림       │          │
│  └───────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 3. 기술 스택

| 영역 | 기술 | 비고 |
|------|------|------|
| **딥러닝 프레임워크** | PyTorch | LSTM + Multi-head Attention |
| **ML 베이스라인** | scikit-learn, XGBoost | 초기 벤치마킹용 |
| **하이퍼파라미터 최적화** | Optuna | 베이지안 최적화 |
| **API 서버** | FastAPI + Uvicorn | RESTful 예측 서비스 |
| **데이터 처리** | Pandas, NumPy, SciPy | 전처리 파이프라인 |
| **시각화** | Matplotlib | EDA 및 결과 분석 |
| **형상 관리** | Git / GitHub | 브랜치 전략 (main, data) |
| **개발 환경** | Python 3.10, CUDA GPU | 서버 컴퓨터 학습 |

---

## Slide 4. 데이터 소개

### 수집 데이터

| 데이터 | 기간 | 해상도 | 주요 변수 |
|--------|------|--------|-----------|
| **유입유량** | 2025.09 ~ 2025.12 | 1분 | flow_TankA/B, level_TankA/B |
| **수질(TMS)** | 2024.08 ~ 2025.09 | 1분 | TOC, SS, TN, TP, PH, FLUX |
| **기상(AWS)** | 2024.08 ~ 2026.01 | 1분 | 기온, 강수량(15분/1h/12h/일), 습도, 이슬점 |

### 데이터 특성

- **유입유량**: 시간대·요일별 강한 주기성 (hour×weekday η² = 0.34~0.44)
- **수질 PH**: 월별·계절별 주기성 (month η² = 0.52, iso_week η² = 0.61)
- **기상 데이터**: 3개 관측소 × 거리 가중 (1.02km, 1.24km, 4.61km)
- **수질 지표 간 상관**: 낮음 (|r| < 0.2) → 독립 모델 설계 근거

<!-- 삽입 이미지: results/correlation/flow.png, results/correlation/tms.png -->
<!-- 삽입 이미지: results/timeseries/flow_plot.png, results/timeseries/tms_plot.png -->

---

## Slide 5. 데이터 전처리

### 전처리 파이프라인

```
원본 데이터 (1분)
    │
    ├─ 1) 리샘플링: 1분 → 30분 (mean / sum)
    │
    ├─ 2) 이상치 제거
    │     ├─ 도메인 기반: 방류 허용 기준 × 2 초과 시 제거
    │     └─ 통계 기반: Z-score / IQR
    │
    ├─ 3) 결측치 보간 (구간별 차등 전략)
    │     ├─ 단기 (1~3시간): Forward Fill
    │     ├─ 중기 (4~12시간): EWMA (span=6)
    │     └─ 장기 (12시간+): Long-span EWMA (span=24)
    │
    └─ 4) 정규화: StandardScaler (타겟별 독립)
```

### 전처리 전후 비교

<!-- 삽입 이미지: results/preprocess/FLOW_before_after.png -->
<!-- 삽입 이미지: results/preprocess/TMS_before_after.png -->

**핵심 설계 의도**: 장기 결측을 NaN으로 유지하면 학습 데이터 손실 → Long-span EWMA로 데이터 보존

---

## Slide 6. 특성 엔지니어링 (40+ 피처)

### 도메인 특화 피처 설계

| 카테고리 | 생성 피처 | 예시 |
|----------|-----------|------|
| **강수 피처** | 선행강우지수(ARI), 건습기 플래그, 강수 변화량 | ARI_decay_0.85 (24h 누적) |
| **관측소 통합** | 3개 AWS 공간 통계 (mean/max/std) | rain_60m_mean_station |
| **기상 피처** | 증기압차(VPD), 기온-습도 상호작용, 이슬점 편차 | VPD, TA_HM_interaction |
| **공정 피처** | 수위 합/차, 수위 lag (1~36h), rolling 통계 | level_sum_lag_6h |
| **시간 피처** | sin/cos 인코딩, hour×weekday 교호작용, 계절 | hour_sin, weekday_hour_interaction |
| **타겟 Lag** | 과거값 lag, rolling mean/std, 차분, EWMA | target_lag_24, target_rolling_mean_12 |

### 데이터 누수(Leakage) 방지

```python
# 모든 타겟 lag 피처에 shift(1) 적용 → 미래 정보 차단
df[f'{col}_lag_{k}'] = df[col].shift(k + 1)  # +1로 안전 마진 확보

# 예측 시 원본 타겟 컬럼 제거
safe_features = [f for f in features if f not in raw_target_columns]
```

### 피처 선택: Walk-Forward Validation

- 시간순 분할로 과적합 방지
- 누적 중요도 기반 최소 충분 피처셋 선택
- 타겟별 독립 선택 → 각 모델 최적 피처 확보

---

## Slide 7. 모델 아키텍처

### LSTM + Multi-head Attention

```
Input (batch, seq_len=48, n_features)
    │
    ▼
┌──────────────────────────────┐
│  LSTM (2~4 layers)           │
│  hidden_size: 64~128         │
│  dropout: 0.3                │
│  + Layer Normalization       │
└──────────┬───────────────────┘
           │
    ┌──────┴──────┐
    │  Attention?  │  ← 타겟별 선택적 적용
    │  (8 heads)   │
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │  FC Layer    │
    │  → Output    │
    │  (24 steps)  │
    └─────────────┘

Output: 12시간 예측 (30분 × 24 step)
```

### 타겟별 모델 구성

| 모델 | hidden | layers | Attention | 특이사항 |
|------|--------|--------|-----------|----------|
| **Flow** | 128 | 4 | 8-head | 시간 피처 중요 |
| **TN** | 64 | 2 | X | 가장 높은 성능 |
| **PH** | 64 | 2 | X | 계절성 피처 활용 |
| **SS** | 64 | 2 | O | 분포 급변 대응 |
| **TP** | 64 | 2 | O | 값 범위 매우 좁음 |
| **FLUX** | 128 | 3 | O | 누적값 차분 처리 |
| **TOC** | 128 | 3 | X | 개선 진행 중 |

### 학습 전략

- **손실함수**: Huber Loss (이상치에 강건)
- **옵티마이저**: Adam + CosineAnnealingWarmRestarts (주기적 재시작 → local minima 탈출)
- **조기 종료**: 연속 5회 val_loss 증가 AND val_loss > train_loss × 3
- **Gradient Clipping**: 기울기 폭발 방지

---

## Slide 8. 자기회귀(Autoregressive) 예측

### 12시간 예측 전략

```
                    ┌── 30분 간격 ──┐
시점:  t-24h ─────── t (현재) ─────── t+12h
       │←── 입력 윈도우 (48 step) ──→│
                     │←── 예측 구간 (24 step) ──→│

Step 1: 입력 [t-24h ~ t] → 예측 [t+0.5h]
Step 2: 예측값을 lag 피처로 업데이트 → 예측 [t+1.0h]
  ...
Step 24: → 예측 [t+12.0h]
```

- 매 스텝 예측값을 다음 스텝의 **lag 피처로 주입**
- 단순 24-step 출력 대비 **시간적 일관성** 확보
- FastAPI에서 실시간 자기회귀 파이프라인 구현

---

## Slide 9. 성능 결과

### 최종 성능 (R² Score)

| 타겟 | R² Score | 개선 과정 | 목표 |
|------|----------|-----------|------|
| **TN (총질소)** | **0.9011** | -0.16 → 0.78 → 0.90 | 0.90 달성 |
| **PH** | **0.8432** | -0.17 → 0.56 → 0.84 | |
| **Flow (유입량)** | **0.7899** | 0.30 → 0.62 → 0.79 | |
| **SS (부유물질)** | **0.6630** | -0.52 → 0.21 → 0.66 | |
| **TP (총인)** | **0.6252** | -2.15 → -0.41 → 0.63 | |
| **FLUX** | **0.6097** | -0.01 → 0.23 → 0.61 | |
| **TOC** | **0.4238** | -1.86 → 0.30 → 0.42 | 개선 중 |

### 성능 개선 핵심 요인

```
[Stage 1] 베이스라인 LSTM              →  대부분 R² 음수 (예측 불가)
    ↓  타겟 Lag 피처 도입
[Stage 2] + Lag/Rolling/EWMA 피처      →  R² 양수 전환 (예측 가능)
    ↓  하이퍼파라미터 최적화
[Stage 3] + batch↑, hidden↑, LR 조정   →  R² 대폭 상승
    ↓  아키텍처 개선
[Stage 4] + Attention, FC layer, 조기종료 →  최종 성능 달성
```

<!-- 삽입 이미지: results/DL/save/prediction_analysis_tn.png (가장 좋은 성능) -->
<!-- 삽입 이미지: results/DL/prediction_analysis_flow.png -->
<!-- 삽입 이미지: results/DL/save/tn_learning_curve.png -->

---

## Slide 10. 유입량(Flow) 구간별 예측 정확도

| 유입량 구간 | MAPE | 비고 |
|-------------|------|------|
| 중간 (320~400 m³/h) | **5.17%** | 가장 정확 |
| 높음 (> 400 m³/h) | **6.40%** | 양호 |
| 낮음 (< 320 m³/h) | 27.05% | 야간/저유량 구간 |

- 주 운영 구간(320+ m³/h)에서 **MAPE 5~6%** → 실용적 정확도 확보
- 저유량 구간은 절대 오차는 작으나 상대 오차(MAPE)가 큼

<!-- 삽입 이미지: results/DL/prediction_analysis_flow.png -->

---

## Slide 11. API 서비스 (FastAPI)

### 엔드포인트 설계

```
GET  /health          → 서버 상태 확인
GET  /ready           → 모델 로드 상태 · 버전 · 피처 수
POST /predict/flow    → 유입량 12시간 예측
POST /predict/tms     → 수질 6개 지표 12시간 예측
```

### 예측 요청/응답 예시

**Request** (`POST /predict/flow`):
```json
{
  "flow_records": [...],     // 1440건 (24시간 × 1분)
  "aws_368": [...],          // 기상관측소 368 데이터
  "aws_541": [...],          // 기상관측소 541 데이터
  "aws_569": [...]           // 기상관측소 569 데이터
}
```

**Response**:
```json
{
  "predictions": [
    {"time_offset": "0.5h", "value": 385.2},
    {"time_offset": "1.0h", "value": 391.7},
    ...
    {"time_offset": "12.0h", "value": 372.1}
  ]
}
```

### API 내부 처리 흐름

```
JSON 입력 → 데이터 병합 → 리샘플링(30분) → 특성 생성
→ 스케일링 → 자기회귀 예측(24 step) → 역스케일링 → JSON 응답
```

---

## Slide 12. 이상 탐지 시스템

### Isolation Forest 기반 이상 탐지

- 정상 패턴 학습 후 **이상 점수(Anomaly Score)** 산출
- 사용자 정의 임계값 (법적 방류 기준) + 통계적 이상 점수 결합

### 운영 점수 산출 체계

| 항목 | 가중치 | 설명 |
|------|--------|------|
| 기준 초과 확률 | 20% | 7개 지표 중 기준 초과 개수 |
| 기준 대비 여유도 | 40% | 현재 수치와 법적 기준 간 마진 |
| 예측 정확도 | 15% | 예측-실측 오차 (0~100 스케일) |
| 데이터 신뢰도 | 15% | 센서 결측·이상 비율 |
| 외부 요인 | 10% | 계절·강우 영향도 |

---

## Slide 13. 개발 과정에서의 문제 해결

### Challenge 1: TMS 예측 성능 음수 → 양수 전환

**문제**: 초기 LSTM 모델이 6개 TMS 지표 모두에서 R² 음수 (평균보다 못한 예측)

**원인 분석**:
- 피처 부족: 외부 변수만으로는 수질 변동 설명 불가
- 조기 종료 조건 과민: 1회 val_loss 증가 시 즉시 종료

**해결**:
1. **타겟 Lag 피처 도입** → 과거 수질 패턴 활용 (data leakage 방지 설계)
2. **조기 종료 완화** → 연속 5회 + 3배 초과 조건으로 변경
3. **배치 크기 확대** → GPU 활용 극대화 (32 → 512)

**결과**: 6개 지표 모두 R² 양수 전환, TN 0.90 달성

---

### Challenge 2: FLUX 누적값 처리

**문제**: FLUX는 일일 누적 유량 → 단순 예측 시 누적 패턴만 학습

**해결**: 차분(differencing) 후 음수 클리핑 → 실제 유량 변화량 학습

---

### Challenge 3: FastAPI HTTP/2 호환 문제

**문제**: Backend에서 HTTP/2로 호출 시 request body 소실 (422 에러)

**해결**: Backend에서 WebClient로 HTTP/1.1 호출하도록 변경

---

## Slide 14. 프로젝트 구조

```
python/
├── data/
│   ├── actual/              # 원본 측정 데이터 (유입량, 수질, 기상)
│   ├── processed/           # 전처리된 학습 데이터
│   ├── pred/                # 예측 결과 저장
│   └── recommand_features/  # 타겟별 선택 피처 목록
│
├── model/save/              # 학습된 모델 7개 + 스케일러 14개
│
├── notebook/
│   ├── DL/                  # LSTM 학습 노트북 & 분석 스크립트
│   ├── feature/             # 특성 엔지니어링 & 선택 모듈
│   ├── EDA/                 # 탐색적 데이터 분석
│   └── preprocess/          # 전처리 파이프라인
│
├── src/main.py              # FastAPI 예측 서버 (744줄)
├── results/                 # 실험 결과 (학습곡선, 예측 시각화, 상관분석)
└── archive/                 # 이전 버전 코드 (ML 베이스라인 등)
```

---

## Slide 15. 향후 계획

### 단기 목표
- **TOC 성능 개선**: 추가 피처 발굴, 시퀀스 길이 조정 (현재 R² 0.42)
- **결측치/이상치 처리 최적화**: drop-only 전략 성능 비교
- **FC Layer 추가 실험**: FLUX에서 효과 확인 → 타 모델 적용

### 중기 목표
- **Transformer + LSTM 하이브리드**: 하수처리 공정을 Transformer로 학습 후 LSTM 예측
- **앙상블 예측**: 다중 모델 결합으로 안정성 향상
- **모델 재학습 자동화**: 데이터 분포 변화 감지 시 자동 재학습

### 장기 목표
- **실시간 운영 시스템 통합**: Dashboard 연동, 알림 자동화
- **타 하수처리장 확장**: 모델 전이학습 적용

---

## Slide 16. 핵심 역량 요약

### 이 프로젝트에서 보여주는 역량

| 역량 | 구체적 내용 |
|------|-------------|
| **데이터 엔지니어링** | 다중 소스(유입량+수질+기상 3개소) 시간축 정렬 및 리샘플링 |
| **도메인 지식 활용** | 선행강우지수, 방류기준 기반 이상치 처리, 누적유량 차분 |
| **특성 엔지니어링** | 40+ 피처 설계, 데이터 누수 방지, Walk-Forward 피처 선택 |
| **딥러닝 모델링** | LSTM+Attention, 타겟별 아키텍처 최적화, 자기회귀 예측 |
| **실험 관리** | 체계적 개발 노트, 단계별 성능 추적, 하이퍼파라미터 기록 |
| **API 서비스화** | FastAPI 기반 실시간 예측 서버, Backend 연동 |
| **문제 해결** | R² 음수 → 양수 전환, HTTP 호환 문제 해결, GPU 최적화 |

---

## 부록: PPT/PDF 변환 시 이미지 배치 가이드

### 각 슬라이드에 삽입할 이미지

| 슬라이드 | 삽입 이미지 경로 | 용도 |
|----------|-----------------|------|
| Slide 4 (데이터) | `results/timeseries/flow_plot.png` | 유입량 시계열 |
| | `results/timeseries/tms_plot.png` | 수질 시계열 |
| | `results/correlation/flow.png` | 유입량 상관관계 |
| | `results/correlation/tms.png` | 수질 상관관계 |
| Slide 5 (전처리) | `results/preprocess/FLOW_before_after.png` | 전처리 전후 비교 |
| | `results/preprocess/TMS_before_after.png` | 전처리 전후 비교 |
| Slide 9 (성능) | `results/DL/save/prediction_analysis_tn.png` | TN 예측 분석 |
| | `results/DL/prediction_analysis_flow.png` | Flow 예측 분석 |
| | `results/DL/save/tn_learning_curve.png` | TN 학습 곡선 |
| Slide 10 (Flow) | `results/DL/prediction_analysis_flow.png` | Flow 구간별 분석 |
| 추가 참고 | `results/DL/save/prediction_analysis_*.png` | 각 지표 예측 분석 |
| | `results/DL/save/*_learning_curve.png` | 각 지표 학습 곡선 |
| | `results/boxplot/flow_boxplot.png` | 데이터 분포 |
| | `results/distribution/tms_distribution.png` | 데이터 분포 |

---

*작성일: 2026-02-12*
