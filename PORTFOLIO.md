# 포트폴리오 — 하수처리장 유입량·수질(TMS) 예측 AI 서비스

> **WWTP Influent Flow & Water Quality Forecasting AI Service**
> 개발 기간: 2024.08 ~ 2026.02 | 4인 개발 (풀스택 프로젝트 + AI))

---

## 1. 프로젝트 개요

### 배경 및 문제 정의

하수처리장은 유입 하수량과 수질 변동에 따라 처리 공정을 실시간으로 조정해야 한다. 수질 기준(TMS) 초과 시 법적 제재와 환경 오염이 발생하지만, 기존 운영은 과거 경험에만 의존해 선제적 대응이 불가능했다.

### 목표

- 유입유량(Flow)과 수질 6개 지표(TOC, SS, TN, TP, FLUX, pH)를 **향후 12시간 선행 예측**
- 예측 결과를 REST API로 제공하여 기존 운영 시스템과 연동
- 이상 징후 조기 탐지 및 알림

### 핵심 성과 요약

| 지표 | R² Score | 비고 |
|------|----------|------|
| 총질소 (TN) | **0.9011** | 목표(0.90) 달성 |
| 수소이온농도 (pH) | **0.8574** | 목표(0.80) 달성|
| 유입유량 (Flow) | **0.8425** | 목표(0.80) 달성|
| 부유물질 (SS) | **0.6906** | |
| 총인 (TP) | **0.6281** | |
| 방류유량 (FLUX) | **0.6241** | |
| 총유기탄소 (TOC) | **0.5574** | |

---

## 2. 기술 스택

| 분야 | 사용 기술 |
|------|-----------|
| **딥러닝** | PyTorch — LSTM + Multi-head Attention |
| **ML 베이스라인** | scikit-learn (Random Forest, Ridge, Lasso), XGBoost, HistGBR |
| **이상 탐지** | Isolation Forest |
| **하이퍼파라미터 최적화** | Optuna (베이지안), 직접 설계한 Phase-1/2 그리드 탐색 |
| **API 서버** | FastAPI + Uvicorn |
| **데이터 처리** | Pandas, NumPy, SciPy |
| **시각화·대시보드** | Matplotlib, Streamlit |
| **배포** | Hugging Face Spaces (Streamlit 앱) |
| **개발 환경** | Python 3.10, PyTorch 2.x, CUDA GPU |

---

## 3. 데이터 파이프라인

### 수집 데이터

| 데이터 | 기간 | 해상도 | 주요 변수 |
|--------|------|--------|-----------|
| 유입유량 | 2025.09 ~ 2025.12 | 1분 | flow_TankA/B, level_TankA/B, Q_in |
| 수질(TMS) | 2024.08 ~ 2025.09 | 1분 | TOC, SS, TN, TP, pH, FLUX |
| 기상(AWS) | 2024.08 ~ 2026.02 | 1분 | 기온, 강수량(15분/1h/12h/일), 습도, 이슬점 |
| 추가 원천 | 2024.08 ~ 2026.02 | 1분 | 송풍량, 약품투입량, 공정 센서 |

### 전처리 파이프라인

```
원본 데이터 (1분 단위)
  │
  ├─ 1) 시간축 정합 — 정렬·중복 제거
  │
  ├─ 2) 리샘플링 — 1분 → 30분 (mean / sum)
  │     └─ FLUX_VU: 누적값 → 30분 증분값(diff) 변환
  │
  ├─ 3) 이상치 제거
  │     ├─ 도메인 기반: 방류허용기준 × 2 초과 값 제거
  │     └─ 통계 기반: Z-score / IQR
  │
  ├─ 4) 결측치 보간 (구간별 차등 전략)
  │     ├─ 단기 (1~3시간): Forward Fill
  │     ├─ 중기 (4~12시간): EWMA (span=6)
  │     └─ 장기 (12시간+): Long-span EWMA (span=24) — 데이터 손실 최소화
  │
  └─ 5) 정규화: StandardScaler (타겟별 독립)
```

### 특성 엔지니어링 (40+ 피처)

| 카테고리 | 주요 피처 |
|----------|-----------|
| **강수 피처** | 선행강우지수(ARI, decay=0.85), 건습기 플래그, 강수 변화량 |
| **기상 피처** | 증기압차(VPD), 기온-습도 교호작용, 이슬점 편차, 3개 AWS 공간 통계 |
| **공정 피처** | 수위 합/차, 수위 lag (1~36h), rolling 통계 |
| **시간 피처** | sin/cos 인코딩, hour×weekday 교호작용, 계절·주차 변수 |
| **타겟 Lag** | 과거값 lag, rolling mean/std, 차분, EWMA (data leakage 방지) |

**피처 선택**: Walk-Forward Validation 기반 → 시간순 분할로 과적합 방지, 타겟별 독립 최적 피처셋 구성

---

## 4. 모델 아키텍처

### LSTM + Multi-head Attention

```
Input (batch, seq_len=48, n_features)  ← 24시간 입력 (30분 × 48 step)
  │
  ▼
LSTM (2~4 layers, hidden 256~512)
  + Layer Normalization
  + Gradient Clipping
  │
  ├─ (타겟별 선택적) Multi-head Attention (8 heads)
  │
  ▼
FC Head (3~4 layer deep head 또는 3-layer head)
  │
  ▼
Output: 12시간 예측 (30분 × 24 step)
```

**학습 전략**

- 손실함수: Huber Loss — 이상치에 강건
- 옵티마이저: Adam + CosineAnnealingWarmRestarts — 주기적 재시작으로 local minima 탈출
- 조기 종료: 연속 5회 val_loss 증가 AND val_loss > train_loss × 3

### 타겟별 최종 모델 구성

| 타겟 | hidden | layers | Attention | R² |
|------|--------|--------|-----------|-----|
| Flow | 512 | 2 | ✗ | 0.8425 |
| TN | 512 | 4 | ✗ | 0.9011 |
| pH | 512 | 1 | ✗ | 0.8574 |
| SS | 256 | 2 | ✗ | 0.6906 |
| TOC | 512 | 1 | ✗ | 0.5574 |
| TP | 384 | 1 | ✗ | 0.6281 |
| FLUX | 512 | 4 | ✓ | 0.6241 |

### 자기회귀(Autoregressive) 12시간 예측

```
입력 윈도우: [t-24h ~ t] (48 step)
  Step 1 → 예측 [t+0.5h]  →  예측값을 lag 피처로 주입
  Step 2 → 예측 [t+1.0h]  →  예측값을 lag 피처로 주입
  ...
  Step 24 → 예측 [t+12.0h]
```

단순 24-step 일괄 출력 대비 **시간적 일관성 확보** — FastAPI 서버에서 실시간 파이프라인으로 구현

---

## 5. API 서비스 (FastAPI)

### 엔드포인트

```
GET  /health       → 서버 상태 확인
GET  /ready        → 모델 로드 상태 · 버전 · 피처 수 반환
POST /predict/flow → 유입유량 12시간 예측 (30분 단위 24점)
POST /predict/tms  → 수질 6개 지표 12시간 동시 예측
```

### 내부 처리 흐름

```
JSON 입력 (1440건, 24시간 × 1분)
  → 병합 · 리샘플링(30분) → 특성 생성 → StandardScaler
  → Autoregressive 추론 (24 step) → 역스케일링
  → JSON 응답 (예측값 24점 + trajectory)
```

### 응답 예시 (POST /predict/tms)

```json
{
  "request_id": "test-002",
  "ok": true,
  "output": {
    "predictions": {
      "toc": { "0.5h": 12.3, "1.0h": 12.5, "...", "12.0h": 11.8 },
      "tn":  { "0.5h": 8.1,  "...", "12.0h": 8.4 }
    },
    "metadata": { "window_size": 48, "targets": ["toc","ss","tn","tp","flux","ph"] }
  },
  "latency_ms": 520
}
```

---

## 6. 성능 개선 과정

### 단계별 R² 개선 (TN 사례)

```
[Stage 1] LSTM 베이스라인 (외부 변수만)  →  R² = -0.16  (예측 불가)
    ↓  타겟 Lag 피처 도입 (과거 수질 패턴)
[Stage 2] + Lag / Rolling / EWMA 피처     →  R² =  0.78  (양수 전환)
    ↓  하이퍼파라미터 최적화
[Stage 3] + hidden 512, lr 2e-3 조정      →  R² =  0.90  (대폭 상승)
    ↓  아키텍처 개선
[Stage 4] + Deep FC Head, 조기 종료 완화  →  R² =  0.9011 (최종)
```

### ML 베이스라인 비교 (Flow 사례)

| 단계 | 방법 | R² |
|------|------|----|
| ML V1 (baseline) | HistGBR, drop 방식 (데이터 사용률 4.2%) | ~0.30 |
| ML V2 | 도메인 피처 + EWMA 보간 (98.4% 사용) | ~0.62 |
| **DL 최종** | **LSTM + Lag 피처 + HP 최적화** | **0.8425** |

### 유입량 구간별 예측 정확도 (MAPE)

| 구간 | MAPE | 비고 |
|------|------|------|
| 중간 (320~400 m³/h) | 5.17% | 실용적 정확도 확보 |
| 높음 (> 400 m³/h) | 6.40% | 양호 |
| 낮음 (< 320 m³/h) | 27.05% | 야간·저유량 구간 |

---

## 7. 문제 해결 사례

### Case 1. TMS 예측 R² 전체 음수 → 양수 전환

**문제**: 초기 LSTM 모델이 6개 TMS 지표 모두 R² 음수 (단순 평균보다 나쁜 예측)

**원인 분석**:
1. 외부 기상 변수만으로는 수질 변동 설명 불가 → 과거 수질 패턴이 가장 강한 신호
2. 조기 종료 조건 과민 (val_loss 1회 증가 시 즉시 종료) → 학습 조기 중단

**해결**:
1. **타겟 Lag 피처 도입** — `shift(k+1)`로 미래 정보 누수 방지하면서 과거 패턴 활용
2. **조기 종료 완화** — "연속 5회 + val_loss > train_loss × 3" 이중 조건으로 변경
3. **배치 크기 확대** — 32 → 512 → 2048 (GPU 활용 극대화, 학습 안정화)

**결과**: 6개 지표 모두 R² 양수 전환, TN R² 0.90 달성

---

### Case 2. FLUX 누적값 처리

**문제**: FLUX는 일일 누적 유량 → 단순 예측 시 항상 증가하는 누적 패턴만 학습, 실제 유량 변화량 예측 불가

**해결**: 30분 증분값으로 차분(differencing) 후 음수 클리핑 → 실제 유량 변화량 학습
**결과**: R² 0.61 달성 (누적값 그대로 학습 시 near-zero 성능)

---

### Case 3. FastAPI HTTP/2 호환 문제

**문제**: 연동 Backend에서 HTTP/2로 FastAPI 호출 시 request body 소실 → 422 Unprocessable Entity

**해결**: Backend WebClient를 HTTP/1.1 명시적 설정으로 변경
**결과**: 정상 연동 확인

---

### Case 4. 체계적 하이퍼파라미터 탐색 설계

단순 랜덤 탐색 대신 **Phase-1/2 구조화 그리드 탐색** 직접 설계:
- Phase 1: hidden_size × num_layers × learning_rate (18조합, 핵심 파라미터)
- Phase 2: dropout × batch_size × weight_decay (18조합, 보조 파라미터)

타겟별 인사이트 누적 → 이후 타겟에 적용:
- layers=1이 유리한 타겟(TP, TOC, pH) vs layers=2가 유리한 타겟(SS, FLUX, Flow, TN)
- lr=2e-3가 대부분 최적, FLUX만 lr=5e-4 (느린 학습이 효과적)
- wd=1e-3는 Flow에서 일관적으로 성능 하락 → 금지

---

## 8. 대시보드 (Streamlit)

**Hugging Face Spaces**에 배포된 멀티페이지 인터랙티브 대시보드

| 페이지 | 내용 |
|--------|------|
| 홈 | 7개 지표 R² 요약 바 차트, 시스템 구성 개요 |
| 1. 성능 대시보드 | ML baseline → DL 단계별 R² 비교 (ML V1/V2/DL 전체 포함), 학습곡선 |
| 2. 예측 분석 | 예측 vs 실측 시각화, 잔차 분석 |
| 3. 모델 정보 | LSTM 아키텍처, 피처 엔지니어링 파이프라인 설명 |
| 4. 운영 KPI | 7개 지표 가중 종합 점수 대시보드 |
| 5. 라이브 추론 | CSV 업로드 → FastAPI 호출 → 실시간 예측 결과 시각화 |
| 6. 업체모델 비교 | 납품 업체 예측값 vs LSTM 예측값 성능 비교 |

---

## 9. 프로젝트 구조

```
c:\AIproject\
├── data/
│   ├── actual/          # 실측 데이터 (FLOW, TMS, 기상)
│   ├── features/save/   # 타겟별 선택 피처 목록 (7개)
│   ├── output/save/     # 예측 출력 (7개 타겟)
│   └── pred/            # 업체 예측 결과 (비교용)
├── model/save/          # 학습된 모델(.pth) × 7 + 스케일러(.pkl) × 14
├── notebook/
│   ├── EDA/             # 탐색적 데이터 분석 (주기성, 상관관계)
│   ├── feature/         # 피처 엔지니어링·선택 모듈
│   ├── preprocess/      # 전처리 파이프라인 노트북
│   └── training/        # LSTM 학습 노트북 + 타겟별 실험 스크립트
├── src/                 # FastAPI 추론 서버
│   ├── config.py        # 7개 타겟 하이퍼파라미터 (단일 진실 출처)
│   ├── models.py        # LSTMRegressor (flow/TMS 공통 아키텍처)
│   ├── preprocess.py    # 입력 전처리 파이프라인
│   ├── predict.py       # autoregressive_predict()
│   └── main.py          # FastAPI 라우터
├── demo/                # Streamlit 멀티페이지 대시보드
├── results/             # 실험 결과 (학습곡선, 예측 시각화, 상관분석)
└── archive/             # 구버전 ML/DL 코드 및 결과
```

---

## 10. 핵심 역량

| 역량 | 구체적 내용 |
|------|-------------|
| **데이터 엔지니어링** | 유입량·수질·기상 3개 소스 시간축 정렬, 1분→30분 리샘플링, Long→Wide 피벗 |
| **도메인 지식 활용** | 선행강우지수 설계, 방류기준 기반 이상치 처리, FLUX 누적값 차분 변환 |
| **특성 엔지니어링** | 40+ 피처 직접 설계, shift(k+1) data leakage 방지, Walk-Forward 피처 선택 |
| **딥러닝 모델링** | LSTM+Attention 타겟별 아키텍처 최적화, 자기회귀 12시간 예측 구현 |
| **체계적 실험 관리** | Phase-1/2 구조화 그리드 탐색, 타겟 간 인사이트 누적·전이, 개발 일지 기록 |
| **API 서비스화** | FastAPI 실시간 추론 서버 구현, Pydantic 스키마 설계, Backend 연동 |
| **배포 경험** | Hugging Face Spaces 배포 (app.py + requirements.txt 구조 설계, git push) |
| **문제 해결** | R² 음수→양수 전환, HTTP/2 호환 문제, GPU 학습 최적화 |

---

*작성일: 2026-02-25*
