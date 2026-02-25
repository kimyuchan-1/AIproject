# 하수 유입량 · 수질(TMS) 예측 AI 서비스

## 1. Overview
본 프로젝트는 하수처리장의 **미래 유입량**과 **수질(TMS)** 을 사전에 예측하고,
운영 기준을 초과할 가능성이 있을 경우 **사전 경고 및 이상 진단**을 제공하는
AI 기반 의사결정 지원 웹 서비스이다.

- **예측(Forecasting)**: 유입유량(Flow), TMS 세부 지표 (TOC, SS, TN, TP, FLUX, pH)
- **분석(Analytics)**: 시간·계절 패턴 및 기상 변수 상관 분석
- **진단(Diagnosis)**: 실시간 이상 여부 판정 및 알림

---

## 2. Project Goals
### 예측 목표
- 유입량 예측 정확도 **90%**
- TMS 세부 지표 예측 정확도 **80%**

### 분석 목표
- 시간별 / 일별 / 계절별 유입량 변동 패턴 분석
- 기상 요인(강우량, 기온 등)과 유입량의 상관 관계 분석

### 진단 목표
- 사용자 정의 기준 기반 실시간 이상 여부 판정
- 이상 발생 시 즉각적인 경고 제공

---

## 3. Features
### Forecasting
- 유입유량(Q_in) 시계열 예측 — 향후 12시간, 30분 단위
- TMS 지표(TOC, SS, TN, TP, FLUX, pH) 예측 — 향후 12시간, 30분 단위
- Autoregressive 방식 다중 시점 예측

### Analytics
- KPI 대시보드
  - 평균 유입량
  - 변동 범위
  - 월별 / 계절별 추이
- 다변량 상관 분석 결과 시각화
- 기상 변수 ↔ 유입량 관계 분석

### Diagnosis
- Isolation Forest 기반 이상 탐지
- 유입량 및 TMS 지표 이상 여부 실시간 판단
- 이상 발생 시 알림 표시

---

## 4. Data
- **데이터 종류**
  - 유입량 시계열 데이터 (`FLOW_Actual.csv`, `FLOW_extended.csv`)
  - TMS 수질 지표 (`TMS_Actual.csv`, `TMS_extended.csv`) — TOC, pH, SS, FLUX, TN, TP
  - 기상 데이터 — AWS 3개소 (368, 541, 569국)

- **전처리**
  - 1분 단위 원시 데이터 → 30분 단위 리샘플링
  - `FLUX_VU`: 누적값 → 30분 증분값(diff) 변환
  - 결측치 ffill 후 0 대체

- **데이터 분할**
  - 시간 순서 기준 Train / Validation / Test 분할
  - 미래 정보 누수(Time Leakage) 방지

---

## 5. Models & Methods
### Deep Learning (현재 운영)
- **LSTM + Attention**
  - 시계열 장기 의존성 학습
  - Sliding Window 48 스텝(24시간) 입력 → 30분 단위 예측
  - Autoregressive 방식으로 24스텝(12시간) 연속 예측

#### 타겟별 모델 구성 (`src/config.py` 기준)

| 타겟 | hidden | layers | Attention | deep_head |
|------|--------|--------|-----------|-----------|
| flow | 512    | 2      | ✗         | ✓ (4-layer head) |
| toc  | 512    | 1      | ✗         | ✓ (4-layer head) |
| ss   | 256    | 2      | ✗         | ✓ (4-layer head) |
| tn   | 512    | 4      | ✗         | ✗ (3-layer head) |
| tp   | 384    | 1      | ✗         | ✗ (3-layer head) |
| flux | 512    | 4      | ✓         | ✗ (3-layer head) |
| ph   | 512    | 1      | ✗         | ✗ (3-layer head) |

### Anomaly Detection
- Isolation Forest
  - 정상 패턴 학습 후 이상 점수 기반 판별
  - 사용자 기준과 병행 적용

### Legacy (archive/)
- Random Forest, XGBoost 기반 ML 파이프라인 (현재 비운영)

---

## 6. Evaluation

#### 최종 R² 성능 (테스트셋 기준)

| 타겟 | R² |
|------|----|
| 유입유량 (Flow) | 0.8425 |
| 총유기탄소 (TOC) | 0.5574 |
| 부유물질 (SS) | 0.6906 |
| 총질소 (TN) | 0.9011 |
| 총인 (TP) | 0.6281 |
| 방류유량 (FLUX) | 0.6241 |
| 수소이온농도 (pH) | 0.8574 |

#### 평가 지표
- **유입량 예측**: MAE, RMSE, MAPE — |실제값 − 예측값| / 실제값 ≤ 5% 비율을 정확도로 정의
- **TMS 예측**: 지표별 MAE / RMSE / MAPE — 목표 정확도 90% 기준 충족 여부 평가
- **이상 진단**: 이상 이벤트 탐지 사례 기반 검증 및 알림 발생 적합성 검토

---

## 7. System Architecture
```
데이터 수집 (1분 단위, 24시간 = 1440 records)
↓
전처리 · 피처 생성 (30분 리샘플링 + feature_engineering.py)
↓
LSTM 모델 추론 (Autoregressive, 12h horizon)
↓
이상 탐지
↓
FastAPI 서버 (src/main.py) ← REST API
```

---

## 8. Getting Started
### Environment
- Python **3.10+**
- PyTorch **2.x**
- scikit-learn, numpy, pandas, fastapi, uvicorn

### Installation
```bash
conda create -n wwtp python=3.10
conda activate wwtp

# PyTorch는 플랫폼에 맞게 별도 설치 (https://pytorch.org)
pip install torch

pip install -r requirements.txt

# FastAPI 백엔드 실행 (python/ 디렉토리에서)
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

### API Endpoints

#### 서버 상태 확인
```
GET /health

Response (200 OK):
{ "ok": true }
```

#### 모델 서비스 준비 상태 확인
```
GET /ready

Response (200 OK):
{
  "ok": true,
  "model_version": "0.3.0",
  "models_loaded": {
    "flow": { "n_features": <int> },
    "tms": {
      "toc": { "n_features": <int>, "use_attention": false },
      "ss":  { "n_features": <int>, "use_attention": false },
      ...
    }
  },
  "window_size": 48,
  "horizon_unit": "30min"
}
```

#### 유입량(Flow) 예측 — 향후 12시간
```
POST /predict/flow
Content-Type: application/json

Request Body:
{
  "request_id": "test-001",
  "in": {
    "dataList": [
      {
        "SYS_TIME": "2024-01-01 00:00:00",
        "flow_TankA": 0.0,
        "flow_TankB": 0.0,
        "level_TankA": 0.0,
        "level_TankB": 0.0,
        "Q_in": 0.0
      }
      // ... 총 1440개 (1분 단위, 24시간)
    ],
    "awsList": {
      "stn_368": [ { "SYS_TIME": "...", "TA": 0.0, "RN_15m": 0.0, ... } ],
      "stn_541": [ ... ],
      "stn_569": [ ... ]
    }
  }
}

Response (200 OK):
{
  "request_id": "test-001",
  "ok": true,
  "output": {
    "predictions": {
      "0.5h": 1234.5, "1.0h": 1240.0, ... , "12.0h": 1200.0
    },
    "trajectories": { "12h": [ ... ] },
    "metadata": { "window_size": 48, "n_features": <int>, ... }
  },
  "latency_ms": 320,
  "error": null
}
```

#### 수질(TMS) 예측 — 향후 12시간 (TOC, SS, TN, TP, FLUX, pH 동시 예측)
```
POST /predict/tms
Content-Type: application/json

Request Body:
{
  "request_id": "test-002",
  "in": {
    "dataList": [
      {
        "SYS_TIME": "2024-01-01 00:00:00",
        "TOC_VU": 0.0,
        "PH_VU":  0.0,
        "SS_VU":  0.0,
        "FLUX_VU": 0.0,
        "TN_VU":  0.0,
        "TP_VU":  0.0
      }
      // ... 총 1440개 (1분 단위, 24시간)
    ],
    "awsList": {
      "stn_368": [ ... ],
      "stn_541": [ ... ],
      "stn_569": [ ... ]
    }
  }
}

Response (200 OK):
{
  "request_id": "test-002",
  "ok": true,
  "output": {
    "predictions": {
      "toc":  { "0.5h": 12.3, "1.0h": 12.5, ... },
      "ss":   { "0.5h": 30.1, ... },
      "tn":   { ... },
      "tp":   { ... },
      "flux": { ... },
      "ph":   { ... }
    },
    "trajectories": {
      "toc": { "12h": [ ... ] },
      ...
    },
    "metadata": { "window_size": 48, "targets": ["toc","ss","tn","tp","flux","ph"], ... }
  },
  "latency_ms": 520,
  "error": null
}
```

---

## 9. Demo Dashboard (Streamlit)

`streamlit run demo/app.py` 로 실행하는 멀티페이지 인터랙티브 대시보드. **Hugging Face Spaces**에 별도 레포지토리로 배포됨.

| 페이지 | 내용 |
|--------|------|
| 홈 | 7개 지표 R² 요약 바 차트, 시스템 구성 개요 |
| 1. 성능 대시보드 | ML baseline(V1/V2) → DL 단계별 R² 비교, 학습곡선 |
| 2. 예측 분석 | 예측 결과 시각화, 잔차 분석 |
| 3. 모델 정보 | LSTM 아키텍처, 피처 엔지니어링 파이프라인 |
| 4. 운영 KPI | 운영 지표 KPI 대시보드 |
| 5. 라이브 추론 | CSV 업로드 → FastAPI 호출 → 실시간 예측 결과 |
| 6. 업체모델 비교 | 업체 예측값(FLOW_Pred/TMS_Pred) vs LSTM 예측값 성능 비교 |

---

## 10. Repository Structure
```
├── data/
│   ├── metadata.xlsx                # 데이터 메타정보
│   ├── weatherAPI.txt               # 기상청 API 키 정보
│   ├── raw/                         # 원천 데이터 (AWS 원시, FLOW/TMS xlsx)
│   │   ├── AWS_{368,541,569}.csv
│   │   └── 유입유량-TMS 데이터(예측, 실측)_1203.xlsx
│   ├── actual/                      # 실측 데이터
│   │   ├── FLOW_Actual.csv
│   │   ├── TMS_Actual.csv
│   │   ├── Weather.csv              # AWS 통합 기상 데이터
│   │   └── AWS_{368,541,569}.csv    # AWS 기상 관측소별 데이터
│   ├── features/                    # 타겟별 추천 특성 목록
│   │   └── save/                    # {target}_recommended_features.csv (7개 타겟)
│   ├── output/                      # 예측 출력
│   │   └── save/                    # {target}_predictions.csv (7개 타겟)
│   └── pred/                        # 업체 예측 결과
│       ├── FLOW_Pred.csv
│       └── TMS_Pred.csv
├── model/
│   └── save/                        # 학습된 모델 체크포인트 및 스케일러
│       ├── {target}_lstm_model.pth  # (7개 타겟)
│       ├── X_scaler_{target}.pkl    # (7개 타겟)
│       └── y_scaler_{target}.pkl    # (7개 타겟)
├── notebook/
│   ├── EDA/                         # 탐색적 데이터 분석
│   │   └── flow_tms_periodicity_eda.ipynb
│   ├── feature/                     # 피처 엔지니어링 모듈
│   │   ├── feature_engineering.py   # 특성 생성 파이프라인 (공유 모듈)
│   │   └── WF_feature_selection.py  # Walk-Forward 특성 선택
│   ├── preprocess/                  # 전처리 노트북
│   │   ├── preprocess.ipynb
│   │   ├── raw_refactoring.ipynb
│   │   ├── show.ipynb
│   │   ├── correlation.ipynb
│   │   └── split_distribution.ipynb
│   └── training/                    # 모델 학습 노트북 및 스크립트
│       ├── LSTM_FLOW.ipynb          # 유입량 모델 학습
│       ├── LSTM_TMS.ipynb           # TMS 6개 타겟 학습
│       ├── experiments/             # 타겟별 하이퍼파라미터 실험
│       │   └── {target}_experiment.py  # (7개 타겟)
│       ├── scripts/                 # 분석 유틸 스크립트
│       │   ├── analyze_predictions.py  # 예측 결과 분석 및 시각화
│       │   └── diagnosis.py            # 이상 진단
│       └── ML/                      # 머신러닝 베이스라인 (레거시)
│           ├── v1/                  # 1차 ML 베이스라인
│           └── v2/                  # 2차 ML 베이스라인
├── src/                             # FastAPI 추론 서버
│   ├── config.py                    # 7개 타겟 모델 하이퍼파라미터 (단일 진실 출처)
│   ├── loader.py                    # 모델·스케일러·피처 CSV 로드
│   ├── models.py                    # LSTMRegressor (flow/TMS 공통)
│   ├── schemas.py                   # Pydantic I/O 스키마
│   ├── preprocess.py                # 입력 파이프라인 (리샘플링 + 피처 엔지니어링)
│   ├── predict.py                   # autoregressive_predict()
│   └── main.py                      # FastAPI 라우터 (/health, /ready, /predict/*)
├── results/
│   ├── DL/                          # 딥러닝 실험 결과
│   │   ├── {target}_experiment_results.csv  # 하이퍼파라미터 실험 요약 (7개 타겟)
│   │   └── save/                    # 학습곡선·진단·예측 분석 이미지
│   │       ├── {target}_learning_curve.png
│   │       ├── {target}_diagnosis.png
│   │       └── prediction_analysis_{target}.png
│   ├── ML/                          # 머신러닝 실험 결과
│   │   ├── improved/                # 최종 개선 모델 결과
│   │   ├── v1/                      # 1차 베이스라인 결과
│   │   └── v2/                      # 2차 베이스라인 결과
│   ├── preprocess/                  # 전처리 전후 비교
│   ├── correlation/                 # 상관관계 분석 결과
│   ├── boxplot/                     # 변수별 박스플롯
│   ├── distribution/                # 분포 분석
│   └── timeseries/                  # 시계열 시각화
├── archive/                         # 구버전 코드 및 데이터
│   ├── old_DL_versions/             # 구버전 DL 소스
│   ├── old_ML_versions/             # 구버전 ML 소스
│   ├── old_model/                   # 구버전 모델 체크포인트
│   ├── old_notebooks/               # 구버전 노트북
│   ├── old_results/                 # 구버전 결과
│   ├── old_scripts/                 # 구버전 학습 스크립트
│   └── old_src/                     # 구버전 src 모듈 (DL/ML)
├── requirements.txt
├── NOTE.md                          # 개발 일지
├── TODO.md
├── PORTFOLIO.md
└── README.md
```

---
