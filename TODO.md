# 1. 데이터 분석
- [X] 데이터 수집
    - [X] 업체 데이터 (TMS_Actual.csv, FLOW_Actual.csv)
    - [X] 기상청 데이터 (기온, 습도, 강수량, 이슬점 온도)
    - [X] 신규 원천 데이터 3종 전처리 (`raw_refactoring.ipynb`)
        - [X] FLOW_extended Long→Wide 피벗 (TAG_SN 15개, 132,297행, `data/actual/FLOW_extended.csv`)
        - [X] 약품주입량 일별→30분 변환 (`medication1/2.csv`, `data/processed/medication_30min.csv`)
        - [X] process1 + process2 병합 (238,915행 × 8컬럼, `data/processed/process.csv`)

- [X] 데이터 전처리
    - [X] 결측치 처리 (ffill/중기 EWMA/장기 EWMA 전략)
    - [X] 이상치 필터링 (도메인 지식 + IQR/Z-score)
    - [X] Feature engineering (시차 변수, 시간 특성)
    - [X] 시간축 정합 (정렬/중복 제거)
    - [X] 전처리 순서 적용 (정합→보간→이상치→리샘플링→피처→분할→스케일링→선택)

- [X] EDA
    - [X] 데이터 시각화
    - [X] 다변량 상관 분석
    - [X] 시간대/요일별 주기성 분석 (flow, ph)

# 2. 머신러닝
- [X] 베이스라인 (Linear, Ridge, Lasso, Elastic Net)
    - [X] 성능 평가 지표 설정 (MAE, RMSE, R2, MAPE)

- [X] 앙상블 (RandomForest, XGBoost, HistGBR)

- [X] 최종 통합 버전
    - [X] Optuna 하이퍼파라미터 최적화
    - [X] TimeSeriesSplit 교차 검증
    - [X] 피처 선택 (중요도 기반)
    - [X] TMS 모델 그룹화 (modelA, modelB, modelC)
    - [X] 데이터 누수 방지
    - [X] 도메인 특화 피처 (강수, 기상, TMS 상호작용)
    - [X] Learning Curve 시각화

# 3. 딥러닝
- [X] LSTM_FLOW (R2: 0.8425)
    - [X] 30분 리샘플링
    - [X] Target lag 피처 추가
    - [X] 시간 특성 추가 (hour×weekday, weekday, iso_week)
    - [X] 하이퍼파라미터 그리드 탐색 (hidden=512, layers=2, lr=2e-3, R2: 0.8166→0.8659)
    - [X] 모델 저장 및 평가

- [X] LSTM_TMS
    - [X] Target lag 피처 추가 (lag/rolling/diff/EWMA)
    - [X] Early stop 기능 구현
    - [X] 이상치 처리 수정 (배출허용기준 2배)
    - [X] FLUX 차분 처리 (누적값 → 차분)
    - [X] TN (R2: 0.9011) — 그리드 탐색 완료, hidden=512, layers=4, lr=2e-3
    - [X] PH (R2: 0.8574) — 그리드 탐색 완료, hidden=512, layers=1, lr=2e-3, dropout=0.1
    - [X] SS (R2: 0.6906) — 그리드 탐색 완료, hidden=256, layers=2, lr=2e-3
    - [X] TOC (R2: 0.5574) — 그리드 탐색 완료, hidden=512, layers=1, window=48, wd=0.2
    - [X] FLUX (R2: 0.6241) — 그리드 탐색 완료, 기존 최고(0.6296) 미달 (WF 특성 수 부족)
    - [X] TP (R2: 0.6281) — 그리드 탐색 완료, hidden=384, layers=1, lr=1e-3, dropout=0.1

- [X] Transformer_TMS (`transformer_TMS.ipynb`)
    - [X] TransformerRegressor 구현 (Pre-LayerNorm, nhead=8, sinusoidal PE)
    - [X] 전 타깃 실험 (toc/ss/tn/tp/flux/ph)
    - [X] LSTM 대비 성능 열위 확인 → LSTM 모델 유지 결정

# 4. 프로젝트 관리
- [X] 의존성 관리 (requirements.txt)
- [X] 문서화 (QUICK_START.md, README.md)
- [X] FastAPI 백엔드 연동 (main.py)
    - [X] 전처리 파이프라인 노트북과 통일
    - [X] 예측 시간해상도 30분 수정
    - [X] WebClient HTTP/1.1 호환
- [X] Streamlit 대시보드 (`demo/`)
    - [X] 멀티페이지 앱 구성 (홈/성능대시보드/예측분석/모델정보/라이브추론)
    - [X] 성능 대시보드 ML 섹션 추가 (ML baseline vs V2, 데이터 사용률 개선)
    - [X] `constants.py` STAGE_R2 ML/DL 구분 개편
    - [X] 운영 KPI 페이지 추가 (`4_운영_KPI.py`)
    - [X] 업체모델 비교 페이지 추가 (`6_업체모델_비교.py`) — FLOW_Pred/TMS_Pred vs LSTM
    - [X] Hugging Face Spaces 배포 (별도 레포지토리, app.py + requirements.txt 구조)