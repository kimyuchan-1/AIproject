"""
하수처리장 딥러닝 예측 시스템 설정 파일

이 모듈은 LSTM 기반 예측 모델을 위한 모든 하이퍼파라미터, 파일 경로,
시스템 설정을 포함합니다.

요구사항: 1.2, 1.3, 1.4, 6.5, 10.4
"""

import os
from pathlib import Path

# ============================================================================
# 디렉토리 경로
# ============================================================================

# 기본 디렉토리
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results" / "DL"
NOTEBOOK_DIR = BASE_DIR / "notebook" / "DL"

# 데이터 하위 디렉토리 (요구사항 1.2, 1.3, 1.4)
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 모델 저장 디렉토리 (요구사항 6.5)
MODEL_SAVE_DIR = MODEL_DIR
SCALER_SAVE_DIR = MODEL_DIR

# 결과 저장 디렉토리 (요구사항 10.4)
RESULTS_SAVE_DIR = RESULTS_DIR
PLOTS_SAVE_DIR = RESULTS_DIR / "plots"
METRICS_SAVE_DIR = RESULTS_DIR / "metrics"

# ============================================================================
# 모델 하이퍼파라미터
# ============================================================================

# LSTM 구조
LSTM_CONFIG = {
    "hidden_size": 64,          # LSTM 은닉층 유닛 수
    "num_layers": 2,            # 쌓인 LSTM 레이어 수
    "dropout": 0.2,             # 정규화를 위한 드롭아웃 비율
    "output_size": 1,           # 출력 차원 (단일 타겟)
    "bidirectional": False,     # 양방향 LSTM 사용 여부
}

# 슬라이딩 윈도우
WINDOW_SIZE = 24                # 입력 시퀀스의 시간 스텝 수

# ============================================================================
# 학습 하이퍼파라미터
# ============================================================================

TRAINING_CONFIG = {
    "batch_size": 32,           # 학습 배치 크기
    "learning_rate": 0.001,     # 옵티마이저 학습률
    "num_epochs": 100,          # 최대 학습 에포크 수
    "patience": 10,             # 조기 종료 patience
    "optimizer": "adam",        # 옵티마이저 타입: 'adam', 'rmsprop', 'sgd'
    "loss_function": "mse",     # 손실 함수: 'mse' 또는 'mae'
}

# ============================================================================
# 데이터 분할 비율
# ============================================================================

SPLIT_RATIOS = {
    "train": 0.7,               # 학습 세트 비율
    "val": 0.15,                # 검증 세트 비율
    "test": 0.15,               # 테스트 세트 비율
}

# ============================================================================
# 타겟 변수
# ============================================================================

# 유량 예측 타겟
FLOW_TARGET = "Q_in"

# TMS 예측 타겟
TMS_TARGETS = [
    "TOC_VU",
    "PH_VU",
    "SS_VU",
    "FLUX_VU",
    "TN_VU",
    "TP_VU",
]

# ============================================================================
# 디바이스 설정
# ============================================================================

# GPU/CPU 디바이스 선택 (자동 감지)
DEVICE_CONFIG = {
    "use_gpu": True,            # 가능한 경우 GPU 사용 시도
    "gpu_id": 0,                # GPU 디바이스 ID
}

# ============================================================================
# 시각화 설정
# ============================================================================

VISUALIZATION_CONFIG = {
    "dpi": 300,                 # 플롯 해상도
    "figsize": (10, 6),         # 그림 크기 (너비, 높이)
    "font_family": "Malgun Gothic",  # 한글 폰트 지원
    "grid": True,               # 플롯에 그리드 표시
}

# ============================================================================
# 재현성을 위한 랜덤 시드
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# 성능 목표
# ============================================================================

PERFORMANCE_TARGETS = {
    "Q_in_r2": 0.95,            # 유량 예측 R² 목표
    "TMS_r2": 0.90,             # TMS 예측 R² 목표
}

# ============================================================================
# 유틸리티 함수
# ============================================================================

def create_directories():
    """
    필요한 모든 디렉토리가 없으면 생성합니다.
    
    이 함수는 학습 시작 전에 모델 저장 디렉토리와 결과 디렉토리가
    사용 가능한지 확인합니다.
    """
    directories = [
        MODEL_SAVE_DIR,
        SCALER_SAVE_DIR,
        RESULTS_SAVE_DIR,
        PLOTS_SAVE_DIR,
        METRICS_SAVE_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ 디렉토리 준비 완료: {directory}")


def get_model_path(target_variable: str, suffix: str = "") -> Path:
    """
    모델 저장/로드를 위한 파일 경로를 가져옵니다.
    
    Args:
        target_variable: 타겟 변수 이름 (예: 'Q_in', 'TOC_VU')
        suffix: 파일명에 추가할 선택적 접미사 (예: 'best', 'final')
    
    Returns:
        모델 파일의 Path 객체
    """
    filename = f"lstm_{target_variable}"
    if suffix:
        filename += f"_{suffix}"
    filename += ".pth"
    
    return MODEL_SAVE_DIR / filename


def get_scaler_path(scaler_name: str) -> Path:
    """
    스케일러 저장/로드를 위한 파일 경로를 가져옵니다.
    
    Args:
        scaler_name: 스케일러 이름 (예: 'X_scaler', 'y_scaler_Q_in')
    
    Returns:
        스케일러 파일의 Path 객체
    """
    filename = f"{scaler_name}.pkl"
    return SCALER_SAVE_DIR / filename


def get_plot_path(plot_name: str, target_variable: str = None) -> Path:
    """
    플롯 저장을 위한 파일 경로를 가져옵니다.
    
    Args:
        plot_name: 플롯 이름 (예: 'training_history', 'predictions')
        target_variable: 선택적 타겟 변수 이름
    
    Returns:
        플롯 파일의 Path 객체
    """
    filename = plot_name
    if target_variable:
        filename += f"_{target_variable}"
    filename += ".png"
    
    return PLOTS_SAVE_DIR / filename


def get_metrics_path(target_variable: str) -> Path:
    """
    평가 지표 저장을 위한 파일 경로를 가져옵니다.
    
    Args:
        target_variable: 타겟 변수 이름
    
    Returns:
        지표 파일의 Path 객체
    """
    filename = f"metrics_{target_variable}.json"
    return METRICS_SAVE_DIR / filename


def validate_config():
    """
    설정 파라미터를 검증합니다.
    
    Raises:
        ValueError: 설정 파라미터가 유효하지 않은 경우
    """
    # LSTM 설정 검증
    if LSTM_CONFIG["hidden_size"] < 1:
        raise ValueError(f"유효하지 않은 hidden_size: {LSTM_CONFIG['hidden_size']}")
    if LSTM_CONFIG["num_layers"] < 1:
        raise ValueError(f"유효하지 않은 num_layers: {LSTM_CONFIG['num_layers']}")
    if not (0 <= LSTM_CONFIG["dropout"] <= 1):
        raise ValueError(f"유효하지 않은 dropout: {LSTM_CONFIG['dropout']}")
    
    # 윈도우 크기 검증
    if WINDOW_SIZE < 1:
        raise ValueError(f"유효하지 않은 window_size: {WINDOW_SIZE}")
    
    # 학습 설정 검증
    if TRAINING_CONFIG["batch_size"] < 1:
        raise ValueError(f"유효하지 않은 batch_size: {TRAINING_CONFIG['batch_size']}")
    if TRAINING_CONFIG["learning_rate"] <= 0:
        raise ValueError(f"유효하지 않은 learning_rate: {TRAINING_CONFIG['learning_rate']}")
    if TRAINING_CONFIG["num_epochs"] < 1:
        raise ValueError(f"유효하지 않은 num_epochs: {TRAINING_CONFIG['num_epochs']}")
    if TRAINING_CONFIG["patience"] < 1:
        raise ValueError(f"유효하지 않은 patience: {TRAINING_CONFIG['patience']}")
    
    # 분할 비율 검증
    total_ratio = sum(SPLIT_RATIOS.values())
    if not (0.99 <= total_ratio <= 1.01):  # 작은 부동소수점 오차 허용
        raise ValueError(f"분할 비율의 합은 1.0이어야 합니다. 현재: {total_ratio}")
    
    print("✓ 설정 검증 완료")


if __name__ == "__main__":
    """
    이 스크립트를 실행하여 디렉토리를 생성하고 설정을 검증합니다.
    """
    print("=" * 60)
    print("하수처리장 딥러닝 예측 시스템 - 설정")
    print("=" * 60)
    print()
    
    print("디렉토리 생성 중...")
    create_directories()
    print()
    
    print("설정 검증 중...")
    validate_config()
    print()
    
    print("설정 요약:")
    print(f"  윈도우 크기: {WINDOW_SIZE}")
    print(f"  LSTM 은닉층 크기: {LSTM_CONFIG['hidden_size']}")
    print(f"  LSTM 레이어 수: {LSTM_CONFIG['num_layers']}")
    print(f"  드롭아웃: {LSTM_CONFIG['dropout']}")
    print(f"  배치 크기: {TRAINING_CONFIG['batch_size']}")
    print(f"  학습률: {TRAINING_CONFIG['learning_rate']}")
    print(f"  최대 에포크: {TRAINING_CONFIG['num_epochs']}")
    print(f"  조기 종료 Patience: {TRAINING_CONFIG['patience']}")
    print()
    
    print("=" * 60)
    print("설정 완료!")
    print("=" * 60)
