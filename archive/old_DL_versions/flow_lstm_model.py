"""
유입 유량 예측을 위한 LSTM 모델 (y_Q_in = y_flowA + y_flowB)

이 모듈은 시계열 데이터로부터 총 유입량(y_Q_in)을 예측하기 위한 PyTorch LSTM 모델을 구현합니다.
y_Q_in은 y_flowA와 y_flowB의 합입니다.

구조:
1. 필요한 패키지 import
2. Configuration 설정 (디렉토리, 하이퍼파라미터, device)
3. 재현성을 위한 랜덤 시드 설정
4. load_data() 함수 - target과 input 분리
5. Train/validation/test 분할 (60%/20%/20%)
6. FlowLSTM 클래스 정의 (__init__(), forward() 함수)
7. evaluate() 함수 정의
8. 메인 실행 블록
"""

# ============================================================================
# 1. 필요한 패키지 import
# ============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# PyTorch 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Scikit-learn 라이브러리
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 설정 파일 import
from config import (
    PROCESSED_DATA_DIR,
    MODEL_SAVE_DIR,
    RESULTS_SAVE_DIR,
    RANDOM_SEED,
    WINDOW_SIZE,
    LSTM_CONFIG,
    TRAINING_CONFIG,
)

# ============================================================================
# 2. Configuration 설정
# ============================================================================

# 디렉토리 경로
DATA_PATH = PROCESSED_DATA_DIR / "modelFLOW_dataset.csv"
MODEL_DIR = MODEL_SAVE_DIR
RESULTS_DIR = RESULTS_SAVE_DIR

# 디렉토리가 없으면 생성
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 하이퍼파라미터
SEQUENCE_LENGTH = WINDOW_SIZE  # 시퀀스 길이 (과거 몇 개의 시점을 볼 것인지)
HIDDEN_SIZE = LSTM_CONFIG["hidden_size"]  # LSTM 은닉층 유닛 수
NUM_LAYERS = LSTM_CONFIG["num_layers"]  # LSTM 레이어 수
DROPOUT = LSTM_CONFIG["dropout"]  # 드롭아웃 비율
OUTPUT_SIZE = 1  # 출력 크기 (y_Q_in 하나)

BATCH_SIZE = TRAINING_CONFIG["batch_size"]
LEARNING_RATE = TRAINING_CONFIG["learning_rate"]
NUM_EPOCHS = TRAINING_CONFIG["num_epochs"]
PATIENCE = TRAINING_CONFIG["patience"]

# 디바이스 설정 (GPU/CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
# 3. 재현성을 위한 랜덤 시드 설정
# ============================================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"랜덤 시드 설정: {RANDOM_SEED}")

# ============================================================================
# 4. load_data() 함수 - target과 input 분리
# ============================================================================

def load_data(data_path):
    """
    데이터셋을 로드하고 특성(X)과 타겟(y_Q_in)을 분리합니다.
    
    Args:
        data_path: CSV 파일 경로
        
    Returns:
        X: 입력 특성 (numpy array)
        y: 타겟 변수 y_Q_in (numpy array)
        feature_names: 특성 컬럼명 리스트
    """
    print(f"\n데이터 로드 중: {data_path}")
    
    # CSV 파일 로드
    df = pd.read_csv(data_path)
    print(f"데이터셋 크기: {df.shape}")
    
    # 타겟 변수 생성: y_Q_in = y_flowA + y_flowB
    if 'y_flowA' not in df.columns or 'y_flowB' not in df.columns:
        raise ValueError("데이터셋에 'y_flowA'와 'y_flowB' 컬럼이 필요합니다")
    
    df['y_Q_in'] = df['y_flowA'] + df['y_flowB']
    print(f"타겟 변수 생성 완료: y_Q_in (y_flowA + y_flowB)")
    
    # 특성과 타겟 분리
    # SYS_TIME, y_flowA, y_flowB, y_Q_in은 특성에서 제외
    exclude_cols = ['SYS_TIME', 'y_flowA', 'y_flowB', 'y_Q_in']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['y_Q_in'].values.reshape(-1, 1)
    
    print(f"특성 크기: {X.shape}")
    print(f"타겟 크기: {y.shape}")
    print(f"특성 개수: {len(feature_cols)}")
    
    return X, y, feature_cols

# ============================================================================
# 5. Train/Validation/Test 분할 (60%/20%/20%)
# ============================================================================

def split_data(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    데이터를 train, validation, test 세트로 분할합니다.
    
    Args:
        X: 입력 특성
        y: 타겟 변수
        train_ratio: 학습 세트 비율 (기본값: 0.6)
        val_ratio: 검증 세트 비율 (기본값: 0.2)
        test_ratio: 테스트 세트 비율 (기본값: 0.2)
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "분할 비율의 합은 1.0이어야 합니다"
    
    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    # 데이터 분할 (시계열 데이터이므로 셔플하지 않음)
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"\n데이터 분할:")
    print(f"  Train: {len(X_train)} 샘플 ({train_ratio*100:.0f}%)")
    print(f"  Validation: {len(X_val)} 샘플 ({val_ratio*100:.0f}%)")
    print(f"  Test: {len(X_test)} 샘플 ({test_ratio*100:.0f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_sequences(X, y, sequence_length):
    """
    LSTM 입력을 위한 시퀀스를 생성합니다.
    
    Args:
        X: 입력 특성
        y: 타겟 변수
        sequence_length: 각 시퀀스의 시간 스텝 수
        
    Returns:
        X_seq: 입력 특성 시퀀스
        y_seq: 대응하는 타겟 값
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    
    return np.array(X_seq), np.array(y_seq)

def prepare_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, 
                         sequence_length, batch_size):
    """
    학습, 검증, 테스트를 위한 DataLoader를 준비합니다.
    
    Args:
        X_train, X_val, X_test: 특성 배열
        y_train, y_val, y_test: 타겟 배열
        sequence_length: 시간 스텝 수
        batch_size: DataLoader의 배치 크기
        
    Returns:
        train_loader, val_loader, test_loader, X_scaler, y_scaler
    """
    # 특성 정규화
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)
    
    # 타겟 정규화
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)
    
    # 시퀀스 생성
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)
    
    print(f"\n시퀀스 크기:")
    print(f"  Train: {X_train_seq.shape}, {y_train_seq.shape}")
    print(f"  Validation: {X_val_seq.shape}, {y_val_seq.shape}")
    print(f"  Test: {X_test_seq.shape}, {y_test_seq.shape}")
    
    # PyTorch 텐서로 변환
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.FloatTensor(y_train_seq)
    X_val_tensor = torch.FloatTensor(X_val_seq)
    y_val_tensor = torch.FloatTensor(y_val_seq)
    X_test_tensor = torch.FloatTensor(X_test_seq)
    y_test_tensor = torch.FloatTensor(y_test_seq)
    
    # DataLoader 생성
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, X_scaler, y_scaler

# ============================================================================
# 6. FlowLSTM 클래스 정의
# ============================================================================

class FlowLSTM(nn.Module):
    """
    유입 유량 예측을 위한 LSTM 모델 (y_Q_in).
    
    구조:
        - 드롭아웃이 적용된 LSTM 레이어
        - 완전 연결 출력 레이어
    """
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        FlowLSTM 모델을 초기화합니다.
        
        Args:
            input_size: 입력 특성 개수
            hidden_size: LSTM 은닉층 유닛 수
            num_layers: 쌓인 LSTM 레이어 수
            output_size: 출력 특성 개수 (y_Q_in의 경우 1)
            dropout: 정규화를 위한 드롭아웃 비율
        """
        super(FlowLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 배치 정규화 (과적합 방지 및 학습 안정화)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # 드롭아웃 레이어
        self.dropout = nn.Dropout(dropout)
        
        # 완전 연결 레이어 (더 깊은 구조로 변경)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        """
        네트워크를 통한 순전파를 수행합니다.
        
        Args:
            x: 입력 텐서, 크기 (batch_size, sequence_length, input_size)
            
        Returns:
            출력 텐서, 크기 (batch_size, output_size)
        """
        # 은닉 상태와 셀 상태 초기화
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 순전파
        # out 크기: (batch_size, sequence_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # 마지막 시간 스텝의 출력 사용
        out = out[:, -1, :]
        
        # 배치 정규화 적용
        out = self.batch_norm(out)
        
        # 드롭아웃 적용
        out = self.dropout(out)
        
        # 완전 연결 레이어 (2층 구조)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# ============================================================================
# 7. evaluate() 함수 정의
# ============================================================================

def evaluate(model, data_loader, criterion, y_scaler, device):
    """
    데이터셋에서 모델을 평가합니다.
    
    Args:
        model: 학습된 모델
        data_loader: 평가용 DataLoader
        criterion: 손실 함수
        y_scaler: 예측값 역변환을 위한 스케일러
        device: 디바이스 (CPU/GPU)
        
    Returns:
        평가 지표를 포함한 딕셔너리
    """
    model.eval()
    
    predictions = []
    actuals = []
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # 순전파
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            # 예측값과 실제값 저장
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())
    
    # numpy 배열로 변환
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 원래 스케일로 역변환
    predictions_original = y_scaler.inverse_transform(predictions)
    actuals_original = y_scaler.inverse_transform(actuals)
    
    # 평가 지표 계산
    mse = mean_squared_error(actuals_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_original, predictions_original)
    r2 = r2_score(actuals_original, predictions_original)
    avg_loss = total_loss / len(data_loader)
    
    metrics = {
        'loss': avg_loss,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions_original,
        'actuals': actuals_original
    }
    
    return metrics


# ============================================================================
# 8. 메인 실행 블록
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, patience, device, y_scaler):
    """
    조기 종료를 적용하여 LSTM 모델을 학습합니다.
    
    Args:
        model: FlowLSTM 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        num_epochs: 최대 에포크 수
        patience: 조기 종료 patience
        device: 디바이스 (CPU/GPU)
        y_scaler: 타겟 변수 스케일러
        
    Returns:
        학습된 모델과 학습 히스토리
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 그래디언트 클리핑 값 가져오기
    grad_clip = TRAINING_CONFIG.get("grad_clip", None)
    
    print(f"\n{num_epochs} 에포크 동안 학습을 시작합니다...")
    print(f"조기 종료 patience: {patience}")
    if grad_clip:
        print(f"그래디언트 클리핑: {grad_clip}")
    
    for epoch in range(num_epochs):
        # 학습 단계
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # 순전파
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑 (그래디언트 폭발 방지)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 검증 단계
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 진행 상황 출력
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # 조기 종료 체크
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\n에포크 {epoch+1}에서 조기 종료 발동")
            print(f"최고 검증 손실: {best_val_loss:.6f}")
            break
    
    # 최고 성능 모델 로드
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }
    
    return model, history

def plot_training_history(history, save_path):
    """
    학습 및 검증 손실 곡선을 그립니다.
    
    Args:
        history: 학습 히스토리를 포함한 딕셔너리
        save_path: 플롯을 저장할 경로
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"학습 히스토리 플롯 저장 완료: {save_path}")

def plot_predictions(actuals, predictions, dataset_name, save_path):
    """
    실제값 vs 예측값을 그립니다.
    
    Args:
        actuals: 실제값
        predictions: 예측값
        dataset_name: 데이터셋 이름 (예: 'Test')
        save_path: 플롯을 저장할 경로
    """
    plt.figure(figsize=(12, 6))
    
    # 시계열 플롯
    plt.subplot(1, 2, 1)
    plt.plot(actuals, label='Actual', alpha=0.7)
    plt.plot(predictions, label='Predicted', alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('y_Q_in')
    plt.title(f'{dataset_name} Set: Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    
    # 산점도
    plt.subplot(1, 2, 2)
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([actuals.min(), actuals.max()], 
             [actuals.min(), actuals.max()], 
             'r--', lw=2)
    plt.xlabel('Actual y_Q_in')
    plt.ylabel('Predicted y_Q_in')
    plt.title(f'{dataset_name} Set: Scatter Plot')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"예측 플롯 저장 완료: {save_path}")

def main():
    """
    메인 실행 함수
    """
    print("=" * 80)
    print("유입 유량 예측을 위한 LSTM 모델 (y_Q_in)")
    print("=" * 80)
    
    # 데이터 로드
    X, y, feature_names = load_data(DATA_PATH)
    
    # 데이터 분할
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )
    
    # 데이터 로더 준비
    train_loader, val_loader, test_loader, X_scaler, y_scaler = prepare_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
    )
    
    # 데이터로부터 입력 크기 가져오기
    input_size = X.shape[1]
    
    # 모델 초기화
    model = FlowLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT
    ).to(DEVICE)
    
    print(f"\n모델 구조:")
    print(model)
    print(f"\n전체 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 손실 함수 및 옵티마이저
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=TRAINING_CONFIG.get("weight_decay", 0.0001)  # L2 정규화
    )
    
    # 모델 학습
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        device=DEVICE,
        y_scaler=y_scaler
    )
    
    # 학습 히스토리 플롯
    plot_training_history(
        history,
        save_path=RESULTS_DIR / "training_history.png"
    )
    
    # 모든 데이터셋에 대한 평가
    print("\n" + "=" * 80)
    print("평가 결과")
    print("=" * 80)
    
    # 학습 세트
    train_metrics = evaluate(model, train_loader, criterion, y_scaler, DEVICE)
    print(f"\n학습 세트:")
    print(f"  Loss: {train_metrics['loss']:.6f}")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  MAE: {train_metrics['mae']:.4f}")
    print(f"  R²: {train_metrics['r2']:.4f}")
    
    # 검증 세트
    val_metrics = evaluate(model, val_loader, criterion, y_scaler, DEVICE)
    print(f"\n검증 세트:")
    print(f"  Loss: {val_metrics['loss']:.6f}")
    print(f"  RMSE: {val_metrics['rmse']:.4f}")
    print(f"  MAE: {val_metrics['mae']:.4f}")
    print(f"  R²: {val_metrics['r2']:.4f}")
    
    # 테스트 세트
    test_metrics = evaluate(model, test_loader, criterion, y_scaler, DEVICE)
    print(f"\n테스트 세트:")
    print(f"  Loss: {test_metrics['loss']:.6f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    
    # 예측 플롯
    plot_predictions(
        test_metrics['actuals'],
        test_metrics['predictions'],
        dataset_name='Test',
        save_path=RESULTS_DIR / "test_predictions.png"
    )
    
    # 모델 저장
    model_path = MODEL_DIR / "flow_lstm_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'output_size': OUTPUT_SIZE,
        'dropout': DROPOUT,
        'sequence_length': SEQUENCE_LENGTH,
        'feature_names': feature_names,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
    }, model_path)
    print(f"\n모델 저장 완료: {model_path}")
    
    # 스케일러 저장
    import pickle
    with open(MODEL_DIR / "X_scaler.pkl", 'wb') as f:
        pickle.dump(X_scaler, f)
    with open(MODEL_DIR / "y_scaler.pkl", 'wb') as f:
        pickle.dump(y_scaler, f)
    print(f"스케일러 저장 완료: {MODEL_DIR}")
    
    print("\n" + "=" * 80)
    print("학습 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()
