"""
LSTM 모델 정의
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM 기반 시계열 예측 모델
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int,
                 dropout: float = 0.2,
                 bidirectional: bool = False):
        """
        Parameters:
        -----------
        input_size : int
            입력 특성 개수
        hidden_size : int
            LSTM 은닉층 유닛 수
        num_layers : int
            LSTM 레이어 수
        output_size : int
            출력 크기
        dropout : float
            드롭아웃 비율
        bidirectional : bool
            양방향 LSTM 사용 여부
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 배치 정규화
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.batch_norm = nn.BatchNorm1d(lstm_output_size)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 완전 연결 레이어
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(lstm_output_size // 2, output_size)
    
    def forward(self, x):
        """
        순전파
        
        Parameters:
        -----------
        x : Tensor
            입력 텐서 (batch_size, sequence_length, input_size)
            
        Returns:
        --------
        Tensor : 출력 텐서 (batch_size, output_size)
        """
        # LSTM 순전파
        lstm_out, _ = self.lstm(x)
        
        # 마지막 시간 스텝의 출력 사용
        out = lstm_out[:, -1, :]
        
        # 배치 정규화
        out = self.batch_norm(out)
        
        # 드롭아웃
        out = self.dropout(out)
        
        # 완전 연결 레이어
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
