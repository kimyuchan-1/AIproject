"""
모델 학습 모듈
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, Any
from pathlib import Path


class Trainer:
    """LSTM 모델 학습 클래스"""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = "cuda",
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0001,
                 grad_clip: float = 1.0):
        """
        Parameters:
        -----------
        model : nn.Module
            학습할 모델
        device : str
            디바이스 ("cuda" or "cpu")
        learning_rate : float
            학습률
        weight_decay : float
            L2 정규화 계수
        grad_clip : float
            그래디언트 클리핑 값
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.grad_clip = grad_clip
        
        self.train_losses = []
        self.valid_losses = []
        self.best_valid_loss = float("inf")
        self.best_model_state = None
    
    def create_data_loader(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          batch_size: int,
                          shuffle: bool = False) -> DataLoader:
        """
        DataLoader 생성
        
        Parameters:
        -----------
        X : ndarray
            입력 데이터
        y : ndarray
            타겟 데이터
        batch_size : int
            배치 크기
        shuffle : bool
            셔플 여부
            
        Returns:
        --------
        DataLoader
        """
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        1 에포크 학습
        
        Parameters:
        -----------
        train_loader : DataLoader
            학습 데이터 로더
            
        Returns:
        --------
        float : 평균 손실
        """
        self.model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # 순전파
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, valid_loader: DataLoader) -> float:
        """
        검증
        
        Parameters:
        -----------
        valid_loader : DataLoader
            검증 데이터 로더
            
        Returns:
        --------
        float : 평균 손실
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(valid_loader)
    
    def fit(self,
            train_loader: DataLoader,
            valid_loader: DataLoader,
            num_epochs: int,
            patience: int = 10,
            verbose: bool = True) -> Dict[str, Any]:
        """
        모델 학습
        
        Parameters:
        -----------
        train_loader : DataLoader
            학습 데이터 로더
        valid_loader : DataLoader
            검증 데이터 로더
        num_epochs : int
            최대 에포크 수
        patience : int
            조기 종료 patience
        verbose : bool
            진행 상황 출력 여부
            
        Returns:
        --------
        dict : 학습 히스토리
        """
        patience_counter = 0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"학습 시작 (최대 {num_epochs} 에포크, patience={patience})")
            print(f"{'='*60}")
        
        for epoch in range(num_epochs):
            # 학습
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 검증
            valid_loss = self.validate(valid_loader)
            self.valid_losses.append(valid_loss)
            
            # 진행 상황 출력
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - "
                      f"Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}")
            
            # 조기 종료 체크
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"\n에포크 {epoch+1}에서 조기 종료")
                    print(f"최고 검증 손실: {self.best_valid_loss:.6f}")
                break
        
        # 최고 성능 모델 로드
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        if verbose:
            print(f"{'='*60}")
            print("학습 완료")
            print(f"{'='*60}\n")
        
        return {
            "train_losses": self.train_losses,
            "valid_losses": self.valid_losses,
            "best_valid_loss": self.best_valid_loss
        }
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        예측
        
        Parameters:
        -----------
        data_loader : DataLoader
            데이터 로더
            
        Returns:
        --------
        tuple : (예측값, 실제값)
        """
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(y_batch.numpy())
        
        return np.array(predictions), np.array(actuals)
    
    def save_model(self, save_path: Path, **kwargs):
        """
        모델 저장
        
        Parameters:
        -----------
        save_path : Path
            저장 경로
        **kwargs : dict
            추가 저장 정보
        """
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "valid_losses": self.valid_losses,
            "best_valid_loss": self.best_valid_loss,
        }
        save_dict.update(kwargs)
        
        torch.save(save_dict, save_path)
        print(f"모델 저장 완료: {save_path}")
    
    def load_model(self, load_path: Path):
        """
        모델 로드
        
        Parameters:
        -----------
        load_path : Path
            로드 경로
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.valid_losses = checkpoint.get("valid_losses", [])
        self.best_valid_loss = checkpoint.get("best_valid_loss", float("inf"))
        print(f"모델 로드 완료: {load_path}")
