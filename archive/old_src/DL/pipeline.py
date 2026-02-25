"""
LSTM 딥러닝 파이프라인
전체 처리 과정을 순차적으로 실행
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

from .config import PipelineConfig, get_default_config
from .data_loader import load_raw_data, set_datetime_index, align_time_index, merge_dataframes
from .preprocessing import impute_missing, handle_outliers, resample_data
from .features import create_features
from .sliding_window import create_sliding_windows, split_windowed_data
from .model import LSTMModel
from .trainer import Trainer
from .evaluator import (
    evaluate_model, print_metrics, save_metrics, save_predictions,
    plot_training_history, plot_predictions
)


class DLPipeline:
    """LSTM 딥러닝 파이프라인"""
    
    def __init__(self, config: PipelineConfig = None):
        """
        Parameters:
        -----------
        config : PipelineConfig
            파이프라인 설정
        """
        self.config = config if config is not None else get_default_config()
        self.set_random_seed(self.config.random_seed)
        
        # 데이터 저장
        self.df_raw = None
        self.df_aligned = None
        self.df_imputed = None
        self.df_outlier_handled = None
        self.df_resampled = None
        self.df_features = None
        
        # 윈도우 데이터
        self.X_seq = None
        self.y_seq = None
        self.splits = None
        
        # 스케일러
        self.X_scaler = None
        self.y_scaler = None
        
        # 모델 및 학습기
        self.model = None
        self.trainer = None
        
        # 결과
        self.results = None
        self.predictions = None
    
    def set_random_seed(self, seed: int):
        """랜덤 시드 설정"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def step1_load_data(self, data_dir: Path):
        """1. 데이터 로드"""
        print(f"\n{'='*60}")
        print("1. 데이터 로드")
        print(f"{'='*60}")
        
        dfs = load_raw_data(data_dir)
        print(f"로드된 데이터: {list(dfs.keys())}")
        
        # 시간 인덱스 설정
        time_col_map = {
            "flow": "SYS_TIME",
            "tms": "SYS_TIME",
            "aws_368": "datetime",
            "aws_541": "datetime",
            "aws_569": "datetime"
            # weather는 사용하지 않음
        }
        
        for name, df in dfs.items():
            if name in time_col_map and time_col_map[name] in df.columns:
                dfs[name] = set_datetime_index(df, time_col_map[name])
            else:
                print(f"경고: {name}에 시간 컬럼이 없습니다. 사용 가능한 컬럼: {df.columns.tolist()}")
        
        self.df_raw = dfs
        print("데이터 로드 완료")
    
    def step2_align_time(self):
        """2. 시간축 정합 (1분 간격)"""
        print(f"\n{'='*60}")
        print("2. 시간축 정합")
        print(f"{'='*60}")
        
        # FLOW 데이터에 Q_in 생성 (flow_TankA + flow_TankB)
        if "flow" in self.df_raw:
            flow_df = self.df_raw["flow"]
            if "flow_TankA" in flow_df.columns and "flow_TankB" in flow_df.columns:
                flow_df["Q_in"] = flow_df["flow_TankA"] + flow_df["flow_TankB"]
                print("Q_in 생성 완료 (flow_TankA + flow_TankB)")
        
        aligned_dfs = align_time_index(self.df_raw, freq=self.config.data.time_freq)
        self.df_aligned = merge_dataframes(aligned_dfs)
        
        print(f"정합된 데이터 shape: {self.df_aligned.shape}")
        print("시간축 정합 완료")
    
    def step3_impute_missing(self):
        """3. 결측치 처리"""
        print(f"\n{'='*60}")
        print("3. 결측치 처리")
        print(f"{'='*60}")
        
        from .preprocessing import ImputationConfig
        
        impute_config = ImputationConfig(
            short_term_hours=self.config.data.short_term_hours,
            medium_term_hours=self.config.data.medium_term_hours,
            long_term_hours=self.config.data.long_term_hours,
            ewma_span=self.config.data.ewma_span
        )
        
        self.df_imputed, mask_impute = impute_missing(
            self.df_aligned,
            freq=self.config.data.time_freq,
            config=impute_config
        )
        
        print(f"결측치 처리 후 shape: {self.df_imputed.shape}")
        print("결측치 처리 완료")
    
    def step4_handle_outliers(self):
        """4. 이상치 처리"""
        print(f"\n{'='*60}")
        print("4. 이상치 처리")
        print(f"{'='*60}")
        
        from .preprocessing import OutlierConfig
        
        outlier_config = OutlierConfig(
            method=self.config.data.outlier_method,
            iqr_threshold=self.config.data.iqr_threshold,
            zscore_threshold=self.config.data.zscore_threshold,
            require_both=self.config.data.require_both
        )
        
        self.df_outlier_handled, mask_outlier = handle_outliers(
            self.df_imputed,
            config=outlier_config
        )
        
        print(f"이상치 처리 후 shape: {self.df_outlier_handled.shape}")
        print("이상치 처리 완료")
    
    def step5_resample(self):
        """5. 리샘플링"""
        print(f"\n{'='*60}")
        print("5. 리샘플링")
        print(f"{'='*60}")
        
        self.df_resampled = resample_data(
            self.df_outlier_handled,
            freq=self.config.data.resample_freq,
            agg=self.config.data.resample_agg
        )
        
        print(f"리샘플링 후 shape: {self.df_resampled.shape}")
        print(f"리샘플링 주기: {self.config.data.resample_freq}")
        print("리샘플링 완료")
    
    def step6_create_features(self, model_mode: str, target_cols: list):
        """6. 특성 생성"""
        print(f"\n{'='*60}")
        print("6. 특성 생성")
        print(f"{'='*60}")
        print(f"모델 모드: {model_mode}")
        print(f"타겟 컬럼: {target_cols}")
        print(f"리샘플링 후 컬럼 수: {len(self.df_resampled.columns)}")
        print(f"사용 가능한 컬럼: {self.df_resampled.columns.tolist()[:10]}... (처음 10개)")
        
        self.df_features = create_features(
            self.df_resampled,
            model_mode=model_mode,
            target_cols=target_cols,
            add_time=self.config.feature.add_time_features,
            add_sin_cos=self.config.feature.add_sin_cos,
            lag_hours=self.config.feature.lag_hours,
            rolling_windows=self.config.feature.rolling_windows,
            rolling_stats=self.config.feature.rolling_stats
        )
        
        print(f"특성 생성 후 shape: {self.df_features.shape}")
        
        # NaN 발생 확인
        nan_counts = self.df_features.isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        if len(cols_with_nan) > 0:
            print(f"⚠ NaN이 있는 컬럼: {len(cols_with_nan)}개")
            print(f"  상위 5개: {cols_with_nan.head().to_dict()}")
        
        print("특성 생성 완료")
    
    def step7_create_windows(self, target_cols: list):
        """7. 슬라이딩 윈도우 생성"""
        print(f"\n{'='*60}")
        print("7. 슬라이딩 윈도우 생성")
        print(f"{'='*60}")
     
        # 타겟에 결측이 있는 행만 제거
        df_clean = self.df_features.dropna(subset=target_cols)
        print(f"타겟 결측 제거 후 shape: {df_clean.shape}")
        
        if len(df_clean) == 0:
            raise ValueError(
                f"타겟 컬럼 {target_cols}에 유효한 행이 없습니다. "
                "원본 데이터 또는 전처리 단계를 확인하세요."
            )
        
        # X, y 분리
        X_df = df_clean.drop(columns=target_cols)
        y_df = df_clean[target_cols]
        
        # 특성 행렬의 NaN 처리
        print(f"특성 NaN 개수 (처리 전): {X_df.isna().sum().sum()}")
        
        # Forward fill로 채우기
        X_df = X_df.ffill()
        
        # 여전히 NaN이 있으면 backward fill
        if X_df.isna().sum().sum() > 0:
            print(f"  Forward fill 후 남은 NaN: {X_df.isna().sum().sum()}")
            X_df = X_df.bfill()
        
        # 그래도 NaN이 있으면 0으로 채우기
        if X_df.isna().sum().sum() > 0:
            print(f"  Backward fill 후 남은 NaN: {X_df.isna().sum().sum()}")
            X_df = X_df.fillna(0)
        
        print(f"특성 NaN 개수 (처리 후): {X_df.isna().sum().sum()}")
        
        # 데이터 통계 확인
        print(f"\n데이터 통계:")
        print(f"  X shape: {X_df.shape}")
        print(f"  y shape: {y_df.shape}")
        print(f"  X 범위: [{X_df.min().min():.2f}, {X_df.max().max():.2f}]")
        print(f"  y 범위: [{y_df.min().min():.2f}, {y_df.max().max():.2f}]")
        print(f"  X에 inf 있음: {np.isinf(X_df.values).any()}")
        print(f"  y에 inf 있음: {np.isinf(y_df.values).any()}")
        
        X = X_df.values
        y = y_df.values
        
        # 슬라이딩 윈도우 생성
        self.X_seq, self.y_seq = create_sliding_windows(
            X, y,
            window_size=self.config.window.window_size,
            horizon=self.config.window.horizon,
            stride=self.config.window.stride
        )
        
        print(f"\n윈도우 shape: X={self.X_seq.shape}, y={self.y_seq.shape}")
        print(f"윈도우 크기: {self.config.window.window_size}")
        print("슬라이딩 윈도우 생성 완료")
        print("슬라이딩 윈도우 생성 완료")
    
    def step8_scale_data(self):
        """8. 데이터 스케일링"""
        print(f"\n{'='*60}")
        print("8. 데이터 스케일링")
        print(f"{'='*60}")
        
        # 데이터 분할
        self.splits = split_windowed_data(
            self.X_seq, self.y_seq,
            train_ratio=self.config.split.train_ratio,
            valid_ratio=self.config.split.valid_ratio,
            test_ratio=self.config.split.test_ratio
        )
        
        X_train, y_train = self.splits["train"]
        X_valid, y_valid = self.splits["valid"]
        X_test, y_test = self.splits["test"]
        
        print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")
        
        # X 스케일링 (3D -> 2D -> 스케일링 -> 3D)
        n_train, window_size, n_features = X_train.shape
        n_valid = X_valid.shape[0]
        n_test = X_test.shape[0]
        
        X_train_2d = X_train.reshape(-1, n_features)
        X_valid_2d = X_valid.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)
        
        self.X_scaler = StandardScaler()
        X_train_scaled_2d = self.X_scaler.fit_transform(X_train_2d)
        X_valid_scaled_2d = self.X_scaler.transform(X_valid_2d)
        X_test_scaled_2d = self.X_scaler.transform(X_test_2d)
        
        X_train_scaled = X_train_scaled_2d.reshape(n_train, window_size, n_features)
        X_valid_scaled = X_valid_scaled_2d.reshape(n_valid, window_size, n_features)
        X_test_scaled = X_test_scaled_2d.reshape(n_test, window_size, n_features)
        
        # y 스케일링
        self.y_scaler = StandardScaler()
        y_train_scaled = self.y_scaler.fit_transform(y_train)
        y_valid_scaled = self.y_scaler.transform(y_valid)
        y_test_scaled = self.y_scaler.transform(y_test)
        
        # 스케일링된 데이터로 업데이트
        self.splits["train"] = (X_train_scaled, y_train_scaled)
        self.splits["valid"] = (X_valid_scaled, y_valid_scaled)
        self.splits["test"] = (X_test_scaled, y_test_scaled)
        
        print("데이터 스케일링 완료")
    
    def step9_train_model(self):
        """9. 모델 학습"""
        print(f"\n{'='*60}")
        print("9. 모델 학습")
        print(f"{'='*60}")
        
        # 모델 생성
        X_train, y_train = self.splits["train"]
        input_size = X_train.shape[2]
        output_size = y_train.shape[1]
        
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.config.lstm.hidden_size,
            num_layers=self.config.lstm.num_layers,
            output_size=output_size,
            dropout=self.config.lstm.dropout,
            bidirectional=self.config.lstm.bidirectional
        )
        
        print(f"모델 구조:")
        print(self.model)
        print(f"\n전체 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 학습기 생성
        self.trainer = Trainer(
            model=self.model,
            device=self.config.device,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            grad_clip=self.config.training.grad_clip
        )
        
        # DataLoader 생성
        train_loader = self.trainer.create_data_loader(
            X_train, y_train,
            batch_size=self.config.training.batch_size,
            shuffle=False
        )
        
        X_valid, y_valid = self.splits["valid"]
        valid_loader = self.trainer.create_data_loader(
            X_valid, y_valid,
            batch_size=self.config.training.batch_size,
            shuffle=False
        )
        
        # 학습
        history = self.trainer.fit(
            train_loader, valid_loader,
            num_epochs=self.config.training.num_epochs,
            patience=self.config.training.patience,
            verbose=True
        )
        
        print("모델 학습 완료")
        return history
    
    def step10_evaluate(self):
        """10. 모델 평가"""
        print(f"\n{'='*60}")
        print("10. 모델 평가")
        print(f"{'='*60}")
        
        # 예측
        self.predictions = {}
        
        for split_name, (X, y) in self.splits.items():
            data_loader = self.trainer.create_data_loader(
                X, y,
                batch_size=self.config.training.batch_size,
                shuffle=False
            )
            
            y_pred_scaled, y_true_scaled = self.trainer.predict(data_loader)
            
            # 역스케일링
            y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
            y_true = self.y_scaler.inverse_transform(y_true_scaled)
            
            self.predictions[split_name] = (y_true, y_pred)
        
        # 평가 지표 계산
        self.results = evaluate_model(self.predictions)
        print_metrics(self.results)
        
        print("모델 평가 완료")
    
    def step11_save_results(self, save_dir: Path):
        """11. 결과 저장"""
        print(f"\n{'='*60}")
        print("11. 결과 저장")
        print(f"{'='*60}")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 평가 지표 저장
        save_metrics(self.results, save_dir / "metrics.json")
        
        # 예측값 저장
        for split_name, (y_true, y_pred) in self.predictions.items():
            save_predictions(
                y_true, y_pred,
                save_dir / f"predictions_{split_name}.csv"
            )
        
        # 학습 히스토리 플롯
        plot_training_history(
            self.trainer.train_losses,
            self.trainer.valid_losses,
            save_dir / "training_history.png"
        )
        
        # 예측 플롯
        y_true_test, y_pred_test = self.predictions["test"]
        plot_predictions(
            y_true_test, y_pred_test,
            save_dir / "predictions_test.png",
            title="Test Set Predictions"
        )
        
        # 모델 저장
        self.trainer.save_model(
            save_dir / "model.pth",
            input_size=self.X_seq.shape[2],
            output_size=self.y_seq.shape[1],
            config=self.config
        )
        
        # 스케일러 저장
        with open(save_dir / "X_scaler.pkl", "wb") as f:
            pickle.dump(self.X_scaler, f)
        with open(save_dir / "y_scaler.pkl", "wb") as f:
            pickle.dump(self.y_scaler, f)
        
        print(f"결과 저장 완료: {save_dir}")
    
    def run(self, data_dir: Path, model_name: str, save_dir: Path):
        """
        전체 파이프라인 실행
        
        Parameters:
        -----------
        data_dir : Path
            데이터 디렉토리
        model_name : str
            모델 이름 (flow, modelA, modelB, modelC)
        save_dir : Path
            결과 저장 디렉토리
        """
        from .model_config import get_model_spec
        
        # 모델 사양 가져오기
        model_spec = get_model_spec(model_name)
        
        print(f"\n{'#'*60}")
        print(f"LSTM 딥러닝 파이프라인 시작 - {model_spec.name}")
        print(f"타겟: {', '.join(model_spec.target_cols)}")
        print(f"설명: {model_spec.description}")
        print(f"{'#'*60}")
        
        # 1. 데이터 로드
        self.step1_load_data(data_dir)
        
        # 2. 시간축 정합
        self.step2_align_time()
        
        # 3. 결측치 처리
        self.step3_impute_missing()
        
        # 4. 이상치 처리
        self.step4_handle_outliers()
        
        # 5. 리샘플링
        self.step5_resample()
        
        # 6. 특성 생성 (모델별)
        self.step6_create_features(model_spec.mode, model_spec.target_cols)
        
        # 7. 슬라이딩 윈도우 생성
        self.step7_create_windows(model_spec.target_cols)
        
        # 8. 데이터 스케일링
        self.step8_scale_data()
        
        # 9. 모델 학습
        self.step9_train_model()
        
        # 10. 모델 평가
        self.step10_evaluate()
        
        # 11. 결과 저장
        save_dir_with_model = save_dir / model_name
        self.step11_save_results(save_dir_with_model)
        
        print(f"\n{'#'*60}")
        print(f"LSTM 딥러닝 파이프라인 완료 - {model_spec.name}")
        print(f"{'#'*60}\n")
