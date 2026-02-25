"""
LSTM 모델 학습 스크립트
4개 모델 지원: flow, modelA, modelB, modelC
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.DL.pipeline import DLPipeline
from src.DL.config import PipelineConfig, DataConfig, FeatureConfig, WindowConfig, SplitConfig, LSTMConfig, TrainingConfig
from src.DL.model_config import get_available_models, print_model_info


def main():
    """메인 실행 함수"""
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="LSTM 모델 학습")
    parser.add_argument("--model", type=str, default="flow", 
                       choices=get_available_models(),
                       help="모델 이름 (flow, modelA, modelB, modelC)")
    parser.add_argument("--window-size", type=int, default=24,
                       help="슬라이딩 윈도우 크기 (기본: 24시간)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="최대 에포크 수 (기본: 100)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="배치 크기 (기본: 32)")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="디바이스 (cuda 또는 cpu)")
    parser.add_argument("--list-models", action="store_true",
                       help="사용 가능한 모델 목록 출력")
    
    args = parser.parse_args()
    
    # 모델 목록 출력
    if args.list_models:
        print_model_info()
        return
    
    # 경로 설정
    data_dir = project_root / "data" / "actual"
    save_dir = project_root / "results" / "DL"
    
    # 설정 생성
    config = PipelineConfig(
        data=DataConfig(
            time_freq="1min",
            short_term_hours=3,
            medium_term_hours=12,
            long_term_hours=48,
            ewma_span=6,
            outlier_method="iqr",
            iqr_threshold=1.5,
            zscore_threshold=3.0,
            require_both=True,
            resample_freq="1h",
            resample_agg="mean"
        ),
        feature=FeatureConfig(
            add_time_features=True,
            add_sin_cos=True,
            lag_hours=[1, 2, 3, 6, 12, 24],
            rolling_windows=[3, 6, 12, 24],
            rolling_stats=["mean", "std"]
        ),
        window=WindowConfig(
            window_size=args.window_size,
            horizon=1,
            stride=1
        ),
        split=SplitConfig(
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15
        ),
        lstm=LSTMConfig(
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            bidirectional=False
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            learning_rate=0.001,
            num_epochs=args.epochs,
            patience=10,
            grad_clip=1.0,
            weight_decay=0.0001
        ),
        random_seed=42,
        device=args.device
    )
    
    # 파이프라인 실행
    pipeline = DLPipeline(config)
    pipeline.run(data_dir, args.model, save_dir)


if __name__ == "__main__":
    main()
