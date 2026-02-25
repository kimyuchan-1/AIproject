"""
결과 저장 모듈
예측값, 시퀀스 데이터셋, 모델 등을 저장
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime


def save_predictions(
    y_true: Union[pd.DataFrame, np.ndarray],
    y_pred: np.ndarray,
    split_name: str,
    target_cols: list,
    save_dir: str,
    index: Optional[pd.Index] = None
) -> str:
    """
    예측값을 CSV 파일로 저장
    
    Parameters:
    -----------
    y_true : DataFrame or ndarray
        실제 값
    y_pred : ndarray
        예측 값
    split_name : str
        데이터 분할 이름 (train/valid/test)
    target_cols : list
        타겟 컬럼 이름
    save_dir : str
        저장 디렉토리
    index : Index, optional
        시간 인덱스
        
    Returns:
    --------
    str : 저장된 파일 경로
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # DataFrame 생성
    if isinstance(y_true, pd.DataFrame):
        df_true = y_true.copy()
        if index is not None:
            df_true.index = index
    else:
        df_true = pd.DataFrame(y_true, columns=target_cols, index=index)
    
    df_pred = pd.DataFrame(y_pred, columns=[f"{col}_pred" for col in target_cols], index=df_true.index)
    
    # 실제값과 예측값 병합
    df_result = pd.concat([df_true, df_pred], axis=1)
    
    # 오차 계산
    for col in target_cols:
        df_result[f"{col}_error"] = df_result[col] - df_result[f"{col}_pred"]
        df_result[f"{col}_error_pct"] = (df_result[f"{col}_error"] / df_result[col]) * 100
    
    # 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_{split_name}_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)
    df_result.to_csv(filepath)
    
    print(f"  💾 예측값 저장: {filepath}")
    return filepath


def save_sequence_dataset(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    split_name: str,
    feature_names: list,
    target_cols: list,
    window_size: int,
    save_dir: str,
    save_format: str = "npz"
) -> str:
    """
    시퀀스 데이터셋을 저장
    
    Parameters:
    -----------
    X_seq : ndarray
        입력 시퀀스 (samples, window_size, features) 또는 (samples, features)
    y_seq : ndarray
        타겟 시퀀스 (samples, targets)
    split_name : str
        데이터 분할 이름 (train/valid/test)
    feature_names : list
        특성 이름 리스트
    target_cols : list
        타겟 컬럼 이름
    window_size : int
        윈도우 크기
    save_dir : str
        저장 디렉토리
    save_format : str
        저장 형식 ('npz', 'pickle', 'csv')
        
    Returns:
    --------
    str : 저장된 파일 경로
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if save_format == "npz":
        # NumPy 압축 형식 (권장 - 빠르고 용량 작음)
        filename = f"sequence_{split_name}_{timestamp}.npz"
        filepath = os.path.join(save_dir, filename)
        
        np.savez_compressed(
            filepath,
            X=X_seq,
            y=y_seq,
            feature_names=feature_names,
            target_cols=target_cols,
            window_size=window_size,
            split_name=split_name
        )
        
    elif save_format == "pickle":
        # Pickle 형식
        filename = f"sequence_{split_name}_{timestamp}.pkl"
        filepath = os.path.join(save_dir, filename)
        
        data = {
            'X': X_seq,
            'y': y_seq,
            'feature_names': feature_names,
            'target_cols': target_cols,
            'window_size': window_size,
            'split_name': split_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    elif save_format == "csv":
        # CSV 형식 (사람이 읽기 쉬움, 용량 큼)
        filename = f"sequence_{split_name}_{timestamp}.csv"
        filepath = os.path.join(save_dir, filename)
        
        # 3D를 2D로 평탄화
        if X_seq.ndim == 3:
            n_samples, window_size, n_features = X_seq.shape
            X_flat = X_seq.reshape(n_samples, -1)
            
            # 특성 이름 생성
            flat_feature_names = []
            for t in range(window_size - 1, -1, -1):
                time_label = f"t-{t}" if t > 0 else "t0"
                for feat in feature_names:
                    flat_feature_names.append(f"{feat}_{time_label}")
        else:
            X_flat = X_seq
            flat_feature_names = feature_names
        
        # DataFrame 생성
        df_X = pd.DataFrame(X_flat, columns=flat_feature_names)
        df_y = pd.DataFrame(y_seq, columns=target_cols)
        df_result = pd.concat([df_X, df_y], axis=1)
        
        df_result.to_csv(filepath, index=False)
    
    else:
        raise ValueError(f"지원하지 않는 형식: {save_format}. 'npz', 'pickle', 'csv' 중 선택하세요.")
    
    print(f"  💾 시퀀스 데이터 저장: {filepath}")
    print(f"     - 형식: {save_format}")
    print(f"     - X shape: {X_seq.shape}")
    print(f"     - y shape: {y_seq.shape}")
    
    return filepath


def load_sequence_dataset(filepath: str) -> Dict[str, Any]:
    """
    저장된 시퀀스 데이터셋 로드
    
    Parameters:
    -----------
    filepath : str
        파일 경로
        
    Returns:
    --------
    dict : 로드된 데이터
    """
    ext = Path(filepath).suffix
    
    if ext == ".npz":
        data = np.load(filepath, allow_pickle=True)
        return {
            'X': data['X'],
            'y': data['y'],
            'feature_names': data['feature_names'].tolist(),
            'target_cols': data['target_cols'].tolist(),
            'window_size': int(data['window_size']),
            'split_name': str(data['split_name'])
        }
    
    elif ext == ".pkl":
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    elif ext == ".csv":
        df = pd.read_csv(filepath)
        # CSV는 메타데이터가 없으므로 수동으로 분리 필요
        print("⚠️  CSV 형식은 메타데이터가 없습니다. X와 y를 수동으로 분리하세요.")
        return {'data': df}
    
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")


def save_model_and_metadata(
    model: Any,
    scaler: Any,
    top_features: list,
    metadata: Dict[str, Any],
    save_dir: str,
    model_name: str = "best_model"
) -> Dict[str, str]:
    """
    모델, 스케일러, 메타데이터 저장
    
    Parameters:
    -----------
    model : Any
        학습된 모델
    scaler : Any
        스케일러 (StandardScaler 등)
    top_features : list
        선택된 특성 리스트
    metadata : dict
        메타데이터 (mode, window_size, horizon 등)
    save_dir : str
        저장 디렉토리
    model_name : str
        모델 이름
        
    Returns:
    --------
    dict : 저장된 파일 경로들
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filepaths = {}
    
    # 모델 저장
    model_path = os.path.join(save_dir, f"{model_name}_{timestamp}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    filepaths['model'] = model_path
    print(f"  💾 모델 저장: {model_path}")
    
    # 스케일러 저장
    if scaler is not None:
        scaler_path = os.path.join(save_dir, f"scaler_{timestamp}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        filepaths['scaler'] = scaler_path
        print(f"  💾 스케일러 저장: {scaler_path}")
    
    # 특성 리스트 저장
    if top_features is not None:
        features_path = os.path.join(save_dir, f"features_{timestamp}.txt")
        with open(features_path, 'w') as f:
            f.write('\n'.join(top_features))
        filepaths['features'] = features_path
        print(f"  💾 특성 리스트 저장: {features_path}")
    
    # 메타데이터 저장
    metadata_path = os.path.join(save_dir, f"metadata_{timestamp}.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    filepaths['metadata'] = metadata_path
    print(f"  💾 메타데이터 저장: {metadata_path}")
    
    return filepaths


def save_all_results(
    result: Dict[str, Any],
    save_dir: str,
    save_predictions_flag: bool = True,
    save_sequences_flag: bool = True,
    save_model_flag: bool = True,
    sequence_format: str = "npz"
) -> Dict[str, Any]:
    """
    파이프라인 결과를 모두 저장
    
    Parameters:
    -----------
    result : dict
        파이프라인 실행 결과
    save_dir : str
        저장 디렉토리
    save_predictions_flag : bool
        예측값 저장 여부
    save_sequences_flag : bool
        시퀀스 데이터 저장 여부
    save_model_flag : bool
        모델 저장 여부
    sequence_format : str
        시퀀스 저장 형식 ('npz', 'pickle', 'csv')
        
    Returns:
    --------
    dict : 저장된 파일 경로들
    """
    saved_files = {
        'predictions': {},
        'sequences': {},
        'models': {}
    }
    
    print(f"\n{'='*60}")
    print("결과 저장 중...")
    print(f"{'='*60}")
    
    # 1. 예측값 저장
    if save_predictions_flag and 'fitted_models' in result and result['fitted_models'] and 'splits' in result:
        print("\n[1/3] 예측값 저장 중...")
        
        best_model_name = result['metric_table'].iloc[0]['model']
        fitted_model = result['fitted_models'][best_model_name]
        
        # fitted_model이 dict인 경우 (다중 타겟) - 각 타겟별로 예측
        if isinstance(fitted_model, dict):
            print("  다중 타겟 모델 - 타겟별 예측값 저장...")
            for split_name, (X, y) in result['splits'].items():
                # 각 타겟별로 예측
                y_pred_list = []
                for target_name in result['target_cols']:
                    model = fitted_model[target_name]
                    y_pred_single = model.predict(X)
                    y_pred_list.append(y_pred_single)
                
                # 다중 타겟 예측값 결합
                y_pred = np.column_stack(y_pred_list)
                
                filepath = save_predictions(
                    y_true=y,
                    y_pred=y_pred,
                    split_name=split_name,
                    target_cols=result['target_cols'],
                    save_dir=os.path.join(save_dir, 'predictions'),
                    index=y.index if hasattr(y, 'index') else None
                )
                saved_files['predictions'][split_name] = filepath
        else:
            # 단일 타겟 또는 일반 모델
            for split_name, (X, y) in result['splits'].items():
                y_pred = fitted_model.predict(X)
                
                filepath = save_predictions(
                    y_true=y,
                    y_pred=y_pred,
                    split_name=split_name,
                    target_cols=result['target_cols'],
                    save_dir=os.path.join(save_dir, 'predictions'),
                    index=y.index if hasattr(y, 'index') else None
                )
                saved_files['predictions'][split_name] = filepath
    elif save_predictions_flag:
        print("\n[1/3] 예측값 저장 건너뛰기 (fitted_models 없음)")
    
    # 2. 시퀀스 데이터 저장 (Sliding Window인 경우)
    if save_sequences_flag and 'X_seq' in result and 'y_seq' in result:
        print("\n[2/3] 시퀀스 데이터 저장 중...")
        
        # 원본 시퀀스 저장 (분할 전)
        filepath = save_sequence_dataset(
            X_seq=result['X_seq'],
            y_seq=result['y_seq'],
            split_name='all',
            feature_names=result['X_original'].columns.tolist() if hasattr(result['X_original'], 'columns') else [],
            target_cols=result['target_cols'],
            window_size=result.get('window_size', 24),
            save_dir=os.path.join(save_dir, 'sequences'),
            save_format=sequence_format
        )
        saved_files['sequences']['all'] = filepath
        
        # 분할된 시퀀스 저장 (선택사항)
        # Train/Valid/Test 각각 저장하려면 여기에 추가
    
    # 3. 모델 및 메타데이터 저장
    if save_model_flag and 'fitted_models' in result and result['fitted_models']:
        print("\n[3/3] 모델 및 메타데이터 저장 중...")
        
        best_model_name = result['metric_table'].iloc[0]['model']
        fitted_model = result['fitted_models'][best_model_name]
        
        metadata = {
            'mode': result.get('mode'),
            'target_cols': result.get('target_cols'),
            'window_size': result.get('window_size'),
            'horizon': result.get('horizon'),
            'stride': result.get('stride'),
            'best_model_name': best_model_name,
            'test_r2': result['metric_table'].iloc[0]['R2_mean'],
            'test_rmse': result['metric_table'].iloc[0]['RMSE_mean'],
            'n_samples_original': len(result.get('X_original', [])),
            'n_windows': len(result.get('X_seq', [])) if 'X_seq' in result else None,
            'n_features_original': result['X_original'].shape[1] if 'X_original' in result else None,
            'n_features_selected': len(result.get('top_features', [])) if result.get('top_features') else None,
            'is_multi_target': isinstance(fitted_model, dict)
        }
        
        # 다중 타겟인 경우 첫 번째 타겟 모델 저장 (또는 전체 dict 저장)
        if isinstance(fitted_model, dict):
            print(f"  다중 타겟 모델 - {len(fitted_model)}개 타겟별 모델 저장...")
            # 전체 dict를 저장
            model_to_save = fitted_model
        else:
            model_to_save = fitted_model
        
        filepaths = save_model_and_metadata(
            model=model_to_save,
            scaler=result.get('scaler'),
            top_features=result.get('top_features'),
            metadata=metadata,
            save_dir=os.path.join(save_dir, 'models'),
            model_name=best_model_name
        )
        saved_files['models'] = filepaths
    elif save_model_flag:
        print("\n[3/3] 모델 저장 건너뛰기 (fitted_models 없음)")
    
    print(f"\n{'='*60}")
    print("저장 완료!")
    print(f"{'='*60}")
    print(f"저장 위치: {save_dir}")
    
    return saved_files


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    """
    결과 저장 모듈 테스트
    """
    print("결과 저장 모듈 테스트\n")
    
    # 예시 데이터 생성
    np.random.seed(42)
    n_samples = 100
    window_size = 24
    n_features = 5
    n_targets = 2
    
    # 시퀀스 데이터
    X_seq = np.random.rand(n_samples, window_size, n_features)
    y_seq = np.random.rand(n_samples, n_targets)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    target_cols = ['target_0', 'target_1']
    
    # 1. 시퀀스 데이터 저장 (NPZ)
    print("1. NPZ 형식으로 저장:")
    filepath_npz = save_sequence_dataset(
        X_seq, y_seq, 'test', feature_names, target_cols, window_size,
        save_dir='test_results/sequences',
        save_format='npz'
    )
    
    # 2. 시퀀스 데이터 로드
    print("\n2. NPZ 파일 로드:")
    loaded_data = load_sequence_dataset(filepath_npz)
    print(f"   로드된 X shape: {loaded_data['X'].shape}")
    print(f"   로드된 y shape: {loaded_data['y'].shape}")
    
    # 3. 예측값 저장
    print("\n3. 예측값 저장:")
    y_true = pd.DataFrame(y_seq, columns=target_cols)
    y_pred = y_seq + np.random.randn(*y_seq.shape) * 0.1
    
    filepath_pred = save_predictions(
        y_true, y_pred, 'test', target_cols,
        save_dir='test_results/predictions'
    )
    
    print("\n✅ 결과 저장 모듈 테스트 완료!")
    print(f"   테스트 파일 위치: test_results/")
