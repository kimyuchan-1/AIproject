"""
데이터 로드 모듈
CSV 파일 로드 및 시간 인덱스 설정
"""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def load_raw_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    원본 데이터 로드
    
    Parameters:
    -----------
    data_dir : Path
        데이터 디렉토리 경로
        
    Returns:
    --------
    dict : 데이터프레임 딕셔너리
    """
    dfs = {}
    
    # FLOW 데이터
    flow_path = data_dir / "FLOW_Actual.csv"
    if flow_path.exists():
        dfs["flow"] = pd.read_csv(flow_path)
    
    # TMS 데이터
    tms_path = data_dir / "TMS_Actual.csv"
    if tms_path.exists():
        dfs["tms"] = pd.read_csv(tms_path)
    
    # AWS 데이터 (suffix 추가)
    for station_id in ["368", "541", "569"]:
        aws_path = data_dir / f"AWS_{station_id}.csv"
        if aws_path.exists():
            df = pd.read_csv(aws_path)
            # datetime 컬럼 제외하고 suffix 추가
            if "datetime" in df.columns:
                time_col = df["datetime"]
                df = df.drop(columns=["datetime"])
                df = df.add_suffix(f"_{station_id}")
                df["datetime"] = time_col
            else:
                df = df.add_suffix(f"_{station_id}")
            dfs[f"aws_{station_id}"] = df
    
    # Weather 데이터는 사용하지 않음
    # weather_path = data_dir / "Weather.csv"
    # if weather_path.exists():
    #     dfs["weather"] = pd.read_csv(weather_path)
    
    return dfs


def set_datetime_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    시간 컬럼을 DatetimeIndex로 설정
    
    Parameters:
    -----------
    df : DataFrame
        입력 데이터프레임
    time_col : str
        시간 컬럼명
        
    Returns:
    --------
    DataFrame : DatetimeIndex가 설정된 데이터프레임
    """
    out = df.copy()
    
    # 시간 컬럼이 존재하는지 확인
    if time_col not in out.columns:
        raise ValueError(f"시간 컬럼 '{time_col}'이 데이터프레임에 없습니다. 사용 가능한 컬럼: {out.columns.tolist()}")
    
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col])
    out = out.set_index(time_col).sort_index()
    return out


def align_time_index(dfs: Dict[str, pd.DataFrame], freq: str = "1min") -> Dict[str, pd.DataFrame]:
    """
    시간 인덱스 정합 (1분 간격)
    
    Parameters:
    -----------
    dfs : dict
        데이터프레임 딕셔너리
    freq : str
        시간 간격
        
    Returns:
    --------
    dict : 정합된 데이터프레임 딕셔너리
    """
    aligned_dfs = {}
    
    # 전체 시간 범위 찾기
    min_time = min(df.index.min() for df in dfs.values() if len(df) > 0)
    max_time = max(df.index.max() for df in dfs.values() if len(df) > 0)
    
    # 공통 시간 인덱스 생성
    common_index = pd.date_range(start=min_time, end=max_time, freq=freq)
    
    # 각 데이터프레임을 공통 인덱스에 맞춤
    for name, df in dfs.items():
        aligned_dfs[name] = df.reindex(common_index)
    
    return aligned_dfs


def merge_dataframes(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    여러 데이터프레임 병합
    
    Parameters:
    -----------
    dfs : dict
        데이터프레임 딕셔너리
        
    Returns:
    --------
    DataFrame : 병합된 데이터프레임
    """
    if not dfs:
        raise ValueError("병합할 데이터프레임이 없습니다")
    
    # 첫 번째 데이터프레임부터 시작
    result = list(dfs.values())[0].copy()
    
    # 나머지 데이터프레임 병합
    for df in list(dfs.values())[1:]:
        result = result.join(df, how="outer")
    
    return result
