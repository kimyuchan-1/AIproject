"""
모델 정의 및 래퍼 모듈
다양한 회귀 모델 정의 및 멀티아웃풋 래퍼
Optuna 하이퍼파라미터 최적화 지원
"""

from typing import Dict, Any, Optional
import optuna
from optuna.samplers import TPESampler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
import numpy as np
import pandas as pd


def build_model_zoo(random_state: int = 42) -> Dict[str, Any]:
    """
    사용 가능한 모델들의 딕셔너리 생성
    
    Parameters:
    -----------
    random_state : int
        랜덤 시드
        
    Returns:
    --------
    dict : 모델 딕셔너리
    """
    zoo = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.001, random_state=random_state, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state, max_iter=5000),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "HistGBR": HistGradientBoostingRegressor(
            random_state=random_state, learning_rate=0.05, max_iter=500
        ),
    }
    return zoo


def wrap_multioutput_if_needed(model: Any, 
                               y: pd.DataFrame) -> Any:
    """
    필요시 멀티아웃풋 래퍼 적용
    
    Parameters:
    -----------
    model : sklearn model
        래핑할 모델
    y : DataFrame
        타겟 데이터
        
    Returns:
    --------
    model : 래핑된 모델 또는 원본 모델
    """
    if y.shape[1] <= 1:
        return model

    # HistGBR은 래퍼가 필요함
    if isinstance(model, HistGradientBoostingRegressor):
        return MultiOutputRegressor(model)
    return model



# ========================================
# Optuna 하이퍼파라미터 최적화
# ========================================

class OptunaModelWrapper:
    """
    Optuna를 사용한 하이퍼파라미터 최적화 래퍼
    
    TimeSeriesSplit 교차 검증을 사용하여 최적 파라미터 탐색
    """
    
    def __init__(self, 
                 model_name: str, 
                 cv_splits: int = 3, 
                 n_trials: int = 50, 
                 random_state: int = 42):
        """
        Parameters:
        -----------
        model_name : str
            모델 이름 ('Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'HistGBR', 'XGBoost')
        cv_splits : int
            TimeSeriesSplit 분할 수
        n_trials : int
            Optuna 시도 횟수
        random_state : int
            랜덤 시드
        """
        self.model_name = model_name
        self.cv_splits = cv_splits
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_model_: Optional[Any] = None
        self.study_: Optional[optuna.Study] = None
    
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """모델별 하이퍼파라미터 제안"""
        param_configs = {
            'Ridge': {
                'alpha': ('float', 0.01, 100.0, True)
            },
            'Lasso': {
                'alpha': ('float', 0.0001, 10.0, True),
                'max_iter': ('int', 1000, 10000)
            },
            'ElasticNet': {
                'alpha': ('float', 0.0001, 10.0, True),
                'l1_ratio': ('float', 0.0, 1.0, False),
                'max_iter': ('int', 1000, 10000)
            },
            'RandomForest': {
                'n_estimators': ('int', 100, 500),
                'max_depth': ('int', 5, 30),
                'min_samples_split': ('int', 2, 20),
                'min_samples_leaf': ('int', 1, 10)
            },
            'HistGBR': {
                'learning_rate': ('float', 0.01, 0.3, True),
                'max_iter': ('int', 100, 1000),
                'max_depth': ('int', 3, 15),
                'min_samples_leaf': ('int', 10, 100)
            },
            'XGBoost': {
                'learning_rate': ('float', 0.01, 0.3, True),
                'max_depth': ('int', 3, 10),
                'min_child_weight': ('int', 1, 10),
                'subsample': ('float', 0.6, 1.0, False),
                'colsample_bytree': ('float', 0.6, 1.0, False),
                'gamma': ('float', 0.0, 5.0, False)
            }
        }
        
        config = param_configs.get(self.model_name, {})
        params = {}
        
        for param_name, param_spec in config.items():
            if param_spec[0] == 'float':
                log = param_spec[3] if len(param_spec) > 3 else False
                params[param_name] = trial.suggest_float(param_name, param_spec[1], param_spec[2], log=log)
            elif param_spec[0] == 'int':
                params[param_name] = trial.suggest_int(param_name, param_spec[1], param_spec[2])
        
        return params
    
    def _create_model(self, params: Dict[str, Any]) -> Any:
        """파라미터로 모델 생성"""
        model_classes = {
            'Ridge': Ridge,
            'Lasso': Lasso,
            'ElasticNet': ElasticNet,
            'RandomForest': RandomForestRegressor,
            'HistGBR': HistGradientBoostingRegressor,
            'XGBoost': xgb.XGBRegressor
        }
        
        model_class = model_classes.get(self.model_name)
        if model_class is None:
            raise ValueError(f"지원하지 않는 모델: {self.model_name}")
        
        # 공통 파라미터 추가
        if self.model_name in ['Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'HistGBR']:
            params['random_state'] = self.random_state
        
        if self.model_name == 'RandomForest':
            params['n_jobs'] = -1
        elif self.model_name == 'XGBoost':
            params['n_estimators'] = 100
            params['random_state'] = self.random_state
            params['n_jobs'] = -1
            params['random_state'] = self.random_state
            params['n_jobs'] = -1
        
        return model_class(**params)
    
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective 함수"""
        params = self._suggest_params(trial)
        model = self._create_model(params)
        
        # TimeSeriesSplit 교차 검증
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
        
        return -scores.mean()  # MSE를 최소화
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'OptunaModelWrapper':
        """최적 하이퍼파라미터 탐색 및 모델 학습"""
        # Optuna study 생성
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = TPESampler(seed=self.random_state)
        self.study_ = optuna.create_study(direction='minimize', sampler=sampler)
        
        # 최적화 실행
        self.study_.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=False
        )
        
        # 최적 파라미터로 모델 학습
        self.best_params_ = self.study_.best_params
        self.best_model_ = self._create_model(self.best_params_)
        self.best_model_.fit(X, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측"""
        if self.best_model_ is None:
            raise ValueError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")
        return self.best_model_.predict(X)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """파라미터 반환 (sklearn 호환)"""
        return {
            'model_name': self.model_name,
            'cv_splits': self.cv_splits,
            'n_trials': self.n_trials,
            'random_state': self.random_state
        }
    
    def set_params(self, **params: Any) -> 'OptunaModelWrapper':
        """파라미터 설정 (sklearn 호환)"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


def build_model_zoo_with_optuna(cv_splits: int = 3, 
                                n_trials: int = 50, 
                                random_state: int = 42) -> Dict[str, Any]:
    """
    Optuna 하이퍼파라미터 최적화를 사용하는 모델 Zoo
    
    Parameters:
    -----------
    cv_splits : int
        TimeSeriesSplit 분할 수
    n_trials : int
        Optuna 시도 횟수
    random_state : int
        랜덤 시드
        
    Returns:
    --------
    dict : 모델 딕셔너리
    """
    zoo = {
        "LinearRegression": LinearRegression(),  # 파라미터 없음
        "Ridge": OptunaModelWrapper('Ridge', cv_splits, n_trials, random_state),
        "Lasso": OptunaModelWrapper('Lasso', cv_splits, n_trials, random_state),
        "ElasticNet": OptunaModelWrapper('ElasticNet', cv_splits, n_trials, random_state),
        "RandomForest": OptunaModelWrapper('RandomForest', cv_splits, n_trials, random_state),
        "HistGBR": OptunaModelWrapper('HistGBR', cv_splits, n_trials, random_state),
        "XGBoost": OptunaModelWrapper('XGBoost', cv_splits, n_trials, random_state),
    }
    return zoo
