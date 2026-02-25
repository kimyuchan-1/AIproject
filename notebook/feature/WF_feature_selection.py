import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from tqdm import tqdm

class WalkForwardFeatureSelector:
    """
    Walk-Forward Validation 기반 특성 선택
    - 각 폴드에서 특성 중요도 계산
    - 폴드별 최적 특성 세트 도출
    - 특성 안정성 평가
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_splits: int = 5,
        train_size: int = 350,      # 600 시간 (약 25일)
        val_size: int = 100,        # 150 시간 (약 6일)
        test_size: int = 100,       # 150 시간 (약 6일)
        window_step: int = 100,     # 200 시간마다 이동
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        X : np.ndarray
            특성 행렬 (n_samples, n_features)
        y : np.ndarray
            타겟 벡터 (n_samples,) 또는 (n_samples, n_outputs)
        feature_names : List[str]
            특성 이름 리스트
        n_splits : int
            Walk-forward 폴드 수
        train_size : int
            학습 세트 크기 (시간 단위)
        val_size : int
            검증 세트 크기
        test_size : int
            테스트 세트 크기
        window_step : int
            윈도우 이동 크기
        random_state : int
            난수 시드
        """
        # ====== 데이터 검증 ======
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)

        # NaN/Inf 체크
        if np.any(np.isnan(self.X)) or np.any(np.isinf(self.X)):
            n_nan = np.sum(np.isnan(self.X))
            n_inf = np.sum(np.isinf(self.X))
            raise ValueError(
                f"X contains invalid values: {n_nan} NaN, {n_inf} Inf. "
                f"Please clean your data before feature selection."
            )

        if np.any(np.isnan(self.y)) or np.any(np.isinf(self.y)):
            n_nan = np.sum(np.isnan(self.y))
            n_inf = np.sum(np.isinf(self.y))
            raise ValueError(
                f"y contains invalid values: {n_nan} NaN, {n_inf} Inf. "
                f"Please clean your data before feature selection."
            )

        if self.y.ndim == 1:
            self.y = self.y[:, None]

        self.feature_names = np.array(feature_names)
        self.n_features = len(feature_names)
        self.n_samples = len(X)

        # 특성 이름과 X shape 일치 확인
        if self.n_features != self.X.shape[1]:
            raise ValueError(
                f"Feature names length ({self.n_features}) does not match "
                f"X columns ({self.X.shape[1]})"
            )

        # 샘플 수 일치 확인
        if len(self.X) != len(self.y):
            raise ValueError(
                f"X and y have different sample counts: "
                f"X={len(self.X)}, y={len(self.y)}"
            )

        self.n_splits = n_splits
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.window_step = window_step
        self.random_state = random_state

        # 전역 랜덤 시드 설정 (재현성 보장)
        np.random.seed(self.random_state)

        self.fold_results = {}
        self.feature_importance_folds = []
        self.fold_metrics = []
        self.best_features_per_fold = []
        
    def _create_splits(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Walk-forward 폴드 생성"""
        splits = []
        total_window = self.train_size + self.val_size + self.test_size
        
        start_idx = 0
        fold_count = 0
        
        while start_idx + total_window <= self.n_samples and fold_count < self.n_splits:
            # Train / Val / Test 인덱스
            train_end = start_idx + self.train_size
            val_end = train_end + self.val_size
            test_end = val_end + self.test_size
            
            train_idx = np.arange(start_idx, train_end)
            val_idx = np.arange(train_end, val_end)
            test_idx = np.arange(val_end, test_end)
            
            splits.append((train_idx, val_idx, test_idx))
            
            # 다음 폴드로 이동
            start_idx += self.window_step
            fold_count += 1
        
        if len(splits) == 0:
            raise ValueError(
                f"데이터가 부족합니다. 필요한 최소 크기: "
                f"{total_window}, 실제: {self.n_samples}"
            )
        
        self.n_splits = len(splits)
        print(f"✓ {self.n_splits}개 폴드 생성")
        print(f"  총 윈도우 크기: {total_window}")
        print(f"  Train: {self.train_size}, Val: {self.val_size}, Test: {self.test_size}")
        
        return splits
    
    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = "rf"
    ):
        """특성 중요도 계산용 모델 학습"""
        # ====== 데이터 검증 ======
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Empty training data provided")

        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            raise ValueError("X_train contains NaN or Inf values")

        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            raise ValueError("y_train contains NaN or Inf values")

        if model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0
            )
        elif model_type == "gb":
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state,
                verbose=0
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # 다중 출력 처리
        if y_train.ndim > 1 and y_train.shape[1] > 1:
            # 각 출력별로 독립적으로 학습
            models = []
            for i in range(y_train.shape[1]):
                m = model.__class__(**model.get_params())
                m.fit(X_train, y_train[:, i])
                models.append(m)
            return models
        else:
            y_train_1d = y_train.ravel()
            model.fit(X_train, y_train_1d)
            return model
    
    def _get_feature_importance(self, model) -> np.ndarray:
        """모델에서 특성 중요도 추출"""
        if isinstance(model, list):
            # 다중 출력: 각 모델의 중요도 평균
            importances = []
            for m in model:
                importances.append(m.feature_importances_)
            return np.mean(importances, axis=0)
        else:
            return model.feature_importances_
    
    def _evaluate_model(
        self, 
        model, 
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """모델 성능 평가"""
        if isinstance(model, list):
            # 다중 출력
            preds = np.column_stack([m.predict(X_val) for m in model])
            y_val_eval = y_val
        else:
            preds = model.predict(X_val)
            y_val_eval = y_val.ravel()
        
        mse = mean_squared_error(y_val_eval, preds)
        mae = mean_absolute_error(y_val_eval, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val_eval, preds)
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }
    
    def _select_features_by_threshold(
        self,
        importances: np.ndarray,
        threshold: float = 0.80
    ) -> np.ndarray:
        """
        중요도 임계값으로 특성 선택

        Parameters
        ----------
        importances : np.ndarray
            특성 중요도 배열
        threshold : float
            누적 중요도 임계값 (기본: 0.80 = 80%)

        Returns
        -------
        np.ndarray
            선택된 특성 인덱스
        """
        # ====== 입력 검증 ======
        if np.any(np.isnan(importances)) or np.any(np.isinf(importances)):
            raise ValueError("Feature importances contain NaN or Inf values")

        if np.sum(importances) == 0:
            raise ValueError("Sum of feature importances is zero")

        # 정렬된 인덱스 (내림차순)
        sorted_idx = np.argsort(importances)[::-1]

        # 누적 중요도 계산
        cum_importance = np.cumsum(importances[sorted_idx])

        # 안전한 정규화 (분모가 0인 경우 방지)
        total_importance = cum_importance[-1]
        if total_importance > 0:
            cum_importance = cum_importance / total_importance
        else:
            # 모든 중요도가 0인 경우: 상위 10개 또는 전체 특성의 10% 선택
            n_features_to_select = max(1, min(10, len(importances) // 10))
            selected_idx = sorted_idx[:n_features_to_select]
            print(f"  ⚠️ Warning: All feature importances are zero. "
                  f"Selecting top {n_features_to_select} features by default.")
            return np.sort(selected_idx)

        # 누적 중요도 threshold 달성하는 특성 수 결정
        mask = cum_importance >= threshold
        if np.any(mask):
            n_features_to_select = np.argmax(mask) + 1
        else:
            # threshold를 달성하지 못하는 경우: 모든 특성 선택
            n_features_to_select = len(importances)
            print(f"  ⚠️ Warning: Cannot reach {threshold*100:.0f}% cumulative importance. "
                  f"Selecting all {n_features_to_select} features.")

        # 최소 1개 이상의 특성 선택 보장
        n_features_to_select = max(1, n_features_to_select)
        selected_idx = sorted_idx[:n_features_to_select]

        return np.sort(selected_idx)
    
    def run(
        self,
        model_type: str = "rf",
        importance_threshold: float = 0.80,
        verbose: bool = True
    ) -> Dict:
        """
        Walk-Forward Validation 실행

        Parameters
        ----------
        model_type : str
            'rf' (Random Forest) 또는 'gb' (Gradient Boosting)
        importance_threshold : float
            누적 중요도 임계값 (기본: 0.80 = 80%)
        verbose : bool
            상세 출력 여부

        Returns
        -------
        Dict
            폴드별 결과, 특성 중요도, 추천 특성
        """
        # ====== 입력 검증 ======
        if model_type not in ["rf", "gb"]:
            raise ValueError(f"model_type must be 'rf' or 'gb', got '{model_type}'")

        if not 0.0 < importance_threshold <= 1.0:
            raise ValueError(
                f"importance_threshold must be in (0, 1], got {importance_threshold}"
            )

        splits = self._create_splits()
        
        print(f"\n{'='*70}")
        print(f"Walk-Forward Validation 시작 (Model: {model_type.upper()})")
        print(f"{'='*70}\n")
        
        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(tqdm(
            splits, 
            desc="Fold Progress",
            total=self.n_splits
        )):
            X_train, y_train = self.X[train_idx], self.y[train_idx]
            X_val, y_val = self.X[val_idx], self.y[val_idx]
            # X_test, y_test = self.X[test_idx], self.y[test_idx]  # 현재 미사용 (향후 확장용)
            
            # 스케일링 (안전성 개선)
            scaler = StandardScaler()
            try:
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                # X_test_scaled = scaler.transform(X_test)  # 현재 미사용 (향후 확장용)

                # 스케일링 후 NaN/Inf 체크
                if (np.any(np.isnan(X_train_scaled)) or np.any(np.isinf(X_train_scaled)) or
                    np.any(np.isnan(X_val_scaled)) or np.any(np.isinf(X_val_scaled))):
                    raise ValueError(
                        f"Fold {fold_idx + 1}: Scaling produced NaN or Inf values. "
                        f"Check for constant features or extreme values."
                    )
            except Exception as e:
                print(f"\n❌ Error in fold {fold_idx + 1} during scaling: {e}")
                print(f"  X_train stats: min={X_train.min():.2e}, max={X_train.max():.2e}, "
                      f"std={X_train.std():.2e}")
                raise
            
            # 모델 학습
            model = self._train_model(X_train_scaled, y_train, model_type=model_type)
            
            # 특성 중요도 추출
            importance = self._get_feature_importance(model)
            self.feature_importance_folds.append(importance)
            
            # 특성 선택 (안전성 개선)
            try:
                selected_idx = self._select_features_by_threshold(
                    importance,
                    threshold=importance_threshold
                )
                self.best_features_per_fold.append(selected_idx)
            except Exception as e:
                print(f"\n❌ Error in fold {fold_idx + 1} during feature selection: {e}")
                print(f"  Feature importance stats: min={importance.min():.2e}, "
                      f"max={importance.max():.2e}, sum={importance.sum():.2e}")
                raise
            
            # 성능 평가 (모든 특성 vs 선택된 특성)
            val_metrics_all = self._evaluate_model(model, X_val_scaled, y_val)
            
            # 선택된 특성으로만 재학습 및 평가
            if len(selected_idx) > 0:
                X_train_sel = X_train_scaled[:, selected_idx]
                X_val_sel = X_val_scaled[:, selected_idx]
                model_sel = self._train_model(X_train_sel, y_train, model_type=model_type)
                val_metrics_sel = self._evaluate_model(model_sel, X_val_sel, y_val)
            else:
                # 선택된 특성이 없는 경우 (예외 상황)
                print(f"\n⚠️ Warning: No features selected in fold {fold_idx + 1}")
                val_metrics_sel = val_metrics_all.copy()
            
            fold_result = {
                "fold": fold_idx + 1,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "importance": importance,
                "selected_features": selected_idx,
                "n_selected": len(selected_idx),
                "val_metrics_all": val_metrics_all,
                "val_metrics_selected": val_metrics_sel,
                "feature_reduction": 1.0 - len(selected_idx) / self.n_features
            }
            
            self.fold_results[fold_idx] = fold_result
            self.fold_metrics.append({
                "fold": fold_idx + 1,
                **{f"all_{k}": v for k, v in val_metrics_all.items()},
                **{f"sel_{k}": v for k, v in val_metrics_sel.items()},
            })
            
            if verbose:
                print(f"\n[Fold {fold_idx + 1}/{self.n_splits}]")
                print(f"  선택된 특성: {len(selected_idx)}/{self.n_features} "
                      f"({fold_result['feature_reduction']*100:.1f}% 감소)")
                print(f"  Val RMSE (All): {val_metrics_all['rmse']:.4f}")
                print(f"  Val RMSE (Selected): {val_metrics_sel['rmse']:.4f}")
                print(f"  성능 변화: {((val_metrics_sel['rmse']/val_metrics_all['rmse']-1)*100):+.2f}%")
        
        return self._summarize_results()
    
    def _summarize_results(self) -> Dict:
        """결과 요약"""
        feature_importance_mean = np.mean(self.feature_importance_folds, axis=0)
        feature_importance_std = np.std(self.feature_importance_folds, axis=0)
        
        # 안정적인 특성 선택 (모든 폴드에서 선택됨)
        feature_selection_count = np.zeros(self.n_features)
        for selected_idx in self.best_features_per_fold:
            feature_selection_count[selected_idx] += 1
        
        stable_features = np.where(
            feature_selection_count == self.n_splits
        )[0]
        
        # 상위 특성 (평균 중요도 기준)
        top_k = min(50, self.n_features)
        top_features = np.argsort(feature_importance_mean)[::-1][:top_k]
        
        # 메트릭 집계
        metrics_df = pd.DataFrame(self.fold_metrics)
        
        summary = {
            "n_folds": self.n_splits,
            "n_features_original": self.n_features,
            "n_selected_mean": np.mean([len(idx) for idx in self.best_features_per_fold]),
            "n_selected_std": np.std([len(idx) for idx in self.best_features_per_fold]),
            "feature_importance_mean": feature_importance_mean,
            "feature_importance_std": feature_importance_std,
            "stable_features": stable_features,
            "top_features": top_features,
            "all_metrics_summary": metrics_df.describe(),
        }
        
        return summary
    
    def get_recommended_features(self, stability_ratio: float = 0.6) -> np.ndarray:
        """
        추천 특성 세트 반환
        
        Parameters
        ----------
        stability_ratio : float
            선택 안정성 임계값 (기본: 60% 이상의 폴드에서 선택)
        
        Returns
        -------
        np.ndarray
            추천 특성 인덱스
        """
        feature_selection_count = np.zeros(self.n_features)
        for selected_idx in self.best_features_per_fold:
            feature_selection_count[selected_idx] += 1
        
        stability_threshold = self.n_splits * stability_ratio
        recommended = np.where(feature_selection_count >= stability_threshold)[0]
        
        return np.sort(recommended)
    
    def plot_feature_importance(self, top_k: int = 30, figsize: Tuple = (12, 8)):
        """특성 중요도 시각화"""
        feature_importance_mean = np.mean(self.feature_importance_folds, axis=0)
        feature_importance_std = np.std(self.feature_importance_folds, axis=0)
        
        sorted_idx = np.argsort(feature_importance_mean)[::-1][:top_k]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(sorted_idx))
        ax.barh(
            y_pos,
            feature_importance_mean[sorted_idx],
            xerr=feature_importance_std[sorted_idx],
            capsize=3,
            alpha=0.7,
            color='steelblue'
        )
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.feature_names[sorted_idx])
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance (Mean ± Std)")
        ax.set_title(f"Top {top_k} Features (Walk-Forward Validation)")
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def plot_feature_stability(self, figsize: Tuple = (14, 6)):
        """특성 선택 안정성 시각화"""
        feature_selection_count = np.zeros(self.n_features)
        for selected_idx in self.best_features_per_fold:
            feature_selection_count[selected_idx] += 1
        
        stability_ratio = feature_selection_count / self.n_splits * 100
        
        sorted_idx = np.argsort(stability_ratio)[::-1][:50]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['green' if v >= 60 else 'orange' if v >= 40 else 'red' 
                  for v in stability_ratio[sorted_idx]]
        
        ax.barh(
            np.arange(len(sorted_idx)),
            stability_ratio[sorted_idx],
            color=colors,
            alpha=0.7
        )
        
        ax.set_yticks(np.arange(len(sorted_idx)))
        ax.set_yticklabels(self.feature_names[sorted_idx])
        ax.invert_yaxis()
        ax.set_xlabel("Selection Stability (%)")
        ax.set_title("Feature Selection Stability (Across Folds)")
        ax.axvline(x=60, color='green', linestyle='--', alpha=0.5, label='Stable (60%+)')
        ax.axvline(x=40, color='orange', linestyle='--', alpha=0.5, label='Moderate (40%+)')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def plot_metrics_across_folds(self, figsize: Tuple = (12, 5)):
        """폴드별 메트릭 변화"""
        metrics_df = pd.DataFrame(self.fold_metrics)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # RMSE 비교
        axes[0].plot(metrics_df['fold'], metrics_df['all_rmse'], 
                    marker='o', label='All Features', linewidth=2)
        axes[0].plot(metrics_df['fold'], metrics_df['sel_rmse'], 
                    marker='s', label='Selected Features', linewidth=2)
        axes[0].set_xlabel("Fold")
        axes[0].set_ylabel("RMSE")
        axes[0].set_title("RMSE Across Folds")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # R² 비교
        axes[1].plot(metrics_df['fold'], metrics_df['all_r2'], 
                    marker='o', label='All Features', linewidth=2)
        axes[1].plot(metrics_df['fold'], metrics_df['sel_r2'], 
                    marker='s', label='Selected Features', linewidth=2)
        axes[1].set_xlabel("Fold")
        axes[1].set_ylabel("R²")
        axes[1].set_title("R² Score Across Folds")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig, axes
    
    def save_results(self, save_dir: str = "results/feature_selection"):
        """결과 저장"""
        from pathlib import Path
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 특성 중요도 저장
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance_mean": np.mean(self.feature_importance_folds, axis=0),
            "importance_std": np.std(self.feature_importance_folds, axis=0),
        }).sort_values("importance_mean", ascending=False)
        importance_df.to_csv(save_path / "feature_importance.csv", index=False)
        
        # 메트릭 저장
        metrics_df = pd.DataFrame(self.fold_metrics)
        metrics_df.to_csv(save_path / "fold_metrics.csv", index=False)
        
        # 추천 특성 저장
        recommended = self.get_recommended_features()
        recommended_df = pd.DataFrame({
            "feature_name": self.feature_names[recommended],
            "feature_index": recommended
        })
        recommended_df.to_csv(save_path / "recommended_features.csv", index=False)
        
        print(f"✓ 결과 저장됨: {save_path}")
        
        return save_path