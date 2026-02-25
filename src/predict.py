import torch
import torch.nn as nn

from .config import device


def autoregressive_predict(
    input_tensor: torch.Tensor,
    n_steps: int,
    pred_model: nn.Module,
    y_scaler_obj,
    target_feat_idx: int | None = None,
) -> list[float]:
    """
    Autoregressive 방식으로 다중 시점 예측

    Args:
        input_tensor: (1, 48, n_features) 초기 입력
        n_steps: 예측할 시점 수 (1h=2, 3h=6, 12h=24)
        pred_model: 예측에 사용할 모델
        y_scaler_obj: 역정규화에 사용할 y 스케일러
        target_feat_idx: 특성 벡터에서 타겟 lag 특성의 인덱스 (None이면 업데이트 안 함)

    Returns:
        predictions: 각 시점의 예측값 (역정규화된 실제 값)
    """
    predictions = []
    current_input = input_tensor.clone()

    with torch.no_grad():
        for _ in range(n_steps):
            pred_scaled = pred_model(current_input.to(device))  # (1, 1)

            pred_original = y_scaler_obj.inverse_transform(pred_scaled.cpu().numpy())
            predictions.append(float(pred_original[0, 0]))

            # 마지막 시점의 특성 벡터 복사 후 윈도우 시프트
            last_features = current_input[0, -1, :].cpu().numpy()

            if target_feat_idx is not None:
                last_features[target_feat_idx] = pred_scaled.cpu().numpy()[0, 0]

            new_features = torch.tensor(last_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            current_input = torch.cat([current_input[:, 1:, :], new_features], dim=1)

    return predictions


def find_target_lag_idx(feature_names: list[str], target_col: str) -> int | None:
    """추천 특성 목록에서 타겟의 가장 작은 lag 특성 인덱스를 찾음"""
    prefix = f"{target_col}_tlag_"
    tlag_features = [(i, f) for i, f in enumerate(feature_names) if f.startswith(prefix)]
    if not tlag_features:
        return None
    tlag_features.sort(key=lambda x: int(x[1].split("_")[-1]))
    return tlag_features[0][0]
