import sys
import pickle
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from .config import (
    SAVE_DIR,
    FEATURE_DIR,
    TMS_TARGETS,
    FLOW_CONFIG,
    device,
)
from .models import LSTMRegressor, StandardScaler

# pickle 호환성: 스케일러 로딩 전에 __main__에 StandardScaler 등록
sys.modules['__main__'].StandardScaler = StandardScaler


# ====== 로딩 유틸 함수 ======
def load_feature_names(csv_path: Path) -> list[str]:
    feat_df = pd.read_csv(csv_path)
    return feat_df["feature_name"].tolist()


def load_scaler(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_model_weights(model: nn.Module, ckpt_path: Path, map_location=None) -> nn.Module:
    if map_location is None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ====== 모델 및 스케일러 로드 ======
print("Loading models and scalers...")

# --- TMS 모델 로드 (6개) ---
tms_models: dict = {}
tms_x_scalers: dict = {}
tms_y_scalers: dict = {}
tms_features: dict = {}

for target_name, cfg in TMS_TARGETS.items():
    feature_names = load_feature_names(FEATURE_DIR / f"{target_name}_recommended_features.csv")
    n_features = len(feature_names)
    tms_features[target_name] = feature_names

    mdl = LSTMRegressor(
        n_features=n_features,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        out_size=1,
        use_attention=cfg["use_attention"],
        deep_head=cfg.get("deep_head", False),
    ).to(device)

    tms_models[target_name] = load_model_weights(mdl, SAVE_DIR / f"{target_name}_lstm_model.pth", device)
    tms_x_scalers[target_name] = load_scaler(SAVE_DIR / f"X_scaler_{target_name}.pkl")
    tms_y_scalers[target_name] = load_scaler(SAVE_DIR / f"y_scaler_{target_name}.pkl")

    print(f"  Loaded {target_name}: {n_features} features, attention={cfg['use_attention']}")

# --- Flow 모델 로드 ---
FLOW_FEATURE_NAMES = load_feature_names(FEATURE_DIR / "flow_recommended_features.csv")
FLOW_N_FEATURES = len(FLOW_FEATURE_NAMES)

flow_model = LSTMRegressor(
    n_features=FLOW_N_FEATURES,
    hidden_size=FLOW_CONFIG["hidden_size"],
    num_layers=FLOW_CONFIG["num_layers"],
    dropout=FLOW_CONFIG["dropout"],
    out_size=1,
    use_attention=FLOW_CONFIG["use_attention"],
    deep_head=FLOW_CONFIG.get("deep_head", False),
).to(device)

flow_model = load_model_weights(flow_model, SAVE_DIR / "flow_lstm_model.pth", device)
flow_x_scaler = load_scaler(SAVE_DIR / "X_scaler_flow.pkl")
flow_y_scaler = load_scaler(SAVE_DIR / "y_scaler_flow.pkl")

print(f"  Loaded flow: {FLOW_N_FEATURES} features")
print("All models loaded successfully")
