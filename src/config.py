import sys
from pathlib import Path
import torch

# 프로젝트 루트를 sys.path에 추가
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# ====== 경로 설정 ======
MODEL_DIR = BASE_DIR / "model"
SAVE_DIR = MODEL_DIR / "save"
DATA_DIR = BASE_DIR / "data"
FEATURE_DIR = DATA_DIR / "features" / "save"

# ====== 디바이스 설정 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 모델 설정 (노트북 MODE_CONFIGS 기준) ======
# deep_head=True: 4-layer head (flow/toc/ss), False: 3-layer head (tn/tp/flux/ph)
# use_attention=True: flux, flow 만 Attention 사용
TMS_TARGETS = {
    "toc":  {"target_col": "TOC_VU",  "hidden_size": 512, "num_layers": 1, "dropout": 0.2, "use_attention": False, "deep_head": True},
    "ss":   {"target_col": "SS_VU",   "hidden_size": 256, "num_layers": 2, "dropout": 0.2, "use_attention": False, "deep_head": True},
    "tn":   {"target_col": "TN_VU",   "hidden_size": 512, "num_layers": 4, "dropout": 0.2, "use_attention": False, "deep_head": False},
    "tp":   {"target_col": "TP_VU",   "hidden_size": 384, "num_layers": 1, "dropout": 0.1, "use_attention": False, "deep_head": False},
    "flux": {"target_col": "FLUX_VU", "hidden_size": 512, "num_layers": 4, "dropout": 0.2, "use_attention": True,  "deep_head": False},
    "ph":   {"target_col": "PH_VU",   "hidden_size": 512, "num_layers": 1, "dropout": 0.1, "use_attention": False, "deep_head": False},
}

FLOW_CONFIG = {"hidden_size": 512, "num_layers": 2, "dropout": 0.2, "use_attention": False, "deep_head": True}
